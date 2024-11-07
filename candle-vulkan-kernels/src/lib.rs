use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, RwLock};
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::{Device, Queue};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::shader::ShaderModule;
use vulkano::sync::{self, GpuFuture};
use crate::VulkanKernelError::CommandError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Affine,
    Indexing,
    Unary,
    Binary,
    Ternary,
    Cast,
    Reduce,
    Conv,
    Random,
    Quantized,
    Sort,
}

#[derive(thiserror::Error, Debug)]
pub enum VulkanKernelError {
    #[error("Could not lock kernel map: {0}")]
    LockError(String),
    #[error("Error while loading library: {0}")]
    LoadLibraryError(String),
    #[error("Error while loading function: {0:?}")]
    LoadFunctionError(String),
    #[error("Failed to create compute function")]
    FailedToCreateComputeFunction,
    #[error("Failed to create pipeline")]
    FailedToCreatePipeline(String),
    #[error("Invalid matmul arguments {lhs_stride:?} {rhs_stride:?} {mnk:?}")]
    MatMulNonContiguous {
        lhs_stride: Vec<usize>,
        rhs_stride: Vec<usize>,
        mnk: (usize, usize, usize),
    },
    #[error("Error while waiting for completion: {0:?}")]
    CommandError(String),
}

impl<T> From<std::sync::PoisonError<T>> for VulkanKernelError {
    fn from(e: std::sync::PoisonError<T>) -> Self {
        Self::LockError(e.to_string())
    }
}

pub struct GpuFutureHolder {
    future: Arc<Mutex<Option<Box<dyn GpuFuture + Send>>>>,
}

impl fmt::Debug for GpuFutureHolder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuFutureHolder")
            .field("future", &"<GpuFuture>")
            .finish()
    }
}

impl GpuFutureHolder {
    pub fn new() -> Self {
        Self {
            future: Arc::new(Mutex::new(None)),
        }
    }

    fn lock(&self) -> Result<MutexGuard<Option<Box<dyn GpuFuture + Send>>>, VulkanKernelError> {
        self.future.lock().map_err(|e| e.into())
    }

    pub fn set_future(&self, future: Box<dyn GpuFuture + Send>) -> Result<(), VulkanKernelError> {
        let mut guard = self.lock()?;
        *guard = Some(future);
        Ok(())
    }

    pub fn sync_if_needed(&self) -> Result<(), VulkanKernelError> {
        let mut guard = self.lock()?;
        if let Some(future) = guard.take() {
            future.then_signal_fence_and_flush()
                .map_err(|e| CommandError(e.to_string()))?
                .wait(None)
                .map_err(|e| CommandError(e.to_string()))?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Kernels {
    shader_modules: RwLock<HashMap<Source, Arc<ShaderModule>>>,
    pipelines: RwLock<HashMap<Source, Arc<ComputePipeline>>>,
}

impl Kernels {
    pub fn new(device: Arc<Device>) -> Self {
        let mut shader_modules = HashMap::new();
        // Load and cache the shader modules here
        let random_shader = random::load(device.clone())
            .expect("Failed to load random shader module");
        shader_modules.insert(Source::Random, random_shader);

        Self {
            shader_modules: RwLock::new(shader_modules),
            pipelines: RwLock::new(HashMap::new()),
        }
    }

    pub fn load_pipeline(
        &self,
        device: Arc<Device>,
        source: Source,
    ) -> Result<Arc<ComputePipeline>, VulkanKernelError> {
        // Use the cached shader module
        let shader_module = {
            let shader_modules = self.shader_modules.read()?;
            shader_modules
                .get(&source)
                .ok_or_else(|| VulkanKernelError::LoadLibraryError("Shader module not found".to_string()))?
                .clone()
        };

        // Check if the pipeline is already cached
        let mut pipelines = self.pipelines.write()?;
        if let Some(pipeline) = pipelines.get(&source) {
            return Ok(pipeline.clone());
        }

        // Create the pipeline using the shader module
        let stage = PipelineShaderStageCreateInfo::new(
            shader_module.entry_point("main")
                .ok_or(VulkanKernelError::FailedToCreatePipeline("No entry point".to_string()))?,
        );

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())
                .map_err(|e| VulkanKernelError::FailedToCreatePipeline(e.to_string()))?,
        )
            .map_err(|e| VulkanKernelError::FailedToCreatePipeline(e.to_string()))?;

        let pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
            .map_err(|e| VulkanKernelError::FailedToCreatePipeline(e.to_string()))?;

        pipelines.insert(source, pipeline.clone());
        Ok(pipeline)
    }
}

mod random {
    use vulkano_shaders::shader;

    shader! {
        ty: "compute",
        path: "src/random.comp"
    }
}

#[allow(clippy::too_many_arguments)]
pub fn call_random_normal(
    device: Arc<Device>,
    queue: Arc<Queue>,
    kernels: &Kernels,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    name: &'static str,
    mean: f32,
    stddev: f32,
    length: usize,
    seed: Subbuffer<[u32]>,
    buffer: Subbuffer<[f32]>,
    future_holder: &GpuFutureHolder,
) -> Result<(), VulkanKernelError> {
    let pipeline = kernels.load_pipeline(device.clone(), Source::Random).map_err(|e| VulkanKernelError::CommandError(e.to_string()))?;
    let layout = pipeline.layout().set_layouts().get(0).ok_or(VulkanKernelError::CommandError("no set layouts".to_string()))?;
    let descriptor_set = PersistentDescriptorSet::new(
        &*descriptor_set_allocator,
        layout.clone(),
        [
            WriteDescriptorSet::buffer(0, seed),
            WriteDescriptorSet::buffer(1, buffer),
        ],
        [],
    ).map_err(|e| VulkanKernelError::CommandError(e.to_string()))?;

    let mut builder = AutoCommandBufferBuilder::primary(
        &*command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    ).map_err(|e| VulkanKernelError::CommandError(e.to_string()))?;

    builder
        .bind_pipeline_compute(pipeline.clone()).map_err(|e| VulkanKernelError::CommandError(e.to_string()))?
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.layout().clone(),
            0,
            descriptor_set,
        ).map_err(|e| VulkanKernelError::CommandError(e.to_string()))?
        .push_constants(pipeline.layout().clone(), 0, mean).map_err(|e| VulkanKernelError::CommandError(e.to_string()))?
        .push_constants(pipeline.layout().clone(), 4, stddev).map_err(|e| VulkanKernelError::CommandError(e.to_string()))?;
    builder.dispatch([length as u32, 1, 1]).map_err(|e| VulkanKernelError::CommandError(e.to_string()))?;

    let command_buffer = builder.build().map_err(|e| VulkanKernelError::CommandError(e.to_string()))?;
    let future = sync::now(device).then_execute(queue, command_buffer).map_err(|e| VulkanKernelError::CommandError(e.to_string()))?.then_signal_fence_and_flush().map_err(|e| VulkanKernelError::CommandError(e.to_string()))?;
    future_holder.set_future(Box::new(future))?;
    Ok(())
}
