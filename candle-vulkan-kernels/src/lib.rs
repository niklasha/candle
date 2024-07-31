use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo};
use vulkano::device::{Device, Queue};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::shader::ShaderModule;
use vulkano::sync::{self, GpuFuture};

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

#[derive(Debug)]
pub struct Kernels {
    pipelines: RwLock<HashMap<Source, Arc<ComputePipeline>>>,
}

impl Default for Kernels {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernels {
    pub fn new() -> Self {
        let pipelines = RwLock::new(HashMap::new());
        Self { pipelines }
    }

    pub fn load_pipeline(
        &self,
        device: Arc<Device>,
        source: Source,
        shader: Arc<ShaderModule>,
    ) -> Result<Arc<ComputePipeline>, VulkanKernelError> {
        let mut pipelines = self.pipelines.write()?;
        if let Some(pipeline) = pipelines.get(&source) {
            return Ok(pipeline.clone());
        }

        let stage = PipelineShaderStageCreateInfo::new(shader.entry_point("main").ok_or(VulkanKernelError::FailedToCreatePipeline("no entrypoint".to_string()))?);
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone()).map_err(|e| VulkanKernelError::FailedToCreatePipeline(e.to_string()))?,
        ).map_err(|e| VulkanKernelError::FailedToCreatePipeline(e.to_string()))?;
        let pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        ).map_err(|e| VulkanKernelError::FailedToCreatePipeline(e.to_string()))?;

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
    name: &'static str,
    mean: f32,
    stddev: f32,
    length: usize,
    seed: Subbuffer<[u32]>,
    buffer: Subbuffer<[f32]>,
) -> Result<(), VulkanKernelError> {
    let shader = random::load(device.clone()).map_err(|e| VulkanKernelError::LoadLibraryError(e.to_string()))?;

    let pipeline = kernels.load_pipeline(device.clone(), Source::Random, shader).map_err(|e| VulkanKernelError::CommandError(e.to_string()))?;
    let layout = pipeline.layout().set_layouts().get(0).ok_or(VulkanKernelError::CommandError("no set layouts".to_string()))?;
    let allocator = StandardDescriptorSetAllocator::new(device.clone(), StandardDescriptorSetAllocatorCreateInfo::default());
    let descriptor_set = PersistentDescriptorSet::new(
        &allocator,
        layout.clone(),
        [
            WriteDescriptorSet::buffer(0, seed),
            WriteDescriptorSet::buffer(1, buffer),
        ],
        [],
    ).map_err(|e| VulkanKernelError::CommandError(e.to_string()))?;

    let mut builder = AutoCommandBufferBuilder::primary(
        &StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ),
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
    unsafe { builder.dispatch([length as u32, 1, 1]).map_err(|e| VulkanKernelError::CommandError(e.to_string()))?; }

    let command_buffer = builder.build().map_err(|e| VulkanKernelError::CommandError(e.to_string()))?;
    let future = sync::now(device).then_execute(queue, command_buffer).map_err(|e| VulkanKernelError::CommandError(e.to_string()))?.then_signal_fence_and_flush().map_err(|e| VulkanKernelError::CommandError(e.to_string()))?;
    future.wait(None).map_err(|e| VulkanKernelError::CommandError(e.to_string()))?;
    Ok(())
}
