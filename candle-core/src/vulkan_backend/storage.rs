#![allow(dead_code)]

use crate::backend::{BackendDevice, BackendStorage};
use crate::cpu_backend::{binary_map, binary_map_vec};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, VulkanDevice, VulkanError};
use candle_vulkan_kernels::Source;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryCommandBufferAbstract,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::sync::GpuFuture;

#[derive(Clone, Debug)]
pub struct VulkanStorage {
    /// The actual subbuffer containing the data.  It is type erased since VulkanStorage is untyped.
    /// It is an Option, since a zero-sized buffer is invalid in Vulkan, but not in Candle, so we use None representing that case.
    buffer: Arc<Option<Subbuffer<[u8]>>>,
    /// a reference to the device owning this buffer
    device: VulkanDevice,
    /// The count of allocated elements in the buffer
    count: usize,
    /// The dtype is kept since buffers are untyped.
    dtype: DType,
}

impl VulkanStorage {
    pub(crate) fn new(
        buffer: Option<Subbuffer<[u8]>>,
        device: VulkanDevice,
        count: usize,
        dtype: DType,
    ) -> Self {
        Self {
            buffer: Arc::new(buffer),
            device,
            count,
            dtype,
        }
    }

    pub fn to_cpu<T: BufferContents + Clone + Copy + Send>(&self) -> Result<Vec<T>> {
        // self.pending_future
        //     .sync_if_needed()
        //     .map_err(VulkanError::from)?;
        self.device.to_cpu(self.buffer.clone())
    }

    fn execute_compute_kernel(
        &self,
        pipeline: &Arc<ComputePipeline>,
        input_buffers: Vec<Subbuffer<[u8]>>,
        output_buffers: Vec<Subbuffer<[u8]>>,
        elem_count: usize,
    ) -> Result<()> {
        let device = self.device();

        let mut builder = AutoCommandBufferBuilder::primary(
            device.command_buffer_allocator.clone(),
            device.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(VulkanError::ValidatedVulkanError)?;

        // Bind pipeline and descriptors
        let offset = input_buffers.len();
        let mut bindings = input_buffers
            .into_iter()
            .enumerate()
            .map(|(i, buf)| WriteDescriptorSet::buffer(i as u32, buf))
            .chain(
                output_buffers
                    .into_iter()
                    .enumerate()
                    .map(|(i, buf)| WriteDescriptorSet::buffer((i + offset) as u32, buf)),
            )
            .collect::<Vec<_>>();

        unsafe {
            builder
                .bind_pipeline_compute(pipeline.clone())
                .map_err(VulkanError::ValidationError)?
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    pipeline.layout().clone(),
                    0,
                    DescriptorSet::new(
                        device.descriptor_set_allocator.clone(),
                        pipeline.layout().set_layouts()[0].clone(),
                        bindings,
                        [],
                    )
                    .map_err(VulkanError::ValidatedVulkanError)?,
                )
                .map_err(VulkanError::ValidationError)?
                .dispatch([(elem_count as u32 + 255) / 256, 1, 1])
                .map_err(|e| VulkanError::ValidationError(e.into()))?;
        }

        // Execute and sync
        let command_buffer = builder.build().map_err(VulkanError::ValidatedVulkanError)?;
        let future = command_buffer
            .execute(device.queue.clone())
            .map_err(VulkanError::CommandBufferExecError)?;

        future
            .then_signal_fence_and_flush()
            .map_err(VulkanError::ValidatedVulkanError)?
            .wait(None)
            .map_err(VulkanError::ValidatedVulkanError)?;

        Ok(())
    }

    // XXX This is an in-place version, which may be faster for some ops.
    // // Bind pipeline and dispatch compute
    // builder
    //     .bind_pipeline_compute(self.device.neg_pipeline.clone())
    //     .map_err(VulkanError::ValidationError)?
    //     .bind_descriptor_sets(
    //         PipelineBindPoint::Compute,
    //         self.device.neg_pipeline.layout().clone(),
    //         0,
    //         PersistentDescriptorSet::new(
    //             &self.device.descriptor_set_allocator,
    //             self.device.neg_pipeline.layout().set_layouts()[0].clone(),
    //             [WriteDescriptorSet::buffer(0, buffer.clone())],
    //             [],
    //         )
    //             .map_err(VulkanError::ValidatedVulkanError)?,
    //     )
    //     .map_err(VulkanError::ValidationError)?
    //     .dispatch([(elem_count as u32 + 255) / 256, 1, 1])
    //     .map_err(|e| VulkanError::Message(format!("Dispatch failed: {e}")))?;

    fn unary_op_impl(
        &self,
        layout: &Layout,
        pipeline: &Arc<ComputePipeline>,
        target_dtype: DType,
    ) -> Result<Self> {
        let dtype = self.dtype();

        if let Some(buffer) = (*self.buffer).clone() {
            let elem_count = layout.shape().elem_count();
            let device = self.device();
            let new_storage = unsafe { device.alloc_uninit(layout.shape(), target_dtype)? };

            self.execute_compute_kernel(
                pipeline,
                vec![buffer],
                vec![(*new_storage.buffer).clone().unwrap()],
                elem_count,
            )?;

            Ok(new_storage)
        } else {
            // Zero-sized buffer, return zero-sized buffer
            Ok(self.clone())
        }
    }

    fn binary_op_impl(
        &self,
        layout: &Layout,
        rhs: &Self,
        rhs_layout: &Layout,
        pipeline: &Arc<ComputePipeline>,
    ) -> Result<Self> {
        let lhs_dtype = self.dtype();
        let rhs_dtype = rhs.dtype();

        // Only handle F32 for now
        if lhs_dtype != DType::F32 || rhs_dtype != DType::F32 {
            return Err(VulkanError::Message(format!(
                "Unsupported dtype pair: {:?} {:?}",
                lhs_dtype, rhs_dtype,
            )))?;
        }

        if let (Some(lhs_buffer), Some(rhs_buffer)) =
            ((*self.buffer).clone(), (*rhs.buffer).clone())
        {
            let elem_count = layout.shape().elem_count();
            let device = self.device();
            let new_storage = unsafe { device.alloc_uninit(layout.shape(), lhs_dtype)? };

            self.execute_compute_kernel(
                pipeline,
                vec![lhs_buffer, rhs_buffer],
                vec![(*new_storage.buffer).clone().unwrap()],
                elem_count,
            )?;

            Ok(new_storage)
        } else {
            // Zero-sized buffer, return zero-sized buffer
            Ok(self.clone())
        }
    }

    fn reduce_op_impl(&self, layout: &Layout, pipeline: &Arc<ComputePipeline>) -> Result<Self> {
        let dtype = self.dtype();

        // Only handle F32 for now
        if dtype != DType::F32 {
            return Err(VulkanError::Message(format!(
                "Unsupported dtype: {:?}",
                dtype
            )))?;
        }

        if let Some(buffer) = (*self.buffer).clone() {
            let elem_count = layout.shape().elem_count();
            let device = self.device();
            let new_storage = unsafe { device.alloc_uninit(layout.shape(), dtype)? };

            self.execute_compute_kernel(
                pipeline,
                vec![buffer],
                vec![(*new_storage.buffer).clone().unwrap()],
                elem_count,
            )?;

            Ok(new_storage)
        } else {
            // Zero-sized buffer, return zero-sized buffer
            Ok(self.clone())
        }
    }

    fn affine_elu_op_impl(
        &self,
        layout: &Layout,
        pipeline: &Arc<ComputePipeline>,
        mul: f64,   // used only for affine
        add: f64,   // used only for affine
        alpha: f64, // used only for ELU
    ) -> Result<Self> {
        let elem_count = layout.shape().elem_count();
        let device = self.device();
        let new_storage = unsafe { device.alloc_uninit(layout.shape(), self.dtype)? };

        let mul_f32 = mul as f32;
        let add_f32 = add as f32;
        let alpha_f32 = alpha as f32;

        // Extract the full shape and stride. We assume a maximum rank of 4.
        let shape_slice = layout.shape();
        let stride_slice = layout.stride();
        let mut shape_arr = [1u32; 4];
        let mut stride_arr = [1u32; 4];
        for i in 0..shape_slice.rank().min(4) {
            shape_arr[i] = (shape_slice.dim(i).unwrap())
                .try_into()
                .map_err(|_| VulkanError::Message("Shape conversion failed".to_string()))?;
        }
        for i in 0..stride_slice.len().min(4) {
            stride_arr[i] = (*stride_slice.get(i).unwrap()) as u32;
        }
        let rank = shape_slice.rank() as u32;

        let mut builder = AutoCommandBufferBuilder::primary(
            device.command_buffer_allocator.clone(),
            device.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(VulkanError::ValidatedVulkanError)?;

        let input_buffer = (*self.buffer)
            .clone()
            .ok_or_else(|| VulkanError::Message("Missing input buffer".into()))?;
        let output_buffer = (*new_storage.buffer)
            .clone()
            .ok_or_else(|| VulkanError::Message("Missing output buffer".into()))?;

        let bindings = vec![
            WriteDescriptorSet::buffer(0, input_buffer),
            WriteDescriptorSet::buffer(1, output_buffer),
        ];

        let pds = DescriptorSet::new(
            device.descriptor_set_allocator.clone(),
            pipeline.layout().set_layouts()[0].clone(),
            bindings,
            [],
        )
        .map_err(VulkanError::ValidatedVulkanError)?;

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct AffinePushConstants {
            rank: u32,
            _pad0: [u32; 3],
            shape: [u32; 4],
            stride: [u32; 4],
            mul: f32,
            add: f32,
            alpha: f32,
        }
        let push_constants = AffinePushConstants {
            rank,
            _pad0: [0; 3],
            shape: shape_arr,
            stride: stride_arr,
            mul: mul_f32,
            add: add_f32,
            alpha: alpha_f32,
        };

        unsafe {
            builder
                .bind_pipeline_compute(pipeline.clone())
                .map_err(VulkanError::ValidationError)?
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    pipeline.layout().clone(),
                    0,
                    pds,
                )
                .map_err(VulkanError::ValidationError)?
                .push_constants(pipeline.layout().clone(), 0, push_constants)
                .map_err(VulkanError::ValidationError)?
                .dispatch([((elem_count as u32) + 255) / 256, 1, 1])
                .map_err(VulkanError::ValidationError)?;
        }

        let command_buffer = builder.build().map_err(VulkanError::ValidatedVulkanError)?;
        let future = command_buffer
            .execute(device.queue.clone())
            .map_err(VulkanError::CommandBufferExecError)?;
        future
            .then_signal_fence_and_flush()
            .map_err(VulkanError::ValidatedVulkanError)?
            .wait(None)
            .map_err(VulkanError::ValidatedVulkanError)?;

        Ok(new_storage)
    }
}

macro_rules! fail {
    () => {
        todo!("vulkan support is incomplete, this function is not yet implemented")
    };
}

impl crate::backend::BackendStorage for VulkanStorage {
    type Device = VulkanDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        // Simple clone of storage (could be implemented as buffer copy later)
        Ok(self.clone())
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        match self.dtype {
            DType::U8 => Ok(CpuStorage::U8(self.to_cpu()?)),
            DType::U32 => Ok(CpuStorage::U32(self.to_cpu()?)),
            DType::I64 => Ok(CpuStorage::I64(self.to_cpu()?)),
            DType::F16 => Ok(CpuStorage::F16(self.to_cpu()?)),
            DType::BF16 => Ok(CpuStorage::BF16(self.to_cpu()?)),
            DType::F32 => Ok(CpuStorage::F32(self.to_cpu()?)),
            DType::F64 => Ok(CpuStorage::F64(self.to_cpu()?)),
        }
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        if self.dtype != DType::F32 {
            fail!()
        }
        let pipeline = self
            .device
            .affine_elu_pipelines
            .get("affine")
            .ok_or_else(|| VulkanError::Message("No affine pipeline".to_string()))?;
        self.affine_elu_op_impl(layout, pipeline, mul, add, 0.0)
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        fail!()
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        if self.dtype != DType::F32 {
            fail!()
        }
        let pipeline = self
            .device
            .affine_elu_pipelines
            .get("elu")
            .ok_or_else(|| VulkanError::Message("No elu pipeline".to_string()))?;
        self.affine_elu_op_impl(layout, pipeline, 0.0, 0.0, alpha)
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, s: &[usize]) -> Result<Self> {
        match (op, self.dtype) {
            (ReduceOp::Max, DType::F32) => {
                if let Some(pipeline) = self.device.reduce_pipelines.get(op.name()) {
                    self.reduce_op_impl(layout, pipeline)
                } else {
                    fail!()
                }
            }
            _ => fail!(),
        }
    }

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        fail!()
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        match (self.dtype, dtype) {
            (a, b) if a == b => Ok(self.clone()),
            (DType::F32, DType::F16) => {
                self.unary_op_impl(layout, &self.device.cast_pipelines[0], dtype)
            }
            (DType::F16, DType::F32) => {
                self.unary_op_impl(layout, &self.device.cast_pipelines[1], dtype)
            }
            (DType::U32, DType::F32) => {
                self.unary_op_impl(layout, &self.device.cast_pipelines[2], dtype)
            }
            _ => fail!(),
        }
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        if self.dtype == DType::F32 {
            if let Some(pipeline) = self.device.unary_pipelines.get(B::NAME) {
                self.unary_op_impl(layout, pipeline, self.dtype())
            } else {
                fail!()
            }
        } else if self.dtype == DType::F16 {
            let key = format!("{}_f16", B::NAME);
            if let Some(pipeline) = self.device.unary_pipelines.get(key.as_str()) {
                self.unary_op_impl(layout, pipeline, self.dtype())
            } else {
                fail!()
            }
        } else {
            fail!()
        }
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        match (self.dtype(), rhs.dtype()) {
            (DType::F32, DType::F32) => {
                if let Some(pipeline) = self.device.binary_pipelines.get(B::NAME) {
                    self.binary_op_impl(layout, rhs, rhs_layout, pipeline)
                } else {
                    fail!()
                }
            }
            _ => {
                fail!()
            }
        }
    }

    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        fail!()
    }

    fn conv1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        fail!()
    }

    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        fail!()
    }

    fn conv2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        fail!()
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        fail!()
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        fail!()
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        fail!()
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        fail!()
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        fail!()
    }

    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        fail!()
    }

    fn scatter_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        fail!()
    }

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self> {
        fail!()
    }

    fn index_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        fail!()
    }

    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        fail!()
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()> {
        fail!()
    }

    fn copy2d(
        &self,
        _: &mut Self,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
    ) -> Result<()> {
        fail!()
    }
}
