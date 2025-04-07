#![allow(dead_code)]

use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape, VulkanDevice, VulkanError};
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::sync::GpuFuture;

// The maximum rank of tensors supported by the Vulkan backend.
const MAX_RANK: usize = 8;

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

    fn lock(&self) -> Result<MutexGuard<Option<Box<dyn GpuFuture + Send>>>> {
        self.future.lock().map_err(|e| VulkanError::from(e).into())
    }

    pub fn set_future(&self, future: Box<dyn GpuFuture + Send>) -> Result<()> {
        let mut guard = self.lock()?;
        *guard = Some(future);
        Ok(())
    }

    pub fn sync_if_needed(&self) -> Result<()> {
        let mut guard = self.lock()?;
        if let Some(future) = guard.take() {
            future
                .then_signal_fence_and_flush()
                .map_err(VulkanError::ValidatedVulkanError)?
                .wait(None)
                .map_err(VulkanError::ValidatedVulkanError)?;
            *guard = None;
        }
        Ok(())
    }
}

impl fmt::Display for CmpOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            CmpOp::Eq => "Eq",
            CmpOp::Lt => "Lt",
            CmpOp::Gt => "Gt",
            CmpOp::Ne => "Ne",
            CmpOp::Le => "Le",
            CmpOp::Ge => "Ge",
        };
        write!(f, "{}", s)
    }
}

impl CmpOp {
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Le => "le",
            Self::Ge => "ge",
            Self::Lt => "lt",
            Self::Gt => "gt",
        }
    }
}

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
    pub(crate) pending_future: Arc<GpuFutureHolder>,
}

impl VulkanStorage {
    // Define MAX_RANK constant, consistent with common.comp and other ops
    const MAX_RANK: usize = 8;

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
            pending_future: Arc::new(GpuFutureHolder::new()),
        }
    }

    pub fn to_cpu<T: BufferContents + Clone + Copy + Send>(&self) -> Result<Vec<T>> {
        self.pending_future.sync_if_needed()?;
        self.device.to_cpu(self.buffer.clone())
    }

    fn execute_compute_kernel<PC: bytemuck::Pod + Send + Sync>(
        &self,
        pipeline: &Arc<ComputePipeline>,
        input_buffers: Vec<Subbuffer<[u8]>>,
        output_buffers: Vec<Subbuffer<[u8]>>,
        dispatch_dims: [u32; 3],
        push_constants: PC,
        direct_dispatch: bool,
    ) -> Result<()> {
        let device = self.device();

        let mut builder = AutoCommandBufferBuilder::primary(
            device.command_buffer_allocator.clone(),
            device.queue.queue_family_index(),
            CommandBufferUsage::SimultaneousUse,
        )
        .map_err(VulkanError::ValidatedVulkanError)?;

        let offset = input_buffers.len();
        let bindings = input_buffers
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

        let dims = if direct_dispatch {
            dispatch_dims
        } else {
            let total_threads = dispatch_dims[0];
            let threads_per_group = 256;
            let max_counts = device
                .device()
                .physical_device()
                .properties()
                .max_compute_work_group_count;
            Self::compute_3d_dispatch_dims(total_threads, threads_per_group, max_counts)
        };
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
            .push_constants(pipeline.layout().clone(), 0, push_constants)
            .map_err(VulkanError::ValidationError)?;
        unsafe { builder.dispatch(dims) }
            .inspect_err(|e| {
                eprintln!("{:?}", dims);
                if dims[0] > 65536 {
                    panic!("POFF");
                }
            })
            .map_err(|e| VulkanError::ValidationError(e.into()))?;

        let command_buffer = builder.build().map_err(VulkanError::ValidatedVulkanError)?;
        let future = command_buffer
            .execute(device.queue.clone())
            .map_err(VulkanError::CommandBufferExecError);
        if future.is_err() {
            println!("ERR");
        }
        let future = future?;
        self.pending_future.set_future(Box::new(future))?;
        Ok(())
    }

    fn compute_3d_dispatch_dims(
        total_threads: u32,
        threads_per_group: u32,
        max_group_count: [u32; 3],
    ) -> [u32; 3] {
        let total_groups = (total_threads + threads_per_group - 1) / threads_per_group;

        let max_x = max_group_count[0].min(65535); // Vulkan spec limit
        let max_y = max_group_count[1];
        let max_z = max_group_count[2];

        let mut x = total_groups.min(max_x);
        let mut y = 1;
        let mut z = 1;
        let mut remaining = total_groups / x;

        if remaining > 1 {
            y = remaining.min(max_y);
            remaining /= y;

            if remaining > 1 {
                z = remaining.min(max_z);
            }
        }

        [x, y, z]
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
    //     .dispatch([(elem_count as u32 + 255) / 512, 1, 1])
    //     .map_err(|e| VulkanError::Message(format!("Dispatch failed: {e}")))?;

    pub fn unary_op_impl(
        &self,
        layout: &Layout,
        pipeline: &Arc<ComputePipeline>,
        target_dtype: DType,
    ) -> Result<Self> {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PushConstants {
            rank: u32,
            base: u32,
            shape: [u32; MAX_RANK],
            stride: [u32; MAX_RANK],
        }

        if let Some(buffer) = (*self.buffer).clone() {
            let shape_slice = layout.shape();
            let rank = shape_slice.rank();
            if rank > MAX_RANK {
                return Err(VulkanError::Message(format!(
                    "Vulkan backend only supports rank up to {}, got {}",
                    MAX_RANK, rank
                ))
                .into());
            }

            let elem_count = layout.shape().elem_count();
            let device = self.device();
            let new_storage = unsafe { device.alloc_uninit(layout.shape(), target_dtype)? };

            // Extract the full shape and stride. We assume a maximum rank of MAX_RANK.
            let stride_slice = layout.stride();
            let mut shape_arr = [1u32; MAX_RANK];
            let mut stride_arr = [1u32; MAX_RANK];
            for i in 0..rank.min(MAX_RANK) {
                shape_arr[i] = shape_slice
                    .dim(i)
                    .unwrap()
                    .try_into()
                    .map_err(|_| VulkanError::Message("Shape conversion failed".to_string()))?;
            }
            for i in 0..stride_slice.len().min(MAX_RANK) {
                stride_arr[i] = (*stride_slice.get(i).unwrap()) as u32;
            }
            let base = layout.start_offset() as u32;

            let push_constants = PushConstants {
                rank: rank as u32,
                base,
                shape: shape_arr,
                stride: stride_arr,
            };
            self.pending_future.sync_if_needed()?;
            let future = new_storage.execute_compute_kernel(
                pipeline,
                vec![buffer],
                vec![(*new_storage.buffer).clone().unwrap()],
                [elem_count as u32, 1, 1],
                push_constants,
                false,
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
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PushConstants {
            a_rank: u32,
            a_base: u32,
            a_shape: [u32; MAX_RANK],
            a_stride: [u32; MAX_RANK],
            b_rank: u32,
            b_base: u32,
            b_shape: [u32; MAX_RANK],
            b_stride: [u32; MAX_RANK],
        }

        if let (Some(lhs_buffer), Some(rhs_buffer)) =
            ((*self.buffer).clone(), (*rhs.buffer).clone())
        {
            let a_shape_slice = layout.shape();
            let a_rank = a_shape_slice.rank();
            let b_shape_slice = rhs_layout.shape();
            let b_rank = b_shape_slice.rank();
            if a_rank > MAX_RANK || b_rank > MAX_RANK {
                return Err(VulkanError::Message(format!(
                    "Vulkan backend only supports rank up to {}, got {} and {}",
                    MAX_RANK, a_rank, b_rank
                ))
                .into());
            }

            let elem_count = layout.shape().elem_count();
            let device = self.device();
            let lhs_dtype = self.dtype();
            let new_storage = unsafe { device.alloc_uninit(layout.shape(), lhs_dtype)? };

            let a_stride_slice = layout.stride();
            let mut a_shape_arr = [1u32; MAX_RANK];
            let mut a_stride_arr = [1u32; MAX_RANK];
            for i in 0..a_rank.min(MAX_RANK) {
                a_shape_arr[i] = a_shape_slice
                    .dim(i)
                    .unwrap()
                    .try_into()
                    .map_err(|_| VulkanError::Message("Shape conversion failed".to_string()))?;
            }
            for i in 0..a_stride_slice.len().min(MAX_RANK) {
                a_stride_arr[i] = (*a_stride_slice.get(i).unwrap()) as u32;
            }
            let a_base = layout.start_offset() as u32;
            let b_stride_slice = rhs_layout.stride();
            let mut b_shape_arr = [1u32; MAX_RANK];
            let mut b_stride_arr = [1u32; MAX_RANK];
            for i in 0..b_rank.min(MAX_RANK) {
                b_shape_arr[i] = b_shape_slice
                    .dim(i)
                    .unwrap()
                    .try_into()
                    .map_err(|_| VulkanError::Message("Shape conversion failed".to_string()))?;
            }
            for i in 0..b_stride_slice.len().min(MAX_RANK) {
                b_stride_arr[i] = (*b_stride_slice.get(i).unwrap()) as u32;
            }
            let b_base = rhs_layout.start_offset() as u32;

            let push_constants = PushConstants {
                a_rank: a_rank as u32,
                a_base,
                a_shape: a_shape_arr,
                a_stride: a_stride_arr,
                b_rank: b_rank as u32,
                b_base,
                b_shape: b_shape_arr,
                b_stride: b_stride_arr,
            };

            self.pending_future.sync_if_needed()?;
            rhs.pending_future.sync_if_needed()?;
            new_storage.execute_compute_kernel(
                pipeline,
                vec![lhs_buffer, rhs_buffer],
                vec![(*new_storage.buffer).clone().unwrap()],
                [elem_count as u32, 1, 1],
                push_constants,
                false,
            )?;

            Ok(new_storage)
        } else {
            // Zero-sized buffer, return zero-sized buffer
            Ok(self.clone())
        }
    }

    fn reduce_op_impl(
        &self,
        layout: &Layout,
        reduce_axes: &[usize],
        partial_pipeline: &Arc<ComputePipeline>, // partial reduction shader
        combine_pipeline: &Arc<ComputePipeline>, // combining shader
        to_index: bool,                          // false for sum, true for argmax/argmin
    ) -> Result<Self> {
        use std::convert::TryInto;

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
        struct ReducePushConstants {
            base: u32,
            rank: u32,
            shape: [u32; MAX_RANK],
            stride: [u32; MAX_RANK],
            reduce_axes: [u32; MAX_RANK], // XXX make this a bitmap?
        }

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
        struct CombinePushConstants {
            num_partials: u32,
        }

        let dtype = self.dtype();
        let result_dtype = if to_index { DType::U32 } else { dtype };

        if let Some(buffer) = (*self.buffer).clone() {
            let device = self.device();

            // Build tensor metadata.
            let shape_slice = layout.shape();
            let stride_slice = layout.stride();
            let rank = shape_slice.rank();
            if rank > MAX_RANK {
                return Err(VulkanError::Message(format!(
                    "Vulkan backend only supports rank up to {}, got {}",
                    MAX_RANK, rank
                ))
                .into());
            }
            let mut shape_arr = [1u32; MAX_RANK];
            let mut stride_arr = [1u32; MAX_RANK];
            for i in 0..rank.min(MAX_RANK) {
                shape_arr[i] = shape_slice
                    .dim(i)
                    .unwrap()
                    .try_into()
                    .map_err(|_| VulkanError::Message("Shape conversion failed".to_string()))?;
            }
            for i in 0..stride_slice.len().min(MAX_RANK) {
                stride_arr[i] = (*stride_slice.get(i).unwrap()) as u32;
            }
            let base = layout.start_offset() as u32;
            // Build reduce_axes array; unused entries are filled with u32::MAX.
            let reduce_axes_arr: [u32; MAX_RANK] = reduce_axes
                .iter()
                .map(|&ax| ax as u32)
                .chain(std::iter::repeat(u32::MAX))
                .take(MAX_RANK)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();

            let push_constants = ReducePushConstants {
                base,
                rank: rank as u32,
                shape: shape_arr,
                stride: stride_arr,
                reduce_axes: reduce_axes_arr,
            };

            // Compute the flattened reduction size (product over all reduction axes).
            let mut flat_reduction_size = 1u32;
            for &ax in reduce_axes {
                flat_reduction_size *= shape_arr[ax];
            }

            // Use a workgroup size of 256.
            let wg_size = 256u32;
            let segments_per_batch = (flat_reduction_size + wg_size - 1) / wg_size;

            // Compute number of batches as product of dimensions not being reduced.
            let mut num_batches = 1u32;
            for d in 0..(rank as usize) {
                if !reduce_axes.contains(&d) {
                    num_batches *= shape_arr[d];
                }
            }

            // Total workgroups for first pass.
            let total_workgroups = num_batches * segments_per_batch;

            // Allocate two temporary buffers for the partial results:
            // one for candidate values and one for candidate indices.
            let buffer_shape = Shape::from(&[total_workgroups as usize]);
            let partial_values = unsafe { device.alloc_uninit(&buffer_shape, result_dtype)? };
            let partial_indices = unsafe { device.alloc_uninit(&buffer_shape, DType::U32)? };

            self.pending_future.sync_if_needed()?;
            // Dispatch the partial reduction shader.
            partial_values.execute_compute_kernel(
                partial_pipeline,
                vec![buffer],
                vec![
                    (*partial_values.buffer).clone().unwrap(),
                    (*partial_indices.buffer).clone().unwrap(),
                ],
                [total_workgroups, 1, 1],
                push_constants,
                true,
            )?;

            // If more than one segment per batch, dispatch the combining shader.
            let (final_values, final_indices) = if segments_per_batch > 1 {
                let combine_constants = CombinePushConstants {
                    num_partials: segments_per_batch,
                };

                // Allocate final combine buffers with shape [num_batches].
                let final_shape = Shape::from(&[num_batches as usize]);
                let final_values = unsafe { device.alloc_uninit(&final_shape, result_dtype)? };
                let final_indices = unsafe { device.alloc_uninit(&final_shape, DType::U32)? };

                partial_values.pending_future.sync_if_needed()?;
                final_values.execute_compute_kernel(
                    combine_pipeline,
                    vec![
                        (*partial_values.buffer).clone().unwrap(),
                        (*partial_indices.buffer).clone().unwrap(),
                    ],
                    vec![
                        (*final_values.buffer).clone().unwrap(),
                        (*final_indices.buffer).clone().unwrap(),
                    ],
                    [num_batches, 1, 1],
                    combine_constants,
                    true,
                )?;
                (final_values, final_indices)
            } else {
                (partial_values, partial_indices)
            };

            // Compute the output shape by removing reduction axes.
            let mut output_dims = Vec::new();
            for d in 0..(rank as usize) {
                if !reduce_axes.contains(&d) {
                    output_dims.push(shape_arr[d] as usize);
                }
            }
            let output_shape = Shape::from(&[num_batches as usize]);

            // Allocate final storage.
            let mut new_storage = unsafe { device.alloc_uninit(&output_shape, result_dtype)? };
            new_storage.pending_future = final_values.pending_future;

            // For operations like argmax/argmin we want indices,
            // for sum (and others) we want values.
            if to_index {
                new_storage.buffer = final_indices.buffer.clone();
            } else {
                new_storage.buffer = final_values.buffer.clone();
            }

            Ok(new_storage)
        } else {
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
        let base = layout.start_offset() as u32;

        let mut builder = AutoCommandBufferBuilder::primary(
            device.command_buffer_allocator.clone(),
            device.queue.queue_family_index(),
            CommandBufferUsage::SimultaneousUse,
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
            base: u32,
            rank: u32,
            _pad0: [u32; 2],
            shape: [u32; 4],
            stride: [u32; 4],
            mul: f32,
            add: f32,
            alpha: f32,
        }
        let push_constants = AffinePushConstants {
            base,
            rank,
            _pad0: [0; 2],
            shape: shape_arr,
            stride: stride_arr,
            mul: mul_f32,
            add: add_f32,
            alpha: alpha_f32,
        };

        self.pending_future.sync_if_needed()?;
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
            .map_err(VulkanError::ValidationError)?;
        unsafe { builder.dispatch([((elem_count as u32) + 255) / 256, 1, 1]) }
            .map_err(VulkanError::ValidationError)?;

        let command_buffer = builder.build().map_err(VulkanError::ValidatedVulkanError)?;
        let future = command_buffer
            .execute(device.queue.clone())
            .map_err(VulkanError::CommandBufferExecError)?;
        new_storage.pending_future.set_future(Box::new(future))?;

        Ok(new_storage)
    }

    fn gather_op_impl(
        &self,
        dst: &Self,
        index: &Self,
        src_layout: &Layout,
        index_layout: &Layout,
        pipeline: &Arc<ComputePipeline>,
        dim: usize,
    ) -> Result<Self> {
        if let (Some(src_buf), Some(idx_buf), Some(dst_buf)) = (
            (*self.buffer).clone(),
            (*index.buffer).clone(),
            (*dst.buffer).clone(),
        ) {
            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct GatherPushConstants {
                total_out_elems: u32,
                base: u32,
                rank: u32,
                selected_dim: u32,
                input_strides: [u32; 4],
                output_strides: [u32; 4],
            }

            let base = src_layout.start_offset() as u32;
            let rank = src_layout.shape().rank();
            let total_out_elems = index_layout.shape().elem_count() as u32;

            let mut padded_in_strides = [0u32; 4];
            let mut padded_out_strides = [0u32; 4];
            for i in 0..rank.min(4) {
                padded_in_strides[i] = src_layout.stride()[i] as u32;
                padded_out_strides[i] = index_layout.stride()[i] as u32;
            }

            let push_constants = GatherPushConstants {
                total_out_elems,
                rank: rank as u32,
                base,
                selected_dim: dim as u32,
                input_strides: padded_in_strides,
                output_strides: padded_out_strides,
            };

            self.pending_future.sync_if_needed()?;
            index.pending_future.sync_if_needed()?;
            dst.execute_compute_kernel(
                pipeline,
                vec![src_buf, idx_buf],
                vec![dst_buf],
                [total_out_elems, 1, 1],
                push_constants,
                false,
            )?;

            Ok(dst.clone())
        } else {
            Ok(self.clone())
        }
    }

    fn scatter_set_op_impl(
        &mut self,
        layout: &Layout,
        ids: &Self,
        src: &Self,
        src_layout: &Layout,
        pipeline: &Arc<ComputePipeline>,
        dim: usize,
    ) -> Result<()> {
        if let (Some(dst_buf), Some(idx_buf), Some(src_buf)) = (
            (*self.buffer).clone(),
            (*ids.buffer).clone(),
            (*src.buffer).clone(),
        ) {
            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct ScatterSetPushConstants {
                total_src_elems: u32,
                rank: u32,
                in_base: u32,
                out_base: u32,
                input_strides: [u32; 4],
                output_strides: [u32; 4],
                selected_dim: u32,
            }

            let rank = src_layout.shape().rank();
            let in_base = src_layout.start_offset() as u32;
            let total_src_elems = src_layout.shape().elem_count() as u32;
            let out_base = layout.start_offset() as u32;

            // Prepare padded strides
            let mut padded_input_strides = [0u32; 4];
            let mut padded_output_strides = [0u32; 4];

            for i in 0..rank.min(4) {
                padded_input_strides[i] = src_layout.stride()[i] as u32;
                padded_output_strides[i] = layout.stride()[i] as u32;
            }

            let push_constants = ScatterSetPushConstants {
                total_src_elems,
                rank: rank as u32,
                in_base,
                out_base,
                selected_dim: dim as u32,
                input_strides: padded_input_strides,
                output_strides: padded_output_strides,
            };

            self.pending_future.sync_if_needed()?;
            ids.pending_future.sync_if_needed()?;
            src.pending_future.sync_if_needed()?;
            self.execute_compute_kernel(
                pipeline,
                vec![src_buf, idx_buf],
                vec![dst_buf],
                [total_src_elems, 1, 1],
                push_constants,
                false,
            )?;

            Ok(())
        } else {
            Ok(())
        }
    }

    fn scatter_add_set_op_impl(
        &mut self,
        layout: &Layout,
        ids: &Self,
        src: &Self,
        src_layout: &Layout,
        pipeline: &Arc<ComputePipeline>,
        dim: usize,
    ) -> Result<()> {
        if let (Some(dst_buf), Some(idx_buf), Some(src_buf)) = (
            (*self.buffer).clone(),
            (*ids.buffer).clone(),
            (*src.buffer).clone(),
        ) {
            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct ScatterAddSetPushConstants {
                total_src_elems: u32,
                rank: u32,
                selected_dim: u32,
                _pad: u32,
                input_strides: [u32; 4],
                output_strides: [u32; 4],
            }

            let rank = src_layout.shape().rank();
            let total_src_elems = src_layout.shape().elem_count() as u32;

            // Prepare padded strides
            let mut padded_input_strides = [0u32; 4];
            let mut padded_output_strides = [0u32; 4];

            for i in 0..rank.min(4) {
                padded_input_strides[i] = src_layout.stride()[i] as u32;
                padded_output_strides[i] = layout.stride()[i] as u32;
            }

            let push_constants = ScatterAddSetPushConstants {
                total_src_elems,
                rank: rank as u32,
                selected_dim: dim as u32,
                _pad: 0,
                input_strides: padded_input_strides,
                output_strides: padded_output_strides,
            };

            self.execute_compute_kernel(
                pipeline,
                vec![src_buf, idx_buf],
                vec![dst_buf],
                [total_src_elems, 1, 1],
                push_constants,
                false,
            )?;

            Ok(())
        } else {
            Ok(())
        }
    }

    fn index_select_op_impl(
        &self,
        dst: &Self,
        index: &Self,
        src_layout: &Layout,
        index_layout: &Layout,
        pipeline: &Arc<ComputePipeline>,
        dim: usize,
    ) -> Result<Self> {
        // Ensure that the buffers exist.
        if let (Some(src_buffer), Some(index_buffer), Some(dst_buffer)) = (
            (*self.buffer).clone(),
            (*index.buffer).clone(),
            (*dst.buffer).clone(),
        ) {
            // Push constant struct matching the shader.
            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct IndexSelectPushConstants {
                total_out_elems: u32,
                rank: u32,
                base: u32,
                selected_dim: u32,
                input_strides: [u32; 4],
                output_strides: [u32; 4],
            }

            // Get the input shape and strides from the source layout.
            let src_shape: Vec<usize> = src_layout.shape().dims().to_vec();
            let rank = src_shape.len();
            let base = src_layout.start_offset() as u32;
            let input_strides: Vec<u32> = src_layout.stride().iter().map(|&s| s as u32).collect();

            // Compute the output shape by replacing the selected dimension with the index tensor's length.
            let mut out_shape = src_shape.clone();
            out_shape[dim] = index_layout.shape().dim(0)?;
            // Total number of output elements:
            let total_out_elems: u32 = out_shape.iter().map(|&d| d as u32).product();

            // Compute output strides in row-major order.
            // For a shape [s0, s1, ..., s_{R-1}], row-major strides are:
            // stride[R-1] = 1; stride[i] = stride[i+1] * s[i+1]
            let mut output_strides: Vec<u32> = vec![0; rank];
            if rank > 0 {
                output_strides[rank - 1] = 1;
                for i in (0..rank - 1).rev() {
                    output_strides[i] = output_strides[i + 1] * (out_shape[i + 1] as u32);
                }
            }

            // Pad the strides to 4 elements (assuming rank <= 4).
            let mut padded_input_strides = input_strides.clone();
            padded_input_strides.resize(4, 0);
            let mut padded_output_strides = output_strides.clone();
            padded_output_strides.resize(4, 0);

            let push_constants = IndexSelectPushConstants {
                total_out_elems,
                rank: rank as u32,
                base,
                selected_dim: dim as u32,
                input_strides: [
                    padded_input_strides[0],
                    padded_input_strides[1],
                    padded_input_strides[2],
                    padded_input_strides[3],
                ],
                output_strides: [
                    padded_output_strides[0],
                    padded_output_strides[1],
                    padded_output_strides[2],
                    padded_output_strides[3],
                ],
            };

            // We'll dispatch 1D: one thread per output element.
            // Each workgroup has 256 threads.
            let dispatch_x = (total_out_elems + 255) / 256;
            let dispatch_dims = [dispatch_x, 1, 1];

            self.pending_future.sync_if_needed()?;
            index.pending_future.sync_if_needed()?;
            dst.execute_compute_kernel(
                pipeline,
                vec![src_buffer, index_buffer], // binding 0: source, binding 1: indices
                vec![dst_buffer],
                dispatch_dims,
                push_constants,
                true,
            )?;

            Ok(dst.clone())
        } else {
            Ok(self.clone())
        }
    }

    fn index_add_op_impl(
        &self,
        index: &Self,
        src: &Self,
        dst_layout: &Layout,
        src_layout: &Layout,
        pipeline: &Arc<ComputePipeline>,
        dim: usize,
    ) -> Result<Self> {
        if let (Some(dst_buffer), Some(index_buffer), Some(src_buffer)) = (
            (*self.buffer).clone(),
            (*index.buffer).clone(),
            (*src.buffer).clone(),
        ) {
            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct IndexAddPushConstants {
                total_src_elems: u32,
                rank: u32,
                in_base: u32,
                out_base: u32,
                input_strides: [u32; 4],
                output_strides: [u32; 4],
                selected_dim: u32,
            }

            let src_shape = src_layout.shape().dims();
            let dst_shape = dst_layout.shape().dims();
            let rank = src_shape.len();
            let in_base = src_layout.start_offset() as u32;
            let out_base = dst_layout.start_offset() as u32;

            if rank != dst_shape.len() {
                Err(VulkanError::Message("Rank mismatch in index_add".into()))?;
            }

            let total_src_elems = src_shape.iter().copied().product::<usize>() as u32;

            let mut padded_input_strides = vec![0u32; 4];
            let mut padded_output_strides = vec![0u32; 4];

            for i in 0..rank.min(4) {
                padded_input_strides[i] = src_layout.stride()[i] as u32;
                padded_output_strides[i] = dst_layout.stride()[i] as u32;
            }

            let push_constants = IndexAddPushConstants {
                total_src_elems,
                rank: rank as u32,
                in_base,
                out_base,
                input_strides: [
                    padded_input_strides[0],
                    padded_input_strides[1],
                    padded_input_strides[2],
                    padded_input_strides[3],
                ],
                output_strides: [
                    padded_output_strides[0],
                    padded_output_strides[1],
                    padded_output_strides[2],
                    padded_output_strides[3],
                ],
                selected_dim: dim as u32,
            };

            self.pending_future.sync_if_needed()?;
            index.pending_future.sync_if_needed()?;
            src.pending_future.sync_if_needed()?;
            self.execute_compute_kernel(
                pipeline,
                vec![src_buffer, index_buffer],
                vec![dst_buffer],
                [total_src_elems, 1, 1],
                push_constants,
                false,
            )?;

            Ok(self.clone())
        } else {
            Ok(self.clone())
        }
    }

    /// Generic copy_op_impl helper.
    ///
    /// This function dispatches a compute kernel using a prebuilt pipeline.
    /// - `push_constants`: the push constant data (of type PC) to be passed.
    /// - `dispatch_dims`: the dispatch dimensions as [u32;3].
    /// - `pipeline`: the pipeline to use.
    fn copy_op_impl<PC: bytemuck::Pod + std::marker::Send + std::marker::Sync>(
        &self,
        dst: &mut Self,
        push_constants: PC,
        dispatch_dims: [u32; 3],
        pipeline: &Arc<ComputePipeline>,
    ) -> Result<()> {
        if self.buffer.is_none() {
            return Ok(());
        }
        let device = self.device();
        // Obtain source and destination buffers without moving out of the Arc.
        let src_buffer = self
            .buffer
            .as_ref()
            .clone()
            .ok_or_else(|| VulkanError::Message("Missing source buffer".into()))?;
        let dst_buffer = dst
            .buffer
            .as_ref()
            .clone()
            .ok_or_else(|| VulkanError::Message("Missing destination buffer".into()))?;
        let bindings = vec![
            WriteDescriptorSet::buffer(0, src_buffer),
            WriteDescriptorSet::buffer(1, dst_buffer),
        ];
        // Create descriptor set.
        let pds = vulkano::descriptor_set::DescriptorSet::new(
            device.descriptor_set_allocator.clone(),
            pipeline.layout().set_layouts()[0].clone(),
            bindings,
            [],
        )
        .map_err(VulkanError::ValidatedVulkanError)?;

        let mut builder = AutoCommandBufferBuilder::primary(
            device.command_buffer_allocator.clone(),
            device.queue.queue_family_index(),
            CommandBufferUsage::SimultaneousUse,
        )
        .map_err(VulkanError::ValidatedVulkanError)?;
        self.pending_future.sync_if_needed()?;
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
            .map_err(VulkanError::ValidationError)?;
        unsafe { builder.dispatch(dispatch_dims) }.map_err(VulkanError::ValidationError)?;
        let command_buffer = builder.build().map_err(VulkanError::ValidatedVulkanError)?;
        let future = command_buffer
            .execute(device.queue.clone())
            .map_err(VulkanError::CommandBufferExecError)?;
        dst.pending_future.set_future(Box::new(future))?;
        Ok(())
    }

    /// Low-level helper that dispatches the cmp shader.
    /// It assumes that:
    /// - `rhs` is the second operand.
    /// - `dst` is preallocated to hold the output (of type U32, with one element per input).
    /// - `pipeline` is the compute pipeline for the given comparison operator.
    /// - `elem_count` is the number of elements to process.
    fn cmp_op_impl(
        &self,
        rhs: &Self,
        layout: &Layout,
        rhs_layout: &Layout,
        dst: &mut Self,
        pipeline: &Arc<ComputePipeline>,
    ) -> Result<()> {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PushConstants {
            a_base: u32,
            a_rank: u32,
            _pad0: [u32; 2],
            a_shape: [u32; 4],
            a_stride: [u32; 4],
            b_base: u32,
            b_rank: u32,
            _pad1: [u32; 2],
            b_shape: [u32; 4],
            b_stride: [u32; 4],
        }

        let lhs_dtype = self.dtype();

        if let (Some(lhs_buffer), Some(rhs_buffer), Some(dst_buffer)) = (
            (*self.buffer).clone(),
            (*rhs.buffer).clone(),
            (*dst.buffer).clone(),
        ) {
            let elem_count = layout.shape().elem_count();
            let device = self.device();
            let new_storage = unsafe { device.alloc_uninit(layout.shape(), lhs_dtype)? };

            let a_shape_slice = layout.shape();
            let a_stride_slice = layout.stride();
            let mut a_shape_arr = [1u32; 4];
            let mut a_stride_arr = [1u32; 4];
            for i in 0..a_shape_slice.rank().min(4) {
                a_shape_arr[i] = (a_shape_slice.dim(i).unwrap())
                    .try_into()
                    .map_err(|_| VulkanError::Message("Shape conversion failed".to_string()))?;
            }
            for i in 0..a_stride_slice.len().min(4) {
                a_stride_arr[i] = (*a_stride_slice.get(i).unwrap()) as u32;
            }
            let a_rank = a_shape_slice.rank() as u32;
            let a_base = layout.start_offset() as u32;
            let b_shape_slice = rhs_layout.shape();
            let b_stride_slice = rhs_layout.stride();
            let mut b_shape_arr = [1u32; 4];
            let mut b_stride_arr = [1u32; 4];
            for i in 0..b_shape_slice.rank().min(4) {
                b_shape_arr[i] = (b_shape_slice.dim(i).unwrap())
                    .try_into()
                    .map_err(|_| VulkanError::Message("Shape conversion failed".to_string()))?;
            }
            for i in 0..b_stride_slice.len().min(4) {
                b_stride_arr[i] = (*b_stride_slice.get(i).unwrap()) as u32;
            }
            let b_rank = b_shape_slice.rank() as u32;
            let b_base = rhs_layout.start_offset() as u32;

            let push_constants = PushConstants {
                a_base,
                a_rank,
                _pad0: [0; 2],
                a_shape: a_shape_arr,
                a_stride: a_stride_arr,
                b_base,
                b_rank,
                _pad1: [0; 2],
                b_shape: b_shape_arr,
                b_stride: b_stride_arr,
            };
            self.pending_future.sync_if_needed()?;
            rhs.pending_future.sync_if_needed()?;
            dst.execute_compute_kernel(
                pipeline,
                vec![lhs_buffer, rhs_buffer],
                vec![dst_buffer],
                [elem_count as u32, 1, 1],
                push_constants,
                false,
            )?;
        }
        Ok(())
    }

    pub fn arg_sort_op_impl(
        &self,
        layout: &Layout,
        pipeline: &Arc<ComputePipeline>,
        ascending: bool,
    ) -> Result<VulkanStorage> {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct ArgSortPushConstants {
            base: u32,
            nrows: u32,
            ncols: u32,
            ncols_pad: u32,
            ascending: u32,
        }

        let el = layout.shape().elem_count();
        let base = layout.start_offset() as u32;
        let ncols = layout.shape().dims().last().copied().unwrap_or(1);
        let nrows = el / ncols;
        let ncols_pad = ncols.next_power_of_two();

        if ncols_pad > 1024 {
            crate::bail!("arg_sort: padded row size {ncols_pad} exceeds 1024");
        }

        let push_constants = ArgSortPushConstants {
            base,
            nrows: nrows as u32,
            ncols: ncols as u32,
            ncols_pad: ncols_pad as u32,
            ascending: if ascending { 1 } else { 0 },
        };

        let device = self.device();
        let output = unsafe { device.alloc_uninit(&layout.shape().clone().into(), DType::U32)? };

        self.pending_future.sync_if_needed()?;
        output.execute_compute_kernel(
            &pipeline,
            vec![(*self.buffer).clone().unwrap()],
            vec![(*output.buffer).clone().unwrap()],
            [nrows as u32, 1, 1],
            push_constants,
            true,
        )?;

        Ok(output)
    }

    fn where_cond_op_impl(
        &self,
        layout: &Layout,
        cond: &Self,
        t: &Self,
        f: &Self,
        pipeline: &Arc<ComputePipeline>,
    ) -> Result<Self> {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PushConstants {
            elem_count: u32,
            base: u32,
        }

        let elem_count = layout.shape().elem_count();
        let base = layout.start_offset() as u32;
        let device = self.device();
        let new_storage = unsafe { device.alloc_uninit(layout.shape(), self.dtype)? };

        let push_constants = PushConstants {
            elem_count: elem_count as u32,
            base,
        };

        cond.pending_future.sync_if_needed()?;
        t.pending_future.sync_if_needed()?;
        f.pending_future.sync_if_needed()?;
        new_storage.execute_compute_kernel(
            pipeline,
            vec![
                (*cond.buffer).clone().unwrap(),
                (*t.buffer).clone().unwrap(),
                (*f.buffer).clone().unwrap(),
            ],
            vec![(*new_storage.buffer).clone().unwrap()],
            [elem_count as u32, 1, 1],
            push_constants,
            false,
        )?;

        Ok(new_storage)
    }

    fn conv1d_op_impl(
        &self,                  // Input tensor storage
        kernel: &Self,          // Kernel tensor storage
        layout: &Layout,        // Input layout
        kernel_layout: &Layout, // Kernel layout
        params: &crate::conv::ParamsConv1D,
        pipeline: &Arc<ComputePipeline>,
    ) -> Result<Self> {
        // Assumes crate::Error return

        // Define push constant struct including layout info for both inputs
        #[repr(C)]
        #[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct Conv1DPushConstants {
            // Input Layout Info (Rank 3, padded to 4)
            in_base: u32,
            in_rank: u32, // Expected value: 3
            ker_base: u32,
            ker_rank: u32,        // Expected value: 3
            in_stride: [u32; 4],  // Strides for [B, Cin, Lin, 1]
            ker_stride: [u32; 4], // Strides for [Cout, Cin, Ksize, 1]

            // Dimensions & Convolution Parameters
            elem_count: u32, // Total output elements for bounds check
            b_size: u32,
            l_in: u32,
            c_out: u32,
            c_in: u32,
            k_size: u32,
            l_out: u32,
            padding: u32,
            stride: u32,
            dilation: u32,
        }
        // Ensure total size and alignment are checked/managed if needed

        // --- Extract Input Layout ---
        let in_shape_slice = layout.shape();
        let in_stride_slice = layout.stride();
        let in_rank = in_shape_slice.rank(); // Expected 3
        let mut in_stride_arr = [1u32; 4]; // Default stride 1 for unused dims
        for i in 0..in_rank {
            in_stride_arr[i] = in_stride_slice[i] as u32;
        }
        let in_base = layout.start_offset() as u32;

        // --- Extract Kernel Layout ---
        let ker_shape_slice = kernel_layout.shape();
        let ker_stride_slice = kernel_layout.stride();
        let ker_rank = ker_shape_slice.rank(); // Expected 3
        let mut ker_stride_arr = [1u32; 4]; // Default stride 1
        for i in 0..ker_rank {
            ker_stride_arr[i] = ker_stride_slice[i] as u32;
        }
        let ker_base = kernel_layout.start_offset() as u32;

        // --- Calculate Output Shape & Allocate ---
        let device = self.device();
        // Use params which are already adjusted for groups by candle-core
        let out_dims_vec = params.out_dims(); // Returns Vec<usize> [B, Cout_per_group, Lout]
        let out_shape: Shape = out_dims_vec.into(); // Convert to Shape
        let out_layout = Layout::contiguous(&out_shape); // Output is contiguous
        let new_storage = unsafe { device.alloc_uninit(&out_shape, self.dtype())? };

        // Get output length from calculated shape
        let l_out_calc = out_shape.dims()[2] as u32; // Lout is dim 2

        // --- Populate Push Constants ---
        let push_constants = Conv1DPushConstants {
            in_base,
            in_rank: in_rank as u32,
            ker_base,
            ker_rank: ker_rank as u32,
            in_stride: in_stride_arr,
            ker_stride: ker_stride_arr,

            elem_count: out_layout.shape().elem_count() as u32,
            b_size: params.b_size as u32,
            l_in: params.l_in as u32,
            c_out: params.c_out as u32, // Per-group C_out from params
            c_in: params.c_in as u32,   // Per-group C_in from params
            k_size: params.k_size as u32,
            l_out: l_out_calc, // Use calculated L_out
            padding: params.padding as u32,
            stride: params.stride as u32,
            dilation: params.dilation as u32,
        };

        // Print push constants for debugging if needed
        // println!("Conv1D PushConstants: {:?}", push_constants);

        // --- Synchronization and Dispatch ---
        self.pending_future.sync_if_needed()?;
        kernel.pending_future.sync_if_needed()?; // Sync kernel buffer too

        // Get output buffer (must exist)
        let output_buffer = (*new_storage.buffer).clone().ok_or_else(|| {
            VulkanError::Message("Output buffer allocation failed unexpectedly".into())
        })?; // Convert VulkanError

        // Dispatch one thread per output element
        new_storage.execute_compute_kernel(
            pipeline,
            vec![
                (*self.buffer).clone().unwrap(),   // Input buffer
                (*kernel.buffer).clone().unwrap(), // Kernel buffer
            ],
            vec![output_buffer],               // Output buffer
            [push_constants.elem_count, 1, 1], // Dispatch size (total threads)
            push_constants,                    // The populated push constants
            false,                             // Let execute_compute_kernel calculate workgroups
        )?;

        Ok(new_storage)
    }

    fn conv_transpose1d_op_impl(
        &self,                  // The input tensor storage
        layout: &Layout,        // Input tensor layout
        kernel: &Self,          // The kernel tensor storage
        kernel_layout: &Layout, // Kernel tensor layout
        params: &crate::conv::ParamsConvTranspose1D,
        pipeline: &Arc<ComputePipeline>,
    ) -> Result<Self> {
        // Define the push constant struct matching the shader's expected layout
        #[repr(C)]
        #[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct ConvTranspose1DPushConstants {
            // Input Tensor Layout (shape: [B, C_in_T, L_in_T]) - Max Rank 4
            in_base: u32,
            in_rank: u32,
            _pad_in: [u32; 2],   // Padding for alignment
            in_shape: [u32; 4],  // Padded shape [B, C_in_T, L_in_T, 1]
            in_stride: [u32; 4], // Padded strides

            // Kernel Tensor Layout (shape: [C_in_T, C_out_T, K]) - Max Rank 4 (Assuming groups=1)
            ker_base: u32,
            ker_rank: u32,
            _pad_ker: [u32; 2],   // Padding for alignment
            ker_shape: [u32; 4],  // Padded shape [C_in_T, C_out_T, K, 1]
            ker_stride: [u32; 4], // Padded strides

            // Convolution Parameters (Explicit dimensions for clarity)
            b_size: u32, // == in_shape[0]
            c_in: u32,   // == in_shape[1] == ker_shape[0]
            l_in: u32,   // == in_shape[2]
            c_out: u32,  // == ker_shape[1]
            k_size: u32, // == ker_shape[2]
            l_out: u32,  // Output dimension L

            // Convolution Algorithm Parameters
            padding: u32,
            stride: u32, // Forward stride
            dilation: u32,
            output_padding: u32, // (Unused in current shader)
        }

        // --- Extract Layout Info for Input Tensor ---
        let in_shape_slice = layout.shape();
        let in_stride_slice = layout.stride();
        let in_rank = in_shape_slice.rank() as u32;
        let mut in_shape_arr = [1u32; 4];
        let mut in_stride_arr = [1u32; 4]; // Use 1 for default stride like other ops
        for i in 0..(in_rank as usize).min(4) {
            in_shape_arr[i] = in_shape_slice.dims()[i]
                .try_into()
                .map_err(|_| VulkanError::Message("Input shape conversion failed".to_string()))?;
            // Make sure stride slice has enough elements before accessing
            if i < in_stride_slice.len() {
                in_stride_arr[i] = in_stride_slice[i] as u32;
            } else {
                // Handle cases where stride might be shorter than rank (shouldn't happen for conv usually)
                // If rank > stride.len(), calculate trailing contiguous strides or set sensible defaults.
                // For rank 3 shape [B, C, L], strides [S0, S1, S2], if rank=3, stride.len()=3, we are fine.
                // If somehow rank=4, stride.len=3, we might need to set stride[3]=1, but shape[3] is 1 anyway.
                // Sticking with default 1 seems safest if index out of bounds.
                in_stride_arr[i] = 1; // Default for potentially missing stride dimensions
            }
        }
        let in_base = layout.start_offset() as u32;

        // --- Extract Layout Info for Kernel Tensor ---
        let ker_shape_slice = kernel_layout.shape();
        let ker_stride_slice = kernel_layout.stride();
        let ker_rank = ker_shape_slice.rank() as u32;
        // Assuming groups=1, kernel rank should be 3: [C_in_T, C_out_T, K]
        if ker_rank != 3 {
            // Or handle groups here if supporting them
            return Err(VulkanError::Message(format!(
                "conv_transpose1d shader expects kernel rank 3 (got {})",
                ker_rank
            ))
            .into());
        }
        let mut ker_shape_arr = [1u32; 4];
        let mut ker_stride_arr = [1u32; 4];
        for i in 0..(ker_rank as usize).min(4) {
            // Will loop 3 times
            ker_shape_arr[i] = ker_shape_slice.dims()[i]
                .try_into()
                .map_err(|_| VulkanError::Message("Kernel shape conversion failed".to_string()))?;
            if i < ker_stride_slice.len() {
                ker_stride_arr[i] = ker_stride_slice[i] as u32;
            } else {
                ker_stride_arr[i] = 1; // Default
            }
        }
        let ker_base = kernel_layout.start_offset() as u32;

        // --- Allocate Output Buffer ---
        let device = self.device();
        // Calculate output shape using params (handles stride, padding etc.)
        let out_layout = Layout::contiguous(params.out_dims());
        let new_storage = unsafe { device.alloc_uninit(out_layout.shape(), self.dtype)? };
        let l_out_calc = params.l_out(); // Calculate final output length

        // --- Populate Push Constants ---
        let push_constants = ConvTranspose1DPushConstants {
            // Input layout
            in_base,
            in_rank,
            _pad_in: [0; 2],
            in_shape: in_shape_arr,
            in_stride: in_stride_arr,

            // Kernel layout
            ker_base,
            ker_rank,
            _pad_ker: [0; 2],
            ker_shape: ker_shape_arr,
            ker_stride: ker_stride_arr,

            // Convolution parameters (redundant with shapes but maybe clearer for shader)
            b_size: params.b_size as u32,
            c_in: params.c_in as u32,
            l_in: params.l_in as u32,
            c_out: params.c_out as u32,
            k_size: params.k_size as u32,
            l_out: l_out_calc as u32, // Use calculated output length

            // Algorithm parameters
            padding: params.padding as u32,
            stride: params.stride as u32, // Renamed from stride_conv
            dilation: params.dilation as u32,
            output_padding: params.output_padding as u32,
        };

        // --- Synchronization and Dispatch ---
        self.pending_future.sync_if_needed()?;
        kernel.pending_future.sync_if_needed()?; // Sync kernel too

        // Get output buffer (must exist since we just allocated it)
        let output_buffer = (*new_storage.buffer)
            .clone()
            .ok_or_else(|| VulkanError::Message("Output buffer allocation failed".into()))?;

        // Dispatch: One thread per output element
        let total_output_elements = out_layout.shape().elem_count() as u32;

        new_storage.execute_compute_kernel(
            pipeline,
            vec![
                (*self.buffer).clone().unwrap(),   // Input buffer
                (*kernel.buffer).clone().unwrap(), // Kernel buffer
            ],
            vec![output_buffer],           // Output buffer
            [total_output_elements, 1, 1], // Dispatch size (total threads)
            push_constants,                // The populated push constants
            false,                         // Let execute_compute_kernel calculate workgroups
        )?;

        Ok(new_storage)
    }

    fn conv2d_op_impl(
        &self,                  // Input tensor storage [B, Cin, Hin, Win]
        kernel: &Self,          // Kernel tensor storage [Cout, Cin/Groups, KH, KW]
        layout: &Layout,        // Input tensor layout
        kernel_layout: &Layout, // Kernel tensor layout
        params: &crate::conv::ParamsConv2D,
        pipeline: &Arc<ComputePipeline>,
    ) -> Result<Self> {
        // --- Push Constant Struct Definition ---
        #[repr(C)]
        #[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct Conv2DPushConstants {
            // Input Layout [B, Cin_per_group, Hin, Win] (Conceptually)
            in_base: u32,
            in_rank: u32,
            _pad_in: [u32; 2],
            in_shape: [u32; 4],
            in_stride: [u32; 4],
            // Kernel Layout [Cout_per_group, Cin_per_group, KH, KW]
            ker_base: u32,
            ker_rank: u32,
            _pad_ker: [u32; 2],
            ker_shape: [u32; 4],
            ker_stride: [u32; 4],
            // Dimensions
            b_size: u32,
            c_in: u32,
            h_in: u32,
            w_in: u32,
            c_out: u32,
            k_h: u32,
            k_w: u32,
            h_out: u32,
            w_out: u32,
            // Conv Params (Single values)
            padding: u32,
            stride: u32,
            dilation: u32,
        }

        // --- Rank Checks ---
        if layout.shape().rank() != 4 || kernel_layout.shape().rank() != 4 {
            return Err(VulkanError::Message(format!(
                "conv2d requires rank 4 tensors (input: {}, kernel: {})",
                layout.shape().rank(),
                kernel_layout.shape().rank()
            ))
            .into());
        }

        // --- Extract Input Layout ---
        let in_shape_slice = layout.shape();
        let in_stride_slice = layout.stride();
        let mut in_shape_arr = [1u32; 4];
        let mut in_stride_arr = [1u32; 4];
        // ... (similar loop as conv_transpose1d_op_impl to fill these based on rank 4) ...
        for i in 0..4 {
            in_shape_arr[i] = in_shape_slice.dims()[i] as u32; // Simplified assuming rank 4
            in_stride_arr[i] = in_stride_slice[i] as u32;
        }
        let in_base = layout.start_offset() as u32;

        // --- Extract Kernel Layout ---
        let ker_shape_slice = kernel_layout.shape(); // [Cout, Cin/G, KH, KW]
        let ker_stride_slice = kernel_layout.stride();
        let mut ker_shape_arr = [1u32; 4];
        let mut ker_stride_arr = [1u32; 4];
        // ... (similar loop as conv_transpose1d_op_impl to fill these based on rank 4) ...
        for i in 0..4 {
            ker_shape_arr[i] = ker_shape_slice.dims()[i] as u32; // Simplified assuming rank 4
            ker_stride_arr[i] = ker_stride_slice[i] as u32;
        }
        let ker_base = kernel_layout.start_offset() as u32;

        // --- Calculate Output Shape & Allocate ---
        let device = self.device();
        let out_shape = params.out_dims(); // [B, Cout, Hout, Wout]
        let h_out_calc = out_shape[2];
        let w_out_calc = out_shape[3];
        let out_layout = Layout::contiguous(params.out_dims());
        let new_storage = unsafe { device.alloc_uninit(out_layout.shape(), self.dtype)? };
        let output_buffer = (*new_storage.buffer).clone().unwrap(); // Should exist

        // --- Populate Push Constants ---
        let push_constants = Conv2DPushConstants {
            in_base,
            in_rank: 4, // Hardcoded for conv2d
            _pad_in: [0; 2],
            in_shape: in_shape_arr,
            in_stride: in_stride_arr,

            ker_base,
            ker_rank: 4, // Hardcoded for conv2d
            _pad_ker: [0; 2],
            ker_shape: ker_shape_arr, // Note: shape[1] is Cin/Groups
            ker_stride: ker_stride_arr,

            b_size: params.b_size as u32,
            c_in: params.c_in as u32, // This IS the per-group count
            h_in: params.i_h as u32,  // Use i_h/i_w from params
            w_in: params.i_w as u32,
            c_out: params.c_out as u32, // This IS the per-group count
            k_h: params.k_h as u32,
            k_w: params.k_w as u32,
            h_out: h_out_calc as u32, // Calculated H out
            w_out: w_out_calc as u32, // Calculated W out

            padding: params.padding as u32,   // Single value
            stride: params.stride as u32,     // Single value
            dilation: params.dilation as u32, // Single value
        };

        // --- Synchronization and Dispatch ---
        self.pending_future.sync_if_needed()?;
        kernel.pending_future.sync_if_needed()?;

        let total_output_elements = out_layout.shape().elem_count() as u32;
        new_storage.execute_compute_kernel(
            pipeline,
            vec![
                (*self.buffer).clone().unwrap(),
                (*kernel.buffer).clone().unwrap(),
            ],
            vec![output_buffer],
            [total_output_elements, 1, 1],
            push_constants,
            false,
        )?;

        Ok(new_storage)
    }

    fn conv_transpose2d_op_impl(
        &self,                  // Input tensor storage [B, CinT, Hin, Win]
        kernel: &Self,          // Kernel tensor storage [CinT, CoutT/Groups, KH, KW]
        layout: &Layout,        // Input tensor layout
        kernel_layout: &Layout, // Kernel tensor layout
        params: &crate::conv::ParamsConvTranspose2D,
        pipeline: &Arc<ComputePipeline>,
    ) -> Result<Self> {
        // Note: Result<Self> implies Result<Self, crate::Error>
        #[repr(C)]
        #[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct ConvTranspose2DPushConstants {
            // Input Layout [B, CinT, Hin, Win]
            in_base: u32,
            in_rank: u32,
            _pad_in: [u32; 2],
            in_shape: [u32; 4],
            in_stride: [u32; 4],
            // Kernel Layout [CinT, CoutT, KH, KW] (Since groups=1 always for now)
            ker_base: u32,
            ker_rank: u32,
            _pad_ker: [u32; 2],
            ker_shape: [u32; 4],
            ker_stride: [u32; 4],
            // Dimensions
            b_size: u32,
            c_in_t: u32,
            h_in: u32,
            w_in: u32,
            c_out_t: u32,
            k_h: u32,
            k_w: u32,
            h_out: u32,
            w_out: u32,
            // Conv Params
            padding: u32,
            stride: u32,
            dilation: u32,
            // No output_padding needed for shader, no groups
        }

        // <<< --- START: ADDED LAYOUT EXTRACTION --- >>>
        // --- Extract Input Layout ---
        let in_shape_slice = layout.shape();
        let in_stride_slice = layout.stride();
        let mut in_shape_arr = [1u32; 4];
        let mut in_stride_arr = [1u32; 4];
        for i in 0..4 {
            // Assume rank 4 based on checks
            in_shape_arr[i] = in_shape_slice.dims()[i] as u32;
            in_stride_arr[i] = in_stride_slice[i] as u32;
        }
        let in_base = layout.start_offset() as u32;

        // --- Extract Kernel Layout ---
        let ker_shape_slice = kernel_layout.shape(); // Shape is [CinT, CoutT, KH, KW]
        let ker_stride_slice = kernel_layout.stride();
        let mut ker_shape_arr = [1u32; 4];
        let mut ker_stride_arr = [1u32; 4];
        for i in 0..4 {
            // Assume rank 4
            ker_shape_arr[i] = ker_shape_slice.dims()[i] as u32;
            ker_stride_arr[i] = ker_stride_slice[i] as u32;
        }
        let ker_base = kernel_layout.start_offset() as u32;
        // <<< --- END: ADDED LAYOUT EXTRACTION --- >>>

        // --- Calculate Output Shape & Allocate ---
        let device = self.device();
        let out_shape_vec = params.out_dims(); // Returns Vec<usize>
        let out_shape: Shape = out_shape_vec.into(); // Convert Vec<usize> to Shape
        let h_out_calc = out_shape.dims()[2]; // Get from Shape
        let w_out_calc = out_shape.dims()[3]; // Get from Shape
                                              // Fix: Use the calculated Shape for Layout::contiguous
        let out_layout = Layout::contiguous(&out_shape);
        let new_storage = unsafe { device.alloc_uninit(out_layout.shape(), self.dtype)? };
        let output_buffer = (*new_storage.buffer).clone().unwrap();

        // --- Populate Push Constants ---
        let push_constants = ConvTranspose2DPushConstants {
            // Use the variables declared and initialized above
            in_base,
            in_rank: 4,
            _pad_in: [0; 2],
            in_shape: in_shape_arr,
            in_stride: in_stride_arr,
            ker_base,
            ker_rank: 4,
            _pad_ker: [0; 2],
            ker_shape: ker_shape_arr,
            ker_stride: ker_stride_arr,

            b_size: params.b_size as u32,
            c_in_t: params.c_in as u32, // Transpose Input Channels
            h_in: params.i_h as u32,
            w_in: params.i_w as u32,
            c_out_t: params.c_out as u32, // Transpose Output Channels
            k_h: params.k_h as u32,
            k_w: params.k_w as u32,
            h_out: h_out_calc as u32,
            w_out: w_out_calc as u32,

            padding: params.padding as u32,   // Single value
            stride: params.stride as u32,     // Single value
            dilation: params.dilation as u32, // Single value
        };

        // --- Synchronization and Dispatch ---
        self.pending_future.sync_if_needed()?;
        kernel.pending_future.sync_if_needed()?;

        let total_output_elements = out_layout.shape().elem_count() as u32;
        new_storage.execute_compute_kernel(
            pipeline,
            vec![
                (*self.buffer).clone().unwrap(),
                (*kernel.buffer).clone().unwrap(),
            ],
            vec![output_buffer],
            [total_output_elements, 1, 1],
            push_constants,
            false,
        )?;

        Ok(new_storage)
    }

    fn pool2d_op_impl(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        pipeline: &Arc<ComputePipeline>,
    ) -> Result<Self> {
        // Assumes crate::Error return
        let (k_h, k_w) = kernel_size;
        let (s_h, s_w) = stride;
        // --- Input Validation & Shape Calculation ---
        let in_shape = layout.shape();
        let in_dims = in_shape.dims();
        let rank = in_shape.rank();

        // Validate Rank (Require at least 4 dims for H and W)
        if rank < 2 {
            // Technically needs only H, W, so rank >= 2
            Err(VulkanError::Message(
                "Vulkan Pool2D requires at least 2 dimensions".into(),
            ))?;
        }
        if rank > MAX_RANK {
            Err(VulkanError::Message(format!(
                "Vulkan Pool2D only supports rank up to {}, got {}",
                MAX_RANK, rank
            )))?;
        }
        // Ensure kernel and stride are not zero
        if k_h == 0 || k_w == 0 {
            Err(VulkanError::Message(
                "pooling kernel size cannot be zero".into(),
            ))?;
        }
        if s_h == 0 || s_w == 0 {
            Err(VulkanError::Message("pooling stride cannot be zero".into()))?;
        }

        // Identify Height and Width dimensions (last two)
        let h_dim_idx = rank - 2;
        let w_dim_idx = rank - 1;
        let h_in = in_dims[h_dim_idx];
        let w_in = in_dims[w_dim_idx];

        // Calculate output dimensions (no padding support assumed)
        // Formula: floor((Input - Kernel) / Stride) + 1
        let h_out = if h_in >= k_h {
            (h_in - k_h) / s_h + 1
        } else {
            0
        };
        let w_out = if w_in >= k_w {
            (w_in - k_w) / s_w + 1
        } else {
            0
        };

        // Construct output shape, keeping batch/channel/other leading dims
        let mut out_dims_vec: Vec<usize> = in_dims[..h_dim_idx].to_vec(); // Copy leading dimensions
        out_dims_vec.push(h_out);
        out_dims_vec.push(w_out);
        let out_shape: Shape = out_dims_vec.into();

        // Handle potentially empty output gracefully (e.g., return tensor of correct shape but 0 elements)
        // The shader's bounds check will handle this, but allocation might fail?
        if h_out == 0 || w_out == 0 {
            println!(
                "Warning: Pool2D output dimension is zero (H={} W={}). Returning empty tensor.",
                h_out, w_out
            );
            // Return empty tensor matching output shape and dtype
            return self.device().zeros_impl(&out_shape, self.dtype());
        }

        // --- Push Constant Struct Definition ---
        #[repr(C)]
        #[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct Pool2DPushConstants {
            // Input Layout
            in_base: u32,
            in_rank: u32,               // Actual rank of the input tensor
            in_shape: [u32; MAX_RANK],  // Input shape padded with 1s
            in_stride: [u32; MAX_RANK], // Input strides padded with 1s

            // Output Layout (Contiguous NCHW-like based on rank)
            out_rank: u32,               // Same as input rank
            out_shape: [u32; MAX_RANK],  // Output shape padded with 1s
            out_stride: [u32; MAX_RANK], // Contiguous strides for output padded with 1s

            // Pooling Parameters
            k_h: u32,
            k_w: u32, // Kernel size H, W
            s_h: u32,
            s_w: u32, // Stride H, W
            // No padding param for now, assuming padding=0

            // Dimensions (for convenience in shader)
            h_in: u32,
            w_in: u32, // Input H, W size (from in_shape)
            h_out: u32,
            w_out: u32, // Output H, W size (from out_shape)

            // Total output elements for bounds check
            total_out_elems: u32,
            // Add padding if needed for struct alignment based on MAX_RANK
            // e.g., if MAX_RANK=8, size is large, might need padding to multiple of 16.
        }

        // --- Allocate Output ---
        let device = self.device();
        let new_storage = unsafe { device.alloc_uninit(&out_shape, self.dtype())? };
        let output_buffer = (*new_storage.buffer)
            .clone()
            .ok_or_else(|| VulkanError::Message("Output buffer allocation failed".into()))?; // Convert error

        // --- Extract Input Layout ---
        let in_stride_slice = layout.stride();
        let mut in_shape_arr = [1u32; MAX_RANK];
        let mut in_stride_arr = [1u32; MAX_RANK];
        for i in 0..rank {
            in_shape_arr[i] = in_dims[i] as u32;
            in_stride_arr[i] = in_stride_slice[i] as u32;
        }
        let in_base = layout.start_offset() as u32;

        // --- Calculate Contiguous Output Strides ---
        let out_dims = out_shape.dims(); // Use calculated output dimensions
        let mut out_shape_arr = [1u32; MAX_RANK];
        let mut out_stride_arr = [1u32; MAX_RANK];
        let mut current_out_stride = 1u32;
        for i in (0..rank).rev() {
            // Use input rank for output layout too
            let dim_size = out_dims[i] as u32;
            out_shape_arr[i] = dim_size;
            out_stride_arr[i] = current_out_stride;
            // Handle dimension size 0 correctly during stride calculation
            if dim_size > 0 {
                current_out_stride = current_out_stride.saturating_mul(dim_size);
            } else {
                // If dim size is 0, subsequent strides become effectively infinite/irrelevant
                // but setting stride to 1 is safer than 0 for the array.
                current_out_stride = 1; // Or keep previous value? Needs care.
            }
        }

        // --- Populate Push Constants ---
        let total_out_elems = out_shape.elem_count() as u32;
        let push_constants = Pool2DPushConstants {
            in_base,
            in_rank: rank as u32,
            in_shape: in_shape_arr,
            in_stride: in_stride_arr,

            out_rank: rank as u32, // Output rank is same
            out_shape: out_shape_arr,
            out_stride: out_stride_arr,

            k_h: k_h as u32,
            k_w: k_w as u32,
            s_h: s_h as u32,
            s_w: s_w as u32,

            h_in: h_in as u32,
            w_in: w_in as u32,
            h_out: h_out as u32,
            w_out: w_out as u32,

            total_out_elems,
        };

        // --- Synchronization and Dispatch ---
        self.pending_future.sync_if_needed()?;

        // Execute kernel
        new_storage.execute_compute_kernel(
            pipeline,
            vec![(*self.buffer).clone().unwrap()], // Input buffer
            vec![output_buffer],                   // Output buffer
            [total_out_elems, 1, 1],               // Dispatch one thread per output element
            push_constants,
            false, // Let helper calculate workgroups based on total_out_elems
        )?;

        Ok(new_storage)
    }

    fn upsample_nearest1d_op_impl(
        &self,
        layout: &Layout,
        scale_l: usize,
        pipeline: &Arc<ComputePipeline>,
    ) -> Result<Self> {
        // --- Push Constant Struct Definition ---
        #[repr(C)]
        #[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct UpsampleNearest1DPushConstants {
            // Input Layout [B, C, Lin]
            in_base: u32,
            in_rank: u32, // Expected: 3
            _pad_in: [u32; 2],
            in_shape: [u32; 4],  // Padded to 4D
            in_stride: [u32; 4], // Padded to 4D

            // Output Dimensions & Scale Factor
            b_size: u32,          // For output gid decoding
            c_size: u32,          // For output gid decoding
            l_out: u32,           // Output Length
            scale_l: u32,         // Scaling factor for L dimension
            total_out_elems: u32, // For bounds checking
        }

        let in_dims = layout.shape().dims();
        let b_size = in_dims[0];
        let c_size = in_dims[1];
        let l_in = in_dims[2];

        let l_out = l_in * scale_l;
        let out_shape = Shape::from(&[b_size, c_size, l_out]); // Rank 3 output

        // --- Allocate Output ---
        let device = self.device();
        let new_storage = unsafe { device.alloc_uninit(&out_shape, self.dtype())? };
        let output_buffer = (*new_storage.buffer).clone().unwrap(); // Should exist

        // --- Extract Input Layout (Pad to 4D for struct consistency) ---
        let in_shape_slice = layout.shape();
        let in_stride_slice = layout.stride();
        let in_rank = in_shape_slice.rank() as u32; // Will be 3
        let mut in_shape_arr = [1u32; 4];
        let mut in_stride_arr = [1u32; 4];
        for i in 0..(in_rank as usize).min(4) {
            // Loop 3 times
            in_shape_arr[i] = in_shape_slice.dims()[i] as u32;
            // Handle potentially shorter stride slice (though unlikely for rank 3)
            if i < in_stride_slice.len() {
                in_stride_arr[i] = in_stride_slice[i] as u32;
            } else {
                // Should not happen if rank == stride.len()
                in_stride_arr[i] = 1; // Default for safety
            }
        }
        let in_base = layout.start_offset() as u32;

        // --- Populate Push Constants ---
        let total_out_elems = out_shape.elem_count() as u32;
        let push_constants = UpsampleNearest1DPushConstants {
            in_base,
            in_rank, // Pass actual rank (3)
            _pad_in: [0; 2],
            in_shape: in_shape_arr,
            in_stride: in_stride_arr,

            b_size: b_size as u32,
            c_size: c_size as u32,
            l_out: l_out as u32,
            scale_l: scale_l as u32,
            total_out_elems,
        };

        // --- Synchronization and Dispatch ---
        self.pending_future.sync_if_needed()?;

        new_storage.execute_compute_kernel(
            pipeline,
            vec![(*self.buffer).clone().unwrap()], // Input buffer
            vec![output_buffer],                   // Output buffer
            [total_out_elems, 1, 1],               // Dispatch one thread per output element
            push_constants,
            false, // Let helper calculate workgroups
        )?;

        Ok(new_storage)
    }

    fn upsample_nearest2d_op_impl(
        &self,
        layout: &Layout,
        out_h: usize,
        out_w: usize,
        pipeline: &Arc<ComputePipeline>,
    ) -> Result<Self> {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct UpsampleNearest2DPushConstants {
            // Input Layout [B, C, Hin, Win]
            in_base: u32,
            in_rank: u32,
            _pad_in: [u32; 2],
            in_shape: [u32; 4],
            in_stride: [u32; 4],
            // Output Dimensions
            b_size: u32,
            c_size: u32,
            h_out: u32,
            w_out: u32,
            // Input Dimensions needed for scale calculation in shader
            h_in: u32,
            w_in: u32,
            total_out_elems: u32,
        }

        // --- Input Validation & Shape Calculation ---
        let in_dims = layout.shape().dims();
        if in_dims.len() != 4 { /* ... error ... */ }
        // Validate out_h, out_w perhaps?
        if out_h == 0 || out_w == 0 {
            return Err(crate::Error::Msg(
                "upsample_nearest2d output dimensions cannot be zero".to_string(),
            )
            .bt());
        }

        let b_size = in_dims[0];
        let c_size = in_dims[1];
        let h_in = in_dims[2];
        let w_in = in_dims[3];

        // Output shape uses the provided out_h, out_w (correct order now)
        let out_shape = Shape::from(&[b_size, c_size, out_h, out_w]);

        // --- Allocate Output ---
        let device = self.device();
        let new_storage = unsafe { device.alloc_uninit(&out_shape, self.dtype())? };
        let output_buffer = (*new_storage.buffer).clone().unwrap();

        // --- Extract Input Layout ---
        let in_shape_slice = layout.shape();
        let in_stride_slice = layout.stride();
        let mut in_shape_arr = [1u32; 4];
        let mut in_stride_arr = [1u32; 4];
        for i in 0..4 {
            in_shape_arr[i] = in_shape_slice.dims()[i] as u32;
            in_stride_arr[i] = in_stride_slice[i] as u32;
        }
        let in_base = layout.start_offset() as u32;

        // --- Populate Push Constants ---
        let total_out_elems = out_shape.elem_count() as u32;
        let push_constants = UpsampleNearest2DPushConstants {
            in_base,
            in_rank: 4,
            _pad_in: [0; 2],
            in_shape: in_shape_arr,
            in_stride: in_stride_arr,

            b_size: b_size as u32,
            c_size: c_size as u32,
            h_out: out_h as u32,
            w_out: out_w as u32,
            h_in: h_in as u32,
            w_in: w_in as u32,
            total_out_elems,
        };
        // --- Synchronization and Dispatch ---
        self.pending_future.sync_if_needed()?;

        new_storage.execute_compute_kernel(
            pipeline,
            vec![(*self.buffer).clone().unwrap()],
            vec![output_buffer],
            [total_out_elems, 1, 1],
            push_constants,
            false,
        )?;

        Ok(new_storage)
    }

    pub fn layernorm_op_impl(
        &self,
        layout: &Layout,
        gamma: &VulkanStorage,
        beta: &VulkanStorage,
        pipeline: &Arc<ComputePipeline>,
        normalized_axis: usize,
        eps: f32,
    ) -> Result<Self> {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PushConstants {
            base: u32,
            rank: u32,
            normalized_axis: u32,
            eps: f32,
            shape: [u32; 4],
            stride: [u32; 4],
        }

        let elem_count = layout.shape().elem_count();
        let device = self.device();
        let new_storage = unsafe { device.alloc_uninit(layout.shape(), self.dtype)? };

        // shape/stride extraction (rank <= 4)
        let shape_slice = layout.shape();
        let stride_slice = layout.stride();
        let mut shape_arr = [1u32; 4];
        let mut stride_arr = [1u32; 4];
        for i in 0..shape_slice.rank().min(4) {
            shape_arr[i] = shape_slice.dim(i).unwrap() as u32;
        }
        for i in 0..stride_slice.len().min(4) {
            stride_arr[i] = stride_slice[i] as u32;
        }

        let push_constants = PushConstants {
            base: layout.start_offset() as u32,
            rank: shape_slice.rank() as u32,
            normalized_axis: normalized_axis as u32,
            eps: eps as f32,
            shape: shape_arr,
            stride: stride_arr,
        };

        self.pending_future.sync_if_needed()?;
        gamma.pending_future.sync_if_needed()?;
        beta.pending_future.sync_if_needed()?;
        new_storage.execute_compute_kernel(
            pipeline,
            vec![
                (*self.buffer).clone().unwrap(),
                (*gamma.buffer).clone().unwrap(),
                (*beta.buffer).clone().unwrap(),
            ],
            vec![(*new_storage.buffer).clone().unwrap()],
            [elem_count as u32, 1, 1],
            push_constants,
            false,
        )?;

        Ok(new_storage)
    }

    pub fn rmsnorm_op_impl(
        &self,
        layout: &Layout,
        gamma: &VulkanStorage,
        pipeline: &Arc<ComputePipeline>,
        axis: usize,
        eps: f32,
    ) -> Result<Self> {
        // Assumes crate::Error
        // Check Rank Limit
        let rank = layout.shape().rank();
        if rank > 4 {
            return Err(VulkanError::Message(format!(
                "Vulkan RMSNorm only supports rank up to 4, got {}",
                rank
            ))
            .into()); // Convert VulkanError to crate::Error as needed
        }
        if axis >= rank {
            // Removed axis >= 4 check as rank is already <= 4
            return Err(VulkanError::Message(format!(
                "RMSNorm axis {} is out of bounds for rank {}",
                axis, rank
            ))
            .into());
        }

        #[repr(C)]
        #[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct PushConstants {
            in_base: u32,
            rank: u32,
            axis: u32,
            eps: f32,
            // Input Layout
            in_shape: [u32; 4],
            in_stride: [u32; 4],
            // Output Layout (contiguous, same shape as input)
            out_shape: [u32; 4],  // = in_shape
            out_stride: [u32; 4], // Contiguous strides based on shape
            norm_dim_size: u32,
        }

        let shape = layout.shape();
        let stride = layout.stride();
        // rank already calculated

        let mut in_shape_arr = [1u32; 4];
        let mut in_stride_arr = [1u32; 4];
        let mut out_shape_arr = [1u32; 4]; // This will be same as in_shape_arr
        let mut out_stride_arr = [1u32; 4];

        // Calculate contiguous output strides while populating shapes/input strides
        let mut current_out_stride = 1u32;
        for i in (0..rank).rev() {
            // Iterate backwards for contiguous stride calculation
            // Use shape.dims() directly which returns usize, then cast
            let dim_size_usize = shape.dims()[i];
            if dim_size_usize > u32::MAX as usize {
                // Safety check
                return Err(VulkanError::Message(format!(
                    "Dimension size {} exceeds u32::MAX",
                    dim_size_usize
                ))
                .into());
            }
            let dim_size = dim_size_usize as u32;

            in_shape_arr[i] = dim_size;
            out_shape_arr[i] = dim_size;
            in_stride_arr[i] = stride[i] as u32;

            out_stride_arr[i] = current_out_stride; // Assign current multiplier
            if dim_size > 0 {
                // Avoid multiplying by 0 if dim_size is 0
                current_out_stride = current_out_stride.saturating_mul(dim_size);
            } else {
                // If dim_size is 0, the stride for dimensions before it doesn't increase.
                // The stride *of* this dimension remains 1 (or the previous stride multiplier).
                // Let's keep it as the previous multiplier. This needs careful thought for rank 0 tensors though.
                // If dim_size is 0, current_out_stride effectively becomes infinite logically.
                // However, element count is 0, so maybe doesn't matter? Stick to simple mul.
                current_out_stride = current_out_stride.saturating_mul(dim_size); // Will become 0
                if current_out_stride == 0 && rank > i + 1 {
                    // If not the outermost dim
                    // How to represent stride for outer dims if inner is 0?
                    // Let's default back to 1? This implies a conceptual size of 1.
                    current_out_stride = 1;
                } else if current_out_stride == 0 && i == 0 {
                    current_out_stride = 1; // Base case
                }
            }
        }
        // Fill remaining strides for rank < 4
        for i in (rank..4).rev() {
            out_stride_arr[i] = current_out_stride;
            // Don't multiply current_out_stride further as shape is 1
        }

        // Correctly get norm_dim_size from the populated array
        let norm_dim_size = in_shape_arr[axis];
        if norm_dim_size == 0 {
            return Err(
                crate::Error::Msg("Cannot normalize over dimension of size 0".to_string()).bt(),
            );
        }
        // num_slices calculation requires usize
        let num_slices = shape.elem_count() / (norm_dim_size as usize);

        let push_constants = PushConstants {
            in_base: layout.start_offset() as u32,
            rank: rank as u32,
            axis: axis as u32,
            eps: eps,
            in_shape: in_shape_arr,
            in_stride: in_stride_arr,
            out_shape: out_shape_arr,
            out_stride: out_stride_arr,
            norm_dim_size,
        };

        let device = self.device();
        let output = unsafe { device.alloc_uninit(shape, self.dtype())? };

        self.pending_future.sync_if_needed()?;
        gamma.pending_future.sync_if_needed()?;

        output.execute_compute_kernel(
            pipeline,
            vec![
                (*self.buffer).clone().unwrap(),
                (*gamma.buffer).clone().unwrap(),
            ],
            vec![(*output.buffer).clone().unwrap()],
            [num_slices as u32, 1, 1],
            push_constants,
            true,
        )?;

        Ok(output)
    }

    pub fn softmax_last_dim_op_impl(
        &self,
        layout: &Layout,
        pipeline: &Arc<ComputePipeline>,
    ) -> Result<Self> {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PushConstants {
            base_offset: u32,
            rank: u32,
            axis: u32,
            _pad0: u32,
            shape: [u32; 4],
            stride: [u32; 4],
        }

        let axis = layout.shape().rank() - 1;

        let shape = layout.shape();
        let stride = layout.stride();

        let mut shape_arr = [1u32; 4];
        let mut stride_arr = [1u32; 4];

        for i in 0..shape.rank().min(4) {
            shape_arr[i] = shape.dim(i)? as u32;
        }
        for i in 0..stride.len().min(4) {
            stride_arr[i] = stride[i] as u32;
        }

        let push_constants = PushConstants {
            base_offset: layout.start_offset() as u32,
            rank: shape.rank() as u32,
            axis: axis as u32,
            _pad0: 0,
            shape: shape_arr,
            stride: stride_arr,
        };

        let device = self.device();
        let output = unsafe { device.alloc_uninit(shape, self.dtype())? };

        // Total number of softmax rows = product of dims except axis
        let mut nrows = 1u32;
        for i in 0..shape.rank() {
            if i != axis {
                nrows *= shape.dim(i)? as u32;
            }
        }

        self.pending_future.sync_if_needed()?;
        output.execute_compute_kernel(
            pipeline,
            vec![(*self.buffer).clone().unwrap()],
            vec![(*output.buffer).clone().unwrap()],
            [nrows, 1, 1],
            push_constants,
            false,
        )?;

        Ok(output)
    }

    pub fn rope_op_impl(
        &self,
        layout: &Layout,
        cos: &Self,
        sin: &Self,
        pipeline: &Arc<ComputePipeline>,
    ) -> Result<Self> {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PushConstants {
            shape: [u32; 4],
            strides: [u32; 4],
            base: u32,
        }

        let input_buf = (*self.buffer)
            .clone()
            .ok_or_else(|| VulkanError::Message("rope_op_impl: missing input buffer".into()))?;
        let cos_buf = (*cos.buffer)
            .clone()
            .ok_or_else(|| VulkanError::Message("rope_op_impl: missing cos buffer".into()))?;
        let sin_buf = (*sin.buffer)
            .clone()
            .ok_or_else(|| VulkanError::Message("rope_op_impl: missing sin buffer".into()))?;

        let shape = layout.shape().dims();
        if shape.len() != 4 {
            return Err(VulkanError::Message(format!(
                "Expected shape [B, H, T, D], got {:?}",
                shape
            )))?;
        }
        let base = layout.start_offset() as u32;

        let out = unsafe { self.device().alloc_uninit(layout.shape(), self.dtype)? };

        let strides = layout.stride();
        if strides.len() != 4 {
            return Err(VulkanError::Message("Expected 4D stride layout".into()))?;
        }

        let strides: [u32; 4] = strides
            .iter()
            .copied()
            .map(|s| s as u32)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let shape: [u32; 4] = shape
            .iter()
            .copied()
            .map(|s| s as u32)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let push_constants = PushConstants {
            shape,
            strides,
            base,
        };

        let total_elems = layout.shape().elem_count() as u32;

        self.pending_future.sync_if_needed()?;
        cos.pending_future.sync_if_needed()?;
        sin.pending_future.sync_if_needed()?;
        out.execute_compute_kernel(
            pipeline,
            vec![input_buf, cos_buf, sin_buf],
            vec![(*out.buffer).clone().unwrap()],
            [total_elems, 1, 1],
            push_constants,
            false,
        )?;

        Ok(out)
    }

    pub fn rope_i_op_impl(
        &self,
        layout: &Layout,
        cos: &VulkanStorage,
        sin: &VulkanStorage,
        pipeline: &Arc<ComputePipeline>,
    ) -> Result<VulkanStorage> {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PushConstants {
            shape: [u32; 4],
            strides: [u32; 4],
            base: u32,
        }

        // Retrieve input buffers.
        let input_buf = (*self.buffer)
            .clone()
            .ok_or_else(|| VulkanError::Message("rope_i_op_impl: missing input buffer".into()))?;
        let cos_buf = (*cos.buffer)
            .clone()
            .ok_or_else(|| VulkanError::Message("rope_i_op_impl: missing cos buffer".into()))?;
        let sin_buf = (*sin.buffer)
            .clone()
            .ok_or_else(|| VulkanError::Message("rope_i_op_impl: missing sin buffer".into()))?;

        // Ensure the input tensor is 4D.
        let dims = layout.shape().dims();
        if dims.len() != 4 {
            Err(VulkanError::Message(format!(
                "rope_i_op_impl: expected 4D tensor, got shape {:?}",
                dims
            )))?;
        }
        let base = layout.start_offset() as u32;
        let shape: [u32; 4] = dims
            .iter()
            .map(|&d| d as u32)
            .collect::<Vec<u32>>()
            .try_into()
            .unwrap();
        let strides_vec: Vec<u32> = layout.stride().iter().map(|&s| s as u32).collect();
        let strides: [u32; 4] = strides_vec.try_into().unwrap();

        let push_constants = PushConstants {
            shape,
            strides,
            base,
        };

        // Allocate output storage with the same shape and data type.
        let out = unsafe { self.device().alloc_uninit(layout.shape(), self.dtype())? };

        // The number of pairs = B * H * T * (D/2)
        let half = shape[3] >> 1;
        let total_pairs = shape[0] * shape[1] * shape[2] * half;

        self.pending_future.sync_if_needed()?;
        cos.pending_future.sync_if_needed()?;
        sin.pending_future.sync_if_needed()?;

        // Dispatch one thread per pair.
        out.execute_compute_kernel(
            pipeline,
            vec![input_buf, cos_buf, sin_buf],
            vec![(*out.buffer).clone().unwrap()],
            [total_pairs, 1, 1],
            push_constants,
            false,
        )?;

        Ok(out)
    }

    pub fn random_impl(
        &self,
        shape: &Shape,
        pipeline: &Arc<ComputePipeline>,
        target_dtype: DType,
        seed: u64,
        arg0: f64, // low / mean
        arg1: f64, // high / stddev
    ) -> Result<Self> {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PushConstants {
            seed: u64,
            arg0: f32,
            arg1: f32,
        }

        if let Some(buffer) = (*self.buffer).clone() {
            let elem_count = shape.elem_count();
            let device = self.device();
            let new_storage = unsafe { device.alloc_uninit(shape, target_dtype)? };

            let arg0 = arg0 as f32;
            let arg1 = arg1 as f32;

            let push_constants = PushConstants { seed, arg0, arg1 };
            new_storage.execute_compute_kernel(
                pipeline,
                vec![],
                vec![(*new_storage.buffer).clone().unwrap()], // XXX
                [elem_count as u32, 1, 1],
                push_constants,
                false,
            )?;
            self.device
                .set_seed(pcg32_advance(seed, elem_count as u64))?;

            Ok(new_storage)
        } else {
            // Zero-sized buffer, return zero-sized buffer
            Ok(self.clone())
        }
    }

    pub fn gemm_impl(
        &self,
        rhs: &Self,
        dst: &Self,      // Output tensor (assumed contiguous NCHW-like [Batch..., M, N])
        layout: &Layout, // Layout for A [Batch..., M, K]
        rhs_layout: &Layout, // Layout for B [Batch..., K, N]
        pipeline: &Arc<ComputePipeline>,
        // M, N, K dimensions identified by the caller (e.g., candle-core matmul)
        // b_dims: &[usize], // List of batch dimension sizes (product is total batches) - OR pass total_batches
        // m_dim_idx, k_dim_idx_a, k_dim_idx_b, n_dim_idx // Indices of M, K, N dims? Less common.
        // Let's stick to the M,N,K values and calculate batching based on rank.
        (_b_total_usize, m_usize, n_usize, k_usize): (usize, usize, usize, usize),
    ) -> Result<()> {
        // Assumes return type Result<(), crate::Error>
        let a_rank = layout.shape().rank();
        let b_rank = rhs_layout.shape().rank();

        // --- Validate Ranks (MatMul requires at least Rank 2) ---
        if a_rank < 2 || b_rank < 2 {
            Err(VulkanError::Message(format!(
                "GEMM requires input ranks >= 2 (got A: {}, B: {})",
                a_rank, b_rank
            )))?;
        }

        // --- Identify M, K (A) and K, N (B) dimension indices ---
        // Standard convention: last two dimensions are matrix dims
        let a_m_dim_idx = a_rank - 2;
        let a_k_dim_idx = a_rank - 1;
        let b_k_dim_idx = b_rank - 2;
        let b_n_dim_idx = b_rank - 1;

        // --- Validate Shapes ---
        let a_dims = layout.shape().dims();
        let b_dims = rhs_layout.shape().dims();
        if a_dims[a_k_dim_idx] != b_dims[b_k_dim_idx] {
            // Check K dimension match
            Err(VulkanError::Message(format!(
                "GEMM K dimension mismatch: A ({}) != B ({})",
                a_dims[a_k_dim_idx], b_dims[b_k_dim_idx]
            )))?;
        }
        // Check batch dimensions match (ignoring M,K,N)
        let num_batch_dims = a_rank - 2;
        if num_batch_dims != b_rank - 2 {
            // Ranks differ in batch part, check if one can broadcast to the other?
            // For simple GEMM, usually require same batch shape.
            Err(VulkanError::Message("GEMM batch rank mismatch".into()))?;
        }
        for i in 0..num_batch_dims {
            if a_dims[i] != b_dims[i] {
                Err(VulkanError::Message(format!(
                    "GEMM batch dimension {} mismatch: A ({}) != B ({})",
                    i, a_dims[i], b_dims[i]
                )))?;
            }
        }

        // Verify passed M, N, K match layout (use usize versions first)
        if a_dims[a_m_dim_idx] != m_usize
            || a_dims[a_k_dim_idx] != k_usize
            || b_dims[b_n_dim_idx] != n_usize
        {
            Err(VulkanError::Message(format!(
                "GEMM M,N,K parameters ({},{},{}) do not match tensor shapes A{:?}, B{:?}",
                m_usize, n_usize, k_usize, a_dims, b_dims
            )))?;
        }

        // Calculate total number of batches
        let total_batches: usize = a_dims[0..num_batch_dims].iter().product();
        // Sanity check parameter _b_total_usize?
        // if total_batches != _b_total_usize { /* Error */ }

        // --- Push Constant Struct ---
        #[repr(C)]
        #[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct GemmPushConstants {
            m: u32,
            n: u32,
            k: u32,
            // Strides identified by *meaning* for the batched GEMM operation
            a_batch_stride: u32, // Stride between logical batches in A's buffer
            a_m_stride: u32,     // Stride for M dimension in A
            a_k_stride: u32,     // Stride for K dimension in A
            b_batch_stride: u32, // Stride between logical batches in B's buffer
            b_k_stride: u32,     // Stride for K dimension in B
            b_n_stride: u32,     // Stride for N dimension in B
            // Base offsets
            a_base: u32,
            b_base: u32,
            // Output layout (assume contiguous [total_batches, M, N])
            ldc: u32,            // Leading dimension C (usually N)
            c_batch_stride: u32, // Stride between batches C (usually M*N)
            // Scalars
            alpha: f32,
            beta: f32,
            // Add padding if needed for alignment
        }

        // --- Determine Strides for Batched GEMM View ---
        let a_strides = layout.stride();
        let b_strides = rhs_layout.stride();

        // Strides for the core matrix dimensions (M, K for A; K, N for B)
        let a_m_stride = a_strides[a_m_dim_idx] as u32;
        let a_k_stride = a_strides[a_k_dim_idx] as u32;
        let b_k_stride = b_strides[b_k_dim_idx] as u32;
        let b_n_stride = b_strides[b_n_dim_idx] as u32;

        // Calculate stride between logical batches.
        // This is the stride of the slowest-moving batch dimension.
        // If num_batch_dims > 0, it's stride[num_batch_dims - 1].
        // If num_batch_dims == 0 (rank 2 input), the concept of batch stride is 0 or irrelevant.
        // The shader uses 'batch * stride', so a stride of 0 works mathematically if total_batches is 1.
        let a_batch_stride = if num_batch_dims > 0 {
            a_strides[num_batch_dims - 1] as u32
        } else {
            0
        };
        let b_batch_stride = if num_batch_dims > 0 {
            b_strides[num_batch_dims - 1] as u32
        } else {
            0
        };
        // Note: This assumes the flattened 'batch' index in the shader (0..total_batches-1)
        // maps correctly using the stride of the dimension *just before* M/K.
        // This might need adjustment if batch dimensions themselves are not contiguous.

        // Base offsets
        let a_base = layout.start_offset() as u32;
        let b_base = rhs_layout.start_offset() as u32;

        // Output layout strides (assuming contiguous dst = [total_batches, M, N])
        let m_u32 = m_usize as u32;
        let n_u32 = n_usize as u32;
        let k_u32 = k_usize as u32; // Cast M, N, K once
        let ldc = n_u32; // Stride between rows (M dim) is N elements
        let c_batch_stride = m_u32 * ldc; // Stride between batches is M*N elements

        if let (Some(lhs_buffer), Some(rhs_buffer)) =
            ((*self.buffer).clone(), (*rhs.buffer).clone())
        {
            // Get dst buffer safely
            let dst_buffer = (*dst.buffer)
                .clone()
                .ok_or_else(|| VulkanError::Message("Destination buffer is missing".into()))?; // Convert error

            // Build the push constants
            let push_constants = GemmPushConstants {
                m: m_u32,
                n: n_u32,
                k: k_u32,
                a_batch_stride,
                a_m_stride,
                a_k_stride,
                b_batch_stride,
                b_k_stride,
                b_n_stride,
                a_base,
                b_base,
                ldc,
                c_batch_stride,
                alpha: 1f32, // Or allow passing alpha/beta
                beta: 0f32,  // Assume C is zeroed - if not, need to pass beta=1
            };

            // Dispatch dimensions based on total batches, M, N
            let tile_size = 16u32; // Common tile size
            let wg_x = (n_u32 + tile_size - 1) / tile_size; // Workgroups along N dimension
            let wg_y = (m_u32 + tile_size - 1) / tile_size; // Workgroups along M dimension
            let wg_z = total_batches as u32; // Workgroups for batches

            // Sync inputs (dst sync is often implicit if newly created, but explicit is safer)
            self.pending_future.sync_if_needed()?;
            rhs.pending_future.sync_if_needed()?;
            dst.pending_future.sync_if_needed()?; // Sync dst if beta != 0 or reuse

            dst.execute_compute_kernel(
                pipeline,
                vec![lhs_buffer, rhs_buffer], // Input buffers A, B
                vec![dst_buffer],             // Output buffer C
                [wg_x, wg_y, wg_z],           // Dispatch grid size
                push_constants,               // Push constants with calculated strides/offsets
                true, // direct_dispatch = true (we calculated exact workgroups)
            )?;
        } else {
            // Handle cases where input buffers might be None (e.g., zero-sized tensors)
            if layout.shape().elem_count() == 0 || rhs_layout.shape().elem_count() == 0 {
                // If either input is empty, output should be zeros (or handle according to GEMM rules)
                // Potentially fill dst with zeros here if needed.
                // For now, just succeed silently if an input buffer is None (likely zero elements)
            } else {
                // This case (Some() check failed but elem_count > 0) shouldn't happen if alloc works
                Err(VulkanError::Message(
                    "Input buffer missing unexpectedly in GEMM".into(),
                ))?;
            }
        }
        Ok(())
    }
}

// PCG32 constants for 64-bit state
const PCG_MULTIPLIER: u64 = 6364136223846793005u64;
const PCG_INCREMENT: u64 = 1442695040888963407u64;

fn pcg32_advance(state: u64, delta: u64) -> u64 {
    let mut acc_mult = 1u64;
    let mut acc_plus = 0u64;
    let mut cur_mult = PCG_MULTIPLIER;
    let mut cur_plus = PCG_INCREMENT;
    let mut delta = delta;
    while delta > 0 {
        if (delta & 1) != 0 {
            acc_mult = acc_mult.wrapping_mul(cur_mult);
            acc_plus = acc_plus.wrapping_mul(cur_mult).wrapping_add(cur_plus);
        }
        cur_plus = (cur_mult.wrapping_add(1)).wrapping_mul(cur_plus);
        cur_mult = cur_mult.wrapping_mul(cur_mult);
        delta >>= 1;
    }
    acc_mult.wrapping_mul(state).wrapping_add(acc_plus)
}

macro_rules! fail {
    () => {
        todo!("vulkan support is incomplete, this function is not yet implemented")
    };
}

impl BackendStorage for VulkanStorage {
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
        let suffix = match self.dtype {
            DType::F32 => "f32",
            DType::BF16 => "bf16",
            DType::I64 => "i64",
            _ => todo!("Unsupported dtype {:?}", self.dtype),
        };
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(self.device.device(), &format!("affine_{}", suffix), None)
            .map_err(VulkanError::from)?;
        self.affine_elu_op_impl(layout, &pipeline, mul, add, 0.0)
    }

    fn powf(&self, layout: &Layout, exp: f64) -> Result<Self> {
        let suffix = match self.dtype {
            DType::F32 => "f32",
            DType::BF16 => "bf16",
            _ => todo!("Unsupported dtype {:?}", self.dtype),
        };
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(self.device.device(), &format!("powf_{}", suffix), None)
            .map_err(VulkanError::from)?;
        self.affine_elu_op_impl(layout, &pipeline, exp, 0.0, 0.0)
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        let suffix = match self.dtype {
            DType::F32 => "f32",
            DType::BF16 => "bf16",
            _ => todo!("Unsupported dtype {:?}", self.dtype),
        };
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(self.device.device(), &format!("elu_{}", suffix), None)
            .map_err(VulkanError::from)?;
        self.affine_elu_op_impl(layout, &pipeline, 0.0, 0.0, alpha)
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, s: &[usize]) -> Result<Self> {
        let suffix = match self.dtype {
            DType::F32 => "f32",
            DType::U32 => "u32",
            DType::I64 => "i64",
            _ => todo!("Unsupported dtype {:?}", self.dtype),
        };
        match op {
            ReduceOp::Max | ReduceOp::Min | ReduceOp::Sum => {
                let partial_key = format!("{}_partial_{}", op.name(), suffix);
                let partial_pipeline = self
                    .device
                    .kernels()
                    .load_pipeline(self.device.device(), &partial_key, None)
                    .map_err(VulkanError::from)?;
                let combine_key = format!("{}_combine_{}", op.name(), suffix);
                let combine_pipeline = self
                    .device
                    .kernels()
                    .load_pipeline(self.device.device(), &combine_key, None)
                    .map_err(VulkanError::from)?;
                self.reduce_op_impl(layout, s, &partial_pipeline, &combine_pipeline, false)
            }
            ReduceOp::ArgMax | ReduceOp::ArgMin => {
                let partial_key = format!("{}_partial_{}", op.name(), suffix);
                let partial_pipeline = self
                    .device
                    .kernels()
                    .load_pipeline(self.device.device(), &partial_key, None)
                    .map_err(VulkanError::from)?;
                let combine_key = format!("{}_combine_{}", op.name(), suffix);
                let combine_pipeline = self
                    .device
                    .kernels()
                    .load_pipeline(self.device.device(), &combine_key, None)
                    .map_err(VulkanError::from)?;
                self.reduce_op_impl(layout, s, &partial_pipeline, &combine_pipeline, true)
            }
        }
    }

    fn cmp(&self, cmp_op: CmpOp, rhs: &Self, layout: &Layout, rhs_layout: &Layout) -> Result<Self> {
        let suffix = match (self.dtype, rhs.dtype) {
            (DType::F32, DType::F32) => "f32",
            (DType::U32, DType::U32) => "u32",
            (DType::I64, DType::I64) => "i64",
            _ => todo!("Unsupported dtype combo {:?} {:?}", self.dtype, rhs.dtype),
        };
        let key = format!("{}_{}", cmp_op.name(), suffix);
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(self.device.device(), &key, None)
            .map_err(VulkanError::from)?;

        // Allocate new storage for the result.
        // We choose U32 to store 1 for true and 0 for false.
        let device = self.device();
        let new_storage = unsafe { device.alloc_uninit(layout.shape(), DType::U8)? };

        // Call the lower-level helper.
        self.cmp_op_impl(rhs, layout, rhs_layout, &mut new_storage.clone(), &pipeline)?;
        Ok(new_storage)
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        if self.dtype == dtype {
            Ok(self.clone())
        } else {
            let kernel = match (self.dtype, dtype) {
                (DType::F32, DType::F16) => "cast_f32_f16",
                (DType::F16, DType::F32) => "cast_f16_f32",
                (DType::U32, DType::F32) => "cast_u32_f32",
                (DType::U32, DType::U8) => "cast_u32_u8",
                (DType::U8, DType::F32) => "cast_u8_f32",
                (DType::BF16, DType::F32) => "cast_bf16_f32",
                (DType::F32, DType::BF16) => "cast_f32_bf16",
                (DType::BF16, DType::U32) => "cast_bf16_u32",
                (DType::U32, DType::BF16) => "cast_u32_bf16",
                (DType::BF16, DType::F16) => "cast_bf16_f16",
                (DType::U32, DType::I64) => "cast_u32_i64",
                (DType::I64, DType::U32) => "cast_i64_u32",
                _ => todo!("Unsupported dtype combo {:?} {:?}", self.dtype, dtype),
            };
            let pipeline = self
                .device
                .kernels()
                .load_pipeline(self.device.device(), kernel, None)
                .map_err(VulkanError::from)?;
            self.unary_op_impl(layout, &pipeline, dtype)
        }
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let suffix = match self.dtype {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            _ => todo!("Unsupported dtype {:?}", self.dtype),
        };
        let key = format!("{}_{}", B::NAME, suffix);
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(self.device.device(), &key, None)
            .map_err(VulkanError::from)?;
        self.unary_op_impl(layout, &pipeline, self.dtype())
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        let suffix = match (self.dtype, rhs.dtype) {
            (DType::F32, DType::F32) => "f32",
            (DType::I64, DType::I64) => "i64",
            (DType::BF16, DType::BF16) => "bf16",
            (DType::F16, DType::F16) => "f16",
            _ => todo!("Unsupported dtype combo {:?} {:?}", self.dtype, rhs.dtype),
        };
        let key = format!("{}_{}", B::NAME, suffix);
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(self.device.device(), &key, None)
            .map_err(VulkanError::from)?;
        self.binary_op_impl(layout, rhs, rhs_layout, &pipeline)
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        _t_l: &Layout,
        f: &Self,
        _f_l: &Layout,
    ) -> Result<Self> {
        let shape = layout.shape();
        let dtype = t.dtype;
        let buffer = unsafe { self.device.alloc_uninit(shape, dtype) }?;
        if t.dtype() != f.dtype() {
            crate::bail!(
                "Invalid where: different dtypes for values {:?} != {:?}",
                t.dtype(),
                f.dtype()
            );
        }
        let suffix = match (self.dtype, t.dtype()) {
            (DType::U8, DType::F32) => "u8_f32",
            (DType::U32, DType::F32) => "u32_f32",
            (DType::U8, DType::BF16) => "u8_bf16",
            (DType::U8, DType::F16) => "u8_f16",
            (DType::U8, DType::I64) => "u8_i64",
            (DType::U8, DType::U32) => "u8_u32",
            (DType::U8, DType::U8) => "u8_u8",
            (left, right) => crate::bail!("Vulkan where_cond {left:?} {right:?} not implemented"),
        };
        let key = format!("where_{}", suffix);
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(self.device.device(), &key, None)
            .map_err(VulkanError::from)?;

        buffer.where_cond_op_impl(layout, self, t, f, &pipeline)?;

        Ok(buffer)
    }

    fn conv1d(
        &self,
        layout: &Layout, // input layout; assumed shape: [b, c_in, l_in]
        kernel: &Self,
        kernel_layout: &Layout, // <<< ADDED kernel_layout parameter
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        // Assumes crate::Error return
        // Dtype check (as before)
        if self.dtype != kernel.dtype {
            crate::bail!(
                "Invalid conv1d: mismatched dtypes {:?} vs {:?}",
                self.dtype,
                kernel.dtype
            );
        }

        // Select shader based on dtype
        let suffix = match self.dtype {
            DType::F32 => "f32",
            DType::BF16 => "bf16",
            DType::F16 => "f16",
            // Add other supported types here
            _ => crate::bail!("Vulkan conv1d unsupported dtype {:?}", self.dtype),
        };
        let key = format!("conv1d_{}", suffix);
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(self.device.device(), &key, None)
            .map_err(VulkanError::from)?; // Map VulkanKernelError -> VulkanError -> crate::Error

        // Call the implementation function, passing both layouts
        self.conv1d_op_impl(kernel, layout, kernel_layout, params, &pipeline)
    }

    fn conv_transpose1d(
        &self,
        layout: &Layout, // input layout; assumed shape: [b, c_in, l_in]
        kernel: &Self,
        kernel_layout: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        if self.dtype != kernel.dtype {
            crate::bail!(
                "Invalid conv_transpose1d: mismatched dtypes {:?} vs {:?}",
                self.dtype,
                kernel.dtype
            );
        }

        // Select the appropriate shader pipeline based on dtype.
        let suffix = match (self.dtype, kernel.dtype) {
            (DType::F32, DType::F32) => "f32",
            (DType::F16, DType::F16) => "f16",
            (DType::BF16, DType::BF16) => "bf16",
            (DType::U32, DType::U32) => "u32",
            (DType::U8, DType::U8) => "u8",
            (left, right) => {
                crate::bail!("Vulkan conv_transpose1d {left:?} {right:?} not implemented")
            }
        };
        let key = format!("conv_transpose1d_{}", suffix);
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(self.device.device(), &key, None)
            .map_err(VulkanError::from)?;

        self.conv_transpose1d_op_impl(layout, kernel, kernel_layout, params, &pipeline)
    }

    fn conv2d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_layout: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        if self.dtype != kernel.dtype { /* ... error ... */ }
        let suffix = match self.dtype {
            // Determine suffix based on dtype
            DType::F32 => "f32",
            DType::BF16 => "bf16",
            // DType::F16 => "f16",
            _ => crate::bail!("Vulkan conv2d unsupported dtype {:?}", self.dtype),
        };
        let key = format!("conv2d_{}", suffix);
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(self.device.device(), &key, None)
            .map_err(VulkanError::from)?;
        self.conv2d_op_impl(kernel, layout, kernel_layout, params, &pipeline)
    }

    fn conv_transpose2d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_layout: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        if self.dtype != kernel.dtype { /* ... error ... */ }
        let suffix = match self.dtype {
            // Determine suffix based on dtype
            DType::F32 => "f32",
            DType::BF16 => "bf16",
            // DType::F16 => "f16",
            _ => crate::bail!("Vulkan conv_transpose2d unsupported dtype {:?}", self.dtype),
        };
        let key = format!("conv_transpose2d_{}", suffix);
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(self.device.device(), &key, None)
            .map_err(VulkanError::from)?; // Convert VulkanKernelError -> VulkanError

        self.conv_transpose2d_op_impl(kernel, layout, kernel_layout, params, &pipeline)
    }

    fn avg_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        let suffix = match self.dtype() {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            _ => crate::bail!("Vulkan avg_pool2d unsupported dtype {:?}", self.dtype()),
        };
        let key = format!("pool2d_avg_{}", suffix);
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(self.device.device(), &key, None) // Pass define
            .map_err(VulkanError::from)?;
        self.pool2d_op_impl(layout, kernel_size, stride, &pipeline)
    }

    fn max_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        let suffix = match self.dtype() {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            _ => crate::bail!("Vulkan max_pool2d unsupported dtype {:?}", self.dtype()),
        };
        let key = format!("pool2d_max_{}", suffix);
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(self.device.device(), &key, None) // Pass define
            .map_err(VulkanError::from)?;
        self.pool2d_op_impl(layout, kernel_size, stride, &pipeline)
    }

    fn upsample_nearest1d(&self, layout: &Layout, scale_l: usize) -> Result<Self> {
        let suffix = match self.dtype() {
            DType::F32 => "f32",
            DType::BF16 => "bf16",
            DType::F16 => "f16",
            DType::U8 => "u8",
            DType::U32 => "u32",
            _ => crate::bail!(
                "Vulkan upsample_nearest1d unsupported dtype {:?}",
                self.dtype()
            ),
        };
        let key = format!("upsample_nearest1d_{}", suffix);
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(self.device.device(), &key, None)
            .map_err(VulkanError::from)?;

        self.upsample_nearest1d_op_impl(layout, scale_l, &pipeline)
    }

    fn upsample_nearest2d(&self, layout: &Layout, out_h: usize, out_w: usize) -> Result<Self> {
        let suffix = match self.dtype() {
            // Use self.dtype()
            DType::F32 => "f32",
            DType::BF16 => "bf16",
            DType::F16 => "f16",
            DType::U8 => "u8",
            DType::U32 => "u32",
            // Add I64 etc. if implemented
            _ => crate::bail!(
                "Vulkan upsample_nearest2d unsupported dtype {:?}",
                self.dtype()
            ),
        };
        let key = format!("upsample_nearest2d_{}", suffix);
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(self.device.device(), &key, None)
            .map_err(VulkanError::from)?; // Map error

        self.upsample_nearest2d_op_impl(layout, out_h, out_w, &pipeline)
    }

    fn gather(
        &self,
        src_layout: &Layout,
        index: &Self,
        idx_layout: &Layout,
        dim: usize,
    ) -> Result<Self> {
        if src_layout.shape().rank() != idx_layout.shape().rank() {
            crate::bail!("gather requires tensors of the same rank");
        }

        let out_shape = idx_layout.shape().dims().to_owned();

        let suffix = match (index.dtype, self.dtype) {
            (DType::U8, DType::F32) => "u8_f32",
            (DType::U32, DType::F32) => "u32_f32",
            (DType::I64, DType::F32) => "i64_f32",
            _ => crate::bail!("gather: unsupported dtype combination"),
        };
        let key = format!("gather_{}", suffix);
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(self.device.device(), &key, None)
            .map_err(VulkanError::from)?;

        let dst = unsafe { self.device().alloc_uninit(&out_shape.into(), self.dtype)? };

        self.gather_op_impl(&dst, index, src_layout, idx_layout, &pipeline, dim)
    }

    fn scatter_set(
        &mut self,
        layout: &Layout,
        index: &Self,
        idx_layout: &Layout,
        src: &Self,
        src_layout: &Layout,
        dim: usize,
    ) -> Result<()> {
        if src_layout.shape().dims() != idx_layout.shape().dims() {
            crate::bail!("scatter_set: index and source shapes must match");
        }

        let dtype = self.dtype;
        let idtype = index.dtype;

        let suffix = match (idtype, dtype) {
            (DType::U8, DType::F32) => "u8_f32",
            (DType::U32, DType::F32) => "u32_f32",
            (DType::I64, DType::F32) => "i64_f32",
            _ => crate::bail!("scatter_set: unsupported dtype combination"),
        };

        let key = format!("scatter_set_{}", suffix);
        let pipeline = self
            .device()
            .kernels()
            .load_pipeline(
                self.device().device(),
                &key,
                Some(&[(
                    "FLOAT32_ATOMIC_ADD",
                    if self
                        .device
                        .device()
                        .enabled_features()
                        .shader_buffer_float32_atomic_add
                    {
                        "1"
                    } else {
                        "0"
                    },
                )]),
            )
            .map_err(VulkanError::from)?;

        self.scatter_set_op_impl(layout, index, src, src_layout, &pipeline, dim)
    }

    fn scatter_add_set(
        &mut self,
        layout: &Layout,
        index: &Self,
        idx_layout: &Layout,
        src: &Self,
        src_layout: &Layout,
        dim: usize,
    ) -> Result<()> {
        if src_layout.shape().dims() != idx_layout.shape().dims() {
            crate::bail!("scatter_add_set: index and source shapes must match");
        }

        let dtype = self.dtype;
        let idtype = index.dtype;

        let suffix = match (idtype, dtype) {
            (DType::U8, DType::F32) => "u8_f32",
            (DType::U32, DType::F32) => "u32_f32",
            (DType::I64, DType::F32) => "i64_f32",
            _ => crate::bail!("scatter_add_set: unsupported dtype combination"),
        };

        let key = format!("scatter_add_set_{}", suffix);
        let pipeline = self
            .device()
            .kernels()
            .load_pipeline(self.device().device(), &key, None)
            .map_err(VulkanError::from)?;

        self.scatter_add_set_op_impl(layout, index, src, src_layout, &pipeline, dim)
    }

    fn index_select(
        &self,
        ids: &Self,
        src_layout: &Layout,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<Self> {
        if !ids_layout.is_contiguous() {
            crate::bail!("Vulkan index_select requires contiguous ids")
        }
        let device = self.device();
        let mut dst_shape = src_layout.shape().dims().to_owned();
        if dim >= dst_shape.len() {
            Err(VulkanError::Message(format!(
                "dim {} out of bounds for shape {:?}",
                dim, dst_shape
            )))?;
        }
        let new_dim_size = ids_layout.shape().dim(0)?;
        dst_shape[dim] = new_dim_size;
        let dst = unsafe { device.alloc_uninit(&dst_shape.into(), self.dtype)? };

        let suffix = match (ids.dtype, self.dtype) {
            (DType::U8, DType::U8) => "u8_u8",
            (DType::U8, DType::U32) => "u8_u32",
            (DType::U8, DType::I64) => "u8_i64",
            (DType::U8, DType::BF16) => "u8_bf16",
            (DType::U8, DType::F32) => "u8_f32",
            (DType::U8, DType::F16) => "u8_f16",

            (DType::U32, DType::U8) => "u32_u8",
            (DType::U32, DType::U32) => "u32_u32",
            (DType::U32, DType::I64) => "u32_i64",
            (DType::U32, DType::F32) => "u32_f32",
            (DType::U32, DType::F16) => "u32_f16",
            (DType::U32, DType::BF16) => "u32_bf16",

            (DType::I64, DType::U8) => "i64_u8",
            (DType::I64, DType::U32) => "i64_u32",
            (DType::I64, DType::I64) => "i64_i64",
            (DType::I64, DType::F32) => "i64_f32",
            (DType::I64, DType::F16) => "i64_f16",
            (DType::I64, DType::BF16) => "i64_bf16",

            (left, right) => {
                crate::bail!("Vulkan contiguous index_select {left:?} {right:?} not implemented")
            }
        };
        let key = format!("index_select_{}", suffix);
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(self.device.device(), &key, None)
            .map_err(VulkanError::from)?;

        self.index_select_op_impl(&dst, ids, src_layout, ids_layout, &pipeline, dim)
    }

    fn index_add(
        &self,
        layout: &Layout,
        ids: &Self,
        ids_layout: &Layout,
        src: &Self,
        src_layout: &Layout,
        dim: usize,
    ) -> Result<Self> {
        if !ids_layout.is_contiguous() {
            crate::bail!("Vulkan index_add requires contiguous ids");
        }

        let dtype = self.dtype;
        let idtype = ids.dtype;

        let suffix = match (idtype, dtype) {
            (DType::U8, DType::U8) => "u8_u8",
            (DType::U8, DType::U32) => "u8_u32",
            (DType::U8, DType::I64) => "u8_i64",
            (DType::U8, DType::BF16) => "u8_bf16",
            (DType::U8, DType::F32) => "u8_f32",
            (DType::U8, DType::F16) => "u8_f16",

            (DType::U32, DType::U8) => "u32_u8",
            (DType::U32, DType::U32) => "u32_u32",
            (DType::U32, DType::I64) => "u32_i64",
            (DType::U32, DType::F32) => "u32_f32",
            (DType::U32, DType::F16) => "u32_f16",
            (DType::U32, DType::BF16) => "u32_bf16",

            (DType::I64, DType::U8) => "i64_u8",
            (DType::I64, DType::U32) => "i64_u32",
            (DType::I64, DType::I64) => "i64_i64",
            (DType::I64, DType::F32) => "i64_f32",
            (DType::I64, DType::F16) => "i64_f16",
            (DType::I64, DType::BF16) => "i64_bf16",

            (left, right) => {
                crate::bail!("Vulkan contiguous index_add {left:?} {right:?} not implemented")
            }
        };

        let key = format!("index_add_{}", suffix);
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(
                self.device.device(),
                &key,
                Some(&[(
                    "FLOAT32_ATOMIC_ADD",
                    if self
                        .device
                        .device()
                        .enabled_features()
                        .shader_buffer_float32_atomic_add
                    {
                        "1"
                    } else {
                        "0"
                    },
                )]),
            )
            .map_err(VulkanError::from)?;

        // Call the low-level op executor
        self.index_add_op_impl(ids, src, layout, src_layout, &pipeline, dim)
    }

    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let suffix = match (self.dtype, rhs.dtype) {
            (DType::F32, DType::F32) => "f32",
            (DType::BF16, DType::BF16) => "bf16",
            (DType::F16, DType::F16) => "f16",
            _ => todo!("Unsupported dtype combo {:?} {:?}", self.dtype, rhs.dtype),
        };
        let key = format!("gemm_{}", suffix);
        // Load the GEMM compute pipeline.
        // The pipeline key (e.g. "gemm_f32") must refer to a shader that implements:
        //   C = alpha * (A x B) + beta * C
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(self.device.device(), &key, None)
            .map_err(VulkanError::from)?;

        let shape = Shape::from(&[b, m, n]);
        let dst = self.device().zeros_impl(&shape, self.dtype())?;

        self.gemm_impl(rhs, &dst, lhs_l, rhs_l, &pipeline, (b, m, n, k))?;

        Ok(dst)
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, layout: &Layout) -> Result<()> {
        let suffix = match self.dtype {
            DType::F32 => "f32",
            DType::U32 => "u32",
            DType::I64 => "i64",
            DType::BF16 => "bf16",
            DType::F16 => "f16",
            _ => todo!("Unsupported dtype {:?}", self.dtype),
        };
        let key = format!("copy_strided_src_{}", suffix);
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(self.device.device(), &key, None)
            .map_err(VulkanError::from)?;

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct CopyStridedSrcPushConstants {
            base: u32,
            rank: u32,
            _pad0: [u32; 2],
            shape: [u32; 4],
            stride: [u32; 4],
            dst_offset: u32,
        }
        // Extract full layout info.
        let shape_slice = layout.shape();
        let stride_slice = layout.stride();
        let mut shape_arr = [1u32; 4];
        let mut stride_arr = [1u32; 4];
        for i in 0..shape_slice.rank().min(4) {
            shape_arr[i] = shape_slice
                .dim(i)
                .map_err(|_| VulkanError::Message("Shape conversion failed".to_string()))?
                as u32;
        }
        for i in 0..stride_slice.len().min(4) {
            stride_arr[i] = (*stride_slice.get(i).unwrap()) as u32;
        }
        let rank = shape_slice.rank() as u32;
        let base = layout.start_offset() as u32;
        let push_constants = CopyStridedSrcPushConstants {
            base,
            rank,
            _pad0: [0; 2],
            shape: shape_arr,
            stride: stride_arr,
            dst_offset: dst_offset as u32,
        };
        // Compute total number of elements from the full shape.
        let total_elements = shape_arr.iter().product::<u32>();

        let dispatch_x = (total_elements + 255) / 256;
        self.copy_op_impl(dst, push_constants, [dispatch_x, 1, 1], &pipeline)
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()> {
        let suffix = match self.dtype {
            DType::F32 => "f32",
            DType::U32 => "u32",
            DType::I64 => "i64",
            DType::BF16 => "bf16",
            DType::F16 => "f16",
            DType::U8 => "u8",
            _ => todo!("Unsupported dtype {:?}", self.dtype),
        };
        let key = format!("copy2d_{}", suffix);
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(self.device.device(), &key, None)
            .map_err(VulkanError::from)?;

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Copy2DPushConstants {
            src_offset: u32,
            dst_offset: u32,
            rows: u32,
            cols: u32,
            src_stride: u32,
            dst_stride: u32,
        }
        // Prepare push constants.
        let push_constants = Copy2DPushConstants {
            src_offset: src_offset as u32,
            dst_offset: dst_offset as u32,
            rows: d1 as u32,
            cols: d2 as u32,
            src_stride: src_stride1 as u32,
            dst_stride: dst_stride1 as u32,
        };
        // For copy2d, we use a 2D dispatch with a local size of 16x16.
        let group_size_x = 16;
        let group_size_y = 16;
        let dispatch_x = ((d2 as u32) + group_size_x - 1) / group_size_x;
        let dispatch_y = ((d1 as u32) + group_size_y - 1) / group_size_y;
        self.copy_op_impl(dst, push_constants, [dispatch_x, dispatch_y, 1], &pipeline)
    }

    fn const_set(&mut self, _: crate::scalar::Scalar, _: &Layout) -> crate::Result<()> {
        fail!()
    }
}
