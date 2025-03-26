#![allow(dead_code)]

use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{bail, CpuStorage, DType, Layout, Result, Shape, VulkanDevice, VulkanError};
use std::fmt;
use std::sync::Arc;
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::DeviceOwned;
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::sync::GpuFuture;

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
            CommandBufferUsage::OneTimeSubmit,
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
            [
                (dispatch_dims[0] + 255) / 256,
                dispatch_dims[1],
                dispatch_dims[2],
            ]
        };
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
                .push_constants(pipeline.layout().clone(), 0, push_constants)
                .map_err(VulkanError::ValidationError)?
                .dispatch(dims)
                .map_err(|e| VulkanError::ValidationError(e.into()))?;
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
            base: u32,
            rank: u32,
            _pad0: [u32; 2],
            shape: [u32; 4],
            stride: [u32; 4],
        }

        if let Some(buffer) = (*self.buffer).clone() {
            let elem_count = layout.shape().elem_count();
            let device = self.device();
            let new_storage = unsafe { device.alloc_uninit(layout.shape(), target_dtype)? };

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

            let push_constants = PushConstants {
                base,
                rank,
                _pad0: [0; 2],
                shape: shape_arr,
                stride: stride_arr,
            };
            self.execute_compute_kernel(
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
            self.execute_compute_kernel(
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
            _pad0: [u32; 2],
            shape: [u32; 4],
            stride: [u32; 4],
            reduce_axes: [u32; 4],
        }

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
        struct CombinePushConstants {
            num_partials: u32,
        }

        // Only support F32/U32 for now.
        let dtype = self.dtype();
        if dtype != DType::F32 && dtype != DType::U32 {
            return Err(VulkanError::Message(format!(
                "Unsupported dtype: {:?}",
                dtype
            )))?;
        }
        let result_dtype = if to_index { DType::U32 } else { dtype };

        if let Some(buffer) = (*self.buffer).clone() {
            let device = self.device();

            // Build tensor metadata.
            let shape_slice = layout.shape();
            let stride_slice = layout.stride();
            let rank = shape_slice.rank() as u32;
            let mut shape_arr = [1u32; 4];
            let mut stride_arr = [1u32; 4];
            for i in 0..(rank as usize).min(4) {
                shape_arr[i] = shape_slice
                    .dim(i)
                    .unwrap()
                    .try_into()
                    .map_err(|_| VulkanError::Message("Shape conversion failed".to_string()))?;
            }
            for i in 0..stride_slice.len().min(4) {
                stride_arr[i] = (*stride_slice.get(i).unwrap()) as u32;
            }
            let base = layout.start_offset() as u32;
            // Build reduce_axes array; unused entries are filled with u32::MAX.
            let reduce_axes_arr: [u32; 4] = reduce_axes
                .iter()
                .map(|&ax| ax as u32)
                .chain(std::iter::repeat(u32::MAX))
                .take(4)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();

            let push_constants = ReducePushConstants {
                base,
                rank,
                _pad0: [0; 2],
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

            // Dispatch the partial reduction shader.
            self.execute_compute_kernel(
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

                self.execute_compute_kernel(
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
                rank: u32,
                selected_dim: u32,
                _pad: u32,
                input_strides: [u32; 4],
                output_strides: [u32; 4],
            }

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
                selected_dim: dim as u32,
                _pad: 0,
                input_strides: padded_in_strides,
                output_strides: padded_out_strides,
            };

            self.execute_compute_kernel(
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

    fn scatter_add_op_impl(
        &self,
        layout: &Layout,
        ids: &Self,
        src: &Self,
        src_layout: &Layout,
        pipeline: &Arc<ComputePipeline>,
        dim: usize,
    ) -> Result<Self> {
        if let (Some(dst_buf), Some(idx_buf), Some(src_buf)) = (
            (*self.buffer).clone(),
            (*ids.buffer).clone(),
            (*src.buffer).clone(),
        ) {
            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct ScatterAddPushConstants {
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

            let push_constants = ScatterAddPushConstants {
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

            Ok(self.clone())
        } else {
            Ok(self.clone())
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
                total_out_elems: u32, // total number of output elements
                rank: u32,            // rank of the tensor
                selected_dim: u32,    // the dimension to select over
                _pad0: u32,
                input_strides: [u32; 4],  // input strides (row-major), padded
                output_strides: [u32; 4], // output strides, padded
            }

            // Get the input shape and strides from the source layout.
            let src_shape: Vec<usize> = src_layout.shape().dims().to_vec();
            let rank = src_shape.len();
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
                selected_dim: dim as u32,
                _pad0: 0,
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

            self.execute_compute_kernel(
                pipeline,
                vec![src_buffer, index_buffer], // binding 0: source, binding 1: indices
                vec![dst_buffer],
                dispatch_dims,
                push_constants,
                true, // direct_dispatch true (we computed global size directly)
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
                selected_dim: u32,
                _pad0: u32,
                input_strides: [u32; 4],
                output_strides: [u32; 4],
            }

            let src_shape = src_layout.shape().dims();
            let dst_shape = dst_layout.shape().dims();
            let rank = src_shape.len();

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
                selected_dim: dim as u32,
                _pad0: 0,
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
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(VulkanError::ValidatedVulkanError)?;
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
                .dispatch(dispatch_dims)
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
        elem_count: usize,
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
            self.execute_compute_kernel(
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
            nrows: u32,
            ncols: u32,
            ncols_pad: u32,
            ascending: u32,
        }

        let el = layout.shape().elem_count();
        let ncols = layout.shape().dims().last().copied().unwrap_or(1);
        let nrows = el / ncols;
        let ncols_pad = ncols.next_power_of_two();

        if ncols_pad > 1024 {
            crate::bail!("arg_sort: padded row size {ncols_pad} exceeds 1024");
        }

        let push_constants = ArgSortPushConstants {
            nrows: nrows as u32,
            ncols: ncols as u32,
            ncols_pad: ncols_pad as u32,
            ascending: if ascending { 1 } else { 0 },
        };

        let device = self.device();
        let output = unsafe { device.alloc_uninit(&layout.shape().clone().into(), DType::U32)? };

        self.execute_compute_kernel(
            &pipeline,
            vec![(*self.buffer).clone().unwrap()],
            vec![(*output.buffer).clone().unwrap()],
            [nrows as u32, 1, 1],
            push_constants,
            true,
        )?;

        Ok(output)
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

        self.execute_compute_kernel(
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
        gamma_layout: &Layout,
        pipeline: &Arc<ComputePipeline>,
        axis: usize,
        eps: f32,
    ) -> Result<Self> {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PushConstants {
            base_offset: u32,
            rank: u32,
            axis: u32,
            eps: f32,
            shape: [u32; 4],
            stride: [u32; 4],
        }

        let shape = layout.shape();
        let stride = layout.stride();
        let mut shape_arr = [1u32; 4];
        let mut stride_arr = [1u32; 4];
        for i in 0..shape.rank().min(4) {
            shape_arr[i] = shape.dim(i).unwrap() as u32;
        }
        for i in 0..stride.len().min(4) {
            stride_arr[i] = stride[i] as u32;
        }

        let push_constants = PushConstants {
            base_offset: layout.start_offset() as u32,
            rank: shape.rank() as u32,
            axis: axis as u32,
            eps: eps,
            shape: shape_arr,
            stride: stride_arr,
        };

        let device = self.device();
        let output = unsafe { device.alloc_uninit(shape, self.dtype)? };
        let count = shape.elem_count();

        self.execute_compute_kernel(
            pipeline,
            vec![
                (*self.buffer).clone().unwrap(),
                (*gamma.buffer).clone().unwrap(),
            ],
            vec![(*output.buffer).clone().unwrap()],
            [((count as u32) / shape.dims()[axis] as u32), 1, 1],
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

        self.execute_compute_kernel(
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
        cos_layout: &Layout,
        sin: &Self,
        sin_layout: &Layout,
        pipeline: &Arc<ComputePipeline>,
    ) -> Result<Self> {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PushConstants {
            b_sz: u32,
            n_heads: u32,
            seq_len: u32,
            head_dim: u32,

            stride_bsz: u32,
            stride_head: u32,
            stride_seq: u32,
            stride_dim: u32,

            stride_cos_seq: u32,
            stride_cos_dim: u32,
            stride_sin_seq: u32,
            stride_sin_dim: u32,
        }

        let input_buf = (*self.buffer).clone().ok_or_else(|| {
            VulkanError::Message("rope_op_impl: missing input buffer".into())
        })?;
        let cos_buf = (*cos.buffer).clone().ok_or_else(|| {
            VulkanError::Message("rope_op_impl: missing cos buffer".into())
        })?;
        let sin_buf = (*sin.buffer).clone().ok_or_else(|| {
            VulkanError::Message("rope_op_impl: missing sin buffer".into())
        })?;

        let shape = layout.shape().dims();
        if shape.len() != 4 {
            return Err(VulkanError::Message(format!(
                "Expected shape [B, H, T, D], got {:?}",
                shape
            )))?;
        }

        let bsz = shape[0];
        let n_heads = shape[1];
        let seq_len = shape[2];
        let head_dim = shape[3];

        let out = unsafe {
            self.device()
                .alloc_uninit(layout.shape(), self.dtype)?
        };

        let strides = layout.stride();
        if strides.len() != 4 {
            return Err(VulkanError::Message("Expected 4D stride layout".into()))?;
        }

        let push_constants = PushConstants {
            b_sz: shape[0] as u32,
            n_heads: shape[1] as u32,
            seq_len: shape[2] as u32,
            head_dim: shape[3] as u32,

            stride_bsz: strides[0] as u32,
            stride_head: strides[1] as u32,
            stride_seq: strides[2] as u32,
            stride_dim: strides[3] as u32,

            stride_cos_seq: cos_layout.stride()[0] as u32,
            stride_cos_dim: cos_layout.stride()[1] as u32,
            stride_sin_seq: sin_layout.stride()[0] as u32,
            stride_sin_dim: sin_layout.stride()[1] as u32,
        };

        let total_elems = layout.shape().elem_count() as u32;

        self.execute_compute_kernel(
            pipeline,
            vec![input_buf, cos_buf, sin_buf],
            vec![(*out.buffer).clone().unwrap()],
            [total_elems, 1, 1],
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
            self.execute_compute_kernel(
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
        dst: &Self,
        layout: &Layout,
        rhs_layout: &Layout,
        pipeline: &Arc<ComputePipeline>,
        (b, m, n, k): (usize, usize, usize, usize),
    ) -> Result<()> {
        // Build a push constant struct for GEMM.
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
        struct GemmPushConstants {
            m: u32,
            n: u32,
            k: u32,
            a_batch_stride: u32, // physical batch stride for A
            a_row_stride: u32,   // physical row stride for A
            a_col_stride: u32,   // physical column stride for A
            b_batch_stride: u32, // physical batch stride for B
            b_row_stride: u32,   // physical row stride for B
            b_col_stride: u32,   // physical column stride for B
            ldc: u32,
            alpha: f32,
            beta: f32,
        }

        if let (Some(lhs_buffer), Some(rhs_buffer)) =
            ((*self.buffer).clone(), (*rhs.buffer).clone())
        {
            let (b, m, n, k) = (b as u32, m as u32, n as u32, k as u32);
            // Extract physical strides from the Layouts.
            // For A: assume shape is [b, m, k] so:
            //   a_batch_stride = lhs_l.stride()[0]
            //   a_row_stride   = lhs_l.stride()[lhs_l.shape().rank() - 2]
            //   a_col_stride   = lhs_l.stride()[lhs_l.shape().rank() - 1]
            let a_rank = layout.shape().rank();
            let a_batch_stride = layout.stride()[0] as u32;
            let a_row_stride = layout.stride()[a_rank - 2] as u32;
            let a_col_stride = layout.stride()[a_rank - 1] as u32;

            // Similarly for B: assume shape is [b, k, n]:
            let b_rank = rhs_layout.shape().rank();
            let b_batch_stride = rhs_layout.stride()[0] as u32;
            let b_row_stride = rhs_layout.stride()[b_rank - 2] as u32;
            let b_col_stride = rhs_layout.stride()[b_rank - 1] as u32;

            // For C, assume it’s allocated contiguously with shape [b, m, n],
            // so the logical row stride is n, and batch stride would be m * n.
            let ldc = n; // each row of C has n elements

            // Build the push constants.
            let push_constants = GemmPushConstants {
                m,
                n,
                k,
                a_batch_stride,
                a_row_stride,
                a_col_stride,
                b_batch_stride,
                b_row_stride,
                b_col_stride,
                ldc,
                alpha: 1f32,
                beta: 0f32,
            };

            let tile_size = 16u32;
            let wg_x = (n + tile_size - 1) / tile_size;
            let wg_y = (m + tile_size - 1) / tile_size;
            let wg_z = b;

            self.execute_compute_kernel(
                pipeline,
                vec![lhs_buffer, rhs_buffer],
                vec![(*dst.buffer).clone().unwrap()],
                [wg_x, wg_y, wg_z],
                push_constants,
                true,
            )?;
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
        let suffix = match self.dtype {
            DType::F32 => "f32",
            _ => todo!("Unsupported dtype {:?}", self.dtype),
        };
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(self.device.device(), &format!("affine_{}", suffix))
            .map_err(VulkanError::from)?;
        self.affine_elu_op_impl(layout, &pipeline, mul, add, 0.0)
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        fail!()
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        let suffix = match self.dtype {
            DType::F32 => "f32",
            _ => todo!("Unsupported dtype {:?}", self.dtype),
        };
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(self.device.device(), &format!("elu_{}", suffix))
            .map_err(VulkanError::from)?;
        self.affine_elu_op_impl(layout, &pipeline, 0.0, 0.0, alpha)
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, s: &[usize]) -> Result<Self> {
        let suffix = match self.dtype {
            DType::F32 => "f32",
            DType::U32 => "u32",
            _ => todo!("Unsupported dtype {:?}", self.dtype),
        };
        match op {
            ReduceOp::Max | ReduceOp::Min | ReduceOp::Sum => {
                let partial_key = format!("{}_partial_{}", op.name(), suffix);
                let partial_pipeline = self
                    .device
                    .kernels()
                    .load_pipeline(self.device.device(), &partial_key)
                    .map_err(VulkanError::from)?;
                let combine_key = format!("{}_combine_{}", op.name(), suffix);
                let combine_pipeline = self
                    .device
                    .kernels()
                    .load_pipeline(self.device.device(), &combine_key)
                    .map_err(VulkanError::from)?;
                self.reduce_op_impl(layout, s, &partial_pipeline, &combine_pipeline, false)
            }
            ReduceOp::ArgMax | ReduceOp::ArgMin => {
                let partial_key = format!("{}_partial_{}", op.name(), suffix);
                let partial_pipeline = self
                    .device
                    .kernels()
                    .load_pipeline(self.device.device(), &partial_key)
                    .map_err(VulkanError::from)?;
                let combine_key = format!("{}_combine_{}", op.name(), suffix);
                let combine_pipeline = self
                    .device
                    .kernels()
                    .load_pipeline(self.device.device(), &combine_key)
                    .map_err(VulkanError::from)?;
                self.reduce_op_impl(layout, s, &partial_pipeline, &combine_pipeline, true)
            }
        }
    }

    fn cmp(&self, cmp_op: CmpOp, rhs: &Self, layout: &Layout, rhs_layout: &Layout) -> Result<Self> {
        let suffix = match (self.dtype, rhs.dtype) {
            (DType::F32, DType::F32) => "f32",
            (DType::I64, DType::I64) => "i64",
            _ => todo!("Unsupported dtype combo {:?} {:?}", self.dtype, rhs.dtype),
        };
        let key = format!("{}_{}", cmp_op.name(), suffix);
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(self.device.device(), &key)
            .map_err(VulkanError::from)?;

        // Allocate new storage for the result.
        // We choose U32 to store 1 for true and 0 for false.
        let elem_count = layout.shape().elem_count();
        let device = self.device();
        let new_storage = unsafe { device.alloc_uninit(layout.shape(), DType::U8)? };

        // Call the lower-level helper.
        self.cmp_op_impl(
            rhs,
            layout,
            rhs_layout,
            &mut new_storage.clone(),
            &pipeline,
            elem_count,
        )?;
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
                _ => todo!("Unsupported dtype combo {:?} {:?}", self.dtype, dtype),
            };
            let pipeline = self
                .device
                .kernels()
                .load_pipeline(self.device.device(), kernel)
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
            .load_pipeline(self.device.device(), &key)
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
            _ => todo!("Unsupported dtype combo {:?} {:?}", self.dtype, rhs.dtype),
        };
        let key = format!("{}_{}", B::NAME, suffix);
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(self.device.device(), &key)
            .map_err(VulkanError::from)?;
        self.binary_op_impl(layout, rhs, rhs_layout, &pipeline)
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
            .load_pipeline(self.device.device(), &key)
            .map_err(VulkanError::from)?;

        let dst = unsafe { self.device().alloc_uninit(&out_shape.into(), self.dtype)? };

        self.gather_op_impl(&dst, index, src_layout, idx_layout, &pipeline, dim)
    }

    fn scatter_add(
        &self,
        layout: &Layout,
        index: &Self,
        idx_layout: &Layout,
        src: &Self,
        src_layout: &Layout,
        dim: usize,
    ) -> Result<Self> {
        if src_layout.shape().dims() != idx_layout.shape().dims() {
            crate::bail!("scatter_add: index and source shapes must match");
        }

        let dtype = self.dtype;
        let idtype = index.dtype;

        let suffix = match (idtype, dtype) {
            (DType::U8, DType::F32) => "u8_f32",
            (DType::U32, DType::F32) => "u32_f32",
            (DType::I64, DType::F32) => "i64_f32",
            _ => crate::bail!("scatter_add: unsupported dtype combination"),
        };

        let key = format!("scatter_add_{}", suffix);
        let pipeline = self
            .device()
            .kernels()
            .load_pipeline(self.device().device(), &key)
            .map_err(VulkanError::from)?;

        self.scatter_add_op_impl(layout, index, src, src_layout, &pipeline, dim)
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
            .load_pipeline(self.device.device(), &key)
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
            .load_pipeline(self.device.device(), &key)
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
            _ => todo!("Unsupported dtype combo {:?} {:?}", self.dtype, rhs.dtype),
        };
        let key = format!("gemm_{}", suffix);
        // Load the GEMM compute pipeline.
        // The pipeline key (e.g. "gemm_f32") must refer to a shader that implements:
        //   C = alpha * (A x B) + beta * C
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(self.device.device(), &key)
            .map_err(VulkanError::from)?;

        let shape = Shape::from(&[b, m, n]);
        let dst = unsafe { self.device().zeros_impl(&shape, self.dtype())? };

        self.gemm_impl(rhs, &dst, lhs_l, rhs_l, &pipeline, (b, m, n, k))?;

        Ok(dst)
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, layout: &Layout) -> Result<()> {
        let suffix = match self.dtype {
            DType::F32 => "f32",
            DType::U32 => "u32",
            DType::I64 => "i64",
            DType::BF16 => "bf16",
            _ => todo!("Unsupported dtype {:?}", self.dtype),
        };
        let key = format!("copy_strided_src_{}", suffix);
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(self.device.device(), &key)
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
            _ => todo!("Unsupported dtype {:?}", self.dtype),
        };
        let key = format!("copy2d_{}", suffix);
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(self.device.device(), &key)
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
}
