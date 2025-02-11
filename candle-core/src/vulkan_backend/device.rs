#![allow(dead_code)]

use crate::backend::BackendStorage;
use crate::op::{BinaryOpT, UnaryOpT};
use crate::{CpuStorage, DType, Result, Shape, VulkanError, VulkanStorage};
use bytemuck::Pod;
use half::{bf16, f16};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryCommandBufferAbstract,
};
use vulkano::descriptor_set::allocator::{
    StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{
    AllocationCreateInfo, DeviceLayout, MemoryTypeFilter, StandardMemoryAllocator,
};
use vulkano::pipeline::compute::{ComputePipeline, ComputePipelineCreateInfo};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{Pipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::sync::GpuFuture;
use vulkano::{DeviceSize, VulkanLibrary};

#[derive(Clone, Debug)]
pub struct VulkanDevice {
    ordinal: usize,
    device: Arc<Device>,
    pub(crate) queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    buffer_allocator: Arc<Mutex<SubbufferAllocator>>,
    pub(crate) command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub(crate) descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    //    zero_init_pipeline: Arc<ComputePipeline>,
    pub(crate) cast_pipelines: Vec<Arc<ComputePipeline>>,
    pub(crate) unary_pipelines: HashMap<&'static str, Arc<ComputePipeline>>,
    pub(crate) binary_pipelines: HashMap<&'static str, Arc<ComputePipeline>>,
    pub(crate) reduce_pipelines: HashMap<&'static str, Arc<ComputePipeline>>,
    pub(crate) affine_elu_pipelines: HashMap<&'static str, Arc<ComputePipeline>>,
    pub(crate) copy_pipelines: HashMap<&'static str, Arc<ComputePipeline>>,
    pub(crate) cmp_pipelines: HashMap<&'static str, Arc<ComputePipeline>>,
}

enum DataSource<'a, T> {
    Slice(&'a [T]),
    Fill { value: T, count: usize },
}

macro_rules! fail {
    () => {
        unimplemented!("vulkan support is incomplete, this function is not yet implemented")
    };
}

macro_rules! unary_shaders {
    ($( ($mod:ident, $op:literal, $inner_type:literal, $outer_type:literal) ),* $(,)?) => {
        $(
            mod $mod {
                vulkano_shaders::shader! {
                    ty: "compute",
                    src: "
                        #version 450
                        #extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
                        #extension GL_EXT_shader_16bit_storage : require
                        #extension GL_AMD_gpu_shader_half_float: enable

                        layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

                        // Buffer bindings
                        layout(set = 0, binding = 0) buffer InputBuffer {
                            OUTER_TYPE input_data[];
                        };
                        layout(set = 0, binding = 1) buffer OutputBuffer {
                            OUTER_TYPE output_data[];
                        };
                        layout(push_constant) uniform PushConstants {
                            uint base;
                            uint rank;         // number of dimensions
                            uvec4 shape;       // padded shape (unused dimensions set to 1)
                            uvec4 stride;      // padded stride (unused dimensions set to 1)
                        } pc;

                        const INNER_TYPE HALF = INNER_TYPE(0.5);
                        const INNER_TYPE ONE = INNER_TYPE(1.0);

                        // This function converts a linear index to a physical index using the full
                        // multi-dimensional layout. We assume row-major ordering.
                        uint get_strided_index(uint lin_idx) {
                            uint remaining = lin_idx;
                            uint phys_idx = 0;
                            if (pc.rank > 0u) {
                                uint prod = 1u;
                                if (pc.rank > 1u) { prod = pc.shape.y * pc.shape.z * pc.shape.w; }
                                uint i0 = remaining / prod;
                                remaining = remaining % prod;
                                phys_idx += i0 * pc.stride.x;
                            }
                            if (pc.rank > 1u) {
                                uint prod = 1u;
                                if (pc.rank > 2u) { prod = pc.shape.z * pc.shape.w; }
                                uint i1 = remaining / prod;
                                remaining = remaining % prod;
                                phys_idx += i1 * pc.stride.y;
                            }
                            if (pc.rank > 2u) {
                                uint prod = 1u;
                                if (pc.rank > 3u) { prod = pc.shape.w; }
                                uint i2 = remaining / prod;
                                remaining = remaining % prod;
                                phys_idx += i2 * pc.stride.z;
                            }
                            if (pc.rank > 3u) {
                                uint i3 = remaining;
                                phys_idx += i3 * pc.stride.w;
                            }
                            return pc.base + phys_idx;
                        }

                        void neg_op() {
                            uint idx = gl_GlobalInvocationID.x;
                            uint a_idx = get_strided_index(idx);
                            output_data[idx] = OUTER_TYPE(-INNER_TYPE(input_data[idx]));
                        }

                        void gelu_op() {
                            const INNER_TYPE COEF_A = INNER_TYPE(0.79788456) + INNER_TYPE(0.000000000802865355);
                            const INNER_TYPE COEF_B = INNER_TYPE(0.04471509) + INNER_TYPE(-0.0000000026068047);
                            uint idx = gl_GlobalInvocationID.x;
                            uint a_idx = get_strided_index(idx);
                            INNER_TYPE x = INNER_TYPE(input_data[idx]);
                            INNER_TYPE x_cubed = x * x * x;
                            INNER_TYPE inner = COEF_A * (x + COEF_B * x_cubed);
                            INNER_TYPE cdf = HALF * (ONE + tanh(inner));
                            output_data[idx] = OUTER_TYPE(x * cdf);
                        }

                        // Custom erf approximation
                        INNER_TYPE erf_approx(INNER_TYPE x) {
                            const INNER_TYPE a1 = INNER_TYPE(0.25482959) + INNER_TYPE(0.000000002091);
                            const INNER_TYPE a2 = INNER_TYPE(-0.28449674) + INNER_TYPE(0.000000003751);
                            const INNER_TYPE a3 = INNER_TYPE(1.42141378) + INNER_TYPE(-0.000000038331);
                            const INNER_TYPE a4 = INNER_TYPE(-1.45315206) + INNER_TYPE(0.00000003259);
                            const INNER_TYPE a5 = INNER_TYPE(1.06140542) + INNER_TYPE(0.000000009758);
                            const INNER_TYPE p  = INNER_TYPE(0.3275911) + INNER_TYPE(0.000000000330);
                            INNER_TYPE s = sign(x);
                            x = abs(x);
                            INNER_TYPE t = ONE / (ONE + p * x);
                            INNER_TYPE y = ONE - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);
                            return s * y;
                        }

                        void gelu_erf_op() {
                            uint idx = gl_GlobalInvocationID.x;
                            uint a_idx = get_strided_index(idx);
                            INNER_TYPE x = INNER_TYPE(input_data[idx]);
                            INNER_TYPE cdf = HALF * (ONE + erf_approx(x / sqrt(2.0)));
                            output_data[idx] = OUTER_TYPE(x * cdf);
                        }

                        void erf_op() {
                            uint idx = gl_GlobalInvocationID.x;
                            uint a_idx = get_strided_index(idx);
                            INNER_TYPE x = INNER_TYPE(input_data[idx]);
                            output_data[idx] = OUTER_TYPE(erf_approx(x));
                        }

                        void silu_op() {
                            uint idx = gl_GlobalInvocationID.x;
                            uint a_idx = get_strided_index(idx);
                            INNER_TYPE x = INNER_TYPE(input_data[idx]);
                            output_data[idx] = OUTER_TYPE(x / (ONE + exp(-x)));
                        }

                        void ceil_op() {
                            uint idx = gl_GlobalInvocationID.x;
                            uint a_idx = get_strided_index(idx);
                            output_data[idx] = OUTER_TYPE(ceil(INNER_TYPE(input_data[idx])));
                        }

                        void floor_op() {
                            uint idx = gl_GlobalInvocationID.x;
                            uint a_idx = get_strided_index(idx);
                            output_data[idx] = OUTER_TYPE(floor(INNER_TYPE(input_data[idx])));
                        }

                        void round_op() {
                            uint idx = gl_GlobalInvocationID.x;
                            uint a_idx = get_strided_index(idx);
                            output_data[idx] = OUTER_TYPE(round(INNER_TYPE(input_data[idx])));
                        }

                        void sign_op() {
                            uint idx = gl_GlobalInvocationID.x;
                            uint a_idx = get_strided_index(idx);
                            output_data[idx] = OUTER_TYPE(sign(INNER_TYPE(input_data[idx])));
                        }

                        void sqr_op() {
                            uint idx = gl_GlobalInvocationID.x;
                            uint a_idx = get_strided_index(idx);
                            INNER_TYPE data = INNER_TYPE(input_data[idx]);
                            output_data[idx] = OUTER_TYPE(data * data);
                        }

                        // Conditional main() selection based on the define.
                        void main() { OP(); }
                    ",
                    define: [("OP", $op), ("INNER_TYPE", $inner_type), ("OUTER_TYPE", $outer_type)]
                }
            }
        )*
    };
}

macro_rules! binary_shaders {
    ($( ($mod:ident, $op:literal, $ty:literal) ),* $(,)?) => {
        $(
            mod $mod {
                vulkano_shaders::shader! {
                    ty: "compute",
                    src: "
                        #version 450
                        #extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
                        #extension GL_EXT_shader_16bit_storage : require
                        #extension GL_AMD_gpu_shader_half_float: enable

                        layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

                        // Buffer bindings
                        layout(set = 0, binding = 0) buffer LhsBuffer {
                            TYPE lhs_data[];
                        };
                        layout(set = 0, binding = 1) buffer RhsBuffer {
                            TYPE rhs_data[];
                        };
                        layout(set = 0, binding = 2) buffer OutputBuffer {
                            TYPE output_data[];
                        };
                        layout(push_constant) uniform PushConstants {
                            uint a_base;
                            uint a_rank;         // number of dimensions
                            uvec4 a_shape;       // padded shape (unused dimensions set to 1)
                            uvec4 a_stride;      // padded stride (unused dimensions set to 1)
                            uint b_base;
                            uint b_rank;         // number of dimensions
                            uvec4 b_shape;       // padded shape (unused dimensions set to 1)
                            uvec4 b_stride;      // padded stride (unused dimensions set to 1)
                        } pc;

                        // This function converts a linear index to a physical index using the full
                        // multi-dimensional layout. We assume row-major ordering.
                        uint get_strided_index_a(uint lin_idx) {
                            uint remaining = lin_idx;
                            uint phys_idx = 0;
                            if (pc.a_rank > 0u) {
                                uint prod = 1u;
                                if (pc.a_rank > 1u) { prod = pc.a_shape.y * pc.a_shape.z * pc.a_shape.w; }
                                uint i0 = remaining / prod;
                                remaining = remaining % prod;
                                phys_idx += i0 * pc.a_stride.x;
                            }
                            if (pc.a_rank > 1u) {
                                uint prod = 1u;
                                if (pc.a_rank > 2u) { prod = pc.a_shape.z * pc.a_shape.w; }
                                uint i1 = remaining / prod;
                                remaining = remaining % prod;
                                phys_idx += i1 * pc.a_stride.y;
                            }
                            if (pc.a_rank > 2u) {
                                uint prod = 1u;
                                if (pc.a_rank > 3u) { prod = pc.a_shape.w; }
                                uint i2 = remaining / prod;
                                remaining = remaining % prod;
                                phys_idx += i2 * pc.a_stride.z;
                            }
                            if (pc.a_rank > 3u) {
                                uint i3 = remaining;
                                phys_idx += i3 * pc.a_stride.w;
                            }
                            return pc.a_base + phys_idx;
                        }

                        uint get_strided_index_b(uint lin_idx) {
                            uint remaining = lin_idx;
                            uint phys_idx = 0;
                            if (pc.b_rank > 0u) {
                                uint prod = 1u;
                                if (pc.b_rank > 1u) { prod = pc.b_shape.y * pc.b_shape.z * pc.b_shape.w; }
                                uint i0 = remaining / prod;
                                remaining = remaining % prod;
                                phys_idx += i0 * pc.b_stride.x;
                            }
                            if (pc.b_rank > 1u) {
                                uint prod = 1u;
                                if (pc.b_rank > 2u) { prod = pc.b_shape.z * pc.b_shape.w; }
                                uint i1 = remaining / prod;
                                remaining = remaining % prod;
                                phys_idx += i1 * pc.b_stride.y;
                            }
                            if (pc.b_rank > 2u) {
                                uint prod = 1u;
                                if (pc.b_rank > 3u) { prod = pc.b_shape.w; }
                                uint i2 = remaining / prod;
                                remaining = remaining % prod;
                                phys_idx += i2 * pc.b_stride.z;
                            }
                            if (pc.b_rank > 3u) {
                                uint i3 = remaining;
                                phys_idx += i3 * pc.b_stride.w;
                            }
                            return pc.b_base + phys_idx;
                        }

                        void add_op() {
                            uint idx = gl_GlobalInvocationID.x;
                            uint a_idx = get_strided_index_a(idx);
                            uint b_idx = get_strided_index_b(idx);
                            output_data[idx] = lhs_data[a_idx] + rhs_data[b_idx];
                        }

                        void sub_op() {
                            uint idx = gl_GlobalInvocationID.x;
                            uint a_idx = get_strided_index_a(idx);
                            uint b_idx = get_strided_index_b(idx);
                            output_data[idx] = lhs_data[a_idx] - rhs_data[b_idx];
                        }

                        void div_op() {
                            uint idx = gl_GlobalInvocationID.x;
                            uint a_idx = get_strided_index_a(idx);
                            uint b_idx = get_strided_index_b(idx);
                            output_data[idx] = lhs_data[a_idx] / rhs_data[b_idx];
                        }

                        void mul_op() {
                            uint idx = gl_GlobalInvocationID.x;
                            uint a_idx = get_strided_index_a(idx);
                            uint b_idx = get_strided_index_b(idx);
                            output_data[idx] = lhs_data[a_idx] * rhs_data[b_idx];
                        }

                        // Conditional main() selection based on the define.
                        void main() { OP(); }
                    ",
                    define: [("OP", $op), ("TYPE", $ty)]
                }
            }
        )*
    };
}

macro_rules! reduce_shaders {
    ($( ($mod:ident, $op:literal, $ty:literal) ),* $(,)?) => {
        $(
            mod $mod {
                vulkano_shaders::shader! {
                    ty: "compute",
                    src: "
                        #version 450

                        // Workgroup size; adjust as needed.
                        layout (local_size_x = 256) in;

                        // Input tensor: a flat array of floats.
                        layout(std430, binding = 0) readonly buffer InputBuffer {
                            TYPE data[];
                        };

                        // Output buffer for partial maximum values.
                        // Each workgroup writes one partial result.
                        layout(std430, binding = 1) writeonly buffer OutputBuffer {
                            TYPE partialResult[];
                        };

                        layout(push_constant) uniform ReductionConstants {
                            uint numElements;
                        } rc;

                        // Shared memory for intra-group reduction.
                        shared TYPE sdata[256];

                        void max_op() {
                            // Compute global and local indices.
                            uint globalId = gl_GlobalInvocationID.x;
                            uint localId  = gl_LocalInvocationID.x;
                            uint totalThreads = gl_NumWorkGroups.x * gl_WorkGroupSize.x;

                            // Each thread loads several elements (striding by total number of threads)
                            // and computes a local maximum.
                            TYPE maxVal = -3.402823466e+38; // Use the smallest possible float (approx. -FLT_MAX)
                            for (uint i = globalId; i < rc.numElements; i += totalThreads) {
                                maxVal = max(maxVal, data[i]);
                            }

                            // Store the per-thread maximum in shared memory.
                            sdata[localId] = maxVal;
                            barrier();  // Ensure all threads have written their value.

                            // Perform parallel reduction within the workgroup.
                            // The stride halves at each iteration.
                            for (uint offset = gl_WorkGroupSize.x / 2; offset > 0; offset >>= 1) {
                                if (localId < offset) {
                                    sdata[localId] = max(sdata[localId], sdata[localId + offset]);
                                }
                                barrier();  // Wait for all threads to update shared memory.
                            }

                            // The first thread in each workgroup writes the partial result.
                            if (localId == 0) {
                                partialResult[gl_WorkGroupID.x] = sdata[0];
                            }
                        }

                        void sum_op() {
                            // Compute global and local indices.
                            uint globalId = gl_GlobalInvocationID.x;
                            uint localId  = gl_LocalInvocationID.x;
                            uint totalThreads = gl_NumWorkGroups.x * gl_WorkGroupSize.x;

                            // Each thread loads several elements and accumulates their sum.
                            TYPE sumVal = 0.0;
                            for (uint i = globalId; i < rc.numElements; i += totalThreads) {
                                sumVal += data[i];
                            }

                            // Store the per-thread sum in shared memory.
                            sdata[localId] = sumVal;
                            barrier();

                            // Perform parallel reduction (sum) within the workgroup.
                            for (uint offset = gl_WorkGroupSize.x / 2; offset > 0; offset >>= 1) {
                                if (localId < offset) {
                                    sdata[localId] += sdata[localId + offset];
                                }
                                barrier();
                            }

                            // The first thread writes the partial sum.
                            if (localId == 0) {
                                partialResult[gl_WorkGroupID.x] = sdata[0];
                            }
                        }

                        // Conditional main() selection based on the define.
                        void main() { OP(); }
                    ",
                    define: [("OP", $op), ("TYPE", $ty)]
                }
            }
        )*
    };
}

macro_rules! affine_elu_shaders {
    ($( ($mod:ident, $op:literal) ),* $(,)?) => {
        $(
            mod $mod {
                vulkano_shaders::shader! {
                    ty: "compute",
                    src: "
                        #version 450
                        layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

                        // Buffer bindings.
                        layout(set = 0, binding = 0) buffer InputBuffer {
                            float input_data[];
                        };
                        layout(set = 0, binding = 1) buffer OutputBuffer {
                            float output_data[];
                        };

                        // Push constants carrying full layout information (up to 4 dimensions)
                        // plus the affine parameters.
                        layout(push_constant) uniform PushConstants {
                            uint base;
                            uint rank;         // number of dimensions
                            uvec4 shape;       // padded shape (unused dimensions set to 1)
                            uvec4 stride;      // padded stride (unused dimensions set to 1)
                            float mul;         // multiplier
                            float add;         // additive constant
                            float alpha;       // ELU alpha
                        } pc;

                        // This function converts a linear index to a physical index using the full
                        // multi-dimensional layout. We assume row-major ordering.
                        uint get_strided_index(uint lin_idx) {
                            uint remaining = lin_idx;
                            uint phys_idx = 0;
                            if (pc.rank > 0u) {
                                uint prod = 1u;
                                if (pc.rank > 1u) { prod = pc.shape.y * pc.shape.z * pc.shape.w; }
                                uint i0 = remaining / prod;
                                remaining = remaining % prod;
                                phys_idx += i0 * pc.stride.x;
                            }
                            if (pc.rank > 1u) {
                                uint prod = 1u;
                                if (pc.rank > 2u) { prod = pc.shape.z * pc.shape.w; }
                                uint i1 = remaining / prod;
                                remaining = remaining % prod;
                                phys_idx += i1 * pc.stride.y;
                            }
                            if (pc.rank > 2u) {
                                uint prod = 1u;
                                if (pc.rank > 3u) { prod = pc.shape.w; }
                                uint i2 = remaining / prod;
                                remaining = remaining % prod;
                                phys_idx += i2 * pc.stride.z;
                            }
                            if (pc.rank > 3u) {
                                uint i3 = remaining;
                                phys_idx += i3 * pc.stride.w;
                            }
                            return pc.base + phys_idx;
                        }

                        void affine_op() {
                            uint lin_idx = gl_GlobalInvocationID.x;
                            uint data_index = get_strided_index(lin_idx);
                            float x = input_data[data_index];
                            output_data[lin_idx] = x * pc.mul + pc.add;
                        }

                        void elu_op() {
                            uint lin_idx = gl_GlobalInvocationID.x;
                            uint data_index = get_strided_index(lin_idx);
                            float x = input_data[data_index];
                            output_data[lin_idx] = (x >= 0.0) ? x : pc.alpha * (exp(x) - 1.0);
                        }

                        void main() {
                            OP();
                    }
                    ",
                    // Pass the op type as a preprocessor definition.
                    define: [("OP", $op)]
                }
            }
        )*
    }
}
macro_rules! copy2d_shaders {
    ($( ($mod:ident, $ty:literal) ),* $(,)?) => {
        $(
            mod $mod {
                vulkano_shaders::shader! {
                    ty: "compute",
                    src: "
                        #version 450
                        #extension GL_ARB_gpu_shader_int64 : require

                        // Use a 16x16 workgroup.
                        layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

                        // Source buffer (read-only).
                        layout(set = 0, binding = 0) readonly buffer SrcBuffer {
                            TYPE src_data[];
                        };
                        // Destination buffer (write-only).
                        layout(set = 0, binding = 1) writeonly buffer DstBuffer {
                            TYPE dst_data[];
                        };

                        // Push constants for region parameters.
                        // Order:
                        //  src_offset, dst_offset,
                        //  rows, cols,
                        //  src_stride, dst_stride.
                        layout(push_constant) uniform Copy2DPushConstants {
                            uint src_offset;
                            uint dst_offset;
                            uint rows;        // d1: number of rows to copy
                            uint cols;        // d2: number of columns to copy
                            uint src_stride;  // source row stride
                            uint dst_stride;  // destination row stride
                        } pc;

                        void main() {
                            uint x = gl_GlobalInvocationID.x; // column index within the region
                            uint y = gl_GlobalInvocationID.y; // row index within the region
                            if (x < pc.cols && y < pc.rows) {
                                uint src_index = pc.src_offset + y * pc.src_stride + x;
                                uint dst_index = pc.dst_offset + y * pc.dst_stride + x;
                                dst_data[dst_index] = src_data[src_index];
                            }
                        }
                    ",
                    define: [("TYPE", $ty)]
                }
            }
        )*
    }
}

macro_rules! copy_strided_src_shaders {
    ($( ($mod:ident, $ty:literal) ),* $(,)?) => {
        $(
            mod $mod {
                vulkano_shaders::shader! {
                    ty: "compute",
                    src: "
                        #version 450
                        #extension GL_ARB_gpu_shader_int64 : require

                        // Use a 256-thread 1D workgroup.
                        layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

                        // Source buffer (read-only).
                        layout(set = 0, binding = 0) readonly buffer SrcBuffer {
                            TYPE src_data[];
                        };
                        // Destination buffer (write-only).
                        layout(set = 0, binding = 1) writeonly buffer DstBuffer {
                            TYPE dst_data[];
                        };

                        // Push constants carrying full layout information for the source tensor
                        // plus a destination offset.
                        // We assume a maximum rank of 4.
                        layout(push_constant) uniform PushConstants {
                            uint base;
                            uint rank;
                            uvec4 shape;
                            uvec4 stride;
                            uint dst_offset;
                        } pc;

                        // Compute the physical index in the source from a linear index using full layout.
                        uint get_strided_index(uint lin_idx) {
                            uint remaining = lin_idx;
                            uint phys_idx = 0;
                            if (pc.rank > 0u) {
                                uint prod = 1u;
                                if (pc.rank > 1u) { prod = pc.shape.y * pc.shape.z * pc.shape.w; }
                                uint i0 = remaining / prod;
                                remaining = remaining % prod;
                                phys_idx += i0 * pc.stride.x;
                            }
                            if (pc.rank > 1u) {
                                uint prod = 1u;
                                if (pc.rank > 2u) { prod = pc.shape.z * pc.shape.w; }
                                uint i1 = remaining / prod;
                                remaining = remaining % prod;
                                phys_idx += i1 * pc.stride.y;
                            }
                            if (pc.rank > 2u) {
                                uint prod = 1u;
                                if (pc.rank > 3u) { prod = pc.shape.w; }
                                uint i2 = remaining / prod;
                                remaining = remaining % prod;
                                phys_idx += i2 * pc.stride.z;
                            }
                            if (pc.rank > 3u) {
                                uint i3 = remaining;
                                phys_idx += i3 * pc.stride.w;
                            }
                            return pc.base + phys_idx;
                        }

                        void main() {
                            uint lin_idx = gl_GlobalInvocationID.x;
                            // Compute total number of elements.
                            uint total_elements = pc.shape.x * pc.shape.y * pc.shape.z * pc.shape.w;
                            if (lin_idx < total_elements) {
                                uint src_index = get_strided_index(lin_idx);
                                // Write contiguously starting at dst_offset.
                                dst_data[pc.dst_offset + lin_idx] = src_data[src_index];
                            }
                        }
                    ",
                    define: [("TYPE", $ty)]
                }
            }
        )*
    }
}

macro_rules! cmp_shaders {
    ($( ($mod:ident, $op:literal, $ty:literal) ),* $(,)?) => {
        $(
            mod $mod {
                vulkano_shaders::shader! {
                    ty: "compute",
                    src: "
                        #version 450
                        layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

                        // Input buffers containing values of type TYPE.
                        layout(set = 0, binding = 0) buffer LhsBuffer {
                            TYPE lhs_data[];
                        };
                        layout(set = 0, binding = 1) buffer RhsBuffer {
                            TYPE rhs_data[];
                        };
                        // Output buffer contains unsigned integers: 1 means true, 0 means false.
                        layout(set = 0, binding = 2) buffer OutBuffer {
                            uint out_data[];
                        };

                        // The comparison operation is chosen by the preprocessor define OP.
                        void main() {
                            uint idx = gl_GlobalInvocationID.x;
                            out_data[idx] = (lhs_data[idx] OP rhs_data[idx]) ? 1u : 0u;
                        }
                    ",
                    define: [("OP", $op), ("TYPE", $ty)]
                }
            }
        )*
    };
}

impl VulkanDevice {
    fn select_physical_device(
        instance: &Arc<Instance>,
        gpu_id: usize,
    ) -> Result<Arc<PhysicalDevice>> {
        let physical_devices = instance
            .enumerate_physical_devices()
            .map_err(VulkanError::VulkanError)?;
        let physical = physical_devices
            .into_iter()
            .nth(gpu_id)
            .ok_or(VulkanError::Message(String::from("ordinal out of range")))?;
        Ok(physical)
    }

    pub(crate) fn to_cpu<T: BufferContents + Clone + Copy + Send>(
        &self,
        buffer: Arc<Option<Subbuffer<[u8]>>>,
    ) -> Result<Vec<T>> {
        if let Some(buffer) = (*buffer).clone() {
            // XXX Use a buffer pool to avoid creating a new buffer every time
            let cpu_buffer = Buffer::new_unsized::<[T]>(
                self.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                    ..Default::default()
                },
                buffer.size(),
            )
            .map_err(VulkanError::ValidatedAllocateBufferError)?;

            // Copy data from the Vulkan buffer to the CPU buffer
            let mut builder = AutoCommandBufferBuilder::primary(
                Arc::new(StandardCommandBufferAllocator::new(
                    self.device.clone(),
                    StandardCommandBufferAllocatorCreateInfo::default(),
                )),
                self.queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .map_err(VulkanError::ValidatedVulkanError)?;

            builder
                .copy_buffer(CopyBufferInfo::buffers(buffer, cpu_buffer.clone()))
                .map_err(VulkanError::ValidationError)?;
            let command_buffer = builder.build().map_err(VulkanError::ValidatedVulkanError)?;

            let future = command_buffer
                .execute(self.queue.clone())
                .map_err(VulkanError::CommandBufferExecError)?;
            future
                .then_signal_fence_and_flush()
                .unwrap()
                .wait(None)
                .map_err(VulkanError::ValidatedVulkanError)?;

            // Map the buffer memory and read the data
            let cpu_buffer_content = cpu_buffer.read().unwrap();

            // Convert buffer content to Vec<T>
            let result: Vec<T> = cpu_buffer_content.iter().copied().collect();
            Ok(result)
        } else {
            Ok(vec![])
        }
    }

    fn allocate_data<T>(&self, dtype: DType, source: DataSource<T>) -> Result<VulkanStorage>
    where
        T: vulkano::buffer::BufferContents + Clone + Pod + 'static,
    {
        // Determine element count and validate sizes
        let (count, type_size) = match &source {
            DataSource::Slice(data) => (data.len(), std::mem::size_of::<T>()),
            DataSource::Fill { count, .. } => (*count, std::mem::size_of::<T>()),
        };

        // Allocate device buffer
        let buffer_size = count * type_size;
        let alignment = std::mem::align_of::<T>();
        let buffer = self.allocate(buffer_size, dtype, alignment)?;

        if let Some(ref buffer) = buffer {
            let mut builder = AutoCommandBufferBuilder::primary(
                self.command_buffer_allocator.clone(),
                self.queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .map_err(VulkanError::ValidatedVulkanError)?;

            // Handle 4-byte types with direct fill if possible
            let filled = match &source {
                DataSource::Fill { value, .. } if type_size == 4 => {
                    let value_u32 = bytemuck::cast(value.clone());
                    builder
                        .fill_buffer(buffer.clone().reinterpret::<[u32]>(), value_u32)
                        .map_err(VulkanError::ValidationError)?;
                    true
                }
                _ => false,
            };
            if !filled {
                // Common path for both slice and fill operations
                let iter: Box<dyn ExactSizeIterator<Item = T>> = match source {
                    DataSource::Slice(data) => Box::new(data.iter().cloned()),
                    DataSource::Fill { value, count } => {
                        Box::new(std::iter::repeat(value).take(count))
                    }
                };

                let cpu_buffer = Buffer::from_iter(
                    self.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_HOST
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    iter,
                )
                .map_err(VulkanError::ValidatedAllocateBufferError)?;

                builder
                    .copy_buffer(CopyBufferInfo::buffers(cpu_buffer, buffer.clone()))
                    .map_err(VulkanError::ValidationError)?;
            }

            // Execute command buffer
            let command_buffer = builder.build().map_err(VulkanError::ValidatedVulkanError)?;
            let future = command_buffer
                .execute(self.queue.clone())
                .map_err(VulkanError::CommandBufferExecError)?;

            future
                .then_signal_fence_and_flush()
                .map_err(VulkanError::ValidatedVulkanError)?
                .wait(None)
                .map_err(VulkanError::ValidatedVulkanError)?;
        }

        Ok(VulkanStorage::new(buffer, self.clone(), count, dtype))
    }

    fn allocate_with_data<T>(&self, dtype: DType, data: &[T]) -> Result<VulkanStorage>
    where
        T: vulkano::buffer::BufferContents + Clone + Pod + 'static,
    {
        self.allocate_data(dtype, DataSource::Slice(data))
    }

    fn allocate_filled<T>(&self, count: usize, dtype: DType, value: T) -> Result<VulkanStorage>
    where
        T: vulkano::buffer::BufferContents + Clone + Pod + 'static,
    {
        self.allocate_data(dtype, DataSource::Fill { value, count })
    }

    // XXX alignment should really be gotten from dtype.
    fn allocate(
        &self,
        buffer_size: usize,
        dtype: DType,
        alignment: usize,
    ) -> Result<Option<Subbuffer<[u8]>>> {
        if buffer_size == 0 {
            return Ok(None);
        }

        Ok(Some(
            self.buffer_allocator
                .lock()
                .map_err(|e| VulkanError::Message(format!("lock error: {}", e)))?
                .allocate(
                    DeviceLayout::from_size_alignment(
                        buffer_size as DeviceSize,
                        alignment as DeviceSize,
                    )
                    .ok_or(VulkanError::Message("invalid layout".to_string()))?,
                )
                .map_err(VulkanError::MemoryAllocatorError)?,
        ))
    }
}

macro_rules! cast_shaders {
    ($( ($mod:ident, $src:literal, $dst:literal) ),* $(,)?) => {
        $(
            mod $mod {
                // This macro invocation creates a shader module at compile time.
                vulkano_shaders::shader! {
                    ty: "compute",
                    src: "
                        #version 450
                        #extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
                        #extension GL_EXT_shader_16bit_storage : require
                        #extension GL_AMD_gpu_shader_half_float: enable

                        layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
                        layout(set = 0, binding = 0) buffer Input {
                            SRC_TYPE input_data[];
                        };
                        layout(set = 0, binding = 1) buffer Output {
                            DST_TYPE output_data[];
                        };
                        layout(push_constant) uniform PushConstants {
                            uint base;
                            uint rank;         // number of dimensions
                            uvec4 shape;       // padded shape (unused dimensions set to 1)
                            uvec4 stride;      // padded stride (unused dimensions set to 1)
                        } pc;

                        // This function converts a linear index to a physical index using the full
                        // multi-dimensional layout. We assume row-major ordering.
                        uint get_strided_index(uint lin_idx) {
                            uint remaining = lin_idx;
                            uint phys_idx = 0;
                            if (pc.rank > 0u) {
                                uint prod = 1u;
                                if (pc.rank > 1u) { prod = pc.shape.y * pc.shape.z * pc.shape.w; }
                                uint i0 = remaining / prod;
                                remaining = remaining % prod;
                                phys_idx += i0 * pc.stride.x;
                            }
                            if (pc.rank > 1u) {
                                uint prod = 1u;
                                if (pc.rank > 2u) { prod = pc.shape.z * pc.shape.w; }
                                uint i1 = remaining / prod;
                                remaining = remaining % prod;
                                phys_idx += i1 * pc.stride.y;
                            }
                            if (pc.rank > 2u) {
                                uint prod = 1u;
                                if (pc.rank > 3u) { prod = pc.shape.w; }
                                uint i2 = remaining / prod;
                                remaining = remaining % prod;
                                phys_idx += i2 * pc.stride.z;
                            }
                            if (pc.rank > 3u) {
                                uint i3 = remaining;
                                phys_idx += i3 * pc.stride.w;
                            }
                            return pc.base + phys_idx;
                        }

                        void main() {
                            uint idx = get_strided_index(gl_GlobalInvocationID.x);
                            output_data[idx] = DST_TYPE(input_data[idx]);
                        }",
                    define: [("SRC_TYPE", $src), ("DST_TYPE", $dst)]
                }
            }
        )*
    };
}

impl crate::backend::BackendDevice for VulkanDevice {
    type Storage = VulkanStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let library = VulkanLibrary::new().map_err(|err| VulkanError::from(err))?;
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                ..Default::default()
            },
        )
        .map_err(VulkanError::ValidatedVulkanError)?;

        // Select the physical device
        let physical_device = Self::select_physical_device(&instance, ordinal)?;

        // Initialize Vulkan resources (queues, command buffers, etc.)
        let device_extensions = DeviceExtensions {
            khr_storage_buffer_storage_class: true,
            ..DeviceExtensions::empty()
        };
        let required_features = DeviceFeatures {
            storage_buffer16_bit_access: true,
            shader_int64: true,
            ..DeviceFeatures::empty()
        };

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                enabled_features: required_features,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: 0,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .map_err(VulkanError::ValidatedVulkanError)?;
        let queue = queues
            .next()
            .ok_or(VulkanError::Message("no queues found".to_string()))?;

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let buffer_allocator = Arc::new(Mutex::new(SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                arena_size: 64 * 1024 * 1024,
                buffer_usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::TRANSFER_SRC,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE, // XXX actually the default
                ..Default::default()
            },
        )));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        ));

        // // Initialize zero-init compute pipeline
        // let zero_init_pipeline = {
        //     mod cs {
        //         vulkano_shaders::shader! {
        //             ty: "compute",
        //             src: "
        //                 #version 450
        //                 layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
        //                 layout(set = 0, binding = 0) buffer Data {
        //                     uint data[];
        //                 };
        //                 void main() {
        //                     uint idx = gl_GlobalInvocationID.x;
        //                     data[idx] = 0;
        //                 }
        //             ",
        //         }
        //     }
        //
        //     let shader = cs::load(device.clone())?;
        //     ComputePipeline::new(
        //         device.clone(),
        //         shader.entry_point("main").unwrap(),
        //         &(),
        //         None,
        //         |_| {},
        //     )?
        // };

        let cast_pipelines = {
            cast_shaders!(
                (float_to_half, "float", "float16_t"),
                (half_to_float, "float16_t", "float"),
                (uint_to_float, "uint", "float")
            );
            let shaders = [
                float_to_half::load(device.clone())
                    .map_err(VulkanError::ValidatedVulkanError)
                    .map_err(|e| {
                        if let VulkanError::ValidatedVulkanError(ref e) = e {
                            println!("Error: {:?}", e);
                        }
                        e
                    })?,
                half_to_float::load(device.clone()).map_err(VulkanError::ValidatedVulkanError)?,
                uint_to_float::load(device.clone()).map_err(VulkanError::ValidatedVulkanError)?,
            ];
            // Create the pipelines
            shaders
                .into_iter()
                .map(|shader| {
                    let stage = PipelineShaderStageCreateInfo::new(
                        shader
                            .entry_point("main")
                            .ok_or(VulkanError::Message("No entry point".to_string()))
                            .unwrap(), // XXX
                    );
                    let layout = PipelineLayout::new(
                        device.clone(),
                        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                            .into_pipeline_layout_create_info(device.clone())
                            .map_err(|e| VulkanError::Message(e.to_string()))
                            .unwrap(), // XXX
                    )
                    .map_err(|e| VulkanError::Message(e.to_string()))
                    .unwrap(); // XXX
                    ComputePipeline::new(
                        device.clone(),
                        None,
                        ComputePipelineCreateInfo::stage_layout(stage, layout),
                    )
                    .map_err(|e| VulkanError::Message(e.to_string()))
                    .unwrap() // XXX
                })
                .collect::<Vec<_>>()
        };

        unary_shaders!(
            (neg_shader, "neg_op", "float", "float"),
            (gelu_shader, "gelu_op", "float", "float"),
            (gelu_erf_shader, "gelu_erf_op", "float", "float"),
            (erf_shader, "erf_op", "float", "float"),
            (silu_shader, "silu_op", "float", "float"),
            (ceil_shader, "ceil_op", "float", "float"),
            (floor_shader, "floor_op", "float", "float"),
            (round_shader, "round_op", "float", "float"),
            (sign_shader, "sign_op", "float", "float"),
            (sqr_shader, "sqr_op", "float", "float"),
            (neg_shader_f16, "neg_op", "float", "float16_t"),
            (gelu_shader_f16, "gelu_op", "float", "float16_t"),
            (gelu_erf_shader_f16, "gelu_erf_op", "float", "float16_t"),
            (erf_shader_f16, "erf_op", "float", "float16_t"),
            (silu_shader_f16, "silu_op", "float", "float16_t"),
            (ceil_shader_f16, "ceil_op", "float", "float16_t"),
            (floor_shader_f16, "floor_op", "float", "float16_t"),
            (round_shader_f16, "round_op", "float", "float16_t"),
            (sign_shader_f16, "sign_op", "float", "float16_t"),
            (sqr_shader_f16, "sqr_op", "float", "float16_t"),
        );

        // unary_shaders!(
        //     (neg_shader_f64, "neg_op", "double"),
        //     (gelu_shader_f64, "gelu_op", "double"),
        //     (gelu_erf_shader_f64, "gelu_erf_op", "double"),
        //     (erf_shader_f64, "erf_op", "double"),
        //     (silu_shader_f64, "silu_op", "double"),
        //     (ceil_shader_f64, "ceil_op", "double"),
        //     (floor_shader_f64, "floor_op", "double"),
        //     (round_shader_f64, "round_op", "double"),
        //     (sign_shader_f64, "sign_op", "double"),
        // );

        macro_rules! load_unary_pipelines {
            ($device:expr, $($name:expr => $mod:ident),* $(,)?) => {{
                use vulkano::pipeline::compute::{ComputePipeline, ComputePipelineCreateInfo};
                use vulkano::pipeline::layout::{PipelineLayout, PipelineDescriptorSetLayoutCreateInfo};
                use vulkano::pipeline::PipelineShaderStageCreateInfo;
                use std::sync::Arc;
                use std::collections::HashMap;
                use crate::VulkanError; // Your existing error handling enum.

                let mut map = HashMap::new();
                $(
                    let shader = $mod::load($device.clone()).map_err(VulkanError::ValidatedVulkanError)?;
                    let entry_point = shader
                        .entry_point("main")
                        .ok_or_else(|| VulkanError::Message(format!("Entry point missing: {}", $name)))?;
                    let stage = PipelineShaderStageCreateInfo::new(entry_point);
                    let layout_info = PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                        .into_pipeline_layout_create_info($device.clone())
                        .map_err(VulkanError::IntoPipelineLayoutCreateInfoError)?;
                    let layout = PipelineLayout::new($device.clone(), layout_info)
                        .map_err(VulkanError::ValidatedVulkanError)?;
                    let pipeline_create_info = ComputePipelineCreateInfo::stage_layout(stage, layout);
                    let pipeline = ComputePipeline::new($device.clone(), None, pipeline_create_info)
                            .map_err(VulkanError::ValidatedVulkanError)?;
                    map.insert($name, pipeline);
                )*
                map
            }};
        }

        let unary_pipelines = load_unary_pipelines!(
            device,
            "neg"          => neg_shader,
            "gelu"         => gelu_shader,
            "gelu_erf"     => gelu_erf_shader,
            "erf"          => erf_shader,
            "silu"         => silu_shader,
            "ceil"         => ceil_shader,
            "floor"        => floor_shader,
            "round"        => round_shader,
            "sign"         => sign_shader,
            "sqr"          => sqr_shader,
            "neg_f16"      => neg_shader_f16,
            "gelu_f16"     => gelu_shader_f16,
            "gelu_erf_f16" => gelu_erf_shader_f16,
            "erf_f16"      => erf_shader_f16,
            "silu_f16"     => silu_shader_f16,
            "ceil_f16"     => ceil_shader_f16,
            "floor_f16"    => floor_shader_f16,
            "round_fq6"    => round_shader_f16,
            "sign_f16"     => sign_shader_f16,
            "sqr_f16"      => sqr_shader_f16,
            // "neg_f64"      => neg_shader_f64,
            // "gelu_f64"     => gelu_shader_f64,
            // "gelu_erf_f64" => gelu_erf_shader_f64,
            // "erf_f64"      => erf_shader_f64,
            // "silu_f64"     => silu_shader_f64,
            // "ceil_f64"     => ceil_shader_f64,
            // "floor_f64"    => floor_shader_f64,
            // "round_f64"    => round_shader_f64,
            // "sign_f64"     => sign_shader_f64,
        );

        binary_shaders!(
            (add_shader, "add_op", "float"),
            (sub_shader, "sub_op", "float"),
            (div_shader, "div_op", "float"),
            (mul_shader, "mul_op", "float"),
        );

        macro_rules! load_binary_pipelines {
            ($device:expr, $($name:expr => $mod:ident),* $(,)?) => {{
                use vulkano::pipeline::compute::{ComputePipeline, ComputePipelineCreateInfo};
                use vulkano::pipeline::layout::{PipelineLayout, PipelineDescriptorSetLayoutCreateInfo};
                use vulkano::pipeline::PipelineShaderStageCreateInfo;
                use std::sync::Arc;
                use std::collections::HashMap;
                use crate::VulkanError; // Your existing error handling enum.

                let mut map = HashMap::new();
                $(
                    let shader = $mod::load($device.clone()).map_err(VulkanError::ValidatedVulkanError)?;
                    let entry_point = shader
                        .entry_point("main")
                        .ok_or_else(|| VulkanError::Message(format!("Entry point missing: {}", $name)))?;
                    let stage = PipelineShaderStageCreateInfo::new(entry_point);
                    let layout_info = PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                        .into_pipeline_layout_create_info($device.clone())
                        .map_err(VulkanError::IntoPipelineLayoutCreateInfoError)?;
                    let layout = PipelineLayout::new($device.clone(), layout_info)
                        .map_err(VulkanError::ValidatedVulkanError)?;
                    let pipeline_create_info = ComputePipelineCreateInfo::stage_layout(stage, layout);
                    let pipeline = ComputePipeline::new($device.clone(), None, pipeline_create_info)
                            .map_err(VulkanError::ValidatedVulkanError)?;
                    map.insert($name, pipeline);
                )*
                map
            }};
        }

        let binary_pipelines = load_binary_pipelines!(
            device,
            "add"          => add_shader,
            "sub"          => sub_shader,
            "div"          => div_shader,
            "mul"          => mul_shader,
        );

        reduce_shaders!(
            (max_shader, "max_op", "float"),
            (sum_shader, "sum_op", "float"),
        );

        macro_rules! load_reduce_pipelines {
            ($device:expr, $($name:expr => $mod:ident),* $(,)?) => {{
                use vulkano::pipeline::compute::{ComputePipeline, ComputePipelineCreateInfo};
                use vulkano::pipeline::layout::{PipelineLayout, PipelineDescriptorSetLayoutCreateInfo};
                use vulkano::pipeline::PipelineShaderStageCreateInfo;
                use std::sync::Arc;
                use std::collections::HashMap;
                use crate::VulkanError; // Your existing error handling enum.

                let mut map = HashMap::new();
                $(
                    let shader = $mod::load($device.clone()).map_err(VulkanError::ValidatedVulkanError)?;
                    let entry_point = shader
                        .entry_point("main")
                        .ok_or_else(|| VulkanError::Message(format!("Entry point missing: {}", $name)))?;
                    let stage = PipelineShaderStageCreateInfo::new(entry_point);
                    let layout_info = PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                        .into_pipeline_layout_create_info($device.clone())
                        .map_err(VulkanError::IntoPipelineLayoutCreateInfoError)?;
                    let layout = PipelineLayout::new($device.clone(), layout_info)
                        .map_err(VulkanError::ValidatedVulkanError)?;
                    let pipeline_create_info = ComputePipelineCreateInfo::stage_layout(stage, layout);
                    let pipeline = ComputePipeline::new($device.clone(), None, pipeline_create_info)
                            .map_err(VulkanError::ValidatedVulkanError)?;
                    map.insert($name, pipeline);
                )*
                map
            }};
        }

        let reduce_pipelines = load_reduce_pipelines!(
            device,
            "max"          => max_shader,
            "sum"          => sum_shader,
        );

        affine_elu_shaders!((affine_shader, "affine_op"), (elu_shader, "elu_op"),);

        macro_rules! load_affine_elu_pipelines {
            ($device:expr, $($name:expr => $mod:ident),* $(,)?) => {{
                 use vulkano::pipeline::compute::{ComputePipeline, ComputePipelineCreateInfo};
                 use vulkano::pipeline::layout::{PipelineLayout, PipelineDescriptorSetLayoutCreateInfo};
                 use vulkano::pipeline::PipelineShaderStageCreateInfo;
                 use std::collections::HashMap;
                 use std::sync::Arc;
                 let mut map = HashMap::new();
                 $(
                     let shader = $mod::load($device.clone())
                         .map_err(VulkanError::ValidatedVulkanError)?;
                     let entry_point = shader.entry_point("main")
                         .ok_or_else(|| VulkanError::Message(format!("Missing entry point for {}", $name)))?;
                     let stage = PipelineShaderStageCreateInfo::new(entry_point);
                     let layout_info = PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                        .into_pipeline_layout_create_info($device.clone())
                        .map_err(VulkanError::IntoPipelineLayoutCreateInfoError)?;
                     let layout = PipelineLayout::new($device.clone(), layout_info)
                        .map_err(VulkanError::ValidatedVulkanError)?;
                     let pipeline_create_info = ComputePipelineCreateInfo::stage_layout(stage, layout);
                     let pipeline = ComputePipeline::new($device.clone(), None, pipeline_create_info)
                        .map_err(VulkanError::ValidatedVulkanError)?;
                     map.insert($name, pipeline);
                 )*
                 map
            }};
        }

        let affine_elu_pipelines = load_affine_elu_pipelines!(
            device,
            "affine" => affine_shader,
            "elu" => elu_shader,
        );

        copy2d_shaders!((copy2d_shader, "float"), (copy2d_shader_64, "int64_t"),);

        copy_strided_src_shaders!(
            (copy_strided_src_shader, "float"),
            (copy_strided_src_shader_64, "int64_t"),
        );

        macro_rules! load_copy_shaders {
            ($device:expr, $($name:expr => $mod:ident),* $(,)?) => {{
                 use vulkano::pipeline::compute::{ComputePipeline, ComputePipelineCreateInfo};
                 use vulkano::pipeline::layout::{PipelineLayout, PipelineDescriptorSetLayoutCreateInfo};
                 use vulkano::pipeline::PipelineShaderStageCreateInfo;
                 use std::collections::HashMap;
                 use std::sync::Arc;
                 let mut map = HashMap::new();
                 $(
                     let shader = $mod::load($device.clone())
                         .map_err(VulkanError::ValidatedVulkanError)?;
                     let entry_point = shader.entry_point("main")
                         .ok_or_else(|| VulkanError::Message(format!("Missing entry point for {}", $name)))?;
                     let stage = PipelineShaderStageCreateInfo::new(entry_point);
                     let layout_info = PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                        .into_pipeline_layout_create_info($device.clone())
                        .map_err(VulkanError::IntoPipelineLayoutCreateInfoError)?;
                     let layout = PipelineLayout::new($device.clone(), layout_info)
                        .map_err(VulkanError::ValidatedVulkanError)?;
                     let pipeline_create_info = ComputePipelineCreateInfo::stage_layout(stage, layout);
                     let pipeline = ComputePipeline::new($device.clone(), None, pipeline_create_info)
                        .map_err(VulkanError::ValidatedVulkanError)?;
                     map.insert($name, pipeline);
                 )*
                 map
            }};
        }

        let copy_pipelines = load_copy_shaders!(
            device,
            "copy2d" => copy2d_shader,
            "copy2d_64" => copy2d_shader_64,
            "copy_strided_src" => copy_strided_src_shader,
            "copy_strided_src_64" => copy_strided_src_shader_64,
        );

        cmp_shaders!((cmp_eq_shader, "==", "float"));

        macro_rules! load_cmp_pipelines {
            ($device:expr, $($name:expr => $mod:ident),* $(,)?) => {
                {
                    use vulkano::pipeline::compute::{ComputePipeline, ComputePipelineCreateInfo};
                    use vulkano::pipeline::layout::{PipelineLayout, PipelineDescriptorSetLayoutCreateInfo};
                    use vulkano::pipeline::PipelineShaderStageCreateInfo;
                    use std::collections::HashMap;
                    use std::sync::Arc;
                    let mut map = HashMap::new();
                    $(
                        let shader = $mod::load($device.clone())
                            .map_err(VulkanError::ValidatedVulkanError)?;
                        let entry_point = shader.entry_point("main")
                            .ok_or_else(|| VulkanError::Message(format!("Missing cmp entry point for {}", $name)))?;
                        let stage = PipelineShaderStageCreateInfo::new(entry_point);
                        let layout_info = PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                            .into_pipeline_layout_create_info($device.clone())
                            .map_err(VulkanError::IntoPipelineLayoutCreateInfoError)?;
                        let layout = PipelineLayout::new($device.clone(), layout_info)
                            .map_err(VulkanError::ValidatedVulkanError)?;
                        let pipeline_create_info = ComputePipelineCreateInfo::stage_layout(stage, layout);
                        let pipeline = ComputePipeline::new($device.clone(), None, pipeline_create_info)
                            .map_err(VulkanError::ValidatedVulkanError)?;
                        map.insert($name, pipeline);
                    )*
                    map
                }
            };
        }

        let cmp_pipelines = load_cmp_pipelines!(
            device,
            "Eq" => cmp_eq_shader,
            // Add more if you want, e.g.:
            // "Lt" => cmp_lt_shader,
            // "Gt" => cmp_gt_shader,
        );

        Ok(Self {
            ordinal,
            device,
            queue,
            memory_allocator,
            buffer_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
            //            pools: Arc::new(Mutex::new(HashMap::new())),
            //            zero_init_pipeline: Arc::new(zero_init_pipeline),
            cast_pipelines,
            unary_pipelines,
            binary_pipelines,
            reduce_pipelines,
            affine_elu_pipelines,
            copy_pipelines,
            cmp_pipelines,
        })
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Vulkan {
            gpu_id: self.ordinal,
        }
    }

    fn same_device(&self, other: &Self) -> bool {
        self.ordinal == other.ordinal
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let count_32 = (shape.elem_count() * dtype.size_in_bytes() + 3) / size_of::<u32>();
        self.allocate_filled(count_32, dtype, 0u32)
    }

    fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let num_elements = shape.elem_count();
        let num_units = match dtype {
            DType::F32 | DType::U32 => num_elements, // 1 unit per element
            DType::F16 | DType::BF16 => (num_elements + 1) / 2, // 2 elements per unit
            DType::U8 => (num_elements + 3) / 4,     // 4 elements per unit
            DType::I64 | DType::F64 => num_elements, // 1 unit per element, but each unit is 64-bit
        };
        Ok(match dtype {
            DType::F32 => self.allocate_filled(num_units, dtype, 1.0f32.to_bits())?,
            DType::U32 => self.allocate_filled(num_units, dtype, 1u32)?,
            DType::F16 => {
                let bits = f16::from_f32(1.0).to_bits();
                self.allocate_filled(num_units, dtype, ((bits as u32) << 16) | (bits as u32))?
            }
            DType::BF16 => {
                let bits = bf16::from_f32(1.0).to_bits();
                self.allocate_filled(num_units, dtype, ((bits as u32) << 16) | (bits as u32))?
            }
            DType::U8 => {
                let byte = 1u8 as u32;
                self.allocate_filled(
                    num_units,
                    dtype,
                    (byte << 24) | (byte << 16) | (byte << 8) | byte,
                )?
            }
            DType::I64 => self.allocate_filled(num_units, dtype, 1i64 as u64)?,
            DType::F64 => self.allocate_filled(num_units, dtype, 1.0f64.to_bits())?,
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let count = shape.elem_count();
        let type_size = dtype.size_in_bytes();
        let buffer = self.allocate(count * type_size, dtype, type_size)?; // XXX alignment might need revisiting
        Ok(VulkanStorage::new(buffer, self.clone(), count, dtype))
    }

    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> Result<Self::Storage> {
        fail!()
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        Ok(match storage {
            CpuStorage::U8(s) => self.allocate_with_data(storage.dtype(), s)?,
            CpuStorage::U32(s) => self.allocate_with_data(storage.dtype(), s)?,
            CpuStorage::I64(s) => self.allocate_with_data(storage.dtype(), s)?,
            CpuStorage::BF16(s) => self.allocate_with_data(storage.dtype(), s)?,
            CpuStorage::F16(s) => self.allocate_with_data(storage.dtype(), s)?,
            CpuStorage::F32(s) => self.allocate_with_data(storage.dtype(), s)?,
            CpuStorage::F64(s) => self.allocate_with_data(storage.dtype(), s)?,
        })
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&storage)
    }

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        fail!()
    }

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        fail!()
    }

    fn set_seed(&self, _: u64) -> Result<()> {
        fail!()
    }

    fn synchronize(&self) -> Result<()> {
        fail!()
    }
}
