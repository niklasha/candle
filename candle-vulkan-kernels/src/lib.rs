use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use vulkano::device::Device;
use vulkano::pipeline::{ComputePipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::shader::ShaderModule;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    BF16,
    F16,
    F32,
    I64,
    U32,
    U8,
}

impl DType {
    fn size_in_bytes(&self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U32 => 4,
            Self::I64 => 8,
            Self::BF16 => 2,
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    // Affine,
    // Binary,
    // Conv,
    // Fill,
    // Gemm,
    // Indexing,
    // MlxSort,
    // Quantized,
    // Random,
    // Reduce,
    // Sort,
    // Ternary,
    // Sdpa,
}

// pub mod copy2d {
//     pub struct Kernel(pub &'static str);
//     pub const FLOAT: Kernel = Kernel("copy2d_f32");
//     pub const HALF: Kernel = Kernel("copy2d_f16");
//     pub const BFLOAT: Kernel = Kernel("copy2d_bf16");
//     pub const I64: Kernel = Kernel("copy2d_i64");
//     pub const U32: Kernel = Kernel("copy2d_u32");
//     pub const U8: Kernel = Kernel("copy2d_u8");
// }
//
// macro_rules! ops{
//     ($($name:ident),+) => {
//         pub mod contiguous {
//         pub struct Kernel(pub &'static str);
//         $(
//         pub mod $name {
//             use super::Kernel;
//             pub const FLOAT: Kernel = Kernel(concat!(stringify!($name), "_f32"));
//             pub const HALF: Kernel = Kernel(concat!(stringify!($name), "_f16"));
//             pub const BFLOAT: Kernel = Kernel(concat!(stringify!($name), "_bf16"));
//             pub const I64: Kernel = Kernel(concat!(stringify!($name), "_i64"));
//             pub const U32: Kernel = Kernel(concat!(stringify!($name), "_u32"));
//             pub const U8: Kernel = Kernel(concat!(stringify!($name), "_u8"));
//         }
//         )+
//             pub mod copy {
//                 use super::Kernel;
//                 pub const FLOAT: Kernel = Kernel("copy_f32");
//                 pub const HALF: Kernel = Kernel("copy_f16");
//                 pub const BFLOAT: Kernel = Kernel("copy_bf16");
//                 pub const I64: Kernel = Kernel("copy_i64");
//                 pub const U32: Kernel = Kernel("copy_u32");
//                 pub const U8: Kernel = Kernel("copy_u8");
//             }
//         }
//
//         pub mod contiguous_tiled {
//         pub struct Kernel(pub &'static str);
//         $(
//         pub mod $name {
//             use super::Kernel;
//             pub const FLOAT: Kernel = Kernel(concat!(stringify!($name), "_f32_tiled"));
//             pub const HALF: Kernel = Kernel(concat!(stringify!($name), "_f16_tiled"));
//             pub const BFLOAT: Kernel = Kernel(concat!(stringify!($name), "_bf16_tiled"));
//             pub const I64: Kernel = Kernel(concat!(stringify!($name), "_i64_tiled"));
//             pub const U32: Kernel = Kernel(concat!(stringify!($name), "_u32_tiled"));
//             pub const U8: Kernel = Kernel(concat!(stringify!($name), "_u8_tiled"));
//         }
//         )+
//             pub mod copy {
//                 use super::Kernel;
//                 pub const FLOAT: Kernel = Kernel("copy_f32_tiled");
//                 pub const HALF: Kernel = Kernel("copy_f16_tiled");
//                 pub const BFLOAT: Kernel = Kernel("copy_bf16_tiled");
//                 pub const I64: Kernel = Kernel("copy_i64_tiled");
//                 pub const U32: Kernel = Kernel("copy_u32_tiled");
//                 pub const U8: Kernel = Kernel("copy_u8_tiled");
//             }
//         }
//
//         pub mod strided {
//         pub struct Kernel(pub &'static str);
//         $(
//         pub mod $name {
//             use super::Kernel;
//             pub const FLOAT: Kernel = Kernel(concat!(stringify!($name), "_f32_strided"));
//             pub const HALF: Kernel = Kernel(concat!(stringify!($name), "_f16_strided"));
//             pub const BFLOAT: Kernel = Kernel(concat!(stringify!($name), "_bf16_strided"));
//             pub const I64: Kernel = Kernel(concat!(stringify!($name), "_i64_strided"));
//             pub const U32: Kernel = Kernel(concat!(stringify!($name), "_u32_strided"));
//             pub const U8: Kernel = Kernel(concat!(stringify!($name), "_u8_strided"));
//         }
//         )+
//             pub mod copy {
//                 use super::Kernel;
//                 pub const FLOAT: Kernel = Kernel("copy_f32_strided");
//                 pub const HALF: Kernel = Kernel("copy_f16_strided");
//                 pub const BFLOAT: Kernel = Kernel("copy_bf16_strided");
//                 pub const I64: Kernel = Kernel("copy_i64_strided");
//                 pub const U32: Kernel = Kernel("copy_u32_strided");
//                 pub const U8: Kernel = Kernel("copy_u8_strided");
//             }
//         }
//     };
// }
//
// pub mod unary {
//     ops!(
//         cos, sin, exp, sqr, sqrt, neg, log, gelu, abs, ceil, floor, relu, round, erf, gelu_erf,
//         tanh, recip, silu, sign, sigmoid
//     );
// }
//
// pub mod binary {
//     ops!(add, sub, mul, div, min, max, eq, ne, le, lt, ge, gt);
// }

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
    #[error("{0:?}")]
    ValidatedVulkanError(#[from] vulkano::Validated<vulkano::VulkanError>),
    // #[error("Invalid matmul arguments {lhs_stride:?} {rhs_stride:?} {mnk:?}")]
    // MatMulNonContiguous {
    //     lhs_stride: Vec<usize>,
    //     rhs_stride: Vec<usize>,
    //     mnk: (usize, usize, usize),
    // },
    // #[error("Sdpa {variation} head size was {got}, expectd {expected:?}")]
    // SdpaHeadSizeMismatch {
    //     variation: &'static str,
    //     got: usize,
    //     expected: Vec<usize>,
    // },
    // #[error("Sdpa {variation} got dtype {got:?}")]
    // SdpaHeadDTypeMismatch {
    //     variation: &'static str,
    //     got: SdpaDType,
    // },
}

impl<T> From<std::sync::PoisonError<T>> for VulkanKernelError {
    fn from(e: std::sync::PoisonError<T>) -> Self {
        Self::LockError(e.to_string())
    }
}

// #[derive(Debug, Clone)]
// pub enum KernelName {
//     Ref(&'static str),
//     Value(String),
// }
//
// impl AsRef<str> for KernelName {
//     fn as_ref(&self) -> &str {
//         match self {
//             Self::Ref(r) => r,
//             Self::Value(v) => v.as_str(),
//         }
//     }
// }
//
// impl std::hash::Hash for KernelName {
//     fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
//         match self {
//             Self::Ref(r) => r.hash(state),
//             Self::Value(v) => v.hash(state),
//         }
//     }
// }
//
// impl PartialEq for KernelName {
//     fn eq(&self, other: &Self) -> bool {
//         let v1: &str = self.as_ref();
//         let v2: &str = other.as_ref();
//         v1 == v2
//     }
// }
//
// impl Eq for KernelName {}
//
// impl From<&'static str> for KernelName {
//     fn from(value: &'static str) -> Self {
//         Self::Ref(value)
//     }
// }
//
// impl From<String> for KernelName {
//     fn from(value: String) -> Self {
//         Self::Value(value)
//     }
// }

#[derive(Debug)]
pub struct Kernels {
    kernels: HashMap<String, Arc<ShaderModule>>,
    pub pipelines: RwLock<HashMap<String, Arc<ComputePipeline>>>,
}

impl Kernels {
    pub fn new(device: Arc<Device>) -> Result<Self, VulkanKernelError> {
        let mut kernels = HashMap::new();

        kernels.insert("cast_f32_f16".to_string(), float_to_float16_t::load(device.clone())?);
        kernels.insert("cast_f16_f32".to_string(), float16_t_to_float::load(device.clone())?);
        kernels.insert("cast_u32_f32".to_string(), uint_to_float::load(device.clone())?);
        kernels.insert("cast_u32_u8".to_string(), uint_to_uint8_t::load(device.clone())?);
        kernels.insert("cast_u8_f32".to_string(), uint8_t_to_float::load(device.clone())?);
        kernels.insert("cast_bf16_f32".to_string(), bf16_to_float::load(device.clone())?);
        kernels.insert("cast_f32_bf16".to_string(), float_to_bf16::load(device.clone())?);
        kernels.insert("cast_bf16_u32".to_string(), bf16_to_uint::load(device.clone())?);
        kernels.insert("cast_u32_bf16".to_string(), uint_to_bf16::load(device.clone())?);
        kernels.insert("cast_bf16_f16".to_string(), bf16_to_float16_t::load(device.clone())?);
        kernels.insert("cast_u32_i64".to_string(), uint_to_int64_t::load(device.clone())?);

        kernels.insert("neg_f32".to_string(), neg_float::load(device.clone())?);
        kernels.insert("abs_f32".to_string(), abs_float::load(device.clone())?);
        kernels.insert("sign_f32".to_string(), sign_float::load(device.clone())?);
        kernels.insert("gelu_f32".to_string(), gelu_float::load(device.clone())?);
        kernels.insert("gelu_erf_f32".to_string(), gelu_erf_float::load(device.clone())?);
        kernels.insert("erf_f32".to_string(), erf_float::load(device.clone())?);
        kernels.insert("silu_f32".to_string(), silu_float::load(device.clone())?);
        kernels.insert("ceil_f32".to_string(), ceil_float::load(device.clone())?);
        kernels.insert("floor_f32".to_string(), floor_float::load(device.clone())?);
        kernels.insert("round_f32".to_string(), round_float::load(device.clone())?);
        kernels.insert("sqr_f32".to_string(), sqr_float::load(device.clone())?);
        kernels.insert("sqrt_f32".to_string(), sqrt_float::load(device.clone())?);
        kernels.insert("sin_f32".to_string(), sin_float::load(device.clone())?);
        kernels.insert("cos_f32".to_string(), cos_float::load(device.clone())?);
        kernels.insert("tan_f32".to_string(), tan_float::load(device.clone())?);
        kernels.insert("sigmoid_f32".to_string(), sigmoid_float::load(device.clone())?);
        kernels.insert("exp_f32".to_string(), exp_float::load(device.clone())?);
        kernels.insert("log_f32".to_string(), log_float::load(device.clone())?);
        kernels.insert("recip_f32".to_string(), recip_float::load(device.clone())?);
        kernels.insert("neg_f16".to_string(), neg_float16_t::load(device.clone())?);
        kernels.insert("abs_f16".to_string(), abs_float16_t::load(device.clone())?);
        kernels.insert("sign_f16".to_string(), sign_float16_t::load(device.clone())?);
        kernels.insert("gelu_f16".to_string(), gelu_float16_t::load(device.clone())?);
        kernels.insert("gelu_erf_f16".to_string(), gelu_erf_float16_t::load(device.clone())?);
        kernels.insert("erf_f16".to_string(), erf_float16_t::load(device.clone())?);
        kernels.insert("silu_f16".to_string(), silu_float16_t::load(device.clone())?);
        kernels.insert("ceil_f16".to_string(), ceil_float16_t::load(device.clone())?);
        kernels.insert("floor_f16".to_string(), floor_float16_t::load(device.clone())?);
        kernels.insert("round_f16".to_string(), round_float16_t::load(device.clone())?);
        kernels.insert("sqr_f16".to_string(), sqr_float16_t::load(device.clone())?);
        kernels.insert("sqrt_f16".to_string(), sqrt_float16_t::load(device.clone())?);
        kernels.insert("sin_f16".to_string(), sin_float16_t::load(device.clone())?);
        kernels.insert("cos_f16".to_string(), cos_float16_t::load(device.clone())?);
        kernels.insert("tan_f16".to_string(), tan_float16_t::load(device.clone())?);
        kernels.insert("sigmoid_f16".to_string(), sigmoid_float16_t::load(device.clone())?);
        kernels.insert("exp_f16".to_string(), exp_float16_t::load(device.clone())?);
        kernels.insert("log_f16".to_string(), log_float16_t::load(device.clone())?);
        kernels.insert("recip_f16".to_string(), recip_float16_t::load(device.clone())?);
        kernels.insert("neg_bf16".to_string(), neg_bf16::load(device.clone())?);
        kernels.insert("abs_bf16".to_string(), abs_bf16::load(device.clone())?);
        kernels.insert("sign_bf16".to_string(), sign_bf16::load(device.clone())?);
        kernels.insert("gelu_bf16".to_string(), gelu_bf16::load(device.clone())?);
        kernels.insert("gelu_erf_bf16".to_string(), gelu_erf_bf16::load(device.clone())?);
        kernels.insert("erf_bf16".to_string(), erf_bf16::load(device.clone())?);
        kernels.insert("silu_bf16".to_string(), silu_bf16::load(device.clone())?);
        kernels.insert("ceil_bf16".to_string(), ceil_bf16::load(device.clone())?);
        kernels.insert("floor_bf16".to_string(), floor_bf16::load(device.clone())?);
        kernels.insert("round_bf16".to_string(), round_bf16::load(device.clone())?);
        kernels.insert("sqr_bf16".to_string(), sqr_bf16::load(device.clone())?);
        kernels.insert("sqrt_bf16".to_string(), sqrt_bf16::load(device.clone())?);
        kernels.insert("sin_bf16".to_string(), sin_bf16::load(device.clone())?);
        kernels.insert("cos_bf16".to_string(), cos_bf16::load(device.clone())?);
        kernels.insert("tan_bf16".to_string(), tan_bf16::load(device.clone())?);
        kernels.insert("sigmoid_bf16".to_string(), sigmoid_bf16::load(device.clone())?);
        kernels.insert("exp_bf16".to_string(), exp_bf16::load(device.clone())?);
        kernels.insert("log_bf16".to_string(), log_bf16::load(device.clone())?);
        kernels.insert("recip_bf16".to_string(), recip_bf16::load(device.clone())?);

        kernels.insert("add_f32".to_string(), add_float::load(device.clone())?);
        kernels.insert("sub_f32".to_string(), sub_float::load(device.clone())?);
        kernels.insert("div_f32".to_string(), div_float::load(device.clone())?);
        kernels.insert("mul_f32".to_string(), mul_float::load(device.clone())?);
        kernels.insert("minimum_f32".to_string(), min_float::load(device.clone())?);
        kernels.insert("maximum_f32".to_string(), max_float::load(device.clone())?);
        kernels.insert("add_bf16".to_string(), add_bf16::load(device.clone())?);
        kernels.insert("sub_bf16".to_string(), sub_bf16::load(device.clone())?);
        kernels.insert("div_bf16".to_string(), div_bf16::load(device.clone())?);
        kernels.insert("mul_bf16".to_string(), mul_bf16::load(device.clone())?);
        kernels.insert("minimum_bf16".to_string(), min_bf16::load(device.clone())?);
        kernels.insert("maximum_bf16".to_string(), max_bf16::load(device.clone())?);

        kernels.insert("sum_partial_f32".to_string(), sum_partial_float::load(device.clone())?);
        kernels.insert("argmax_partial_f32".to_string(), argmax_partial_float::load(device.clone())?);
        kernels.insert("max_partial_f32".to_string(), max_partial_float::load(device.clone())?);
        kernels.insert("argmin_partial_f32".to_string(), argmin_partial_float::load(device.clone())?);
        kernels.insert("min_partial_f32".to_string(), min_partial_float::load(device.clone())?);
        kernels.insert("sum_partial_u32".to_string(), sum_partial_uint::load(device.clone())?);
        kernels.insert("argmax_partial_u32".to_string(), argmax_partial_uint::load(device.clone())?);
        kernels.insert("max_partial_u32".to_string(), max_partial_uint::load(device.clone())?);
        kernels.insert("argmin_partial_u32".to_string(), argmin_partial_uint::load(device.clone())?);
        kernels.insert("min_partial_u32".to_string(), min_partial_uint::load(device.clone())?);
        kernels.insert("sum_combine_f32".to_string(), sum_combine_float::load(device.clone())?);
        kernels.insert("argmax_combine_f32".to_string(), argmax_combine_float::load(device.clone())?);
        kernels.insert("max_combine_f32".to_string(), max_combine_float::load(device.clone())?);
        kernels.insert("argmin_combine_f32".to_string(), argmin_combine_float::load(device.clone())?);
        kernels.insert("min_combine_f32".to_string(), min_combine_float::load(device.clone())?);
        kernels.insert("sum_combine_u32".to_string(), sum_combine_uint::load(device.clone())?);
        kernels.insert("argmax_combine_u32".to_string(), argmax_combine_uint::load(device.clone())?);
        kernels.insert("max_combine_u32".to_string(), max_combine_uint::load(device.clone())?);
        kernels.insert("argmin_combine_u32".to_string(), argmin_combine_uint::load(device.clone())?);
        kernels.insert("min_combine_u32".to_string(), min_combine_uint::load(device.clone())?);

        kernels.insert ("affine_f32".to_string(), affine_float::load(device.clone())?);
        kernels.insert ("affine_bf16".to_string(), affine_bf16::load(device.clone())?);
        kernels.insert ("elu_f32".to_string(), elu_float::load(device.clone())?);

        kernels.insert ("gather_u8_u8".to_string(), gather_uint8_t_uint8_t::load(device.clone())?);
        kernels.insert ("gather_u8_u32".to_string(), gather_uint8_t_uint::load(device.clone())?);
        kernels.insert ("gather_u8_i64".to_string(), gather_uint8_t_int64_t::load(device.clone())?);
        kernels.insert ("gather_u8_bf16".to_string(), gather_uint8_t_bf16::load(device.clone())?);
        kernels.insert ("gather_u8_f32".to_string(),  gather_uint8_t_float::load(device.clone())?);
        //kernels.insert ("gather_u8_f16".to_string(), gather_uint8_t_float16_t::load(device.clone())?);
        kernels.insert ("gather_u32_u8".to_string(), gather_uint_uint8_t::load(device.clone())?);
        kernels.insert ("gather_u32_u32".to_string(), gather_uint_uint::load(device.clone())?);
        kernels.insert ("gather_u32_i64".to_string(), gather_uint_int64_t::load(device.clone())?);
        kernels.insert ("gather_u32_bf16".to_string(), gather_uint_bf16::load(device.clone())?);
        kernels.insert ("gather_u32_f32".to_string(), gather_uint_float::load(device.clone())?);
        //kernels.insert ("gather_u32_f16".to_string(), gather_uint_float16_t::load(device.clone())?);
        kernels.insert ("gather_i64_u8".to_string(), gather_int64_t_uint8_t::load(device.clone())?);
        kernels.insert ("gather_i64_u32".to_string(), gather_int64_t_uint::load(device.clone())?);
        kernels.insert ("gather_i64_i64".to_string(), gather_int64_t_int64_t::load(device.clone())?);
        kernels.insert ("gather_i64_bf16".to_string(), gather_int64_t_bf16::load(device.clone())?);
        kernels.insert ("gather_i64_f32".to_string(), gather_int64_t_float::load(device.clone())?);
        //kernels.insert ("gather_i64_f16".to_string(), gather_int64_t_float16_t::load(device.clone())?);

        //kernels.insert ("scatter_add_u8_u8".to_string(), scatter_add_uint8_t_uint8_t::load(device.clone())?);
        kernels.insert ("scatter_add_u8_u32".to_string(), scatter_add_uint8_t_uint::load(device.clone())?);
        //kernels.insert ("scatter_add_u8_i64".to_string(), scatter_add_uint8_t_int64_t::load(device.clone())?);
        kernels.insert ("scatter_add_u8_bf16".to_string(), scatter_add_uint8_t_bf16::load(device.clone())?);
        kernels.insert ("scatter_add_u8_f32".to_string(),  scatter_add_uint8_t_float::load(device.clone())?);
        //kernels.insert ("scatter_add_u8_f16".to_string(), scatter_add_uint8_t_float16_t::load(device.clone())?);
        //kernels.insert ("scatter_add_u32_u8".to_string(), scatter_add_uint_uint8_t::load(device.clone())?);
        kernels.insert ("scatter_add_u32_u32".to_string(), scatter_add_uint_uint::load(device.clone())?);
        //kernels.insert ("scatter_add_u32_i64".to_string(), scatter_add_uint_int64_t::load(device.clone())?);
        kernels.insert ("scatter_add_u32_bf16".to_string(), scatter_add_uint_bf16::load(device.clone())?);
        kernels.insert ("scatter_add_u32_f32".to_string(), scatter_add_uint_float::load(device.clone())?);
        //kernels.insert ("scatter_add_u32_f16".to_string(), scatter_add_uint_float16_t::load(device.clone())?);
        //kernels.insert ("scatter_add_i64_u8".to_string(), scatter_add_int64_t_uint8_t::load(device.clone())?);
        kernels.insert ("scatter_add_i64_u32".to_string(), scatter_add_int64_t_uint::load(device.clone())?);
        //kernels.insert ("scatter_add_i64_i64".to_string(), scatter_add_int64_t_int64_t::load(device.clone())?);
        kernels.insert ("scatter_add_i64_bf16".to_string(), scatter_add_int64_t_bf16::load(device.clone())?);
        kernels.insert ("scatter_add_i64_f32".to_string(), scatter_add_int64_t_float::load(device.clone())?);
        //kernels.insert ("scatter_add_i64_f16".to_string(), scatter_add_int64_t_float16_t::load(device.clone())?);

        //kernels.insert ("index_add_u8_u8".to_string(), index_add_uint8_t_uint8_t::load(device.clone())?);
        kernels.insert ("index_add_u8_u32".to_string(), index_add_uint8_t_uint::load(device.clone())?);
        //kernels.insert ("index_add_u8_i64".to_string(), index_add_uint8_t_int64_t::load(device.clone())?);
        kernels.insert ("index_add_u8_bf16".to_string(), index_add_uint8_t_bf16::load(device.clone())?);
        kernels.insert ("index_add_u8_f32".to_string(),  index_add_uint8_t_float::load(device.clone())?);
        //kernels.insert ("index_add_u8_f16".to_string(), index_add_uint8_t_float16_t::load(device.clone())?);
        //kernels.insert ("index_add_u32_u8".to_string(), index_add_uint_uint8_t::load(device.clone())?);
        kernels.insert ("index_add_u32_u32".to_string(), index_add_uint_uint::load(device.clone())?);
        //kernels.insert ("index_add_u32_i64".to_string(), index_add_uint_int64_t::load(device.clone())?);
        kernels.insert ("index_add_u32_bf16".to_string(), index_add_uint_bf16::load(device.clone())?);
        kernels.insert ("index_add_u32_f32".to_string(), index_add_uint_float::load(device.clone())?);
        //kernels.insert ("index_add_u32_f16".to_string(), index_add_uint_float16_t::load(device.clone())?);
        //kernels.insert ("index_add_i64_u8".to_string(), index_add_int64_t_uint8_t::load(device.clone())?);
        kernels.insert ("index_add_i64_u32".to_string(), index_add_int64_t_uint::load(device.clone())?);
        //kernels.insert ("index_add_i64_i64".to_string(), index_add_int64_t_int64_t::load(device.clone())?);
        kernels.insert ("index_add_i64_bf16".to_string(), index_add_int64_t_bf16::load(device.clone())?);
        kernels.insert ("index_add_i64_f32".to_string(), index_add_int64_t_float::load(device.clone())?);
        //kernels.insert ("index_add_i64_f16".to_string(), index_add_int64_t_float16_t::load(device.clone())?);

        kernels.insert ("index_select_u8_u8".to_string(), index_select_uint8_t_uint8_t::load(device.clone())?);
        kernels.insert ("index_select_u8_u32".to_string(), index_select_uint8_t_uint::load(device.clone())?);
        kernels.insert ("index_select_u8_i64".to_string(), index_select_uint8_t_int64_t::load(device.clone())?);
        kernels.insert ("index_select_u8_bf16".to_string(), index_select_uint8_t_bf16::load(device.clone())?);
        kernels.insert ("index_select_u8_f32".to_string(),  index_select_uint8_t_float::load(device.clone())?);
        //kernels.insert ("index_select_u8_f16".to_string(), index_select_uint8_t_float16_t::load(device.clone())?);
        kernels.insert ("index_select_u32_u8".to_string(), index_select_uint_uint8_t::load(device.clone())?);
        kernels.insert ("index_select_u32_u32".to_string(), index_select_uint_uint::load(device.clone())?);
        kernels.insert ("index_select_u32_i64".to_string(), index_select_uint_int64_t::load(device.clone())?);
        kernels.insert ("index_select_u32_bf16".to_string(), index_select_uint_bf16::load(device.clone())?);
        kernels.insert ("index_select_u32_f32".to_string(), index_select_uint_float::load(device.clone())?);
        //kernels.insert ("index_select_u32_f16".to_string(), index_select_uint_float16_t::load(device.clone())?);
        kernels.insert ("index_select_i64_u8".to_string(), index_select_int64_t_uint8_t::load(device.clone())?);
        kernels.insert ("index_select_i64_u32".to_string(), index_select_int64_t_uint::load(device.clone())?);
        kernels.insert ("index_select_i64_i64".to_string(), index_select_int64_t_int64_t::load(device.clone())?);
        kernels.insert ("index_select_i64_bf16".to_string(), index_select_int64_t_bf16::load(device.clone())?);
        kernels.insert ("index_select_i64_f32".to_string(), index_select_int64_t_float::load(device.clone())?);
        //kernels.insert ("index_select_i64_f16".to_string(), index_select_int64_t_float16_t::load(device.clone())?);

        kernels.insert ("copy2d_f32".to_string(), copy2d_float::load(device.clone())?);
        kernels.insert ("copy2d_u32".to_string(), copy2d_uint::load(device.clone())?);
        kernels.insert ("copy2d_i64".to_string(), copy2d_int64_t::load(device.clone())?);
        kernels.insert ("copy2d_bf16".to_string(), copy2d_bf16::load(device.clone())?);
        kernels.insert ("copy_strided_src_f32".to_string(), copy_strided_src_float::load(device.clone())?);
        kernels.insert ("copy_strided_src_u32".to_string(), copy_strided_src_uint::load(device.clone())?);
        kernels.insert ("copy_strided_src_i64".to_string(), copy_strided_src_int64_t::load(device.clone())?);
        kernels.insert ("copy_strided_src_bf16".to_string(), copy_strided_src_bf16::load(device.clone())?);

        kernels.insert ("eq_f32".to_string(), eq_float::load(device.clone())?);
        kernels.insert ("ne_f32".to_string(), ne_float::load(device.clone())?);
        kernels.insert ("lt_f32".to_string(), lt_float::load(device.clone())?);
        kernels.insert ("gt_f32".to_string(), gt_float::load(device.clone())?);
        kernels.insert ("le_f32".to_string(), le_float::load(device.clone())?);
        kernels.insert ("ge_f32".to_string(), ge_float::load(device.clone())?);
        kernels.insert ("eq_i64".to_string(), eq_int64_t::load(device.clone())?);
        kernels.insert ("ne_i64".to_string(), ne_int64_t::load(device.clone())?);
        kernels.insert ("lt_i64".to_string(), lt_int64_t::load(device.clone())?);
        kernels.insert ("gt_i64".to_string(), gt_int64_t::load(device.clone())?);
        kernels.insert ("le_i64".to_string(), le_int64_t::load(device.clone())?);
        kernels.insert ("ge_i64".to_string(), ge_int64_t::load(device.clone())?);

        kernels.insert("rand_uniform_f32".to_string(), rand_uniform_float::load(device.clone())?);
        kernels.insert("rand_normal_f32".to_string(), rand_normal_float::load(device.clone())?);

        kernels.insert("gemm_f32".to_string(), gemm_float::load(device.clone())?);
        kernels.insert("gemm_bf16".to_string(), gemm_bf16::load(device.clone())?);

        kernels.insert("arg_sort_f32".to_string(), sort_float::load(device.clone())?);
        kernels.insert("arg_sort_u32".to_string(), sort_uint::load(device.clone())?);
        kernels.insert("arg_sort_i64".to_string(), sort_int64_t::load(device.clone())?);
        kernels.insert("arg_srot_bf16".to_string(), sort_bf16::load(device.clone())?);
        kernels.insert("arg_sort_f16".to_string(), sort_float16_t::load(device.clone())?);
        kernels.insert("arg_sort_u8".to_string(), sort_uint8_t::load(device.clone())?);

        kernels.insert("layernorm_f32".to_string(), layernorm_float::load(device.clone())?);
        kernels.insert("layernorm_f16".to_string(), layernorm_float16_t::load(device.clone())?);
        kernels.insert("layernorm_bf16".to_string(), layernorm_bf16::load(device.clone())?);

        kernels.insert("rmsnorm_f32".to_string(), rmsnorm_float::load(device.clone())?);
        kernels.insert("rmsnorm_f16".to_string(), rmsnorm_float16_t::load(device.clone())?);
        kernels.insert("rmsnorm_bf16".to_string(), rmsnorm_bf16::load(device.clone())?);

        kernels.insert("softmax_f32".to_string(), softmax_float::load(device.clone())?);
        kernels.insert("softmax_f16".to_string(), softmax_float16_t::load(device.clone())?);
        kernels.insert("softmax_bf16".to_string(), softmax_bf16::load(device.clone())?);

        kernels.insert("rope_f32".to_string(), rope_float::load(device.clone())?);
        kernels.insert("rope_f16".to_string(), rope_float16_t::load(device.clone())?);
        kernels.insert("rope_bf16".to_string(), rope_bf16::load(device.clone())?);
        kernels.insert("rope_thd_f32".to_string(), rope_thd_float::load(device.clone())?);
        kernels.insert("rope_thd_f16".to_string(), rope_thd_float16_t::load(device.clone())?);
        kernels.insert("rope_thd_bf16".to_string(), rope_thd_bf16::load(device.clone())?);
        kernels.insert("rope_i_f32".to_string(), rope_i_float::load(device.clone())?);
        kernels.insert("rope_i_f16".to_string(), rope_i_float16_t::load(device.clone())?);
        kernels.insert("rope_i_bf16".to_string(), rope_i_bf16::load(device.clone())?);

        Ok(Self {
            kernels,
            pipelines: RwLock::new(HashMap::new()),
        })
    }

    pub fn load_pipeline(
        &self,
        device: Arc<Device>,
        shader: &str,
    ) -> Result<Arc<ComputePipeline>, VulkanKernelError> {
        // Use the cached shader module
        let shader_module = {
            self.kernels
                .get(shader)
                .ok_or_else(|| VulkanKernelError::LoadLibraryError(format!("{} shader not found", shader)))?
                .clone()
        };

        // Check if the pipeline is already cached
        let mut pipelines = self.pipelines.write()?;
        if let Some(pipeline) = pipelines.get(shader) {
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

        pipelines.insert(shader.to_string(), pipeline.clone());
        Ok(pipeline)
    }
}

macro_rules! cast_kernels {
    ($( ($mod:ident, $src:literal, $dst:literal, $need_uint_cast:literal, $src_bf16:literal, $dst_bf16:literal)),* $(,)?) => {
        $(
            mod $mod {
                // This macro invocation creates a shader module at compile time.
                vulkano_shaders::shader! {
                    ty: "compute",
                    path: "src/cast.comp",
                    define: [("SRC_TYPE", $src), ("DST_TYPE", $dst),("NEED_UINT_CAST", $need_uint_cast), ("SRC_BF16", $src_bf16), ("DST_BF16", $dst_bf16)]
                }
            }
        )*
    };
}
cast_kernels!(
    (float_to_float16_t, "float", "float16_t", "0", "0", "0"),
    (float16_t_to_float, "float16_t", "float", "0", "0", "0"),
    (uint_to_float, "uint", "float", "0", "0", "0"),
    (uint_to_uint8_t, "uint", "uint8_t", "0", "0", "0"),
    (uint8_t_to_float, "uint8_t", "float", "1", "0", "0"),
    (bf16_to_float, "uint16_t", "float", "0", "1", "0"),
    (float_to_bf16, "float", "uint16_t", "0", "0", "1"),
    (bf16_to_uint, "uint16_t", "uint", "0", "1", "0"),
    (uint_to_bf16, "uint", "uint16_t", "0", "0", "1"),
    (bf16_to_float16_t, "uint16_t", "float16_t", "0", "1", "0"),
    (uint_to_int64_t, "uint", "int64_t", "0", "0", "0"),
);

macro_rules! unary_kernels {
    ($( ($mod:ident, $op:literal, $inner_type:literal, $outer_type:literal, $bf16:literal) ),* $(,)?) => {
        $(
            mod $mod {
                vulkano_shaders::shader! {
                    ty: "compute",
                    path: "src/unary.comp",
                    define: [("OP", $op), ("INNER_TYPE", $inner_type), ("OUTER_TYPE", $outer_type), ("BF16", $bf16)]
                }
            }
        )*
    };
}

unary_kernels!(
    (neg_float, "neg_op", "float", "float", "0"),
    (abs_float, "abs_op", "float", "float", "0"),
    (sign_float, "sign_op", "float", "float", "0"),
    (gelu_float, "gelu_op", "float", "float", "0"),
    (gelu_erf_float, "gelu_erf_op", "float", "float", "0"),
    (erf_float, "erf_op", "float", "float", "0"),
    (silu_float, "silu_op", "float", "float", "0"),
    (ceil_float, "ceil_op", "float", "float", "0"),
    (floor_float, "floor_op", "float", "float", "0"),
    (round_float, "round_op", "float", "float", "0"),
    (sqr_float, "sqr_op", "float", "float", "0"),
    (sqrt_float, "sqrt_op", "float", "float", "0"),
    (sin_float, "sin_op", "float", "float", "0"),
    (cos_float, "cos_op", "float", "float", "0"),
    (tan_float, "tan_op", "float", "float", "0"),
    (sigmoid_float, "sigmoid_op", "float", "float", "0"),
    (exp_float, "exp_op", "float", "float", "0"),
    (log_float, "log_op", "float", "float", "0"),
    (recip_float, "recip_op", "float", "float", "0"),
    (neg_float16_t, "neg_op", "float", "float16_t", "0"),
    (abs_float16_t, "abs_op", "float", "float16_t", "0"),
    (sign_float16_t, "sign_op", "float", "float16_t", "0"),
    (gelu_float16_t, "gelu_op", "float", "float16_t", "0"),
    (gelu_erf_float16_t, "gelu_erf_op", "float", "float16_t", "0"),
    (erf_float16_t, "erf_op", "float", "float16_t", "0"),
    (silu_float16_t, "silu_op", "float", "float16_t", "0"),
    (ceil_float16_t, "ceil_op", "float", "float16_t", "0"),
    (floor_float16_t, "floor_op", "float", "float16_t", "0"),
    (round_float16_t, "round_op", "float", "float16_t", "0"),
    (sqr_float16_t, "sqr_op", "float", "float16_t", "0"),
    (sqrt_float16_t, "sqrt_op", "float", "float16_t", "0"),
    (sin_float16_t, "sin_op", "float", "float16_t", "0"),
    (cos_float16_t, "cos_op", "float", "float16_t", "0"),
    (tan_float16_t, "tan_op", "float", "float16_t", "0"),
    (sigmoid_float16_t, "sigmoid_op", "float", "float16_t", "0"),
    (exp_float16_t, "exp_op", "float", "float16_t", "0"),
    (log_float16_t, "log_op", "float", "float16_t", "0"),
    (recip_float16_t, "recip_op", "float", "float16_t", "0"),
    (neg_bf16, "neg_op", "float", "uint16_t", "1"),
    (abs_bf16, "abs_op", "float", "uint16_t", "1"),
    (sign_bf16, "sign_op", "float", "uint16_t", "1"),
    (gelu_bf16, "gelu_op", "float", "uint16_t", "1"),
    (gelu_erf_bf16, "gelu_erf_op", "float", "uint16_t", "1"),
    (erf_bf16, "erf_op", "float", "uint16_t", "1"),
    (silu_bf16, "silu_op", "float", "uint16_t", "1"),
    (ceil_bf16, "ceil_op", "float", "uint16_t", "1"),
    (floor_bf16, "floor_op", "float", "uint16_t", "1"),
    (round_bf16, "round_op", "float", "uint16_t", "1"),
    (sqr_bf16, "sqr_op", "float", "uint16_t", "1"),
    (sqrt_bf16, "sqrt_op", "float", "uint16_t", "1"),
    (sin_bf16, "sin_op", "float", "uint16_t", "1"),
    (cos_bf16, "cos_op", "float", "uint16_t", "1"),
    (tan_bf16, "tan_op", "float", "uint16_t", "1"),
    (sigmoid_bf16, "sigmoid_op", "float", "uint16_t", "1"),
    (exp_bf16, "exp_op", "float", "uint16_t", "1"),
    (log_bf16, "log_op", "float", "uint16_t", "1"),
    (recip_bf16, "recip_op", "float", "uint16_t", "1"),
);

// unary_kernels!(
//     (neg_double, "neg_op", "double", "double"),
//     (gelu_double, "gelu_op", "double", "double"),
//     (gelu_erf_double, "gelu_erf_op", "double", "double"),
//     (erf_double, "erf_op", "double", "double"),
//     (silu_double, "silu_op", "double", "double"),
//     (ceil_double, "ceil_op", "double", "double"),
//     (floor_double, "floor_op", "double", "double"),
//     (round_double, "round_op", "double", "double"),
//     (sign_double, "sign_op", "double", "double"),
// );

macro_rules! binary_kernels {
    ($( ($mod:ident, $op:literal, $inner_type:literal, $outer_type:literal, $bf16:literal) ),* $(,)?) => {
        $(
            mod $mod {
                vulkano_shaders::shader! {
                    ty: "compute",
                    path: "src/binary.comp",
                    define: [("OP", $op), ("INNER_TYPE", $inner_type), ("OUTER_TYPE", $outer_type), ("BF16", $bf16)]
                }
            }
        )*
    };
}

binary_kernels!(
    (add_float, "add_op", "float", "float", "0"),
    (sub_float, "sub_op", "float", "float", "0"),
    (div_float, "div_op", "float", "float", "0"),
    (mul_float, "mul_op", "float", "float", "0"),
    (min_float, "min_op", "float", "float", "0"),
    (max_float, "max_op", "float", "float", "0"),
    (add_bf16, "add_op", "float", "uint16_t", "1"),
    (sub_bf16, "sub_op", "float", "uint16_t", "1"),
    (div_bf16, "div_op", "float", "uint16_t", "1"),
    (mul_bf16, "mul_op", "float", "uint16_t", "1"),
    (min_bf16, "min_op", "float", "uint16_t", "1"),
    (max_bf16, "max_op", "float", "uint16_t", "1"),
);

macro_rules! reduce_partial_kernels {
    ($( ($mod:ident, $op:literal, $ty:literal, $to_index:literal) ),* $(,)?) => {
        $(
            mod $mod {
                vulkano_shaders::shader! {
                    ty: "compute",
                    path: "src/reduce_partial.comp",
                    define: [("OP", $op), ("TYPE", $ty), ("TO_INDEX", $to_index)]
                }
            }
        )*
    };
}

macro_rules! reduce_combine_kernels {
    ($( ($mod:ident, $op:literal, $ty:literal, $to_index:literal) ),* $(,)?) => {
        $(
            mod $mod {
                vulkano_shaders::shader! {
                    ty: "compute",
                    path: "src/reduce_combine.comp",
                    define: [("OP", $op), ("TYPE", $ty), ("TO_INDEX", $to_index)]
                }
            }
        )*
    };
}

reduce_partial_kernels!(
    (sum_partial_float, "0", "float", "0"),
    (argmax_partial_float, "1", "float", "1"),
    (max_partial_float, "1", "float", "0"),
    (argmin_partial_float, "2", "float", "1"),
    (min_partial_float, "2", "float", "0"),
    (sum_partial_uint, "0", "uint", "0"),
    (argmax_partial_uint, "1", "uint", "1"),
    (max_partial_uint, "1", "uint", "0"),
    (argmin_partial_uint, "2", "uint", "1"),
    (min_partial_uint, "2", "uint", "0"),
);

reduce_combine_kernels!(
    (sum_combine_float, "0", "float", "0"),
    (argmax_combine_float, "1", "float", "1"),
    (max_combine_float, "1", "float", "0"),
    (argmin_combine_float, "2", "float", "1"),
    (min_combine_float, "2", "float", "0"),
    (sum_combine_uint, "0", "uint", "0"),
    (argmax_combine_uint, "1", "uint", "1"),
    (max_combine_uint, "1", "uint", "0"),
    (argmin_combine_uint, "2", "uint", "1"),
    (min_combine_uint, "2", "uint", "0"),
);

macro_rules! affine_elu_kernels {
    ($( ($mod:ident, $op:literal, $inner_type:literal, $outer_type:literal, $bf16:literal) ),* $(,)?) => {
        $(
            mod $mod {
                vulkano_shaders::shader! {
                    ty: "compute",
                    path: "src/affine_elu.comp",
                    define: [("OP", $op), ("INNER_TYPE", $inner_type), ("OUTER_TYPE", $outer_type), ("BF16", $bf16)]
                }
            }
        )*
    }
}

affine_elu_kernels!(
    (affine_float, "affine_op", "float", "float", "0"),
    (affine_bf16, "affine_op", "float", "uint16_t", "1"),
    (elu_float, "elu_op", "float", "float", "0"),
);

macro_rules! gather_kernels {
    ($( ($mod:ident, $idx_ty:literal, $ty:literal, $bf16:literal) ),* $(,)?) => {
        $(
            mod $mod {
                vulkano_shaders::shader! {
                    ty: "compute",
                    path: "src/gather.comp",
                    define: [("TYPE", $ty), ("IDX_TYPE", $idx_ty), ("BF16", $bf16)]
                }
            }
        )*
    }
}

gather_kernels!(
    (gather_uint8_t_uint8_t, "uint8_t", "uint8_t", "0"),
    (gather_uint8_t_uint, "uint8_t", "uint", "0"),
    (gather_uint8_t_int64_t, "uint8_t", "int64_t", "0"),
    (gather_uint8_t_bf16, "uint8_t", "float", "1"),
    (gather_uint8_t_float, "uint8_t", "float", "0"),
//    (gather_uint8_t_float16_t, "uint8_t", "float16_t", "0"),
    (gather_uint_uint8_t, "uint", "uint8_t", "0"),
    (gather_uint_uint, "uint", "uint", "0"),
    (gather_uint_int64_t, "uint", "int64_t", "0"),
    (gather_uint_bf16, "uint", "float", "1"),
    (gather_uint_float, "uint", "float", "0"),
//    (gather_uint_float16_t, "uint", "float16_t", "0"),
    (gather_int64_t_uint8_t, "int64_t", "uint8_t", "0"),
    (gather_int64_t_uint, "int64_t", "uint", "0"),
    (gather_int64_t_int64_t, "int64_t", "int64_t", "0"),
    (gather_int64_t_bf16, "int64_t", "float", "1"),
    (gather_int64_t_float, "int64_t", "float", "0"),
//    (index_select_int64_t_float16_t, "int64_t", "float16_t", "0"),
);

macro_rules! scatter_add_kernels {
    ($( ($mod:ident, $idx_ty:literal, $ty:literal, $bf16:literal) ),* $(,)?) => {
        $(
            mod $mod {
                vulkano_shaders::shader! {
                    ty: "compute",
                    path: "src/scatter_add.comp",
                    define: [("TYPE", $ty), ("IDX_TYPE", $idx_ty), ("BF16", $bf16)]
                }
            }
        )*
    }
}

scatter_add_kernels!(
//    (scatter_add_uint8_t_uint8_t, "uint8_t", "uint8_t", "0"),
    (scatter_add_uint8_t_uint, "uint8_t", "uint", "0"),
//    (scatter_add_uint8_t_int64_t, "uint8_t", "int64_t", "0"),
    (scatter_add_uint8_t_bf16, "uint8_t", "float", "1"),
    (scatter_add_uint8_t_float, "uint8_t", "float", "0"),
//    (scatter_add_uint8_t_float16_t, "uint8_t", "float16_t", "0"),
//    (scatter_add_uint_uint8_t, "uint", "uint8_t", "0"),
    (scatter_add_uint_uint, "uint", "uint", "0"),
//    (scatter_add_uint_int64_t, "uint", "int64_t", "0"),
    (scatter_add_uint_bf16, "uint", "float", "1"),
    (scatter_add_uint_float, "uint", "float", "0"),
//    (scatter_add_uint_float16_t, "uint", "float16_t", "0"),
//    (scatter_add_int64_t_uint8_t, "int64_t", "uint8_t", "0"),
    (scatter_add_int64_t_uint, "int64_t", "uint", "0"),
//    (scatter_add_int64_t_int64_t, "int64_t", "int64_t", "0"),
    (scatter_add_int64_t_bf16, "int64_t", "float", "1"),
    (scatter_add_int64_t_float, "int64_t", "float", "0"),
//    (scatter_add_int64_t_float16_t, "int64_t", "float16_t", "0"),
);

macro_rules! index_add_kernels {
    ($( ($mod:ident, $idx_ty:literal, $ty:literal, $bf16:literal) ),* $(,)?) => {
        $(
            mod $mod {
                vulkano_shaders::shader! {
                    ty: "compute",
                    path: "src/index_add.comp",
                    define: [("TYPE", $ty), ("IDX_TYPE", $idx_ty), ("BF16", $bf16)]
                }
            }
        )*
    }
}

index_add_kernels!(
//    (index_add_uint8_t_uint8_t, "uint8_t", "uint8_t", "0"),
    (index_add_uint8_t_uint, "uint8_t", "uint", "0"),
//    (index_add_uint8_t_int64_t, "uint8_t", "int64_t", "0"),
    (index_add_uint8_t_bf16, "uint8_t", "float", "1"),
    (index_add_uint8_t_float, "uint8_t", "float", "0"),
//    (index_add_uint8_t_float16_t, "uint8_t", "float16_t", "0"),
//    (index_add_uint_uint8_t, "uint", "uint8_t", "0"),
    (index_add_uint_uint, "uint", "uint", "0"),
//    (index_add_uint_int64_t, "uint", "int64_t", "0"),
    (index_add_uint_bf16, "uint", "float", "1"),
    (index_add_uint_float, "uint", "float", "0"),
//    (index_add_uint_float16_t, "uint", "float16_t", "0"),
//    (index_add_int64_t_uint8_t, "int64_t", "uint8_t", "0"),
    (index_add_int64_t_uint, "int64_t", "uint", "0"),
//    (index_add_int64_t_int64_t, "int64_t", "int64_t", "0"),
    (index_add_int64_t_bf16, "int64_t", "float", "1"),
    (index_add_int64_t_float, "int64_t", "float", "0"),
//    (index_add_int64_t_float16_t, "int64_t", "float16_t", "0"),
);

macro_rules! index_select_kernels {
    ($( ($mod:ident, $idx_ty:literal, $ty:literal, $bf16:literal) ),* $(,)?) => {
        $(
            mod $mod {
                vulkano_shaders::shader! {
                    ty: "compute",
                    path: "src/index_select.comp",
                    define: [("TYPE", $ty), ("IDX_TYPE", $idx_ty), ("BF16", $bf16)]
                }
            }
        )*
    }
}

index_select_kernels!(
    (index_select_uint8_t_uint8_t, "uint8_t", "uint8_t", "0"),
    (index_select_uint8_t_uint, "uint8_t", "uint", "0"),
    (index_select_uint8_t_int64_t, "uint8_t", "int64_t", "0"),
    (index_select_uint8_t_bf16, "uint8_t", "float", "1"),
    (index_select_uint8_t_float, "uint8_t", "float", "0"),
//    (index_select_uint8_t_float16_t, "uint8_t", "float16_t", "0"),
    (index_select_uint_uint8_t, "uint", "uint8_t", "0"),
    (index_select_uint_uint, "uint", "uint", "0"),
    (index_select_uint_int64_t, "uint", "int64_t", "0"),
    (index_select_uint_bf16, "uint", "float", "1"),
    (index_select_uint_float, "uint", "float", "0"),
//    (index_select_uint_float16_t, "uint", "float16_t", "0"),
    (index_select_int64_t_uint8_t, "int64_t", "uint8_t", "0"),
    (index_select_int64_t_uint, "int64_t", "uint", "0"),
    (index_select_int64_t_int64_t, "int64_t", "int64_t", "0"),
    (index_select_int64_t_bf16, "int64_t", "float", "1"),
    (index_select_int64_t_float, "int64_t", "float", "0"),
//    (index_select_int64_t_float16_t, "int64_t", "float16_t", "0"),
);

macro_rules! copy2d_shaders {
    ($( ($mod:ident, $ty:literal) ),* $(,)?) => {
        $(
            mod $mod {
                vulkano_shaders::shader! {
                    ty: "compute",
                    path: "src/copy2d.comp",
                    define: [("TYPE", $ty)]
                }
            }
        )*
    }
}

copy2d_shaders!(
    (copy2d_float, "float"),
    (copy2d_uint, "uint"),
    (copy2d_int64_t, "int64_t"),
    (copy2d_bf16, "uint16_t"),
);

macro_rules! copy_strided_src_kernels {
    ($( ($mod:ident, $ty:literal) ),* $(,)?) => {
        $(
            mod $mod {
                vulkano_shaders::shader! {
                    ty: "compute",
                    path: "src/copy_strided_src.comp",
                    define: [("TYPE", $ty)]
                }
            }
        )*
    }
}


copy_strided_src_kernels!(
    (copy_strided_src_float, "float"),
    (copy_strided_src_uint, "uint"),
    (copy_strided_src_int64_t, "int64_t"),
    (copy_strided_src_bf16, "uint16_t"),
);

macro_rules! cmp_kernels {
    ($( ($mod:ident, $op:literal, $ty:literal) ),* $(,)?) => {
        $(
            mod $mod {
                vulkano_shaders::shader! {
                    ty: "compute",
                    path: "src/cmp.comp",
                    define: [("OP", $op), ("TYPE", $ty)]
                }
            }
        )*
    };
}

cmp_kernels!(
    (eq_float, "==", "float"),
    (ne_float, "!=", "float"),
    (lt_float, "<", "float"),
    (gt_float, ">", "float"),
    (le_float, "<=", "float"),
    (ge_float, ">=", "float"),
    (eq_int64_t, "==", "int64_t"),
    (ne_int64_t, "!=", "int64_t"),
    (lt_int64_t, "<", "int64_t"),
    (gt_int64_t, ">", "int64_t"),
    (le_int64_t, "<=", "int64_t"),
    (ge_int64_t, ">=", "int64_t"),
);

macro_rules! rand_kernels {
    ($( ($mod:ident, $uniform:literal) ),* $(,)?) => {
        $(
            mod $mod {
                vulkano_shaders::shader! {
                    ty: "compute",
                    path: "src/rand.comp",
                    define: [("UNIFORM", $uniform)]
                }
            }
        )*
    };
}

rand_kernels!(
    (rand_uniform_float, "1"),
    (rand_normal_float, "0"),
);

macro_rules! gemm_kernels {
    ($( ($mod:ident, $ty:literal, $bf16:literal) ),* $(,)?) => {
        $(
            mod $mod {
                vulkano_shaders::shader! {
                    ty: "compute",
                    path: "src/gemm.comp",
                    define: [("TYPE", $ty), ("BF16", $bf16)]
                }
            }
        )*
    };
}

gemm_kernels!(
    (gemm_float, "float", "0"),
    (gemm_bf16, "uint16_t", "1"),
);

macro_rules! sort_kernels {
    ($( ($mod:ident, $ty:literal, $bf16:literal) ),* $(,)?) => {
        $(
            mod $mod {
                vulkano_shaders::shader! {
                    ty: "compute",
                    path: "src/sort.comp",
                    define: [("TYPE", $ty), ("BF16", $bf16)]
                }
            }
        )*
    };
}

sort_kernels!(
    (sort_float, "float", "0"),
    (sort_uint, "uint", "0"),
    (sort_int64_t, "int64_t", "0"),
    (sort_bf16, "uint16_t", "1"),
    (sort_float16_t, "float16_t", "0"),
    (sort_uint8_t, "uint8_t", "0"),
);

macro_rules! layernorm_kernels {
    ($( ($mod:ident, $inner_type:literal, $outer_type:literal, $bf16:literal) ),* $(,)?) => {
        $(
            mod $mod {
                vulkano_shaders::shader! {
                    ty: "compute",
                    path: "src/layernorm.comp",
                    define: [("INNER_TYPE", $inner_type), ("OUTER_TYPE", $outer_type), ("BF16", $bf16)]
                }
            }
        )*
    };
}

layernorm_kernels!(
    (layernorm_float, "float", "float", "0"),
    (layernorm_float16_t, "float", "float16_t", "0"),
    (layernorm_bf16, "float", "uint16_t", "1"),
);

macro_rules! rmsnorm_kernels {
    ($( ($mod:ident, $inner_type:literal, $outer_type:literal, $bf16:literal) ),* $(,)?) => {
        $(
            mod $mod {
                vulkano_shaders::shader! {
                    ty: "compute",
                    path: "src/rmsnorm.comp",
                    define: [("INNER_TYPE", $inner_type), ("OUTER_TYPE", $outer_type), ("BF16", $bf16)]
                }
            }
        )*
    };
}

rmsnorm_kernels!(
    (rmsnorm_float, "float", "float", "0"),
    (rmsnorm_float16_t, "float", "float16_t", "0"),
    (rmsnorm_bf16, "float", "uint16_t", "1"),
);

macro_rules! softmax_kernels {
    ($( ($mod:ident, $inner_type:literal, $outer_type:literal, $bf16:literal) ),* $(,)?) => {
        $(
            mod $mod {
                vulkano_shaders::shader! {
                    ty: "compute",
                    path: "src/softmax.comp",
                    define: [("INNER_TYPE", $inner_type), ("OUTER_TYPE", $outer_type), ("BF16", $bf16)]
                }
            }
        )*
    };
}

softmax_kernels!(
    (softmax_float, "float", "float", "0"),
    (softmax_float16_t, "float", "float16_t", "0"),
    (softmax_bf16, "float", "uint16_t", "1"),
);

macro_rules! rope_kernels {
    ($( ($mod:ident, $inner_type:literal, $outer_type:literal, $bf16:literal, $thd:literal) ),* $(,)?) => {
        $(
            mod $mod {
                vulkano_shaders::shader! {
                    ty: "compute",
                    path: "src/rope.comp",
                    define: [("INNER_TYPE", $inner_type), ("OUTER_TYPE", $outer_type), ("BF16", $bf16), ("THD", $thd)]
                }
            }
        )*
    };
}

rope_kernels!(
    (rope_float, "float", "float", "0", "0"),
    (rope_float16_t, "float", "float16_t", "0", "0"),
    (rope_bf16, "float", "uint16_t", "1", "0"),
    (rope_thd_float, "float", "float", "0", "1"),
    (rope_thd_float16_t, "float", "float16_t", "0", "1"),
    (rope_thd_bf16, "float", "uint16_t", "1", "1"),
);

macro_rules! rope_i_kernels {
    ($( ($mod:ident, $inner_type:literal, $outer_type:literal, $bf16:literal) ),* $(,)?) => {
        $(
            mod $mod {
                vulkano_shaders::shader! {
                    ty: "compute",
                    path: "src/ropei.comp",
                    define: [("INNER_TYPE", $inner_type), ("OUTER_TYPE", $outer_type), ("BF16", $bf16)]
                }
            }
        )*
    };
}

rope_i_kernels!(
    (rope_i_float, "float", "float", "0"),
    (rope_i_float16_t, "float", "float16_t", "0"),
    (rope_i_bf16, "float", "uint16_t", "1"),
);

// #[allow(clippy::too_many_arguments)]
// pub fn call_copy2d(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     name: copy2d::Kernel,
//     input: &Buffer,
//     output: &Buffer,
//     d1: usize,
//     d2: usize,
//     src_s: usize,
//     dst_s: usize,
//     src_o_in_bytes: usize,
//     dst_o_in_bytes: usize,
// ) -> Result<(), MetalKernelError> {
//     let pipeline = kernels.load_pipeline(device, Source::Unary, name.0)?;
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     encoder.set_compute_pipeline_state(&pipeline);
//     set_params!(
//         encoder,
//         (
//             d1 as i64,
//             d2 as i64,
//             src_s as i64,
//             dst_s as i64,
//             (input, src_o_in_bytes),
//             (output, dst_o_in_bytes)
//         )
//     );
//
//     let grid_dims = MTLSize {
//         width: d1 as u64,
//         height: d2 as u64,
//         depth: 1,
//     };
//     let group_dims = get_block_dims(d1 as u64, d2 as u64, 1);
//     encoder.use_resource(input, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_threads(grid_dims, group_dims);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_unary_contiguous_tiled(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     kernel_name: unary::contiguous_tiled::Kernel,
//     length: usize,
//     input: BufferOffset,
//     output: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     let pipeline = kernels.load_pipeline(device, Source::Unary, kernel_name.0)?;
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     let tile_size = 2;
//     let tiles = length.div_ceil(tile_size);
//
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     set_params!(encoder, (length, &input, output));
//
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, tiles);
//     encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_unary_contiguous(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     kernel_name: unary::contiguous::Kernel,
//     length: usize,
//     input: BufferOffset,
//     output: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     let pipeline = kernels.load_pipeline(device, Source::Unary, kernel_name.0)?;
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     set_params!(encoder, (length, &input, output));
//
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);
//     encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_unary_strided(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     name: unary::strided::Kernel,
//     shape: &[usize],
//     input: BufferOffset,
//     strides: &[usize],
//     output: BufferOffset,
// ) -> Result<(), MetalKernelError> {
//     let pipeline = kernels.load_pipeline(device, Source::Unary, name.0)?;
//
//     let length: usize = shape.iter().product();
//     let num_dims: usize = shape.len();
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);
//
//     encoder.set_compute_pipeline_state(&pipeline);
//     set_params!(encoder, (length, num_dims, shape, strides, &input, &output));
//     encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output.buffer, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_binary_contiguous(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     kernel_name: binary::contiguous::Kernel,
//     length: usize,
//     left: BufferOffset,
//     right: BufferOffset,
//     output: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     let pipeline = kernels.load_pipeline(device, Source::Binary, kernel_name.0)?;
//
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     set_params!(encoder, (length, &left, &right, output));
//
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);
//
//     encoder.use_resource(left.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(right.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_binary_strided(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     name: binary::strided::Kernel,
//     shape: &[usize],
//     left_input: BufferOffset,
//     left_strides: &[usize],
//     right_input: BufferOffset,
//     right_strides: &[usize],
//     output: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     let pipeline = kernels.load_pipeline(device, Source::Binary, name.0)?;
//
//     let num_dims: usize = shape.len();
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     let width: usize = shape.iter().product();
//     let length: usize = shape.iter().product();
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, width);
//
//     encoder.set_compute_pipeline_state(&pipeline);
//     set_params!(
//         encoder,
//         (
//             length,
//             num_dims,
//             shape,
//             left_strides,
//             right_strides,
//             &left_input,
//             &right_input,
//             output
//         )
//     );
//     encoder.use_resource(left_input.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(right_input.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_cast_contiguous(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     kernel_name: &'static str,
//     length: usize,
//     input: BufferOffset,
//     output: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     let pipeline = kernels.load_pipeline(device, Source::Cast, kernel_name)?;
//
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     set_params!(encoder, (length, &input, output));
//
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);
//     encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_cast_strided(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     kernel_name: &'static str,
//     shape: &[usize],
//     input: BufferOffset,
//     input_strides: &[usize],
//     output: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     let pipeline = kernels.load_pipeline(device, Source::Cast, kernel_name)?;
//
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     let length: usize = shape.iter().product();
//
//     set_params!(
//         encoder,
//         (length, shape.len(), shape, input_strides, &input, output)
//     );
//
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);
//
//     encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_reduce_contiguous(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     kernel_name: &'static str,
//     length: usize,
//     out_length: usize,
//     input: BufferOffset,
//     output: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
//     let elements_to_sum = length / out_length;
//
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     set_params!(encoder, (length, elements_to_sum, &input, output));
//
//     let thread_group_count = MTLSize {
//         width: out_length as u64,
//         height: 1,
//         depth: 1,
//     };
//
//     let width = std::cmp::min(
//         pipeline.max_total_threads_per_threadgroup(),
//         (elements_to_sum as u64).div_ceil(2),
//     )
//     .next_power_of_two();
//
//     let thread_group_size = MTLSize {
//         width,
//         height: 1,
//         depth: 1,
//     };
//
//     encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_reduce_strided(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     kernel_name: &'static str,
//     shape: &[usize],
//     strides: &[usize],
//     out_length: usize,
//     input: BufferOffset,
//     output: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     let length: usize = shape.iter().product();
//     let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
//     let elements_to_sum = length / out_length;
//
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     set_params!(
//         encoder,
//         (shape.len(), shape, strides, elements_to_sum, &input, output)
//     );
//
//     let thread_group_count = MTLSize {
//         width: out_length as u64,
//         height: 1,
//         depth: 1,
//     };
//
//     let width = std::cmp::min(
//         pipeline.max_total_threads_per_threadgroup(),
//         elements_to_sum as u64,
//     )
//     .next_power_of_two();
//
//     let thread_group_size = MTLSize {
//         width,
//         height: 1,
//         depth: 1,
//     };
//
//     encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_last_softmax(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     kernel_name: &'static str,
//     length: usize,
//     elements_to_sum: usize,
//     input: &Buffer,
//     input_offset: usize,
//     output: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     set_params!(
//         encoder,
//         (length, elements_to_sum, (input, input_offset), output)
//     );
//
//     let out_length = length / elements_to_sum;
//
//     let thread_group_count = MTLSize {
//         width: out_length as u64,
//         height: 1,
//         depth: 1,
//     };
//
//     let width = std::cmp::min(
//         pipeline.max_total_threads_per_threadgroup(),
//         elements_to_sum as u64,
//     )
//     .next_power_of_two();
//
//     let thread_group_size = MTLSize {
//         width,
//         height: 1,
//         depth: 1,
//     };
//
//     encoder.use_resource(input, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_rms_norm(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     kernel_name: &'static str,
//     length: usize,
//     elements_to_sum: usize,
//     eps: f32,
//     input: &Buffer,
//     input_offset: usize,
//     alpha: &Buffer,
//     alpha_offset: usize,
//     output: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     set_params!(
//         encoder,
//         (
//             length,
//             elements_to_sum,
//             (input, input_offset),
//             output,
//             (alpha, alpha_offset),
//             eps
//         )
//     );
//
//     let out_length = length / elements_to_sum;
//
//     let thread_group_count = MTLSize {
//         width: out_length as u64,
//         height: 1,
//         depth: 1,
//     };
//
//     let width = std::cmp::min(
//         pipeline.max_total_threads_per_threadgroup(),
//         elements_to_sum as u64,
//     )
//     .next_power_of_two();
//
//     let thread_group_size = MTLSize {
//         width,
//         height: 1,
//         depth: 1,
//     };
//
//     encoder.use_resource(input, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.set_threadgroup_memory_length(0, (width * 4).max(16) as u64);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_layer_norm(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     kernel_name: &'static str,
//     length: usize,
//     elements_to_sum: usize,
//     eps: f32,
//     input: &Buffer,
//     input_offset: usize,
//     alpha: &Buffer,
//     alpha_offset: usize,
//     beta: &Buffer,
//     beta_offset: usize,
//     output: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     set_params!(
//         encoder,
//         (
//             length,
//             elements_to_sum,
//             (input, input_offset),
//             output,
//             (alpha, alpha_offset),
//             (beta, beta_offset),
//             eps
//         )
//     );
//
//     let out_length = length / elements_to_sum;
//
//     let thread_group_count = MTLSize {
//         width: out_length as u64,
//         height: 1,
//         depth: 1,
//     };
//
//     let width = std::cmp::min(
//         pipeline.max_total_threads_per_threadgroup(),
//         elements_to_sum as u64,
//     )
//     .next_power_of_two();
//
//     let thread_group_size = MTLSize {
//         width,
//         height: 1,
//         depth: 1,
//     };
//
//     encoder.use_resource(input, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.set_threadgroup_memory_length(0, (width * 8).max(32) as u64);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_rope_i(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     kernel_name: &'static str,
//     bh: usize,
//     td: usize,
//     src: &Buffer,
//     src_offset: usize,
//     cos: &Buffer,
//     cos_offset: usize,
//     sin: &Buffer,
//     sin_offset: usize,
//     output: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     set_params!(
//         encoder,
//         (
//             bh,
//             td,
//             (src, src_offset),
//             (cos, cos_offset),
//             (sin, sin_offset),
//             output
//         )
//     );
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, (bh * td) / 2);
//     encoder.use_resource(src, metal::MTLResourceUsage::Read);
//     encoder.use_resource(cos, metal::MTLResourceUsage::Read);
//     encoder.use_resource(sin, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_rope_thd(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     kernel_name: &'static str,
//     b: usize,
//     t: usize,
//     h: usize,
//     d: usize,
//     src: &Buffer,
//     src_offset: usize,
//     cos: &Buffer,
//     cos_offset: usize,
//     sin: &Buffer,
//     sin_offset: usize,
//     output: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     set_params!(
//         encoder,
//         (
//             b,
//             t,
//             h,
//             d,
//             (src, src_offset),
//             (cos, cos_offset),
//             (sin, sin_offset),
//             output
//         )
//     );
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, (b * t * h * d) / 2);
//     encoder.use_resource(src, metal::MTLResourceUsage::Read);
//     encoder.use_resource(cos, metal::MTLResourceUsage::Read);
//     encoder.use_resource(sin, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_rope(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     kernel_name: &'static str,
//     bh: usize,
//     td: usize,
//     d: usize,
//     src: &Buffer,
//     src_offset: usize,
//     cos: &Buffer,
//     cos_offset: usize,
//     sin: &Buffer,
//     sin_offset: usize,
//     output: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     set_params!(
//         encoder,
//         (
//             bh,
//             td,
//             d,
//             (src, src_offset),
//             (cos, cos_offset),
//             (sin, sin_offset),
//             output
//         )
//     );
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, (bh * td) / 2);
//     encoder.use_resource(src, metal::MTLResourceUsage::Read);
//     encoder.use_resource(cos, metal::MTLResourceUsage::Read);
//     encoder.use_resource(sin, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_affine(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     name: &'static str,
//     size: usize,
//     input: BufferOffset,
//     output: &Buffer,
//     mul: f32,
//     add: f32,
// ) -> Result<(), MetalKernelError> {
//     let pipeline = kernels.load_pipeline(device, Source::Affine, name)?;
//
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     set_params!(encoder, (size, mul, add, &input, output));
//
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);
//     encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_affine_strided(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     name: &'static str,
//     shape: &[usize],
//     input: BufferOffset,
//     input_stride: &[usize],
//     output: &Buffer,
//     mul: f32,
//     add: f32,
// ) -> Result<(), MetalKernelError> {
//     let pipeline = kernels.load_pipeline(device, Source::Affine, name)?;
//     let size: usize = shape.iter().product();
//
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     set_params!(
//         encoder,
//         (
//             size,
//             shape.len(),
//             shape,
//             input_stride,
//             mul,
//             add,
//             &input,
//             output
//         )
//     );
//
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);
//     encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_powf(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     name: &'static str,
//     size: usize,
//     input: BufferOffset,
//     output: &Buffer,
//     mul: f32,
// ) -> Result<(), MetalKernelError> {
//     let pipeline = kernels.load_pipeline(device, Source::Affine, name)?;
//
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     set_params!(encoder, (size, mul, &input, output));
//
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);
//     encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_powf_strided(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     name: &'static str,
//     shape: &[usize],
//     input: BufferOffset,
//     input_stride: &[usize],
//     output: &Buffer,
//     mul: f32,
// ) -> Result<(), MetalKernelError> {
//     let pipeline = kernels.load_pipeline(device, Source::Affine, name)?;
//     let size: usize = shape.iter().product();
//
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     set_params!(
//         encoder,
//         (size, shape.len(), shape, input_stride, mul, &input, output)
//     );
//
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);
//     encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_elu(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     name: &'static str,
//     size: usize,
//     input: BufferOffset,
//     output: &Buffer,
//     mul: f32,
// ) -> Result<(), MetalKernelError> {
//     let pipeline = kernels.load_pipeline(device, Source::Affine, name)?;
//
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     set_params!(encoder, (size, mul, &input, output));
//
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);
//     encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_elu_strided(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     name: &'static str,
//     shape: &[usize],
//     input: BufferOffset,
//     input_stride: &[usize],
//     output: &Buffer,
//     mul: f32,
// ) -> Result<(), MetalKernelError> {
//     let pipeline = kernels.load_pipeline(device, Source::Affine, name)?;
//     let size: usize = shape.iter().product();
//
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     set_params!(
//         encoder,
//         (size, shape.len(), shape, input_stride, mul, &input, output)
//     );
//
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);
//     encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_where_cond_strided(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     name: &'static str,
//     shape: &[usize],
//     cond: BufferOffset,
//     cond_stride: &[usize],
//     left: BufferOffset,
//     left_stride: &[usize],
//     right: BufferOffset,
//     right_stride: &[usize],
//     output: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     let pipeline = kernels.load_pipeline(device, Source::Ternary, name)?;
//
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     let size: usize = shape.iter().product();
//     let rank = shape.len();
//
//     set_params!(
//         encoder,
//         (
//             size,
//             rank,
//             shape,
//             cond_stride,
//             left_stride,
//             right_stride,
//             &cond,
//             &left,
//             &right,
//             output
//         )
//     );
//
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);
//
//     encoder.use_resource(cond.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(left.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(right.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_index_select(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     name: &'static str,
//     shape: &[usize],
//     ids_size: usize,
//     dim: usize,
//     contiguous: bool,
//     src_dims: &[usize],
//     src_strides: &[usize],
//     input: BufferOffset,
//     ids: BufferOffset,
//     output: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     let left_size: usize = shape[..dim].iter().product();
//     let right_size: usize = shape[dim + 1..].iter().product();
//     let src_dim_size = shape[dim];
//     let dst_el = ids_size * left_size * right_size;
//
//     let pipeline = kernels.load_pipeline(device, Source::Indexing, name)?;
//
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     set_params!(
//         encoder,
//         (
//             dst_el,
//             left_size,
//             src_dim_size,
//             right_size,
//             ids_size,
//             contiguous,
//             src_dims,
//             src_strides,
//             &input,
//             &ids,
//             output
//         )
//     );
//
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
//
//     encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(ids.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_gather(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     name: &'static str,
//     shape: &[usize],
//     ids_size: usize,
//     dim: usize,
//     input: BufferOffset,
//     ids: BufferOffset,
//     output: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     let left_size: usize = shape[..dim].iter().product();
//     let right_size: usize = shape[dim + 1..].iter().product();
//     let src_dim_size = shape[dim];
//     let dst_el = ids_size * left_size * right_size;
//
//     let pipeline = kernels.load_pipeline(device, Source::Indexing, name)?;
//
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     set_params!(
//         encoder,
//         (
//             dst_el,
//             left_size,
//             src_dim_size,
//             right_size,
//             ids_size,
//             &input,
//             &ids,
//             output
//         )
//     );
//
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
//
//     encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(ids.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_scatter_add(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     name: &'static str,
//     src_shape: &[usize],
//     dst_shape: &[usize],
//     dim: usize,
//     input: BufferOffset,
//     ids: BufferOffset,
//     output: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     let left_size: usize = src_shape[..dim].iter().product();
//     let right_size: usize = src_shape[dim + 1..].iter().product();
//     let src_dim_size = src_shape[dim];
//     let dst_el = left_size * right_size;
//     let dst_dim_size = dst_shape[dim];
//
//     let pipeline = kernels.load_pipeline(device, Source::Indexing, name)?;
//
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     set_params!(
//         encoder,
//         (
//             dst_el,
//             left_size,
//             src_dim_size,
//             right_size,
//             dst_dim_size,
//             &input,
//             &ids,
//             output
//         )
//     );
//
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
//
//     encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(ids.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_index_add(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     name: &'static str,
//     src_shape: &[usize],
//     dst_shape: &[usize],
//     ids_shape: &[usize],
//     dim: usize,
//     input: BufferOffset,
//     ids: BufferOffset,
//     output: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     let left_size: usize = src_shape[..dim].iter().product();
//     let right_size: usize = src_shape[dim + 1..].iter().product();
//     let src_dim_size = src_shape[dim];
//     let dst_el = left_size * right_size;
//     let dst_dim_size = dst_shape[dim];
//     let ids_dim_size = ids_shape[0];
//
//     let pipeline = kernels.load_pipeline(device, Source::Indexing, name)?;
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     set_params!(
//         encoder,
//         (
//             dst_el,
//             left_size,
//             src_dim_size,
//             right_size,
//             dst_dim_size,
//             ids_dim_size,
//             &input,
//             &ids,
//             output
//         )
//     );
//
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
//
//     encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(ids.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[derive(Debug, PartialEq)]
// pub enum Value {
//     USize(usize),
//     Bool(bool),
//     F32(f32),
//     U16(u16),
// }
//
// impl std::hash::Hash for Value {
//     fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
//         match self {
//             Value::F32(v) => v.to_bits().hash(state),
//             Value::USize(v) => v.hash(state),
//             Value::U16(v) => v.hash(state),
//             Value::Bool(v) => v.hash(state),
//         }
//     }
// }
//
// impl Value {
//     fn data_type(&self) -> MTLDataType {
//         match self {
//             Value::USize(_) => MTLDataType::UInt,
//             Value::F32(_) => MTLDataType::Float,
//             Value::U16(_) => MTLDataType::UShort,
//             Value::Bool(_) => MTLDataType::Bool,
//         }
//     }
// }
//
// /// Not true, good enough for our purposes.
// impl Eq for Value {}
//
// #[derive(Debug, Eq, PartialEq, Hash)]
// struct ConstantValues(Vec<(usize, Value)>);
//
// impl ConstantValues {
//     pub fn new(values: Vec<(usize, Value)>) -> Self {
//         Self(values)
//     }
//
//     fn function_constant_values(&self) -> FunctionConstantValues {
//         let f = FunctionConstantValues::new();
//         for (index, value) in &self.0 {
//             let ty = value.data_type();
//             match value {
//                 Value::USize(v) => {
//                     f.set_constant_value_at_index(
//                         v as *const usize as *const c_void,
//                         ty,
//                         *index as u64,
//                     );
//                 }
//                 Value::F32(v) => {
//                     f.set_constant_value_at_index(
//                         v as *const f32 as *const c_void,
//                         ty,
//                         *index as u64,
//                     );
//                 }
//                 Value::U16(v) => {
//                     f.set_constant_value_at_index(
//                         v as *const u16 as *const c_void,
//                         ty,
//                         *index as u64,
//                     );
//                 }
//                 Value::Bool(v) => {
//                     f.set_constant_value_at_index(
//                         v as *const bool as *const c_void,
//                         ty,
//                         *index as u64,
//                     );
//                 }
//             }
//         }
//         f
//     }
// }
//
// #[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
// pub enum SdpaDType {
//     BF16,
//     F16,
//     F32,
// }
//
// /// SDPA full is supported when:
// /// - q head dim == 64, 128
// /// - no mask
// /// - q heads == kv heads
// /// - final type != bf16 (TODO maybe just template this kernel too?)
// /// - q,k,v are contiguous
// #[allow(clippy::too_many_arguments)]
// pub fn call_sdpa_full(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     q_offset: usize,
//     q_shape: &[usize],
//     q_buffer: &Buffer,
//     k_offset: usize,
//     k_buffer: &Buffer,
//     v_offset: usize,
//     v_buffer: &Buffer,
//     output: &Buffer,
//     alpha: f32,
//     softcapping: f32,
//     itype: SdpaDType,
// ) -> Result<(), MetalKernelError> {
//     #[derive(Debug)]
//     #[repr(C)]
//     struct MLXFastAttentionParams {
//         m: i32,
//         n: i32,
//         k: i32,
//
//         ldq: i32, // ldq == ldo
//         ldk: i32,
//         ldv: i32,
//         lds: i32,
//         ldo: i32,
//
//         tiles_n: i32,
//         tiles_m: i32,
//
//         batch_stride_q: i32,
//         batch_stride_k: i32,
//         batch_stride_v: i32,
//         batch_stride_o: i32,
//
//         swizzle_log: i32,
//         gemm_n_iterations_aligned: i32,
//         gemm_k_iterations_aligned: i32,
//         gemm_sv_m_block_iterations: i32,
//
//         batch_ndim: i32,
//         alpha: f32,
//         softcapping: f32,
//     }
//
//     let bk = q_shape.last().unwrap();
//
//     const BN: usize = 16;
//     const BM: usize = 16;
//     const WM: usize = 2;
//     const WN: usize = 2;
//
//     let name = match (bk, itype) {
//         (32, SdpaDType::F16) => "steel_gemm_attention_bm_16_bn_16_bk_32_itype_half",
//         (64, SdpaDType::F16) => "steel_gemm_attention_bm_16_bn_16_bk_64_itype_half",
//         (96, SdpaDType::F16) => "steel_gemm_attention_bm_16_bn_16_bk_96_itype_half",
//         (128, SdpaDType::F16) => "steel_gemm_attention_bm_16_bn_16_bk_128_itype_half",
//         (256, SdpaDType::F16) => "steel_gemm_attention_bm_16_bn_16_bk_256_itype_half",
//         (32, SdpaDType::F32) => "steel_gemm_attention_bm_16_bn_16_bk_32_itype_float",
//         (64, SdpaDType::F32) => "steel_gemm_attention_bm_16_bn_16_bk_64_itype_float",
//         (96, SdpaDType::F32) => "steel_gemm_attention_bm_16_bn_16_bk_96_itype_float",
//         (128, SdpaDType::F32) => "steel_gemm_attention_bm_16_bn_16_bk_128_itype_float",
//         (256, SdpaDType::F32) => "steel_gemm_attention_bm_16_bn_16_bk_256_itype_float",
//         (other, SdpaDType::F16 | SdpaDType::F32) => {
//             return Err(MetalKernelError::SdpaHeadSizeMismatch {
//                 variation: "full",
//                 got: *other,
//                 expected: vec![32, 64, 96, 128, 256],
//             })
//         }
//         (_, SdpaDType::BF16) => {
//             return Err(MetalKernelError::SdpaHeadDTypeMismatch {
//                 variation: "full",
//                 got: SdpaDType::BF16,
//             })
//         }
//     };
//
//     let pipeline = kernels.load_pipeline(device, Source::Sdpa, name)?;
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     // q = (bs, qhead, seq, hidden)
//     // k/v = (bs, kv_head, seq, hidden)
//
//     let qseq = q_shape[q_shape.len() - 2];
//
//     let m = q_shape[q_shape.len() - 2];
//     let n = m;
//     let k = q_shape[q_shape.len() - 1];
//     let bs_out = q_shape[0] * q_shape[1];
//
//     let batch_shape = [q_shape[0] * q_shape[1]];
//     let dk = q_shape[q_shape.len() - 1];
//     let ldq = dk;
//     let ldk = dk;
//     let ldv = dk;
//     let lds = BN;
//     let ldo = dk;
//
//     let tn = 1;
//     let tm = m.div_ceil(BM);
//
//     let b_stride_q = dk * qseq;
//     let b_stride_k = dk * qseq;
//     let b_stride_v = dk * qseq;
//     let b_stride_o = dk * qseq;
//     let swizzle_log = 0;
//     let gemm_n_iterations_aligned = n.div_ceil(BN);
//     let gemm_k_iterations_aligned = k.div_ceil(*bk);
//     let gemm_sv_m_block_iterations = m.div_ceil(BM);
//     let batch_ndim = batch_shape.len();
//
//     let alpha = if softcapping != 1. {
//         alpha / softcapping
//     } else {
//         alpha
//     };
//
//     let params = MLXFastAttentionParams {
//         m: m as i32,
//         n: n as i32,
//         k: k as i32,
//         ldq: ldq as i32,
//         ldk: ldk as i32,
//         ldv: ldv as i32,
//         lds: lds as i32,
//         ldo: ldo as i32,
//         tiles_n: tn,
//         tiles_m: tm as i32,
//         batch_stride_q: b_stride_q as i32,
//         batch_stride_k: b_stride_k as i32,
//         batch_stride_v: b_stride_v as i32,
//         batch_stride_o: b_stride_o as i32,
//         swizzle_log,
//         gemm_n_iterations_aligned: gemm_n_iterations_aligned as i32,
//         gemm_k_iterations_aligned: gemm_k_iterations_aligned as i32,
//         gemm_sv_m_block_iterations: gemm_sv_m_block_iterations as i32,
//         batch_ndim: batch_ndim as i32,
//         alpha,
//         softcapping,
//     };
//     let batch_strides = [b_stride_q, b_stride_k, b_stride_v, b_stride_o];
//
//     impl EncoderParam for MLXFastAttentionParams {
//         fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self) {
//             encoder.set_bytes(
//                 position,
//                 core::mem::size_of::<MLXFastAttentionParams>() as u64,
//                 &data as *const MLXFastAttentionParams as *const c_void,
//             );
//         }
//     }
//
//     set_params!(
//         encoder,
//         (
//             (q_buffer, q_offset),
//             (k_buffer, k_offset),
//             (v_buffer, v_offset),
//             output,
//             params,
//             &batch_shape[..],
//             &batch_strides[..]
//         )
//     );
//
//     let grid_dims = MTLSize {
//         width: 1,
//         height: tm as u64,
//         depth: bs_out as u64,
//     };
//     let group_dims = MTLSize {
//         width: 32,
//         height: WM as u64,
//         depth: WN as u64,
//     };
//     encoder.use_resource(q_buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(k_buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(v_buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(grid_dims, group_dims);
//     Ok(())
// }
//
// /// SDPA full is supported when:
// /// - q head dim == 64, 96, 128
// /// - no mask
// /// - q,k,v are contiguous
// #[allow(clippy::too_many_arguments)]
// pub fn call_sdpa_vector(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     q_offset: usize,
//     q_shape: &[usize],
//     q_buffer: &Buffer,
//     k_offset: usize,
//     k_shape: &[usize],
//     k_stride: &[usize],
//     k_buffer: &Buffer,
//     v_offset: usize,
//     v_stride: &[usize],
//     v_buffer: &Buffer,
//     output: &Buffer,
//     alpha: f32,
//     softcapping: f32,
//     itype: SdpaDType,
// ) -> Result<(), MetalKernelError> {
//     let bk = q_shape.last().unwrap();
//
//     let gqa_factor = (q_shape[1] / k_shape[1]) as i32;
//     let n = k_shape[2] as i32;
//     let b = (q_shape[0] * q_shape[1]) as i32;
//     let kstride = k_stride[1];
//     let vstride = v_stride[1];
//
//     let name = match (bk, itype) {
//         (32, SdpaDType::F16) => "sdpa_vector_float16_t_32",
//         (64, SdpaDType::F16) => "sdpa_vector_float16_t_64",
//         (96, SdpaDType::F16) => "sdpa_vector_float16_t_96",
//         (128, SdpaDType::F16) => "sdpa_vector_float16_t_128",
//         (256, SdpaDType::F16) => "sdpa_vector_float16_t_256",
//         (32, SdpaDType::BF16) => "sdpa_vector_bfloat16_t_32",
//         (64, SdpaDType::BF16) => "sdpa_vector_bfloat16_t_64",
//         (96, SdpaDType::BF16) => "sdpa_vector_bfloat16_t_96",
//         (128, SdpaDType::BF16) => "sdpa_vector_bfloat16_t_128",
//         (256, SdpaDType::BF16) => "sdpa_vector_bfloat16_t_256",
//         (32, SdpaDType::F32) => "sdpa_vector_float_32",
//         (64, SdpaDType::F32) => "sdpa_vector_float_64",
//         (96, SdpaDType::F32) => "sdpa_vector_float_96",
//         (128, SdpaDType::F32) => "sdpa_vector_float_128",
//         (256, SdpaDType::F32) => "sdpa_vector_float_256",
//         (other, _) => {
//             return Err(MetalKernelError::SdpaHeadSizeMismatch {
//                 variation: "vector",
//                 got: *other,
//                 expected: vec![32, 64, 96, 128, 256],
//             })
//         }
//     };
//
//     let alpha = if softcapping != 1. {
//         alpha / softcapping
//     } else {
//         alpha
//     };
//
//     let constants = Some(ConstantValues::new(vec![(
//         20,
//         Value::Bool(/* sdpa_vector_has_mask */ false),
//     )]));
//
//     let pipeline = kernels.load_pipeline_with_constants(device, Source::Sdpa, name, constants)?;
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     // q = (bs, qhead, seq, hidden)
//     // k/v = (bs, kv_head, kv_seq, hidden)
//
//     set_params!(
//         encoder,
//         (
//             (q_buffer, q_offset),
//             (k_buffer, k_offset),
//             (v_buffer, v_offset),
//             output,
//             gqa_factor,
//             n,
//             kstride,
//             vstride,
//             alpha,
//             softcapping
//         )
//     );
//
//     let grid_dims = MTLSize {
//         width: 1,
//         height: b as u64,
//         depth: 1_u64,
//     };
//     let group_dims = MTLSize {
//         width: 1024,
//         height: 1,
//         depth: 1,
//     };
//     encoder.use_resource(q_buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(k_buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(v_buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(grid_dims, group_dims);
//     Ok(())
// }
//
// pub const SDPA_2PASS_BLOCKS: usize = 32;
//
// /// SDPA vector 2pass is supported when:
// /// - q head dim == 64, 96, 128
// /// - no mask
// /// - q,k,v are contiguous
// #[allow(clippy::too_many_arguments)]
// pub fn call_sdpa_vector_2pass(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     q_offset: usize,
//     q_shape: &[usize],
//     q_buffer: &Buffer,
//     k_offset: usize,
//     k_shape: &[usize],
//     k_stride: &[usize],
//     k_buffer: &Buffer,
//     v_offset: usize,
//     v_stride: &[usize],
//     v_buffer: &Buffer,
//     output: &Buffer,
//     intermediate: &Buffer,
//     sums: &Buffer,
//     maxs: &Buffer,
//     alpha: f32,
//     softcapping: f32,
//     itype: SdpaDType,
// ) -> Result<(), MetalKernelError> {
//     let bk = q_shape.last().unwrap();
//
//     // First pass
//     {
//         let name_pass1 = match (bk, itype) {
//             (32, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_32",
//             (64, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_64",
//             (96, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_96",
//             (128, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_128",
//             (256, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_256",
//             (32, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_32",
//             (64, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_64",
//             (96, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_96",
//             (128, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_128",
//             (256, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_256",
//             (32, SdpaDType::F32) => "sdpa_vector_2pass_1_float_32",
//             (64, SdpaDType::F32) => "sdpa_vector_2pass_1_float_64",
//             (96, SdpaDType::F32) => "sdpa_vector_2pass_1_float_96",
//             (128, SdpaDType::F32) => "sdpa_vector_2pass_1_float_128",
//             (256, SdpaDType::F32) => "sdpa_vector_2pass_1_float_256",
//             (other, _) => {
//                 return Err(MetalKernelError::SdpaHeadSizeMismatch {
//                     variation: "vector_2pass_1",
//                     got: *other,
//                     expected: vec![32, 64, 96, 128, 256],
//                 })
//             }
//         };
//
//         let gqa_factor = (q_shape[1] / k_shape[1]) as i32;
//         let n = k_shape[2] as i32;
//         let b = (q_shape[0] * q_shape[1]) as i32;
//         let kstride = k_stride[1];
//         let vstride = v_stride[1];
//
//         let alpha = if softcapping != 1. {
//             alpha / softcapping
//         } else {
//             alpha
//         };
//
//         let constants = Some(ConstantValues::new(vec![(
//             20,
//             Value::Bool(/* sdpa_vector_has_mask */ false),
//         )]));
//
//         let pipeline =
//             kernels.load_pipeline_with_constants(device, Source::Sdpa, name_pass1, constants)?;
//         let encoder = ep.encoder();
//         let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//         encoder.set_compute_pipeline_state(&pipeline);
//
//         // q = (bs, qhead, seq, hidden)
//         // k/v = (bs, kv_head, kv_seq, hidden)
//
//         set_params!(
//             encoder,
//             (
//                 (q_buffer, q_offset),
//                 (k_buffer, k_offset),
//                 (v_buffer, v_offset),
//                 intermediate,
//                 sums,
//                 maxs,
//                 gqa_factor,
//                 n,
//                 kstride,
//                 vstride,
//                 alpha,
//                 softcapping
//             )
//         );
//
//         let grid_dims = MTLSize {
//             width: 1,
//             height: b as u64,
//             depth: SDPA_2PASS_BLOCKS as u64,
//         };
//         let group_dims = MTLSize {
//             width: 8 * 32,
//             height: 1,
//             depth: 1,
//         };
//         encoder.use_resource(q_buffer, metal::MTLResourceUsage::Read);
//         encoder.use_resource(k_buffer, metal::MTLResourceUsage::Read);
//         encoder.use_resource(v_buffer, metal::MTLResourceUsage::Read);
//         encoder.use_resource(intermediate, metal::MTLResourceUsage::Write);
//         encoder.use_resource(sums, metal::MTLResourceUsage::Write);
//         encoder.use_resource(maxs, metal::MTLResourceUsage::Write);
//
//         encoder.dispatch_thread_groups(grid_dims, group_dims);
//     }
//
//     // Final pass
//     {
//         let name_pass2 = match (bk, itype) {
//             (32, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_32",
//             (64, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_64",
//             (96, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_96",
//             (128, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_128",
//             (256, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_256",
//             (32, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_32",
//             (64, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_64",
//             (96, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_96",
//             (128, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_128",
//             (256, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_256",
//             (32, SdpaDType::F32) => "sdpa_vector_2pass_2_float_32",
//             (64, SdpaDType::F32) => "sdpa_vector_2pass_2_float_64",
//             (96, SdpaDType::F32) => "sdpa_vector_2pass_2_float_96",
//             (128, SdpaDType::F32) => "sdpa_vector_2pass_2_float_128",
//             (256, SdpaDType::F32) => "sdpa_vector_2pass_2_float_256",
//             (other, _) => {
//                 return Err(MetalKernelError::SdpaHeadSizeMismatch {
//                     variation: "vector_2pass_2",
//                     got: *other,
//                     expected: vec![32, 64, 96, 128, 256],
//                 })
//             }
//         };
//
//         let b = (q_shape[0] * q_shape[1]) as i32;
//
//         let pipeline = kernels.load_pipeline(device, Source::Sdpa, name_pass2)?;
//         let encoder = ep.encoder();
//         let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//         encoder.set_compute_pipeline_state(&pipeline);
//
//         // q = (bs, qhead, seq, hidden)
//         // k/v = (bs, kv_head, kv_seq, hidden)
//
//         set_params!(encoder, (intermediate, sums, maxs, output));
//
//         let grid_dims = MTLSize {
//             width: 1,
//             height: b as u64,
//             depth: 1,
//         };
//         let group_dims = MTLSize {
//             width: 1024,
//             height: 1,
//             depth: 1,
//         };
//         encoder.use_resource(intermediate, metal::MTLResourceUsage::Write);
//         encoder.use_resource(sums, metal::MTLResourceUsage::Write);
//         encoder.use_resource(maxs, metal::MTLResourceUsage::Write);
//         encoder.use_resource(output, metal::MTLResourceUsage::Write);
//
//         encoder.dispatch_thread_groups(grid_dims, group_dims);
//     }
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_im2col1d_strided(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     name: &'static str,
//     shape: &[usize],
//     strides: &[usize],
//     (k_size, stride, padding, dilation): (usize, usize, usize, usize),
//     input: BufferOffset,
//     output: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
//     let l_out = (shape[2] + 2 * padding - dilation * (k_size - 1) - 1) / stride + 1;
//     let dst_el = shape[0] * l_out * shape[1] * k_size;
//
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
//     encoder.set_compute_pipeline_state(&pipeline);
//     set_params!(
//         encoder,
//         (dst_el, l_out, k_size, stride, padding, dilation, shape, strides, &input, output)
//     );
//     encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_col2im1d(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     name: &'static str,
//     shape: &[usize],
//     k_size: usize,
//     stride: usize,
//     input: BufferOffset,
//     output: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
//     let l_in = shape[1];
//     let c_out = shape[2];
//     let l_out = (l_in - 1) * stride + k_size;
//     let dst_el = shape[0] * c_out * l_out;
//
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
//     encoder.set_compute_pipeline_state(&pipeline);
//     set_params!(
//         encoder,
//         (dst_el, l_out, l_in, c_out, k_size, stride, &input, output)
//     );
//     encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_im2col_strided(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     name: &'static str,
//     shape: &[usize],
//     strides: &[usize],
//     (h_k, w_k, stride, padding, dilation): (usize, usize, usize, usize, usize),
//     input: BufferOffset,
//     output: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
//
//     let h = shape[2];
//     let w = shape[3];
//     let h_out = (h + 2 * padding - dilation * (h_k - 1) - 1) / stride + 1;
//     let w_out = (w + 2 * padding - dilation * (w_k - 1) - 1) / stride + 1;
//
//     let dst_el = shape[0] * h_out * w_out * shape[1] * h_k * w_k;
//
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
//     encoder.set_compute_pipeline_state(&pipeline);
//     set_params!(
//         encoder,
//         (
//             dst_el, h_out, w_out, h_k, w_k, stride, padding, dilation, shape, strides, &input,
//             output
//         )
//     );
//     encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_upsample_nearest_2d(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     name: &'static str,
//     shape: &[usize],
//     strides: &[usize],
//     out_w: usize,
//     out_h: usize,
//     input: BufferOffset,
//     output: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
//     let dst_el = out_w * out_h * shape[0] * shape[1];
//     let scale_w = shape[2] as f32 / out_w as f32;
//     let scale_h = shape[3] as f32 / out_h as f32;
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     encoder.set_compute_pipeline_state(&pipeline);
//     set_params!(
//         encoder,
//         (out_w, out_h, scale_w, scale_h, shape, strides, &input, output)
//     );
//     encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_random_uniform(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     name: &'static str,
//     min: f32,
//     max: f32,
//     length: usize,
//     seed: &Buffer,
//     buffer: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     if min >= max {
//         return Err(MetalKernelError::LoadLibraryError(
//             "min must be less than max".to_string(),
//         ));
//     }
//     let pipeline = kernels.load_pipeline(device, Source::Random, name)?;
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//
//     let odd = (length % 2 != 0) as usize;
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, length / 2 + odd);
//
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     set_params!(encoder, (length, min, max, seed, buffer));
//
//     encoder.use_resource(
//         seed,
//         metal::MTLResourceUsage::Read | metal::MTLResourceUsage::Write,
//     );
//     encoder.use_resource(buffer, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_random_normal(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     name: &'static str,
//     mean: f32,
//     stddev: f32,
//     length: usize,
//     seed: &Buffer,
//     buffer: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     let pipeline = kernels.load_pipeline(device, Source::Random, name)?;
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//
//     let odd = (length % 2 != 0) as usize;
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, length / 2 + odd);
//
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     set_params!(encoder, (length, mean, stddev, seed, buffer));
//
//     encoder.use_resource(
//         seed,
//         metal::MTLResourceUsage::Read | metal::MTLResourceUsage::Write,
//     );
//     encoder.use_resource(buffer, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[derive(Debug, Clone, Copy)]
// pub enum GgmlDType {
//     Q4_0,
//     Q4_1,
//     Q5_0,
//     Q5_1,
//     Q8_0,
//     Q8_1,
//     Q2K,
//     Q3K,
//     Q4K,
//     Q5K,
//     Q6K,
//     Q8K,
//     F16,
//     F32,
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_quantized_matmul_mv_t(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     dtype: GgmlDType,
//     (b, m, n, k): (usize, usize, usize, usize),
//     lhs: &Buffer,
//     lhs_offset: usize,
//     rhs: &Buffer,
//     dst_offset: usize,
//     dst: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     // Everything is in reverse
//     let ne00 = k as i64;
//     let ne01 = n as i64;
//     let ne02 = b as i64;
//     let ne03 = 1i64;
//
//     let nb00 = 0i64;
//     let nb01 = 0i64;
//     let nb02 = 0i64;
//
//     let ne10 = k as i64;
//     let ne11 = m as i64;
//     let ne12 = b as i64;
//     let ne13 = 1i64;
//
//     let nb10 = 0i64;
//     let nb11 = 0i64;
//     let nb12 = 0i64;
//
//     let ne0 = n as i64;
//     let ne1 = m as i64;
//     let r2: u32 = (ne12 / ne02) as u32;
//     let r3: u32 = (ne13 / ne03) as u32;
//
//     let (nth0, nth1, align) = match dtype {
//         GgmlDType::Q4_0
//         | GgmlDType::Q4_1
//         | GgmlDType::Q5_0
//         | GgmlDType::Q5_1
//         | GgmlDType::Q8_0
//         | GgmlDType::Q8_1 => {
//             let nth0 = 8;
//             let nth1 = 8;
//             let align = 8;
//             (nth0, nth1, align)
//         }
//         GgmlDType::Q2K => {
//             // Fixing a bug in Metal for GGML
//             // https://github.com/ggerganov/llama.cpp/blob/b8109bc0139f15a5b321909f47510b89dca47ffc/ggml-metal.m#L1576
//             let nth0 = 2;
//             let nth1 = 32;
//             let align = 4;
//             (nth0, nth1, align)
//         }
//         GgmlDType::Q4K => {
//             let nth0 = 4;
//             let nth1 = 8;
//             let align = 4;
//             (nth0, nth1, align)
//         }
//         GgmlDType::Q3K | GgmlDType::Q5K => {
//             let nth0 = 2;
//             let nth1 = 32;
//             let align = 4;
//             (nth0, nth1, align)
//         }
//         GgmlDType::Q6K => {
//             let nth0 = 2;
//             let nth1 = 32;
//             let align = 2;
//             (nth0, nth1, align)
//         }
//         GgmlDType::F16 | GgmlDType::Q8K => {
//             // Original implem uses rows
//             let nth0 = 32;
//             let nth1 = 1;
//             let align = 8;
//             (nth0, nth1, align)
//         }
//         GgmlDType::F32 => {
//             let nth0 = 32;
//             let nth1 = 1;
//             let align = 8;
//             (nth0, nth1, align)
//         }
//     };
//     let thread_groups_count = MTLSize {
//         width: divide(ne01 as usize, align),
//         height: ne11 as u64,
//         depth: (ne12 * ne13) as u64,
//     };
//     let threads_per_threadgroup = MTLSize {
//         width: nth0,
//         height: nth1,
//         depth: 1,
//     };
//     let name = match dtype {
//         GgmlDType::Q4_0 => "kernel_mul_mv_q4_0_f32",
//         GgmlDType::Q4_1 => "kernel_mul_mv_q4_1_f32",
//         GgmlDType::Q5_0 => "kernel_mul_mv_q5_0_f32",
//         GgmlDType::Q5_1 => "kernel_mul_mv_q5_1_f32",
//         GgmlDType::Q8_0 => "kernel_mul_mv_q8_0_f32",
//         GgmlDType::Q8_1 => "kernel_mul_mv_q8_1_f32",
//         GgmlDType::Q2K => "kernel_mul_mv_q2_K_f32",
//         GgmlDType::Q3K => "kernel_mul_mv_q3_K_f32",
//         GgmlDType::Q4K => "kernel_mul_mv_q4_K_f32",
//         GgmlDType::Q5K => "kernel_mul_mv_q5_K_f32",
//         GgmlDType::Q6K => "kernel_mul_mv_q6_K_f32",
//         GgmlDType::Q8K => "kernel_mul_mv_q8_K_f32",
//         GgmlDType::F16 => "kernel_mul_mv_f16_f32",
//         GgmlDType::F32 => "kernel_mul_mv_f32_f32",
//     };
//
//     let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     encoder.set_compute_pipeline_state(&pipeline);
//
//     set_params!(
//         encoder,
//         (
//             rhs,
//             (lhs, lhs_offset),
//             (dst, dst_offset),
//             ne00,
//             ne01,
//             ne02,
//             nb00,
//             nb01,
//             nb02,
//             ne10,
//             ne11,
//             ne12,
//             nb10,
//             nb11,
//             nb12,
//             ne0,
//             ne1,
//             r2,
//             r3
//         )
//     );
//     encoder.use_resource(lhs, metal::MTLResourceUsage::Read);
//     encoder.use_resource(rhs, metal::MTLResourceUsage::Read);
//     encoder.use_resource(dst, metal::MTLResourceUsage::Write);
//
//     encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
//     Ok(())
// }
//
// fn divide(m: usize, b: usize) -> NSUInteger {
//     m.div_ceil(b) as NSUInteger
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_pool2d(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     name: &'static str,
//     shape: &[usize],
//     strides: &[usize],
//     out_w: usize,
//     out_h: usize,
//     w_k: usize,
//     h_k: usize,
//     w_stride: usize,
//     h_stride: usize,
//     input: &Buffer,
//     output: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     let dst_el = out_w * out_h * shape[0] * shape[1];
//     let pipeline: ComputePipelineState = kernels.load_pipeline(device, Source::Conv, name)?;
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     encoder.set_compute_pipeline_state(&pipeline);
//     set_params!(
//         encoder,
//         (w_k, h_k, w_stride, h_stride, shape, strides, input, output)
//     );
//     encoder.use_resource(input, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_conv_transpose1d(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     name: &'static str,
//     dilation: usize,
//     stride: usize,
//     padding: usize,
//     out_padding: usize,
//     c_out: usize,
//     l_out: usize,
//     b_size: usize,
//     src_shape: &[usize],
//     src_strides: &[usize],
//     kernel_shape: &[usize],
//     kernel_strides: &[usize],
//     input: &Buffer,
//     input_offset: usize,
//     kernel: &Buffer,
//     kernel_offset: usize,
//     output: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     let dst_el = c_out * l_out * b_size;
//     let pipeline: ComputePipelineState = kernels.load_pipeline(device, Source::Conv, name)?;
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     encoder.set_compute_pipeline_state(&pipeline);
//     set_params!(
//         encoder,
//         (
//             l_out,
//             stride,
//             padding,
//             out_padding,
//             dilation,
//             src_shape,
//             src_strides,
//             kernel_shape,
//             kernel_strides,
//             (input, input_offset),
//             (kernel, kernel_offset),
//             output
//         )
//     );
//     encoder.use_resource(input, metal::MTLResourceUsage::Read);
//     encoder.use_resource(kernel, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// pub struct CallConvTranspose2dCfg<'a> {
//     pub dilation: usize,
//     pub stride: usize,
//     pub padding: usize,
//     pub output_padding: usize,
//     pub c_out: usize,
//     pub out_w: usize,
//     pub out_h: usize,
//     pub b_size: usize,
//     pub input_dims: &'a [usize],
//     pub input_stride: &'a [usize],
//     pub kernel_dims: &'a [usize],
//     pub kernel_stride: &'a [usize],
//     pub input_offset: usize,
//     pub kernel_offset: usize,
// }
//
// #[allow(clippy::too_many_arguments)]
// pub fn call_conv_transpose2d(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     name: &'static str,
//     cfg: CallConvTranspose2dCfg,
//     input: &Buffer,
//     kernel: &Buffer,
//     output: &Buffer,
// ) -> Result<(), MetalKernelError> {
//     let dst_el = cfg.c_out * cfg.out_w * cfg.out_h * cfg.b_size;
//     let pipeline: ComputePipelineState = kernels.load_pipeline(device, Source::Conv, name)?;
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     encoder.set_compute_pipeline_state(&pipeline);
//     set_params!(
//         encoder,
//         (
//             cfg.out_w,
//             cfg.out_h,
//             cfg.stride,
//             cfg.padding,
//             cfg.output_padding,
//             cfg.dilation,
//             cfg.input_dims,
//             cfg.input_stride,
//             cfg.kernel_dims,
//             cfg.kernel_stride,
//             (input, cfg.input_offset),
//             (kernel, cfg.kernel_offset),
//             output
//         )
//     );
//     encoder.use_resource(input, metal::MTLResourceUsage::Read);
//     encoder.use_resource(kernel, metal::MTLResourceUsage::Read);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }
//
// pub fn call_const_fill(
//     device: &Device,
//     ep: impl EncoderProvider,
//     kernels: &Kernels,
//     name: &'static str,
//     length: usize,
//     output: &Buffer,
//     v: f32,
// ) -> Result<(), MetalKernelError> {
//     let pipeline = kernels.load_pipeline(device, Source::Fill, name)?;
//     let encoder = ep.encoder();
//     let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
//     encoder.set_compute_pipeline_state(&pipeline);
//     set_params!(encoder, (output, v, length));
//     let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);
//     encoder.use_resource(output, metal::MTLResourceUsage::Write);
//     encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
//     Ok(())
// }

// #[cfg(test)]
// mod tests;
