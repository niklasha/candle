use std::collections::HashMap;
use std::sync::{Arc, PoisonError, RwLock, TryLockError};
use vulkano::device::Device;
use vulkano::pipeline::{ComputePipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::shader::ShaderModule;
use shaderc;

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
    pub fn size_in_bytes(&self) -> usize {
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

/// Simple way to catch lock error without
/// depending on T
#[derive(thiserror::Error, Debug)]
pub enum LockError {
    #[error("{0}")]
    Poisoned(String),
    #[error("Would block")]
    WouldBlock,
}

#[derive(thiserror::Error, Debug)]
pub enum VulkanKernelError {
    #[error("{0}")]
    Message(String),
    #[error("{0:?}")]
    LockError(#[from] LockError),
    #[error("{0:?}")]
    IoError(#[from] std::io::Error),
    #[error("{0:?}")]
    ShadercError(#[from] shaderc::Error),
    #[error("Error while loading library: {0}")]
    LoadLibraryError(String),
    #[error("Error while loading function: {0:?}")]
    LoadFunctionError(String),
    #[error("Failed to create compute function")]
    FailedToCreateComputeFunction,
    #[error("Failed to create pipeline: {0}")]
    FailedToCreatePipeline(String),
    #[error("{0:?}")]
    ValidatedVulkanError(#[from] vulkano::Validated<vulkano::VulkanError>),
}

impl From<String> for VulkanKernelError {
    fn from(e: String) -> Self {
        VulkanKernelError::Message(e)
    }
}

impl From<&str> for VulkanKernelError {
    fn from(e: &str) -> Self {
        VulkanKernelError::Message(e.to_string())
    }
}

impl<T> From<TryLockError<T>> for VulkanKernelError {
    fn from(value: TryLockError<T>) -> Self {
        match value {
            TryLockError::Poisoned(p) => VulkanKernelError::LockError(LockError::Poisoned(p.to_string())),
            TryLockError::WouldBlock => VulkanKernelError::LockError(LockError::WouldBlock),
        }
    }
}

impl<T> From<PoisonError<T>> for VulkanKernelError {
    fn from(p: PoisonError<T>) -> Self {
        VulkanKernelError::LockError(LockError::Poisoned(p.to_string()))
    }
}

/// A simple configuration for a shader kernel.
/// Holds the shader source file path and a vector of default macro definitions.
#[derive(Debug)]
pub struct KernelConfig {
    pub path: String,
    pub defines: Vec<(&'static str, String)>,
}

impl KernelConfig {
    /// Compile the shader using shaderc. Additional defines (if any) are merged with the defaults.
    pub fn compile(
        &self,
        device: Arc<Device>,
        additional: Option<&[(&str, &str)]>,
    ) -> Result<Arc<ShaderModule>, VulkanKernelError> {
        let full_path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), self.path);
        let shader_source = std::fs::read_to_string(full_path).map_err(VulkanKernelError::IoError)?;
        let compiler = shaderc::Compiler::new().ok_or(VulkanKernelError::from("Failed to create shader compiler"))?;
        let mut options =
            shaderc::CompileOptions::new().ok_or(VulkanKernelError::from("Failed to create compile options"))?;

        options.set_include_callback(|requested, _include_type, source_path, _depth| {
            let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
            let full_source_path = if std::path::Path::new(source_path).is_absolute() {
                std::path::Path::new(source_path).to_path_buf()
            } else {
                manifest_dir.join(source_path)
            };
            let base_dir = full_source_path.parent().unwrap_or(manifest_dir);
            let include_path = base_dir.join(requested);
            let canonical = std::fs::canonicalize(&include_path)
                .map_err(|e| format!("Failed to canonicalize {}: {}", include_path.display(), e))?;
            let content = std::fs::read_to_string(&canonical)
                .map_err(|e| format!("Failed to read {}: {}", canonical.display(), e))?;
            Ok(shaderc::ResolvedInclude {
                resolved_name: canonical.to_string_lossy().into_owned(),
                content,
            })
        });

        for (key, val) in &self.defines {
            options.add_macro_definition(key, Some(val));
        }
        if let Some(add_defs) = additional {
            for (key, val) in add_defs {
                options.add_macro_definition(key, Some(val));
            }
        }
        let compiled_shader = compiler.compile_into_spirv(
            &shader_source,
            shaderc::ShaderKind::Compute,
            &self.path,
            "main",
            Some(&options),
        ).map_err(VulkanKernelError::ShadercError)?;
        let module = unsafe {
            ShaderModule::new(device, vulkano::shader::ShaderModuleCreateInfo::new(compiled_shader.as_binary()))
        }?;
        Ok(module)
    }
}

/// A helper macro that creates a (name, KernelConfig) pair.
macro_rules! register_kernel {
    ($name:expr, $path:expr, $(($key:expr, $val:expr)),* $(,)?) => {
        ($name.to_string(), KernelConfig {
            path: $path.to_owned(),
            defines: vec![$(($key, $val.to_string())),*],
        })
    };
}

/// Our Kernels struct holds two maps:
/// 1. A config map (kernel name → KernelConfig) that was populated at startup.
/// 2. A compiled cache of shader modules, plus a pipeline cache.
#[derive(Debug)]
pub struct Kernels {
    configs: HashMap<String, KernelConfig>,
    compiled: RwLock<HashMap<String, Arc<ShaderModule>>>,
    pub pipelines: RwLock<HashMap<String, Arc<ComputePipeline>>>,
}

impl Kernels {
    /// Build the kernels from registered configurations.
    /// (Here we register every kernel variant that you previously defined via your macros.)
    pub fn new() -> Result<Self, VulkanKernelError> {
        let mut configs = HashMap::new();

        // --- CAST KERNELS ---
        {
            let (name, config) = register_kernel!("cast_f32_f16", "src/cast.comp",
                ("SRC_TYPE", "float"),
                ("DST_TYPE", "float16_t"),
                ("NEED_UINT_CAST", "0"),
                ("SRC_BF16", "0"),
                ("DST_BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("cast_f16_f32", "src/cast.comp",
                ("SRC_TYPE", "float16_t"),
                ("DST_TYPE", "float"),
                ("NEED_UINT_CAST", "0"),
                ("SRC_BF16", "0"),
                ("DST_BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("cast_u32_f32", "src/cast.comp",
                ("SRC_TYPE", "uint"),
                ("DST_TYPE", "float"),
                ("NEED_UINT_CAST", "0"),
                ("SRC_BF16", "0"),
                ("DST_BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("cast_u32_u8", "src/cast.comp",
                ("SRC_TYPE", "uint"),
                ("DST_TYPE", "uint8_t"),
                ("NEED_UINT_CAST", "0"),
                ("SRC_BF16", "0"),
                ("DST_BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("cast_u8_f32", "src/cast.comp",
                ("SRC_TYPE", "uint8_t"),
                ("DST_TYPE", "float"),
                ("NEED_UINT_CAST", "1"),
                ("SRC_BF16", "0"),
                ("DST_BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("cast_bf16_f32", "src/cast.comp",
                ("SRC_TYPE", "uint16_t"),
                ("DST_TYPE", "float"),
                ("NEED_UINT_CAST", "0"),
                ("SRC_BF16", "1"),
                ("DST_BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("cast_f32_bf16", "src/cast.comp",
                ("SRC_TYPE", "float"),
                ("DST_TYPE", "uint16_t"),
                ("NEED_UINT_CAST", "0"),
                ("SRC_BF16", "0"),
                ("DST_BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("cast_bf16_u32", "src/cast.comp",
                ("SRC_TYPE", "uint16_t"),
                ("DST_TYPE", "uint"),
                ("NEED_UINT_CAST", "0"),
                ("SRC_BF16", "1"),
                ("DST_BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("cast_u32_bf16", "src/cast.comp",
                ("SRC_TYPE", "uint"),
                ("DST_TYPE", "uint16_t"),
                ("NEED_UINT_CAST", "0"),
                ("SRC_BF16", "0"),
                ("DST_BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("cast_bf16_f16", "src/cast.comp",
                ("SRC_TYPE", "uint16_t"),
                ("DST_TYPE", "float16_t"),
                ("NEED_UINT_CAST", "0"),
                ("SRC_BF16", "1"),
                ("DST_BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("cast_u32_i64", "src/cast.comp",
                ("SRC_TYPE", "uint"),
                ("DST_TYPE", "int64_t"),
                ("NEED_UINT_CAST", "0"),
                ("SRC_BF16", "0"),
                ("DST_BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("cast_i64_u32", "src/cast.comp",
                ("SRC_TYPE", "int64_t"),
                ("DST_TYPE", "uint"),
                ("NEED_UINT_CAST", "0"),
                ("SRC_BF16", "0"),
                ("DST_BF16", "0")
            );
            configs.insert(name, config);
        }

        // --- UNARY KERNELS ---
        {
            for op in ["neg", "abs", "sign", "gelu", "gelu_erf", "erf", "relu", "silu", "ceil", "floor", "round", "sqr", "sqrt", "sin", "cos", "tan", "sigmoid", "exp", "log", "recip", "tanh"] {
                for (dtype, glsl_type, is_bf16) in [("f32", "float", "0"), ("f16", "float16_t", "0"), ("bf16", "uint16_t", "1")] {
                    let (name, config) = register_kernel!(
                        format!("{}_{}", op, dtype),
                        "src/unary.comp",
                        ("OP", format!("{}_op", op)),
                        ("INNER_TYPE", "float"),
                        ("OUTER_TYPE", glsl_type),
                        ("BF16", is_bf16)
                    );
                    configs.insert(name, config);
                }
            }
        }

        // --- BINARY KERNELS ---
        {
            for op in ["add", "sub", "div", "mul", "minimum", "maximum"] {
                for (dtype, inner_type, outer_type, is_bf16) in [("f32", "float", "float", "0"), ("f16", "float16_t", "float16_t", "0"), ("bf16", "float", "uint16_t", "1"), ("i64", "int64_t", "int64_t", "0")] {
                    let (name, config) = register_kernel!(format!("{}_{}", op, dtype), "src/binary.comp",
                        ("OP", format!("{}_op", op)),
                        ("INNER_TYPE", inner_type),
                        ("OUTER_TYPE", outer_type),
                        ("BF16", is_bf16)
                    );
                    configs.insert(name, config);
                }
            }
        }

        // --- REDUCE_PARTIAL KERNELS ---
        {
            for (op_name, op, to_index) in [("sum", "0", "0"), ("argmax", "1", "1"), ("max", "1", "0"), ("argmin", "2", "1"), ("min", "2", "0")] {
                for (dtype, inner_type, outer_type, is_bf16) in [("f32", "float", "float", "0"), ("u32", "uint", "uint", "0"), ("f16", "float16_t", "float16_t", "0"), /*("bf16", "float", "uint16_t", "1"),*/ ("i64", "int64_t", "int64_t", "0")] {
                    let (name, config) = register_kernel!(
                        format!("{}_partial_{}", op_name, dtype),
                        "src/reduce_partial.comp",
                        ("OP", op),
                        ("TYPE", outer_type),
                        ("TO_INDEX", to_index));
                    configs.insert(name, config);
                }
            }
        }

        // --- REDUCE_COMBINE KERNELS ---
        {
            for (op_name, op, to_index) in [("sum", "0", "0"), ("argmax", "1", "1"), ("max", "1", "0"), ("argmin", "2", "1"), ("min", "2", "0")] {
                for (dtype, inner_type, outer_type, is_bf16) in [("f32", "float", "float", "0"), ("u32", "uint", "uint", "0"), ("f16", "float16_t", "float16_t", "0"), /*("bf16", "float", "uint16_t", "1"),*/ ("i64", "int64_t", "int64_t", "0")] {
                    let (name, config) = register_kernel!(
                        format!("{}_combine_{}", op_name, dtype),
                        "src/reduce_combine.comp",
                        ("OP", op),
                        ("TYPE", outer_type),
                        ("TO_INDEX", to_index));
                    configs.insert(name, config);
                }
            }
        }

        // --- AFFINE / ELU / POWF KERNELS ---
        {
            for op in ["affine", "elu", "powf"] {
                for (dtype, inner_type, outer_type, is_bf16) in [("f32", "float", "float", "0"), ("f16", "float16_t", "float16_t", "0"), ("bf16", "float", "uint16_t", "1"), ("i64", "int64_t", "int64_t", "0")] {
                    let (name, config) = register_kernel!(
                        format!("{}_{}", op, dtype),
                        "src/affine_elu.comp",
                        ("OP", format!("{}_op", op)),
                        ("INNER_TYPE", inner_type),
                        ("OUTER_TYPE", outer_type),
                        ("BF16", is_bf16)
                    );
                    configs.insert(name, config);
                }
            }
        }

        // --- GATHER KERNELS ---
        {
            for (idx_dtype, idx_type, max_idx) in [("u8", "uint8_t", "0xFF"), ("u32", "uint", "0xFFFFFFFFU"), ("i64", "int64_t", "0x7FFFFFFFFFFFFFFF")] {
                for (dtype, glsl_type, is_bf16) in [("u8", "uint8_t", "0"), ("u32", "uint", "0"), ("i64", "int64_t", "0"), ("bf16", "uint", "1"), ("f32", "float", "0")] {
                    let (name, config) = register_kernel!(
                        format!("gather_{}_{}", idx_dtype, dtype),
                        "src/gather.comp",
                        ("IDX_TYPE", idx_type),
                        ("MAX_IDX", max_idx),
                        ("TYPE", glsl_type),
                        ("BF16", is_bf16)
                    );
                    configs.insert(name, config);
                }
            }
        }

        // --- SCATTER_SET KERNELS ---
        {
            let (name, config) = register_kernel!("scatter_set_u8_u32", "src/scatter.comp",
                ("IDX_TYPE", "uint8_t"),
                ("TYPE", "uint"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("scatter_set_u8_bf16", "src/scatter.comp",
                ("IDX_TYPE", "uint8_t"),
                ("TYPE", "float"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("scatter_set_u8_f32", "src/scatter.comp",
                ("IDX_TYPE", "uint8_t"),
                ("TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("scatter_set_u32_u32", "src/scatter.comp",
                ("IDX_TYPE", "uint"),
                ("TYPE", "uint"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("scatter_set_u32_bf16", "src/scatter.comp",
                ("IDX_TYPE", "uint"),
                ("TYPE", "float"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("scatter_set_u32_f32", "src/scatter.comp",
                ("IDX_TYPE", "uint"),
                ("TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("scatter_set_i64_u32", "src/scatter.comp",
                ("IDX_TYPE", "int64_t"),
                ("TYPE", "uint"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("scatter_set_i64_bf16", "src/scatter.comp",
                ("IDX_TYPE", "int64_t"),
                ("TYPE", "float"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("scatter_set_i64_f32", "src/scatter.comp",
                ("IDX_TYPE", "int64_t"),
                ("TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
        }

        // --- SCATTER_ADD_SET KERNELS ---
        {
            let (name, config) = register_kernel!("scatter_add_set_u8_u32", "src/scatter_add.comp",
                ("IDX_TYPE", "uint8_t"),
                ("TYPE", "uint"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("scatter_add_set_u8_bf16", "src/scatter_add.comp",
                ("IDX_TYPE", "uint8_t"),
                ("TYPE", "float"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("scatter_add_set_u8_f32", "src/scatter_add.comp",
                ("IDX_TYPE", "uint8_t"),
                ("TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("scatter_add_set_u32_u32", "src/scatter_add.comp",
                ("IDX_TYPE", "uint"),
                ("TYPE", "uint"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("scatter_add_set_u32_bf16", "src/scatter_add.comp",
                ("IDX_TYPE", "uint"),
                ("TYPE", "float"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("scatter_add_set_u32_f32", "src/scatter_add.comp",
                ("IDX_TYPE", "uint"),
                ("TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("scatter_add_set_i64_u32", "src/scatter_add.comp",
                ("IDX_TYPE", "int64_t"),
                ("TYPE", "uint"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("scatter_add_set_i64_bf16", "src/scatter_add.comp",
                ("IDX_TYPE", "int64_t"),
                ("TYPE", "float"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("scatter_add_set_i64_f32", "src/scatter_add.comp",
                ("IDX_TYPE", "int64_t"),
                ("TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
        }

        // --- INDEX_{ADD,SELECT} KERNELS ---
        for shader in ["index_add", "index_select"] {
            let path = format!("src/{}.comp", shader);
            for (idx_dtype, idx_glsl_type) in [("u8", "uint8_t"), ("u32", "uint"), ("i64", "int64_t")] {
                for (dtype, glsl_type) in [("u8", "uint8_t"), ("u32", "uint"), ("i64", "int64_t"), ("f32", "float"), ("f16", "float16_t")] {
                    let (name, config) = register_kernel!(
                        format!("{}_{}_{}", shader, idx_dtype, dtype),
                        &path,
                        ("IDX_TYPE", idx_glsl_type),
                        ("TYPE", glsl_type),
                        ("BF16", "0")
                    );
                    configs.insert(name, config);
                }
                let (name, config) = register_kernel!(
                    format!("{}_{}_bf16", shader, idx_dtype),
                    &path,
                    ("IDX_TYPE", idx_glsl_type),
                    ("TYPE", "uint16_t"),
                    ("BF16", "1")
                );
                configs.insert(name, config);
            }
        }

        // --- CONST_SET KERNELS ---
        // The zero width will cause a zero mask which in the shader is interpreted as meaning 64
        // bits
        for (width, glsl_type) in [(8u64, "uint8_t"), (16, "uint16_t"), (32, "uint"), (0, "int64_t")] {
            let (name, config) = register_kernel!(
                format!("const_set_{}", width),
                "src/const_set.comp",
                ("MASK", ((1u64 << width) - 1).to_string()),
                ("TYPE", glsl_type),
            );
            configs.insert(name, config);
        }

        // --- COPY2D SHADERS ---
        {
            for (dtype, glsl_type) in [("u8", "uint8_t"), ("u32", "uint"), ("i64", "int64_t"), ("f32", "float"), ("f16", "float16_t"), ("bf16", "uint16_t")] {
                let (name, config) = register_kernel!(
                    format!("copy2d_{}", dtype),
                    "src/copy2d.comp",
                    ("TYPE", glsl_type)
                );
                configs.insert(name, config);
            }
        }

        // --- COPY_STRIDED_SRC KERNELS ---
        {
            let (name, config) = register_kernel!("copy_strided_src_f32", "src/copy_strided_src.comp",
                ("TYPE", "float")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("copy_strided_src_u32", "src/copy_strided_src.comp",
                ("TYPE", "uint")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("copy_strided_src_i64", "src/copy_strided_src.comp",
                ("TYPE", "int64_t")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("copy_strided_src_bf16", "src/copy_strided_src.comp",
                ("TYPE", "uint16_t")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("copy_strided_src_f16", "src/copy_strided_src.comp",
                ("TYPE", "float16_t")
            );
            configs.insert(name, config);
        }

        // --- COMPARISON (CMP) KERNELS ---
        {
            for (dtype, glsl_type) in [("u8", "uint8_t"), ("u32", "uint"), ("i64", "int64_t"), ("f32", "float"), ("f16", "float16_t"), ("bf16", "uint16_t")] {
                for (op_name, op) in [("eq", "=="), ("ne", "!="), ("lt", "<"), ("gt", ">"), ("le", "<="), ("ge", ">=")] {
                    let (name, config) = register_kernel!(
                        format!("{}_{}", op_name, dtype),
                        "src/cmp.comp",
                        ("OP", op),
                        ("TYPE", glsl_type)
                    );
                    configs.insert(name, config);
                }
            }
        }

        // --- RANDOM KERNELS ---
        {
            let (name, config) = register_kernel!("rand_uniform_f32", "src/rand.comp",
                ("UNIFORM", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("rand_normal_f32", "src/rand.comp",
                ("UNIFORM", "0")
            );
            configs.insert(name, config);
        }

        // --- GEMM KERNELS ---
        {
            let (name, config) = register_kernel!("gemm_f32", "src/gemm.comp",
                ("TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("gemm_f16", "src/gemm.comp",
                ("TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("gemm_bf16", "src/gemm.comp",
                ("TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
        }

        // --- SORT KERNELS ---
        {
            let (name, config) = register_kernel!("arg_sort_f32", "src/sort.comp",
                ("TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("arg_sort_u32", "src/sort.comp",
                ("TYPE", "uint"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("arg_sort_i64", "src/sort.comp",
                ("TYPE", "int64_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("arg_sort_bf16", "src/sort.comp",
                ("TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("arg_sort_f16", "src/sort.comp",
                ("TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("arg_sort_u8", "src/sort.comp",
                ("TYPE", "uint8_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
        }

        // --- LAYERNORM KERNELS ---
        {
            let (name, config) = register_kernel!("layernorm_f32", "src/layernorm.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("layernorm_f16", "src/layernorm.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("layernorm_bf16", "src/layernorm.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
        }

        // --- RMSNORM KERNELS ---
        {
            let (name, config) = register_kernel!("rmsnorm_f32", "src/rmsnorm.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("rmsnorm_f16", "src/rmsnorm.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("rmsnorm_bf16", "src/rmsnorm.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
        }

        // --- SOFTMAX KERNELS ---
        {
            let (name, config) = register_kernel!("softmax_f32", "src/softmax.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("softmax_f16", "src/softmax.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("softmax_bf16", "src/softmax.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
        }

        // --- ROPE KERNELS ---
        {
            let (name, config) = register_kernel!("rope_f32", "src/rope.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0"),
                ("THD", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("rope_f16", "src/rope.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0"),
                ("THD", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("rope_bf16", "src/rope.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1"),
                ("THD", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("rope_thd_f32", "src/rope.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0"),
                ("THD", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("rope_thd_f16", "src/rope.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0"),
                ("THD", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("rope_thd_bf16", "src/rope.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1"),
                ("THD", "1")
            );
            configs.insert(name, config);
        }

        // --- ROPE_I KERNELS ---
        {
            let (name, config) = register_kernel!("rope_i_f32", "src/ropei.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("rope_i_f16", "src/ropei.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("rope_i_bf16", "src/ropei.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
        }

        // --- WHERE KERNELS ---
        {
            let (name, config) = register_kernel!("where_u8_f32", "src/where.comp",
                ("COND_TYPE", "uint8_t"),
                ("ARG_TYPE", "float")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("where_u32_f32", "src/where.comp",
                ("COND_TYPE", "uint"),
                ("ARG_TYPE", "float")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("where_u8_bf16", "src/where.comp",
                ("COND_TYPE", "uint8_t"),
                ("ARG_TYPE", "uint16_t")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("where_u8_f16", "src/where.comp",
                ("COND_TYPE", "uint8_t"),
                ("ARG_TYPE", "float16_t")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("where_u8_i64", "src/where.comp",
                ("COND_TYPE", "uint8_t"),
                ("ARG_TYPE", "int64_t")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("where_u8_u32", "src/where.comp",
                ("COND_TYPE", "uint8_t"),
                ("ARG_TYPE", "uint")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("where_u8_u8", "src/where.comp",
                ("COND_TYPE", "uint8_t"),
                ("ARG_TYPE", "uint8_t")
            );
            configs.insert(name, config);

            // --- CONV1D KERNELS ---
            {
                let (name, config) = register_kernel!("conv1d_f32", "src/conv1d.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
                configs.insert(name, config);
                // let (name, config) = register_kernel!("conv1d_f16", "src/conv1d.comp",
                //     ("INNER_TYPE", "float"),
                //     ("OUTER_TYPE", "float16_t"),
                //     ("BF16", "0")
                // );
                // configs.insert(name, config);
                // let (name, config) = register_kernel!("conv1d_bf16", "src/conv1d.comp",
                //     ("INNER_TYPE", "float"),
                //     ("OUTER_TYPE", "uint16_t"),
                //     ("BF16", "1")
                // );
                // configs.insert(name, config);
                // let (name, config) = register_kernel!("conv1d_u32", "src/conv1d.comp",
                //     ("INNER_TYPE", "float"),
                //     ("OUTER_TYPE", "uint"),
                //     ("BF16", "0")
                // );
                // configs.insert(name, config);
                // let (name, config) = register_kernel!("conv1d_u8", "src/conv1d.comp",
                //     ("INNER_TYPE", "float"),
                //     ("OUTER_TYPE", "uint8_t"),
                //     ("BF16", "0")
                // );
                // configs.insert(name, config);
            }

            // --- CONV_TRANSPOSE1D KERNELS ---
            {
                let (name, config) = register_kernel!("conv_transpose1d_f32", "src/conv_transpose1d.comp",
                    ("INNER_TYPE", "float"),
                    ("OUTER_TYPE", "float"),
                    ("BF16", "0")
                );
                configs.insert(name, config);
                let (name, config) = register_kernel!("conv_transpose1d_f16", "src/conv_transpose1d.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
                configs.insert(name, config);
                let (name, config) = register_kernel!("conv_transpose1d_bf16", "src/conv_transpose1d.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
                configs.insert(name, config);
                let (name, config) = register_kernel!("conv_transpose1d_u32", "src/conv_transpose1d.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint"),
                ("BF16", "0")
            );
                configs.insert(name, config);
                // let (name, config) = register_kernel!("conv_transpose1d_u8", "src/conv_transpose1d.comp",
                //     ("INNER_TYPE", "float"),
                //     ("OUTER_TYPE", "uint8_t"),
                //     ("BF16", "0")
                // );
                // configs.insert(name, config);
            }

            // --- CONV2D KERNELS ---
            {
                let (name, config) = register_kernel!("conv2d_f32", "src/conv2d.comp",
                    ("INNER_TYPE", "float"),
                    ("OUTER_TYPE", "float"),
                    ("BF16", "0")
                );
                configs.insert(name, config);
                // Add BF16, F16 variants if needed...
                // let (name, config) = register_kernel!("conv2d_bf16", "src/conv2d.comp",
                //    ("INNER_TYPE", "float"), ("OUTER_TYPE", "uint16_t"), ("BF16", "1")
                // );
                // configs.insert(name, config);
            }

            // --- CONV_TRANSPOSE2D KERNELS ---
            {
                let (name, config) = register_kernel!("conv_transpose2d_f32", "src/conv_transpose2d.comp",
                    ("INNER_TYPE", "float"), ("OUTER_TYPE", "float"), ("BF16", "0")
                );
                configs.insert(name, config);
                // Add BF16, F16 variants if needed...
            }

            // --- POOL2D KERNELS ---
            {
                // Define supported types and their properties for pooling
                for (op, is_avg) in [("avg", "1"), ("max", "0")] {
                    // Format: (candle_dtype_suffix, glsl_inner_type, glsl_outer_type, glsl_accum_type, is_bf16)
                    // Note: ACCUM_TYPE is often float for avg pooling f16/bf16 for precision.
                    //       For max pooling, ACCUM_TYPE can match INNER_TYPE unless conversion needed.
                    let types = [
                        ("f32", "float", "float", "0",),
                        ("f16", "float16_t", "float", "0"),
                        ("bf16", "uint16_t", "float", "1"),
                    ];

                    for (dtype_suffix, inner_type, outer_type, is_bf16) in types {
                        // Average Pool Variant
                        let name = format!("pool2d_{}_{}", op, dtype_suffix);
                        let (name, avg_config) = register_kernel!(
                            &name,
                            "src/pool2d.comp",
                            ("AVG", is_avg),
                            ("INNER_TYPE", inner_type),
                            ("OUTER_TYPE", outer_type),
                            ("BF16", is_bf16)
                        );
                        configs.insert(name.to_owned(), avg_config);
                    }
                }
            }

            // --- UPSAMPLE_NEAREST1D KERNELS ---
            {
                let (name, config) = register_kernel!("upsample_nearest1d_f32", "src/upsample_nearest1d.comp",
                    ("INNER_TYPE", "float"), ("OUTER_TYPE", "float"), ("BF16", "0")
                );
                configs.insert(name, config);
                let (name, config) = register_kernel!("upsample_nearest1d_bf16", "src/upsample_nearest1d.comp",
                    ("INNER_TYPE", "float"), ("OUTER_TYPE", "uint16_t"), ("BF16", "1")
                );
                configs.insert(name, config);
                let (name, config) = register_kernel!("upsample_nearest1d_f16", "src/upsample_nearest1d.comp",
                    ("INNER_TYPE", "float"), ("OUTER_TYPE", "float16_t"), ("BF16", "0")
                );
                configs.insert(name, config);
                let (name, config) = register_kernel!("upsample_nearest1d_u8", "src/upsample_nearest1d.comp",
                    ("INNER_TYPE", "uint8_t"), ("OUTER_TYPE", "uint8_t"), ("BF16", "0")
                );
                configs.insert(name, config);
                let (name, config) = register_kernel!("upsample_nearest1d_u32", "src/upsample_nearest1d.comp",
                    ("INNER_TYPE", "uint"), ("OUTER_TYPE", "uint"), ("BF16", "0")
                );
                configs.insert(name, config);
                // Add other types if needed
            }

            // --- UPSAMPLE_NEAREST2D KERNELS ---
            {
                let (name, config) = register_kernel!("upsample_nearest2d_f32", "src/upsample_nearest2d.comp",
                    ("INNER_TYPE", "float"), ("OUTER_TYPE", "float"), ("BF16", "0")
                );
                configs.insert(name, config);
                let (name, config) = register_kernel!("upsample_nearest2d_bf16", "src/upsample_nearest2d.comp",
                    ("INNER_TYPE", "float"), ("OUTER_TYPE", "uint16_t"), ("BF16", "1")
                );
                configs.insert(name, config);
                let (name, config) = register_kernel!("upsample_nearest2d_f16", "src/upsample_nearest2d.comp",
                    ("INNER_TYPE", "float"), ("OUTER_TYPE", "float16_t"), ("BF16", "0")
                );
                configs.insert(name, config);
                let (name, config) = register_kernel!("upsample_nearest2d_u8", "src/upsample_nearest2d.comp",
                    ("INNER_TYPE", "uint8_t"), ("OUTER_TYPE", "uint8_t"), ("BF16", "0") // Assuming direct copy
                );
                configs.insert(name, config);
                let (name, config) = register_kernel!("upsample_nearest2d_u32", "src/upsample_nearest2d.comp",
                    ("INNER_TYPE", "uint"), ("OUTER_TYPE", "uint"), ("BF16", "0") // Assuming direct copy
                );
                configs.insert(name, config);
                // Add other types (I64?) if needed, adjusting INNER_TYPE/OUTER_TYPE
            }
        }

        Ok(Self {
            configs,
            compiled: RwLock::new(HashMap::new()),
            pipelines: RwLock::new(HashMap::new()),
        })
    }

    /// Load (or compile on demand) a shader module given its name.
    pub fn load_shader(
        &self,
        device: Arc<Device>,
        name: &str,
        additional: Option<&[(&str, &str)]>,
    ) -> Result<Arc<ShaderModule>, VulkanKernelError> {
        // First, try to find it in the compiled cache.
        if let Some(module) = self.compiled.read().map_err(VulkanKernelError::from)?.get(name).cloned() {
            return Ok(module);
        }
        // Otherwise, look up its config.
        let config = self.configs.get(name)
            .ok_or_else(|| VulkanKernelError::Message(format!("Kernel config for '{}' not found", name)))?;
        // Compile the module.
        let module = config.compile(device.clone(), additional)?;
        // Insert into the cache.
        self.compiled.write().map_err(VulkanKernelError::from)?.insert(name.to_string(), module.clone());
        Ok(module)
    }

    /// Load (or create) a compute pipeline for the given kernel name.
    pub fn load_pipeline(
        &self,
        device: Arc<Device>,
        shader_name: &str,
        additional_defines: Option<&[(&str, &str)]>,
    ) -> Result<Arc<ComputePipeline>, VulkanKernelError> {
        // First, get (or compile) the shader module.
        let shader_module = self.load_shader(device.clone(), shader_name, additional_defines)?;

        // Check if the pipeline is already cached.
        let mut pipelines = self.pipelines.write()?;
        if let Some(pipeline) = pipelines.get(shader_name) {
            return Ok(pipeline.clone());
        }

        // Create the shader stage.
        let stage = PipelineShaderStageCreateInfo::new(
            shader_module.entry_point("main")
                .ok_or_else(|| VulkanKernelError::FailedToCreatePipeline("No entry point".to_string()))?,
        );

        // Create a pipeline layout.
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())
                .map_err(|e| VulkanKernelError::FailedToCreatePipeline(e.to_string()))?,
        )
            .map_err(|e| VulkanKernelError::FailedToCreatePipeline(e.to_string()))?;

        // Create the compute pipeline.
        let pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
            .map_err(|e| VulkanKernelError::FailedToCreatePipeline(e.to_string()))?;

        pipelines.insert(shader_name.to_string(), pipeline.clone());
        Ok(pipeline)
    }
}
