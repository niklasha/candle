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
    pub defines: Vec<(&'static str, &'static str)>,
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
            defines: vec![$(($key, $val)),*],
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
        }

        // --- UNARY KERNELS ---
        {
            let (name, config) = register_kernel!("neg_f32", "src/unary.comp",
                ("OP", "neg_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("abs_f32", "src/unary.comp",
                ("OP", "abs_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("sign_f32", "src/unary.comp",
                ("OP", "sign_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("gelu_f32", "src/unary.comp",
                ("OP", "gelu_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("gelu_erf_f32", "src/unary.comp",
                ("OP", "gelu_erf_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("erf_f32", "src/unary.comp",
                ("OP", "erf_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("silu_f32", "src/unary.comp",
                ("OP", "silu_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("ceil_f32", "src/unary.comp",
                ("OP", "ceil_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("floor_f32", "src/unary.comp",
                ("OP", "floor_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("round_f32", "src/unary.comp",
                ("OP", "round_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("sqr_f32", "src/unary.comp",
                ("OP", "sqr_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("sqrt_f32", "src/unary.comp",
                ("OP", "sqrt_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("sin_f32", "src/unary.comp",
                ("OP", "sin_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("cos_f32", "src/unary.comp",
                ("OP", "cos_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("tan_f32", "src/unary.comp",
                ("OP", "tan_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("sigmoid_f32", "src/unary.comp",
                ("OP", "sigmoid_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("exp_f32", "src/unary.comp",
                ("OP", "exp_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("log_f32", "src/unary.comp",
                ("OP", "log_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("recip_f32", "src/unary.comp",
                ("OP", "recip_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);

            // F16 variants:
            let (name, config) = register_kernel!("neg_f16", "src/unary.comp",
                ("OP", "neg_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("abs_f16", "src/unary.comp",
                ("OP", "abs_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("sign_f16", "src/unary.comp",
                ("OP", "sign_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("gelu_f16", "src/unary.comp",
                ("OP", "gelu_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("gelu_erf_f16", "src/unary.comp",
                ("OP", "gelu_erf_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("erf_f16", "src/unary.comp",
                ("OP", "erf_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("silu_f16", "src/unary.comp",
                ("OP", "silu_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("ceil_f16", "src/unary.comp",
                ("OP", "ceil_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("floor_f16", "src/unary.comp",
                ("OP", "floor_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("round_f16", "src/unary.comp",
                ("OP", "round_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("sqr_f16", "src/unary.comp",
                ("OP", "sqr_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("sqrt_f16", "src/unary.comp",
                ("OP", "sqrt_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("sin_f16", "src/unary.comp",
                ("OP", "sin_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("cos_f16", "src/unary.comp",
                ("OP", "cos_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("tan_f16", "src/unary.comp",
                ("OP", "tan_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("sigmoid_f16", "src/unary.comp",
                ("OP", "sigmoid_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("exp_f16", "src/unary.comp",
                ("OP", "exp_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("log_f16", "src/unary.comp",
                ("OP", "log_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("recip_f16", "src/unary.comp",
                ("OP", "recip_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);

            // BF16 variants:
            let (name, config) = register_kernel!("neg_bf16", "src/unary.comp",
                ("OP", "neg_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("abs_bf16", "src/unary.comp",
                ("OP", "abs_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("sign_bf16", "src/unary.comp",
                ("OP", "sign_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("gelu_bf16", "src/unary.comp",
                ("OP", "gelu_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("gelu_erf_bf16", "src/unary.comp",
                ("OP", "gelu_erf_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("erf_bf16", "src/unary.comp",
                ("OP", "erf_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("silu_bf16", "src/unary.comp",
                ("OP", "silu_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("ceil_bf16", "src/unary.comp",
                ("OP", "ceil_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("floor_bf16", "src/unary.comp",
                ("OP", "floor_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("round_bf16", "src/unary.comp",
                ("OP", "round_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("sqr_bf16", "src/unary.comp",
                ("OP", "sqr_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("sqrt_bf16", "src/unary.comp",
                ("OP", "sqrt_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("sin_bf16", "src/unary.comp",
                ("OP", "sin_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("cos_bf16", "src/unary.comp",
                ("OP", "cos_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("tan_bf16", "src/unary.comp",
                ("OP", "tan_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("sigmoid_bf16", "src/unary.comp",
                ("OP", "sigmoid_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("exp_bf16", "src/unary.comp",
                ("OP", "exp_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("log_bf16", "src/unary.comp",
                ("OP", "log_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("recip_bf16", "src/unary.comp",
                ("OP", "recip_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
        }

        // --- BINARY KERNELS ---
        {
            let (name, config) = register_kernel!("add_f32", "src/binary.comp",
                ("OP", "add_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("sub_f32", "src/binary.comp",
                ("OP", "sub_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("div_f32", "src/binary.comp",
                ("OP", "div_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("mul_f32", "src/binary.comp",
                ("OP", "mul_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("minimum_f32", "src/binary.comp",
                ("OP", "min_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("maximum_f32", "src/binary.comp",
                ("OP", "max_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("add_bf16", "src/binary.comp",
                ("OP", "add_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("sub_bf16", "src/binary.comp",
                ("OP", "sub_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("div_bf16", "src/binary.comp",
                ("OP", "div_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("mul_bf16", "src/binary.comp",
                ("OP", "mul_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("minimum_bf16", "src/binary.comp",
                ("OP", "min_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("maximum_bf16", "src/binary.comp",
                ("OP", "max_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("add_f16", "src/binary.comp",
                ("OP", "add_op"),
                ("INNER_TYPE", "float16_t"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("sub_f16", "src/binary.comp",
                ("OP", "sub_op"),
                ("INNER_TYPE", "float16_t"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("div_f16", "src/binary.comp",
                ("OP", "div_op"),
                ("INNER_TYPE", "float16_t"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("mul_f16", "src/binary.comp",
                ("OP", "mul_op"),
                ("INNER_TYPE", "float16_t"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("minimum_f16", "src/binary.comp",
                ("OP", "min_op"),
                ("INNER_TYPE", "float16_t"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("maximum_f16", "src/binary.comp",
                ("OP", "max_op"),
                ("INNER_TYPE", "float16_t"),
                ("OUTER_TYPE", "float16_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
        }

        // --- REDUCE_PARTIAL KERNELS ---
        {
            let (name, config) = register_kernel!("sum_partial_f32", "src/reduce_partial.comp",
                ("OP", "0"),
                ("TYPE", "float"),
                ("TO_INDEX", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("argmax_partial_f32", "src/reduce_partial.comp",
                ("OP", "1"),
                ("TYPE", "float"),
                ("TO_INDEX", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("max_partial_f32", "src/reduce_partial.comp",
                ("OP", "1"),
                ("TYPE", "float"),
                ("TO_INDEX", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("argmin_partial_f32", "src/reduce_partial.comp",
                ("OP", "2"),
                ("TYPE", "float"),
                ("TO_INDEX", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("min_partial_f32", "src/reduce_partial.comp",
                ("OP", "2"),
                ("TYPE", "float"),
                ("TO_INDEX", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("sum_partial_u32", "src/reduce_partial.comp",
                ("OP", "0"),
                ("TYPE", "uint"),
                ("TO_INDEX", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("argmax_partial_u32", "src/reduce_partial.comp",
                ("OP", "1"),
                ("TYPE", "uint"),
                ("TO_INDEX", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("max_partial_u32", "src/reduce_partial.comp",
                ("OP", "1"),
                ("TYPE", "uint"),
                ("TO_INDEX", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("argmin_partial_u32", "src/reduce_partial.comp",
                ("OP", "2"),
                ("TYPE", "uint"),
                ("TO_INDEX", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("min_partial_u32", "src/reduce_partial.comp",
                ("OP", "2"),
                ("TYPE", "uint"),
                ("TO_INDEX", "0")
            );
            configs.insert(name, config);
        }

        // --- REDUCE_COMBINE KERNELS ---
        {
            let (name, config) = register_kernel!("sum_combine_f32", "src/reduce_combine.comp",
                ("OP", "0"),
                ("TYPE", "float"),
                ("TO_INDEX", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("argmax_combine_f32", "src/reduce_combine.comp",
                ("OP", "1"),
                ("TYPE", "float"),
                ("TO_INDEX", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("max_combine_f32", "src/reduce_combine.comp",
                ("OP", "1"),
                ("TYPE", "float"),
                ("TO_INDEX", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("argmin_combine_f32", "src/reduce_combine.comp",
                ("OP", "2"),
                ("TYPE", "float"),
                ("TO_INDEX", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("min_combine_f32", "src/reduce_combine.comp",
                ("OP", "2"),
                ("TYPE", "float"),
                ("TO_INDEX", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("sum_combine_u32", "src/reduce_combine.comp",
                ("OP", "0"),
                ("TYPE", "uint"),
                ("TO_INDEX", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("argmax_combine_u32", "src/reduce_combine.comp",
                ("OP", "1"),
                ("TYPE", "uint"),
                ("TO_INDEX", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("max_combine_u32", "src/reduce_combine.comp",
                ("OP", "1"),
                ("TYPE", "uint"),
                ("TO_INDEX", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("argmin_combine_u32", "src/reduce_combine.comp",
                ("OP", "2"),
                ("TYPE", "uint"),
                ("TO_INDEX", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("min_combine_u32", "src/reduce_combine.comp",
                ("OP", "2"),
                ("TYPE", "uint"),
                ("TO_INDEX", "0")
            );
            configs.insert(name, config);
        }

        // --- AFFINE / ELU KERNELS ---
        {
            let (name, config) = register_kernel!("affine_f32", "src/affine_elu.comp",
                ("OP", "affine_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("affine_bf16", "src/affine_elu.comp",
                ("OP", "affine_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "uint16_t"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("elu_f32", "src/affine_elu.comp",
                ("OP", "elu_op"),
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
        }

        // --- GATHER KERNELS ---
        {
            let (name, config) = register_kernel!("gather_u8_u8", "src/gather.comp",
                ("IDX_TYPE", "uint8_t"),
                ("TYPE", "uint8_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("gather_u8_u32", "src/gather.comp",
                ("IDX_TYPE", "uint8_t"),
                ("TYPE", "uint"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("gather_u8_i64", "src/gather.comp",
                ("IDX_TYPE", "uint8_t"),
                ("TYPE", "int64_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("gather_u8_bf16", "src/gather.comp",
                ("IDX_TYPE", "uint8_t"),
                ("TYPE", "uint"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("gather_u8_f32", "src/gather.comp",
                ("IDX_TYPE", "uint8_t"),
                ("TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("gather_u32_u8", "src/gather.comp",
                ("IDX_TYPE", "uint"),
                ("TYPE", "uint8_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("gather_u32_u32", "src/gather.comp",
                ("IDX_TYPE", "uint"),
                ("TYPE", "uint"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("gather_u32_i64", "src/gather.comp",
                ("IDX_TYPE", "uint"),
                ("TYPE", "int64_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("gather_u32_bf16", "src/gather.comp",
                ("IDX_TYPE", "uint"),
                ("TYPE", "float"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("gather_u32_f32", "src/gather.comp",
                ("IDX_TYPE", "uint"),
                ("TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("gather_i64_u8", "src/gather.comp",
                ("IDX_TYPE", "int64_t"),
                ("TYPE", "uint8_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("gather_i64_u32", "src/gather.comp",
                ("IDX_TYPE", "int64_t"),
                ("TYPE", "uint"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("gather_i64_i64", "src/gather.comp",
                ("IDX_TYPE", "int64_t"),
                ("TYPE", "int64_t"),
                ("BF16", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("gather_i64_bf16", "src/gather.comp",
                ("IDX_TYPE", "int64_t"),
                ("TYPE", "float"),
                ("BF16", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("gather_i64_f32", "src/gather.comp",
                ("IDX_TYPE", "int64_t"),
                ("TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
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
            }
        }

        // --- COPY2D SHADERS ---
        {
            let (name, config) = register_kernel!("copy2d_f32", "src/copy2d.comp",
                ("TYPE", "float")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("copy2d_u32", "src/copy2d.comp",
                ("TYPE", "uint")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("copy2d_i64", "src/copy2d.comp",
                ("TYPE", "int64_t")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("copy2d_bf16", "src/copy2d.comp",
                ("TYPE", "uint16_t")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("copy2d_f16", "src/copy2d.comp",
                ("TYPE", "float16_t")
            );
            configs.insert(name, config);
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
            for (dtype, glsl_type) in [("u8", "uint8_t"), ("u32", "uint"), ("i64", "int64_t"), ("f32", "float"), ("f16", "float16_t"), ("bf16", "uint16_T")] {
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
