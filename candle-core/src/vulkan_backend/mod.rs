use crate::backend::{BackendDevice, BackendStorage};
use crate::cpu_backend::CpuStorage;
use crate::error::Result;
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{DType, Layout};
use half::{bf16, f16};
use std::sync::Arc;
use vulkano::buffer::BufferContents;
use vulkano::{
    buffer::Buffer,
    device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    Validated, VulkanLibrary,
};

mod device;
pub use device::VulkanDevice;

/// Vulkan related errors
#[derive(thiserror::Error, Debug)]
pub enum VulkanError {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    LoadingError(#[from] vulkano::LoadingError),

    #[error("{0:?}")]
    ValidatedVulkanError(Validated<vulkano::VulkanError>),

    #[error("{0:?}")]
    VulkanError(vulkano::VulkanError),

    #[error("{0:?}")]
    ValidatedAllocateBufferError(Validated<vulkano::buffer::AllocateBufferError>),

    #[error("{0:?}")]
    ValidationError(Box<vulkano::ValidationError>),

    #[error("{0:?}")]
    CommandBufferExecError(vulkano::command_buffer::CommandBufferExecError),
}

impl From<String> for VulkanError {
    fn from(e: String) -> Self {
        VulkanError::Message(e)
    }
}

#[derive(Clone, Debug)]
pub struct VulkanStorage {
    buffer: Arc<Buffer>,
    device: VulkanDevice,
    count: usize,
    dtype: DType,
}

impl VulkanStorage {
    pub fn new(buffer: Arc<Buffer>, device: VulkanDevice, count: usize, dtype: DType) -> Self {
        Self {
            buffer,
            device,
            count,
            dtype,
        }
    }

    pub fn to_cpu<T: BufferContents + Clone + Copy + Send>(&self) -> Result<Vec<T>> {
        self.device.to_cpu(self.buffer.clone())
    }
}

impl BackendStorage for VulkanStorage {
    type Device = VulkanDevice;

    fn try_clone(&self, _layout: &crate::Layout) -> Result<Self> {
        Ok(self.clone())
    }

    fn dtype(&self) -> crate::DType {
        // Return data type
        crate::DType::F32
    }

    fn device(&self) -> &Self::Device {
        // Return device
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

    fn affine(&self, _layout: &crate::Layout, _scale: f64, _bias: f64) -> Result<Self> {
        // Implement affine transformation
        todo!()
    }

    fn powf(&self, _layout: &crate::Layout, _exp: f64) -> Result<Self> {
        // Implement power function
        todo!()
    }

    fn elu(&self, _layout: &crate::Layout, _alpha: f64) -> Result<Self> {
        // Implement ELU function
        todo!()
    }

    fn reduce_op(&self, _op: ReduceOp, _layout: &crate::Layout, _axes: &[usize]) -> Result<Self> {
        // Implement reduction operation
        todo!()
    }

    fn cmp(
        &self,
        _op: CmpOp,
        _rhs: &Self,
        _lhs_layout: &crate::Layout,
        _rhs_layout: &crate::Layout,
    ) -> Result<Self> {
        // Implement comparison
        todo!()
    }

    fn to_dtype(&self, _layout: &crate::Layout, _dtype: crate::DType) -> Result<Self> {
        // Implement conversion to data type
        todo!()
    }

    fn unary_impl<B: UnaryOpT>(&self, _layout: &crate::Layout) -> Result<Self> {
        // Implement unary operation
        todo!()
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        _rhs: &Self,
        _lhs_layout: &crate::Layout,
        _rhs_layout: &crate::Layout,
    ) -> Result<Self> {
        // Implement binary operation
        todo!()
    }

    fn where_cond(
        &self,
        _cond: &crate::Layout,
        _x: &Self,
        _x_layout: &crate::Layout,
        _y: &Self,
        _y_layout: &crate::Layout,
    ) -> Result<Self> {
        // Implement where condition
        todo!()
    }

    fn conv1d(
        &self,
        _layout: &crate::Layout,
        _kernel: &Self,
        _kernel_layout: &crate::Layout,
        _params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        // Implement 1D convolution
        todo!()
    }

    fn conv_transpose1d(
        &self,
        _layout: &crate::Layout,
        _kernel: &Self,
        _kernel_layout: &crate::Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        // Implement 1D transposed convolution
        todo!()
    }

    fn conv2d(
        &self,
        _layout: &crate::Layout,
        _kernel: &Self,
        _kernel_layout: &crate::Layout,
        _params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        // Implement 2D convolution
        todo!()
    }

    fn conv_transpose2d(
        &self,
        _layout: &crate::Layout,
        _kernel: &Self,
        _kernel_layout: &crate::Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        // Implement 2D transposed convolution
        todo!()
    }

    fn avg_pool2d(
        &self,
        _layout: &crate::Layout,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
    ) -> Result<Self> {
        // Implement 2D average pooling
        todo!()
    }

    fn max_pool2d(
        &self,
        _layout: &crate::Layout,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
    ) -> Result<Self> {
        // Implement 2D max pooling
        todo!()
    }

    fn upsample_nearest1d(&self, _layout: &crate::Layout, _scale: usize) -> Result<Self> {
        // Implement 1D nearest upsampling
        todo!()
    }

    fn upsample_nearest2d(
        &self,
        _layout: &crate::Layout,
        _scale_h: usize,
        _scale_w: usize,
    ) -> Result<Self> {
        // Implement 2D nearest upsampling
        todo!()
    }

    fn gather(
        &self,
        _layout: &crate::Layout,
        _indices: &Self,
        _indices_layout: &crate::Layout,
        _axis: usize,
    ) -> Result<Self> {
        // Implement gather
        todo!()
    }

    fn scatter_add(
        &self,
        _layout: &crate::Layout,
        _indices: &Self,
        _indices_layout: &crate::Layout,
        _updates: &Self,
        _updates_layout: &crate::Layout,
        _axis: usize,
    ) -> Result<Self> {
        // Implement scatter add
        todo!()
    }

    fn index_add(
        &self,
        _layout: &crate::Layout,
        _indices: &Self,
        _indices_layout: &crate::Layout,
        _updates: &Self,
        _updates_layout: &crate::Layout,
        _axis: usize,
    ) -> Result<Self> {
        // Implement index add
        todo!()
    }

    fn copy2d(
        &self,
        _: &mut Self,
        _d1: usize,
        _d2: usize,
        _src_stride1: usize,
        _dst_stride1: usize,
        _src_offset: usize,
        _dst_offset: usize,
    ) -> Result<()> {
        // Implement 2D copy
        todo!()
    }

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self> {
        todo!()
    }

    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        todo!()
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()> {
        todo!()
    }
}

impl BackendDevice for VulkanDevice {
    type Storage = VulkanStorage;

    fn new(ordinal: usize) -> Result<Self> {
        // Create a Vulkan instance
        let library = VulkanLibrary::new().map_err(VulkanError::LoadingError)?;
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
        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: 0,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .map_err(VulkanError::ValidatedVulkanError)?;
        let queue = queues.next().unwrap();

        Ok(VulkanDevice {
            gpu_id: ordinal,
            device,
            queue,
        })
    }

    fn location(&self) -> crate::DeviceLocation {
        // Return device location
        crate::DeviceLocation::Vulkan {
            gpu_id: self.gpu_id,
        }
    }

    fn same_device(&self, _other: &Self) -> bool {
        // Implement comparison logic
        todo!()
    }

    fn zeros_impl(&self, shape: &crate::Shape, dtype: DType) -> Result<Self::Storage> {
        let size = (shape.elem_count() * dtype.size_in_bytes() + 3) / 4;
        let buffer = self.allocate_buffer(size, 0)?;
        Ok(VulkanStorage::new(
            buffer,
            self.clone(),
            shape.elem_count(),
            dtype,
        ))
    }

    fn ones_impl(&self, shape: &crate::Shape, dtype: DType) -> Result<Self::Storage> {
        let num_elements = shape.elem_count();

        // Adjust the number of elements based on the dtype
        let num_units = match dtype {
            DType::F32 | DType::U32 => num_elements, // 1 unit per element
            DType::F16 | DType::BF16 => (num_elements + 1) / 2, // 2 elements per unit
            DType::U8 => (num_elements + 3) / 4,     // 4 elements per unit
            DType::I64 | DType::F64 => num_elements, // 1 unit per element, but each unit is 64-bit
        };

        // Convert the value 1 to the appropriate u32 bit pattern based on dtype
        let buffer = match dtype {
            DType::F32 => self.allocate_buffer(num_units, 1.0f32.to_bits())?,
            DType::U32 => self.allocate_buffer(num_units, 1u32)?,
            DType::F16 => {
                let bits = f16::from_f32(1.0).to_bits();
                self.allocate_buffer(num_units, ((bits as u32) << 16) | (bits as u32))?
            }
            DType::BF16 => {
                let bits = bf16::from_f32(1.0).to_bits();
                self.allocate_buffer(num_units, ((bits as u32) << 16) | (bits as u32))?
            }
            DType::U8 => {
                let byte = 1u8 as u32;
                self.allocate_buffer(num_units, (byte << 24) | (byte << 16) | (byte << 8) | byte)?
            }
            DType::I64 => self.allocate_buffer_64(num_units, 1i64 as u64)?,
            DType::F64 => self.allocate_buffer_64(num_units, 1.0f64.to_bits())?,
        };

        Ok(VulkanStorage::new(
            buffer,
            self.clone(),
            num_elements,
            dtype,
        ))
    }

    unsafe fn alloc_uninit(
        &self,
        _shape: &crate::Shape,
        _dtype: crate::DType,
    ) -> Result<Self::Storage> {
        // Implement uninitialized allocation
        todo!()
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        let (count, buffer) = match storage {
            CpuStorage::U8(storage) => (storage.len(), self.new_buffer_with_data(storage)),
            CpuStorage::U32(storage) => (storage.len(), self.new_buffer_with_data(storage)),
            CpuStorage::I64(storage) => (storage.len(), self.new_buffer_with_data(storage)),
            CpuStorage::BF16(storage) => (storage.len(), self.new_buffer_with_data(storage)),
            CpuStorage::F16(storage) => (storage.len(), self.new_buffer_with_data(storage)),
            CpuStorage::F32(storage) => (storage.len(), self.new_buffer_with_data(storage)),
            CpuStorage::F64(storage) => (storage.len(), self.new_buffer_with_data(storage)),
        };
        Ok(Self::Storage::new(
            buffer?,
            self.clone(),
            count,
            storage.dtype(),
        ))
    }

    fn rand_uniform(
        &self,
        _shape: &crate::Shape,
        _dtype: crate::DType,
        _low: f64,
        _high: f64,
    ) -> Result<Self::Storage> {
        // Implement random uniform initialization
        todo!()
    }

    fn rand_normal(
        &self,
        _shape: &crate::Shape,
        _dtype: crate::DType,
        _mean: f64,
        _std: f64,
    ) -> Result<Self::Storage> {
        // Implement random normal initialization
        todo!()
    }

    fn set_seed(&self, _seed: u64) -> Result<()> {
        // Implement seed setting
        todo!()
    }

    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> Result<Self::Storage> {
        todo!()
    }

    fn synchronize(&self) -> Result<()> {
        todo!()
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&storage)
    }
}
