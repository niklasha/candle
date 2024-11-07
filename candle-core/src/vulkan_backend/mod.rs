use crate::backend::{BackendDevice, BackendStorage};
use crate::cpu_backend::CpuStorage;
use crate::error::Result;
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{DType, Layout, Shape};
use candle_vulkan_kernels::{GpuFutureHolder, Kernels};
use half::{bf16, f16};
use std::sync::{Arc, Mutex, PoisonError, TryLockError};
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::{
    buffer::Buffer,
    device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    Validated, VulkanLibrary,
};

mod device;
pub use device::VulkanDevice;

/// Simple way to catch lock error without
/// depending on T
#[derive(thiserror::Error, Debug)]
pub enum LockError {
    #[error("{0}")]
    Poisoned(String),
    #[error("Would block")]
    WouldBlock,
}

impl<T> From<TryLockError<T>> for VulkanError {
    fn from(value: TryLockError<T>) -> Self {
        match value {
            TryLockError::Poisoned(p) => VulkanError::LockError(LockError::Poisoned(p.to_string())),
            TryLockError::WouldBlock => VulkanError::LockError(LockError::WouldBlock),
        }
    }
}

impl<T> From<PoisonError<T>> for VulkanError {
    fn from(p: PoisonError<T>) -> Self {
        VulkanError::LockError(LockError::Poisoned(p.to_string()))
    }
}

/// Vulkan related errors
#[derive(thiserror::Error, Debug)]
pub enum VulkanError {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    LoadingError(#[from] vulkano::LoadingError),
    #[error(transparent)]
    KernelError(#[from] candle_vulkan_kernels::VulkanKernelError),
    #[error("{0:?}")]
    LockError(LockError),
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
    #[error("{0:?}")]
    HostAccessError(vulkano::sync::HostAccessError),
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
    pending_future: Arc<GpuFutureHolder>,
}

impl VulkanStorage {
    pub fn new(buffer: Arc<Buffer>, device: VulkanDevice, count: usize, dtype: DType) -> Self {
        Self {
            buffer,
            device,
            count,
            dtype,
            pending_future: Arc::new(GpuFutureHolder::new()),
        }
    }

    pub fn to_cpu<T: BufferContents + Clone + Copy + Send>(&self) -> Result<Vec<T>> {
        self.pending_future
            .sync_if_needed()
            .map_err(VulkanError::from)?;
        self.device.to_cpu(self.buffer.clone())
    }
}

impl BackendStorage for VulkanStorage {
    type Device = VulkanDevice;

    fn try_clone(&self, _layout: &Layout) -> Result<Self> {
        Ok(self.clone())
    }

    fn dtype(&self) -> DType {
        DType::F32
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

    fn affine(&self, _layout: &Layout, _scale: f64, _bias: f64) -> Result<Self> {
        // Implement affine transformation
        todo!()
    }

    fn powf(&self, _layout: &Layout, _exp: f64) -> Result<Self> {
        // Implement power function
        todo!()
    }

    fn elu(&self, _layout: &Layout, _alpha: f64) -> Result<Self> {
        // Implement ELU function
        todo!()
    }

    fn reduce_op(&self, _op: ReduceOp, _layout: &Layout, _axes: &[usize]) -> Result<Self> {
        // Implement reduction operation
        todo!()
    }

    fn cmp(
        &self,
        _op: CmpOp,
        _rhs: &Self,
        _lhs_layout: &Layout,
        _rhs_layout: &Layout,
    ) -> Result<Self> {
        // Implement comparison
        todo!()
    }

    fn to_dtype(&self, _layout: &Layout, _dtype: DType) -> Result<Self> {
        // Implement conversion to data type
        todo!()
    }

    fn unary_impl<B: UnaryOpT>(&self, _layout: &Layout) -> Result<Self> {
        // Implement unary operation
        todo!()
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        _rhs: &Self,
        _lhs_layout: &Layout,
        _rhs_layout: &Layout,
    ) -> Result<Self> {
        // Implement binary operation
        todo!()
    }

    fn where_cond(
        &self,
        _cond: &Layout,
        _x: &Self,
        _x_layout: &Layout,
        _y: &Self,
        _y_layout: &Layout,
    ) -> Result<Self> {
        // Implement where condition
        todo!()
    }

    fn conv1d(
        &self,
        _layout: &Layout,
        _kernel: &Self,
        _kernel_layout: &Layout,
        _params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        // Implement 1D convolution
        todo!()
    }

    fn conv_transpose1d(
        &self,
        _layout: &Layout,
        _kernel: &Self,
        _kernel_layout: &Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        // Implement 1D transposed convolution
        todo!()
    }

    fn conv2d(
        &self,
        _layout: &Layout,
        _kernel: &Self,
        _kernel_layout: &Layout,
        _params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        // Implement 2D convolution
        todo!()
    }

    fn conv_transpose2d(
        &self,
        _layout: &Layout,
        _kernel: &Self,
        _kernel_layout: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        // Implement 2D transposed convolution
        todo!()
    }

    fn avg_pool2d(
        &self,
        _layout: &Layout,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
    ) -> Result<Self> {
        // Implement 2D average pooling
        todo!()
    }

    fn max_pool2d(
        &self,
        _layout: &Layout,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
    ) -> Result<Self> {
        // Implement 2D max pooling
        todo!()
    }

    fn upsample_nearest1d(&self, _layout: &Layout, _scale: usize) -> Result<Self> {
        // Implement 1D nearest upsampling
        todo!()
    }

    fn upsample_nearest2d(
        &self,
        _layout: &Layout,
        _scale_h: usize,
        _scale_w: usize,
    ) -> Result<Self> {
        // Implement 2D nearest upsampling
        todo!()
    }

    fn gather(
        &self,
        _layout: &Layout,
        _indices: &Self,
        _indices_layout: &Layout,
        _axis: usize,
    ) -> Result<Self> {
        // Implement gather
        todo!()
    }

    fn scatter_add(
        &self,
        _layout: &Layout,
        _indices: &Self,
        _indices_layout: &Layout,
        _updates: &Self,
        _updates_layout: &Layout,
        _axis: usize,
    ) -> Result<Self> {
        // Implement scatter add
        todo!()
    }

    fn index_add(
        &self,
        _layout: &Layout,
        _indices: &Self,
        _indices_layout: &Layout,
        _updates: &Self,
        _updates_layout: &Layout,
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

        // Create command buffer allocator
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let descriptor_set_allocator =
            StandardDescriptorSetAllocator::new(device.clone(), Default::default());

        let queue = queues.next().unwrap();
        let kernels = Arc::new(Kernels::new(device.clone()));
        let seed: Subbuffer<[u32]> = Self::allocate_filled_subbuffer_raw(
            &command_buffer_allocator,
            &memory_allocator,
            queue.clone(),
            1,
            299792458,
        )?;

        Ok(VulkanDevice {
            gpu_id: ordinal,
            device,
            queue,
            kernels,
            seed: Arc::new(Mutex::new(seed)),
            command_buffer_allocator,
            memory_allocator,
            descriptor_set_allocator: Arc::new(descriptor_set_allocator),
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

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let size = (shape.elem_count() * dtype.size_in_bytes() + 3) / 4;
        let buffer = self.allocate_filled_buffer(size, 0)?;
        Ok(VulkanStorage::new(
            buffer,
            self.clone(),
            shape.elem_count(),
            dtype,
        ))
    }

    fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
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
            DType::F32 => self.allocate_filled_buffer(num_units, 1.0f32.to_bits())?,
            DType::U32 => self.allocate_filled_buffer(num_units, 1u32)?,
            DType::F16 => {
                let bits = f16::from_f32(1.0).to_bits();
                self.allocate_filled_buffer(num_units, ((bits as u32) << 16) | (bits as u32))?
            }
            DType::BF16 => {
                let bits = bf16::from_f32(1.0).to_bits();
                self.allocate_filled_buffer(num_units, ((bits as u32) << 16) | (bits as u32))?
            }
            DType::U8 => {
                let byte = 1u8 as u32;
                self.allocate_filled_buffer(
                    num_units,
                    (byte << 24) | (byte << 16) | (byte << 8) | byte,
                )?
            }
            DType::I64 => self.allocate_filled_buffer_64(num_units, 1i64 as u64)?,
            DType::F64 => self.allocate_filled_buffer_64(num_units, 1.0f64.to_bits())?,
        };

        Ok(VulkanStorage::new(
            buffer,
            self.clone(),
            num_elements,
            dtype,
        ))
    }

    unsafe fn alloc_uninit(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
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
        shape: &Shape,
        dtype: DType,
        low: f64,
        high: f64,
    ) -> Result<Self::Storage> {
        let name = match dtype {
            DType::F32 => "rand_normal_f32",
            //DType::F16 => "rand_normal_f16",
            //DType::BF16 => "rand_normal_bf16",
            dtype => crate::bail!("rand_uniform not implemented for {dtype:?}"),
        };
        let buffer = self.allocate_subbuffer(shape.elem_count() * dtype.size_in_bytes())?;
        let storage = VulkanStorage::new(
            buffer.buffer().clone(),
            self.clone(),
            shape.elem_count(),
            dtype,
        );
        let seed = self.seed.lock().map_err(VulkanError::from)?;
        // XXX
        candle_vulkan_kernels::call_random_normal(
            self.device.clone(),
            self.queue.clone(),
            &self.kernels,
            self.command_buffer_allocator.clone(),
            self.descriptor_set_allocator.clone(),
            name,
            ((high + low) / 2f64) as f32,
            ((high - low) / 4f64) as f32,
            shape.elem_count(),
            seed.clone(),
            buffer.clone(),
            &storage.pending_future,
        )
        .map_err(VulkanError::from)?;

        Ok(storage)
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        stddev: f64,
    ) -> Result<Self::Storage> {
        let name = match dtype {
            DType::F32 => "rand_normal_f32",
            //DType::F16 => "rand_normal_f16",
            //DType::BF16 => "rand_normal_bf16",
            dtype => crate::bail!("rand_uniform not implemented for {dtype:?}"),
        };
        let buffer = self.allocate_subbuffer(shape.elem_count() * dtype.size_in_bytes())?;
        let storage = Self::Storage::new(
            buffer.buffer().clone(),
            self.clone(),
            shape.elem_count(),
            dtype,
        );
        let seed = self.seed.lock().map_err(VulkanError::from)?;
        candle_vulkan_kernels::call_random_normal(
            self.device.clone(),
            self.queue.clone(),
            &self.kernels,
            self.command_buffer_allocator.clone(),
            self.descriptor_set_allocator.clone(),
            name,
            mean as f32,
            stddev as f32,
            shape.elem_count(),
            seed.clone(),
            buffer.clone(),
            &storage.pending_future,
        )
        .map_err(VulkanError::from)?;

        Ok(storage)
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        let seed: u32 = seed.try_into().map_err(|_| {
            VulkanError::Message("Vulkan seed must be less than or equal to u32::MAX".to_string())
        })?;

        let seed_buffer = self.seed.try_lock().map_err(VulkanError::from)?;
        todo!()
        // let contents = seed_buffer.contents();
        // unsafe {
        //     std::ptr::copy([seed].as_ptr(), contents as *mut u32, 1);
        // }
        // seed_buffer.did_modify_range(metal::NSRange::new(0, 4));

        // Ok(())
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
