#![allow(dead_code)]

use crate::backend::BackendStorage;
use crate::{CpuStorage, CpuStorageRef, DType, Result, Shape, VulkanError, VulkanStorage};
use bytemuck::Pod;
use candle_vulkan_kernels::Kernels;
use half::{bf16, f16};
use std::io::Read;
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
use vulkano::pipeline::compute::ComputePipeline;
use vulkano::sync::GpuFuture;
use vulkano::{DeviceSize, VulkanLibrary};

#[derive(Clone, Debug)]
pub struct VulkanDevice {
    ordinal: usize,
    device: Arc<Device>,
    pub(crate) queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    buffer_allocator: Arc<Mutex<SubbufferAllocator>>,
    pub(crate) kernels: Arc<Kernels>,
    pub(crate) command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub(crate) descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub(crate) global_seed: Arc<Mutex<u64>>,
}

enum DataSource<'a, T> {
    Slice(&'a [T]),
    Fill { value: T, count: usize },
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

    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    pub fn device(&self) -> Arc<Device> {
        self.device.clone()
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
                buffer.size() / size_of::<T>() as DeviceSize,
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
        let buffer = self.allocate(buffer_size, alignment)?;

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
    fn allocate(&self, buffer_size: usize, alignment: usize) -> Result<Option<Subbuffer<[u8]>>> {
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

impl crate::backend::BackendDevice for VulkanDevice {
    type Storage = VulkanStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let library = VulkanLibrary::new().map_err(VulkanError::from)?;
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
            uniform_and_storage_buffer8_bit_access: true,
            storage_buffer16_bit_access: true,
            shader_int64: true,
            shader_float16: true,
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

        let kernels = Arc::new(Kernels::new(device.clone()).map_err(VulkanError::from)?);

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

        Ok(Self {
            ordinal,
            device,
            queue,
            memory_allocator,
            buffer_allocator,
            kernels,
            command_buffer_allocator,
            descriptor_set_allocator,
            global_seed: Arc::new(Mutex::new(0)),
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
        let buffer = self.allocate(count * type_size, type_size)?; // XXX alignment might need revisiting
        Ok(VulkanStorage::new(buffer, self.clone(), count, dtype))
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        match T::cpu_storage_ref(s) {
            CpuStorageRef::U8(storage) => self.allocate_with_data(T::DTYPE, storage),
            CpuStorageRef::U32(storage) => self.allocate_with_data(T::DTYPE, storage),
            CpuStorageRef::I64(storage) => self.allocate_with_data(T::DTYPE, storage),
            CpuStorageRef::BF16(storage) => self.allocate_with_data(T::DTYPE, storage),
            CpuStorageRef::F16(storage) => self.allocate_with_data(T::DTYPE, storage),
            CpuStorageRef::F32(storage) => self.allocate_with_data(T::DTYPE, storage),
            CpuStorageRef::F64(storage) => self.allocate_with_data(T::DTYPE, storage),
        }
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

    fn rand_uniform(
        &self,
        shape: &Shape,
        dtype: DType,
        low: f64,
        high: f64,
    ) -> Result<Self::Storage> {
        let suffix = match dtype {
            DType::F32 => "f32",
            dtype => crate::bail!("rand_uniform not implemented for {dtype:?}"),
        };
        let key = format!("rand_uniform_{}", suffix);
        let pipeline = self
            .kernels()
            .load_pipeline(self.device(), &key)
            .map_err(VulkanError::from)?;

        let num_elements = shape.elem_count();
        let storage = unsafe { self.alloc_uninit(shape, dtype) }?;

        let seed = {
            let lock = self.global_seed.lock().map_err(VulkanError::from)?;
            *lock
        };
        storage.random_impl(shape, &pipeline, dtype, seed, low, high)?;

        Ok(storage)
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        stddev: f64,
    ) -> Result<Self::Storage> {
        let suffix = match dtype {
            DType::F32 => "f32",
            dtype => crate::bail!("rand_normal not implemented for {dtype:?}"),
        };
        let key = format!("rand_normal_{}", suffix);
        let pipeline = self
            .kernels()
            .load_pipeline(self.device(), &key)
            .map_err(VulkanError::from)?;

        let num_elements = shape.elem_count();
        let storage = unsafe { self.alloc_uninit(shape, dtype) }?;

        let seed = {
            let lock = self.global_seed.lock().map_err(VulkanError::from)?;
            *lock
        };
        storage.random_impl(shape, &pipeline, dtype, seed, mean, stddev)?;

        Ok(storage)
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        // let seed: u32 = seed.try_into().map_err(|_| {
        //     VulkanError::Message("Vulkan seed must be less than or equal to u32::MAX".to_string())
        // })?;

        let mut global_seed = self.global_seed.lock().map_err(VulkanError::from)?;
        *global_seed = seed;
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        crate::bail!("synchronize not implemented");
        Ok(())
    }
}
