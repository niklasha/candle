use super::VulkanError;
use crate::{DType, Result};
use bytemuck::Pod;
use std::sync::Arc;
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryCommandBufferAbstract,
    },
    device::{physical::PhysicalDevice, Device, Queue},
    instance::Instance,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    sync::GpuFuture,
    DeviceSize,
};

#[derive(Clone, Debug)]
pub struct VulkanDevice {
    pub(super) gpu_id: usize,
    pub(super) device: Arc<Device>,
    pub(super) queue: Arc<Queue>,
}

impl VulkanDevice {
    pub(super) fn new_buffer_with_data<T>(&self, data: &[T]) -> Result<Arc<Buffer>>
    where
        T: BufferContents + Pod + Sync + Send + Clone,
    {
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(self.device.clone()));

        let cpu_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                // Specify that this buffer will be used as a transfer source.
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                // Specify use for upload to the device.
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data.iter().cloned(),
        )
        .map_err(VulkanError::ValidatedAllocateBufferError)?;

        let gpu_buffer = Buffer::new_slice::<T>(
            memory_allocator.clone(),
            BufferCreateInfo {
                // Specify use as a storage buffer and transfer destination.
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                // Specify use by the device only.
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            data.len() as DeviceSize,
        )
        .map_err(VulkanError::ValidatedAllocateBufferError)?;

        // Copy data from the CPU buffer to the Vulkan buffer
        let mut builder = AutoCommandBufferBuilder::primary(
            &StandardCommandBufferAllocator::new(
                self.device.clone(),
                StandardCommandBufferAllocatorCreateInfo::default(),
            ),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(VulkanError::ValidatedVulkanError)?;

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                cpu_buffer.clone(),
                gpu_buffer.clone(),
            ))
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

        Ok(gpu_buffer.buffer().clone())
    }

    pub(super) fn allocate_buffer(&self, size: usize, value: u32) -> Result<Arc<Buffer>> {
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(self.device.clone()));
        let allocator_create_info = StandardCommandBufferAllocatorCreateInfo::default();
        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(self.device.clone(), allocator_create_info);

        let buffer = Buffer::new_unsized::<[u32]>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            size as DeviceSize,
        )
        .map_err(VulkanError::ValidatedAllocateBufferError)?;

        // Fill the buffer with the specified value
        let mut builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(VulkanError::ValidatedVulkanError)?;

        builder
            .fill_buffer(buffer.clone(), value)
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

        Ok(buffer.buffer().clone())
    }

    pub(super) fn allocate_buffer_64(&self, size: usize, value: u64) -> Result<Arc<Buffer>> {
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(self.device.clone()));

        // Determine the buffer size in bytes
        let buffer_size = size * std::mem::size_of::<u64>();

        let cpu_buffer = Buffer::new_unsized::<[u8]>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            buffer_size as DeviceSize,
        )
        .map_err(VulkanError::ValidatedAllocateBufferError)?;

        // Map the buffer and fill with 64-bit values
        {
            let mut buffer_content = cpu_buffer.write().unwrap();
            for chunk in buffer_content.chunks_exact_mut(8) {
                let bytes = value.to_ne_bytes();
                chunk.copy_from_slice(&bytes);
            }
        }

        let gpu_buffer = Buffer::new_unsized::<[u8]>(
            memory_allocator.clone(),
            BufferCreateInfo {
                // Specify use as a storage buffer and transfer destination.
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                // Specify use by the device only.
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            buffer_size as DeviceSize,
        )
        .map_err(VulkanError::ValidatedAllocateBufferError)?;

        // Copy data from the CPU buffer to the Vulkan buffer
        let mut builder = AutoCommandBufferBuilder::primary(
            &StandardCommandBufferAllocator::new(
                self.device.clone(),
                StandardCommandBufferAllocatorCreateInfo::default(),
            ),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(VulkanError::ValidatedVulkanError)?;

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                cpu_buffer.clone(),
                gpu_buffer.clone(),
            ))
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

        Ok(gpu_buffer.buffer().clone())
    }

    pub(super) fn to_cpu<T: BufferContents + Clone + Copy + Send>(
        &self,
        buffer: Arc<Buffer>,
    ) -> Result<Vec<T>> {
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(self.device.clone()));

        // Create a CPU-accessible buffer
        let cpu_buffer = Buffer::new_unsized::<[T]>(
            memory_allocator.clone(),
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
            &StandardCommandBufferAllocator::new(
                self.device.clone(),
                StandardCommandBufferAllocatorCreateInfo::default(),
            ),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(VulkanError::ValidatedVulkanError)?;

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                Subbuffer::new(buffer),
                cpu_buffer.clone(),
            ))
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
        let result: Vec<T> = cpu_buffer_content.iter().map(|&v| v.clone()).collect();
        Ok(result)
    }

    pub(super) fn select_physical_device(
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

    pub fn create_command_buffer(
        &self,
    ) -> Result<Arc<vulkano::command_buffer::PrimaryAutoCommandBuffer>> {
        let queue_family = self.queue.queue_family_index();
        let allocator_create_info = StandardCommandBufferAllocatorCreateInfo::default();
        let allocator =
            StandardCommandBufferAllocator::new(self.device.clone(), allocator_create_info);
        let builder = AutoCommandBufferBuilder::primary(
            &allocator,
            queue_family,
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(VulkanError::ValidatedVulkanError)?;

        let command_buffer = builder.build().map_err(VulkanError::ValidatedVulkanError)?;
        Ok(command_buffer)
    }

    pub fn allocate_memory(&self, size: u64) -> Result<vulkano::memory::DeviceMemory> {
        use vulkano::memory::{DeviceMemory, MemoryAllocateInfo};

        let allocate_info = MemoryAllocateInfo {
            allocation_size: size,
            memory_type_index: 0,
            ..Default::default()
        };

        let memory = DeviceMemory::allocate(self.device.clone(), allocate_info)
            .map_err(VulkanError::ValidatedVulkanError)?;

        Ok(memory)
    }
}

impl DType {
    pub fn size(&self) -> usize {
        match self {
            DType::F32 => std::mem::size_of::<f32>(),
            DType::U8 => std::mem::size_of::<u8>(),
            DType::U32 => std::mem::size_of::<u32>(),
            DType::I64 => std::mem::size_of::<i64>(),
            DType::F64 => std::mem::size_of::<f64>(),
            DType::BF16 => std::mem::size_of::<u16>(),
            DType::F16 => std::mem::size_of::<u16>(),
        }
    }
}
