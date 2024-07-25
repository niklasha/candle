use super::VulkanError;
use crate::{DType, Result};
use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage,
        /* CopyBufferInfo, */ PrimaryCommandBufferAbstract,
    },
    device::{physical::PhysicalDevice, Device, Queue},
    instance::Instance,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    sync::GpuFuture,
};

#[derive(Clone, Debug)]
pub struct VulkanDevice {
    pub(super) gpu_id: usize,
    pub(super) device: Arc<Device>,
    pub(super) queue: Arc<Queue>,
}

impl VulkanDevice {
    // pub fn new_buffer_with_data<T: BufferContents>(&self, data: &[T]) -> Result<Arc<Buffer>> {
    //     let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(self.device.clone()));
    //     let allocator_create_info = StandardCommandBufferAllocatorCreateInfo::default();
    //     let command_buffer_allocator =
    //         StandardCommandBufferAllocator::new(self.device.clone(), allocator_create_info);

    //     let buffer_size = data.len() * std::mem::size_of::<T>();

    //     let temporary_accessible_buffer = Buffer::from_iter(
    //         memory_allocator.clone(),
    //         BufferCreateInfo {
    //             // Specify that this buffer will be used as a transfer source.
    //             usage: BufferUsage::TRANSFER_SRC,
    //             ..Default::default()
    //         },
    //         AllocationCreateInfo {
    //             // Specify use for upload to the device.
    //             memory_type_filter: MemoryTypeFilter::PREFER_HOST
    //             | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
    //             ..Default::default()
    //         },
    //         data,
    //     ).map_err(VulkanError::ValidatedAllocateBufferError)?;

    //     let device_local_buffer = Buffer::new_slice::<T>(
    //         memory_allocator.clone(),
    //         BufferCreateInfo {
    //             // Specify use as a storage buffer and transfer destination.
    //             usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
    //             ..Default::default()
    //         },
    //         AllocationCreateInfo {
    //             // Specify use by the device only.
    //             memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
    //             ..Default::default()
    //         },
    //         data.len() as DeviceSize,
    //     )?;

    //     let mut cbb = AutoCommandBufferBuilder::primary(
    //         &command_buffer_allocator,
    //         self.queue.queue_family_index(),
    //         CommandBufferUsage::OneTimeSubmit,
    //     ).map_err(VulkanError::ValidatedVulkanError)?;
    //     cbb.copy_buffer(CopyBufferInfo::buffers(
    //         temporary_accessible_buffer,
    //         device_local_buffer.clone(),
    //     ))?;
    //     let cb = cbb.build()?;

    //     // Execute the copy command and wait for completion before proceeding.
    //     cb.execute(self.queue.clone())?
    //         .then_signal_fence_and_flush()?
    //         .wait(None /* timeout */)?;

    //     Ok(device_local_buffer.buffer().clone())
    // }

    pub(super) fn allocate_buffer(&self, size: usize, value: u32) -> Result<Arc<Buffer>> {
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(self.device.clone()));
        let allocator_create_info = StandardCommandBufferAllocatorCreateInfo::default();
        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(self.device.clone(), allocator_create_info);

        let buffer = Buffer::new_unsized::<[u32]>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            size as u64,
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
