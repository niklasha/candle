use super::VulkanError;
use crate::{DType, Result};
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::{
    device::{physical::PhysicalDevice, Device, Queue},
    instance::Instance,
};

#[derive(Clone, Debug)]
pub struct VulkanDevice {
    pub(super) gpu_id: usize,
    pub(super) device: Arc<Device>,
    pub(super) queue: Arc<Queue>,
}

impl VulkanDevice {
    pub(super) fn allocate_buffer(&self, size: usize) -> Result<Arc<Subbuffer<[u8]>>> {
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(self.device.clone()));

        let buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            (0..size).map(|_| 0u8),
        )
        .map_err(VulkanError::ValidatedAllocateBufferError)?;

        Ok(Arc::new(buffer))
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
        use vulkano::command_buffer::allocator::StandardCommandBufferAllocatorCreateInfo;

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
