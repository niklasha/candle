use crate::error::Result;
use std::sync::Arc;
use vulkano::{device::{physical::PhysicalDevice, Device}, instance::Instance};

use super::VulkanError;

#[derive(Clone, Debug)]
pub struct VulkanDevice {
    pub(super) gpu_id: usize,
    pub(super) device: Arc<Device>,
    pub(super) queue: Arc<vulkano::device::Queue>,
}

impl VulkanDevice {
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
        use vulkano::command_buffer::allocator::{
            StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
        };
        use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};

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
