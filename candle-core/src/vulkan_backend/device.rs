#![allow(dead_code)]

use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape, VulkanError, VulkanStorage};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryCommandBufferAbstract,
};
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{
    AllocationCreateInfo, DeviceLayout, MemoryTypeFilter, StandardMemoryAllocator,
};
use vulkano::memory::MemoryType;
use vulkano::sync::GpuFuture;
use vulkano::{DeviceSize, VulkanLibrary};

#[derive(Clone, Debug)]
pub struct VulkanDevice {
    ordinal: usize,
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    buffer_allocator: Arc<SubbufferAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    //    pools: Arc<Mutex<HashMap<(DType), Arc<Mutex<MemoryPool>>>>>,
    //    zero_init_pipeline: Arc<ComputePipeline>,
}

// #[derive(Debug)]
// struct MemoryPool {
//     /// Group of buffer arenas for this dtype/memory type combination
//     arenas: Vec<BufferArena>,
//     /// Current allocation strategy for this pool
//     strategy: AllocationStrategy,
//     dtype: DType,
// }
//
// impl MemoryPool {
//     fn new(
//         dtype: DType,
//         strategy: AllocationStrategy,
//         allocator: Arc<StandardMemoryAllocator>,
//         device: Arc<Device>,
//     ) -> Result<Self> {
//         let initial_size = match &strategy {
//             AllocationStrategy::ExponentialGrowth { next_size, .. } => *next_size,
//             AllocationStrategy::FixedSize { size } => *size,
//         };
//
//         let element_size = dtype.size_in_bytes() as u64;
//         let logical_size = initial_size * element_size;
//
//         let arena = BufferArena::new(logical_size, dtype, device, allocator)?;
//
//         Ok(Self {
//             arenas: vec![arena],
//             strategy,
//             dtype,
//         })
//     }
// }
//
// #[derive(Debug)]
// struct BufferArena {
//     /// Parent buffer allocation
//     buffer: Arc<Buffer>,
//     /// Free regions tracked as (offset, size)
//     free_regions: Vec<(u64, u64)>,
//     /// Alignment requirement for this arena
//     alignment: u64,
//     /// Total size of the buffer
//     total_size: u64,
// }
//
// impl BufferArena {
//     fn new(
//         logical_size: u64,
//         dtype: DType,
//         device: Arc<Device>,
//         allocator: Arc<StandardMemoryAllocator>,
//     ) -> Result<Self> {
//         // Get physical device properties
//         let properties = device.physical_device().properties();
//
//         // Determine alignment requirements based on dtype and buffer usage
//         let element_size = dtype.size_in_bytes() as u64;
//         let alignment = properties
//             .min_storage_buffer_offset_alignment
//             .as_devicesize()
//             .max(element_size);
//
//         // Calculate aligned size that meets both dtype and Vulkan requirements
//         let aligned_size = ((logical_size + alignment - 1) / alignment) * alignment;
//
//         // Create device layout with proper alignment
//         let layout =
//             vulkano::memory::allocator::DeviceLayout::from_size_alignment(aligned_size, alignment)
//                 .ok_or(VulkanError::Message("size/alignment invalid".to_string()))?;
//
//         // Create buffer with storage buffer usage
//         let buffer = Buffer::new(
//             allocator.clone(),
//             BufferCreateInfo {
//                 usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
//                 ..Default::default()
//             },
//             AllocationCreateInfo {
//                 memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
//                 ..Default::default()
//             },
//             layout,
//         )
//         .map_err(VulkanError::ValidatedAllocateBufferError)?;
//
//         // Get final allocated size from memory
//         let total_size = buffer.size();
//
//         Ok(Self {
//             buffer,
//             free_regions: vec![(0, total_size)],
//             alignment,
//             total_size,
//         })
//
//     }
//
//     fn try_allocate(&mut self, size: u64) -> Option<Subbuffer<[u8]>> {
//         // Align requested size to arena's alignment
//         let aligned_size = ((size + self.alignment - 1) / self.alignment) * self.alignment;
//
//         // First-fit allocation
//         for i in 0..self.free_regions.len() {
//             let (offset, region_size) = self.free_regions[i];
//
//             if region_size >= aligned_size {
//                 // Split the region
//                 let remaining = region_size - aligned_size;
//
//                 if remaining == 0 {
//                     // Remove current region
//                     self.free_regions.remove(i);
//                 } else {
//                     // Add remaining space back if any
//                     self.free_regions[i] = (offset + aligned_size, remaining);
//                 }
//
//                 // Create subbuffer for the allocated region
//                 return Some(self.buffer.slice(offset..offset + aligned_size));
//             }
//         }
//
//         None
//     }
//
//     fn deallocate(&mut self, offset: u64, size: u64) {
//         // Merge with adjacent regions
//         let mut new_region = (offset, size);
//
//         // Check previous regions
//         if let Some((prev_offset, prev_size)) = self.free_regions.iter()
//             .find(|&&(o, s)| o + s == new_region.0)
//         {
//             new_region.0 = *prev_offset;
//             new_region.1 += prev_size;
//             self.free_regions.retain(|r| r != &(*prev_offset, *prev_size));
//         }
//
//         // Check next regions
//         if let Some((next_offset, next_size)) = self.free_regions.iter()
//             .find(|&&(o, _)| o == new_region.0 + new_region.1)
//         {
//             new_region.1 += next_size;
//             self.free_regions.retain(|r| r != &(*next_offset, *next_size));
//         }
//
//         self.free_regions.push_back(new_region);
//     }
// }
//
// #[derive(Debug, Clone)]
// enum AllocationStrategy {
//     ExponentialGrowth { next_size: u64, max_size: u64 },
//     FixedSize { size: u64 },
// }

macro_rules! fail {
    () => {
        unimplemented!("vulkan support is incomplete, this function is not yet implemented")
    };
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

    pub(crate) fn to_cpu<T: BufferContents + Clone + Copy + Send>(
        &self,
        buffer: Arc<Subbuffer<[u32]>>,
    ) -> Result<Vec<T>> {
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
                (*buffer).clone(),
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
        let result: Vec<T> = cpu_buffer_content.iter().copied().collect();
        Ok(result)
    }

    // /// Gets or creates a memory pool for the specified dtype and memory type
    // fn get_pool(&self, dtype: DType, memory_type: MemoryType) -> Result<Arc<Mutex<MemoryPool>>> {
    //     let mut pools = self
    //         .pools
    //         .lock()
    //         .map_err(|err| VulkanError::Message(err.to_string()))?;
    //     match pools.entry((dtype/*, memory_type*/)) {
    //         Entry::Occupied(entry) => Ok(entry.get().clone()),
    //         Entry::Vacant(vacant) => {
    //             let pool = MemoryPool::new(
    //                 dtype,
    //                 AllocationStrategy::ExponentialGrowth {
    //                     next_size: 64 * 1024 * 1024,      // 64MB initial
    //                     max_size: 1 * 1024 * 1024 * 1024, // 1GB max
    //                 },
    //                 self.memory_allocator.clone(),
    //                 self.device.clone(),
    //             )?;
    //             let arc = Arc::new(Mutex::new(pool));
    //             vacant.insert(arc.clone());
    //             Ok(arc)
    //         }
    //     }
    // }

    fn allocate_subbuffer_raw<T: BufferContents + ?Sized>(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        size: usize,
    ) -> Result<Subbuffer<T>> {
        let buffer = Buffer::new_unsized(
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

        Ok(buffer)
    }

    // fn allocate_filled_subbuffer_raw(
    //     command_buffer_allocator: &Arc<StandardCommandBufferAllocator>,
    //     memory_allocator: &Arc<StandardMemoryAllocator>,
    //     queue: Arc<Queue>,
    //     size: usize,
    //     value: u32,
    // ) -> Result<Subbuffer<[u32]>> {
    //     let buffer = Self::allocate_subbuffer_raw(memory_allocator, size)?;
    //
    //     // Fill the buffer with the specified value
    //     let mut builder = AutoCommandBufferBuilder::primary(
    //         command_buffer_allocator,
    //         queue.queue_family_index(),
    //         CommandBufferUsage::OneTimeSubmit,
    //     )
    //     .map_err(VulkanError::ValidatedVulkanError)?;
    //
    //     builder
    //         .fill_buffer(buffer.clone(), value)
    //         .map_err(VulkanError::ValidationError)?;
    //     let command_buffer = builder.build().map_err(VulkanError::ValidatedVulkanError)?;
    //
    //     let future = command_buffer
    //         .execute(queue.clone())
    //         .map_err(VulkanError::CommandBufferExecError)?;
    //     future
    //         .then_signal_fence_and_flush()
    //         .unwrap()
    //         .wait(None)
    //         .map_err(VulkanError::ValidatedVulkanError)?;
    //
    //     Ok(buffer)
    // }
    //
    // fn allocate_filled_subbuffer(
    //     &self,
    //     queue: Arc<Queue>,
    //     size: usize,
    //     value: u32,
    // ) -> Result<Subbuffer<[u32]>> {
    //     Self::allocate_filled_subbuffer_raw(
    //         &self.command_buffer_allocator,
    //         &self.memory_allocator,
    //         queue,
    //         size,
    //         value,
    //     )
    // }
    //
    // fn allocate_filled_buffer(&self, size: usize, value: u32) -> Result<Arc<Buffer>> {
    //     let buffer = self.allocate_filled_subbuffer(self.queue.clone(), size, value)?;
    //     Ok(buffer.buffer().clone())
    // }
}

impl crate::backend::BackendDevice for VulkanDevice {
    type Storage = VulkanStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let library = VulkanLibrary::new().map_err(|err| VulkanError::from(err))?;
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
        let queue = queues
            .next()
            .ok_or(VulkanError::Message("no queues found".to_string()))?;

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let buffer_allocator = Arc::new(SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                arena_size: 64 * 1024 * 1024,
                buffer_usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::TRANSFER_SRC,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE, // XXX actually the default
                ..Default::default()
            },
        ));

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));

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
            command_buffer_allocator,
            //            pools: Arc::new(Mutex::new(HashMap::new())),
            //            zero_init_pipeline: Arc::new(zero_init_pipeline),
        })
    }

    //     fn new(ordinal: usize) -> Result<Self> {
    //         // Create a Vulkan instance
    //         let library = VulkanLibrary::new().map_err(VulkanError::LoadingError)?;
    //         let instance = Instance::new(
    //             library,
    //             InstanceCreateInfo {
    //                 flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
    //                 ..Default::default()
    //             },
    //         )
    //         .map_err(VulkanError::ValidatedVulkanError)?;
    //
    //         // Select the physical device
    //         let physical_device = Self::select_physical_device(&instance, ordinal)?;
    //
    //         // Initialize Vulkan resources (queues, command buffers, etc.)
    //         let device_extensions = DeviceExtensions {
    //             khr_storage_buffer_storage_class: true,
    //             ..DeviceExtensions::empty()
    //         };
    //         let (device, mut queues) = Device::new(
    //             physical_device.clone(),
    //             DeviceCreateInfo {
    //                 enabled_extensions: device_extensions,
    //                 queue_create_infos: vec![QueueCreateInfo {
    //                     queue_family_index: 0,
    //                     ..Default::default()
    //                 }],
    //                 ..Default::default()
    //             },
    //         )
    //         .map_err(VulkanError::ValidatedVulkanError)?;
    //
    //         // Create command buffer allocator
    //         let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
    //             device.clone(),
    //             Default::default(),
    //         ));
    //
    //         let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    //
    //         let descriptor_set_allocator =
    //             StandardDescriptorSetAllocator::new(device.clone(), Default::default());
    //
    //         let queue = queues.next().unwrap();
    //         let kernels = Arc::new(Kernels::new(device.clone()));
    //         let seed: Subbuffer<[u32]> = Self::allocate_filled_subbuffer_raw(
    //             &command_buffer_allocator,
    //             &memory_allocator,
    //             queue.clone(),
    //             1,
    //             299792458,
    //         )?;
    //
    //         Ok(VulkanDevice {
    //             gpu_id: ordinal,
    //             device,
    //             queue,
    //             kernels,
    //             seed: Arc::new(Mutex::new(seed)),
    //             command_buffer_allocator,
    //             memory_allocator,
    //             descriptor_set_allocator: Arc::new(descriptor_set_allocator),
    //         })
    //     }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Vulkan {
            gpu_id: self.ordinal,
        }
    }

    fn same_device(&self, other: &Self) -> bool {
        self.ordinal == other.ordinal
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let buffer = self
            .buffer_allocator
            .allocate(
                DeviceLayout::from_size_alignment(
                    ((((shape.elem_count() * dtype.size_in_bytes()) + 3) / 4) * 4) as DeviceSize,
                    dtype.size_in_bytes() as DeviceSize, // XXX is this correct?
                )
                .ok_or(VulkanError::Message("invalid layout".to_string()))?,
            )
            .map_err(VulkanError::MemoryAllocatorError)?
            .reinterpret::<[u32]>();

        // Fill the buffer with the specified value
        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(VulkanError::ValidatedVulkanError)?;

        builder
            .fill_buffer(buffer.clone().reinterpret::<[u32]>(), 0)
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

        Ok(VulkanStorage::new(
            buffer,
            self.clone(),
            shape.elem_count(),
            dtype,
        ))
    }

    fn ones_impl(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        fail!()
    }

    unsafe fn alloc_uninit(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        fail!()
    }

    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> Result<Self::Storage> {
        fail!()
    }

    fn storage_from_cpu_storage(&self, _: &CpuStorage) -> Result<Self::Storage> {
        fail!()
    }

    fn storage_from_cpu_storage_owned(&self, _: CpuStorage) -> Result<Self::Storage> {
        fail!()
    }

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        fail!()
    }

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        fail!()
    }

    fn set_seed(&self, _: u64) -> Result<()> {
        fail!()
    }

    fn synchronize(&self) -> Result<()> {
        fail!()
    }
}
