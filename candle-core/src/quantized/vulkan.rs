use super::{GgmlDType, QStorage};
use crate::backend::BackendStorage;
use crate::{DType, Result, Shape, VulkanDevice, VulkanStorage};
use std::sync::Arc;

pub struct QVulkanStorage {
    dtype: GgmlDType,
    device: VulkanDevice,
    //    buffer: Arc<Buffer>,
}

impl QVulkanStorage {
    pub fn zeros(device: &VulkanDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size = elem_count * dtype.type_size() / dtype.block_size();
        unimplemented!()
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }

    //    pub fn buffer(&self) -> &Buffer {
    //        &self.buffer
    //    }

    pub fn dequantize(&self, elem_count: usize) -> Result<VulkanStorage> {
        unimplemented!()
    }

    pub fn quantize(&mut self, src: &VulkanStorage) -> Result<()> {
        unimplemented!()
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        unimplemented!()
    }

    pub fn fwd(
        &self,
        self_shape: &Shape,
        storage: &VulkanStorage,
        layout: &crate::Layout,
    ) -> Result<(VulkanStorage, Shape)> {
        unimplemented!()
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &VulkanDevice,
    data: &[T],
) -> Result<QStorage> {
    unimplemented!()
}

// impl From<GgmlDType> for candle_vulkan_kernels::GgmlDType {
//     fn from(value: GgmlDType) -> Self {
//         match value {
//             GgmlDType::Q4_0 => candle_vulkan_kernels::GgmlDType::Q4_0,
//             GgmlDType::Q4_1 => candle_vulkan_kernels::GgmlDType::Q4_1,
//             GgmlDType::Q5_0 => candle_vulkan_kernels::GgmlDType::Q5_0,
//             GgmlDType::Q5_1 => candle_vulkan_kernels::GgmlDType::Q5_1,
//             GgmlDType::Q8_0 => candle_vulkan_kernels::GgmlDType::Q8_0,
//             GgmlDType::Q8_1 => candle_vulkan_kernels::GgmlDType::Q8_1,
//             GgmlDType::Q2K => candle_vulkan_kernels::GgmlDType::Q2K,
//             GgmlDType::Q3K => candle_vulkan_kernels::GgmlDType::Q3K,
//             GgmlDType::Q4K => candle_vulkan_kernels::GgmlDType::Q4K,
//             GgmlDType::Q5K => candle_vulkan_kernels::GgmlDType::Q5K,
//             GgmlDType::Q6K => candle_vulkan_kernels::GgmlDType::Q6K,
//             GgmlDType::Q8K => candle_vulkan_kernels::GgmlDType::Q8K,
//             GgmlDType::F16 => candle_vulkan_kernels::GgmlDType::F16,
//             GgmlDType::F32 => candle_vulkan_kernels::GgmlDType::F32,
//         }
//     }
// }
