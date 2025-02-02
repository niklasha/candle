#![allow(unused)]
use super::GgmlDType;
use crate::dummy_vulkan_backend::{VulkanDevice, VulkanStorage};
use crate::{Error, Result};

pub struct QVulkanStorage {
    dtype: GgmlDType,
    device: VulkanDevice,
}

impl QVulkanStorage {
    pub fn zeros(_: &VulkanDevice, _: usize, _: GgmlDType) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }

    pub fn dequantize(&self, _elem_count: usize) -> Result<VulkanStorage> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn quantize(&mut self, _src: &VulkanStorage) -> Result<()> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        0
    }

    pub fn fwd(
        &self,
        _self_shape: &crate::Shape,
        _storage: &VulkanStorage,
        _layout: &crate::Layout,
    ) -> Result<(VulkanStorage, crate::Shape)> {
        Err(Error::NotCompiledWithVulkanSupport)
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    _device: &VulkanDevice,
    _data: &[T],
) -> Result<super::QStorage> {
    Err(Error::NotCompiledWithVulkanSupport)
}
