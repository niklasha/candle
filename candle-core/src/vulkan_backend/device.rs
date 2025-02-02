#![allow(dead_code)]
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::vulkan_backend::storage::VulkanStorage;
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};

#[derive(Debug, Clone)]
pub struct VulkanDevice {
    pub(crate) gpu_id: usize,
}

impl VulkanDevice {
    pub(crate) fn new(p0: usize) -> Result<Self> {
        todo!()
    }
}

macro_rules! fail {
    () => {
        unimplemented!("vulkan support is incomplete, this function is not yet implemented")
    };
}

impl crate::backend::BackendDevice for VulkanDevice {
    type Storage = VulkanStorage;

    fn new(gpu_id: usize) -> Result<Self> {
        Ok(Self { gpu_id })
    }

    fn set_seed(&self, _: u64) -> Result<()> {
        fail!()
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Vulkan {
            gpu_id: self.gpu_id,
        }
    }

    fn same_device(&self, other: &Self) -> bool {
        self.gpu_id == other.gpu_id
    }

    fn zeros_impl(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        fail!()
    }

    fn ones_impl(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        fail!()
    }

    unsafe fn alloc_uninit(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
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

    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> Result<Self::Storage> {
        fail!()
    }

    fn synchronize(&self) -> Result<()> {
        fail!()
    }
}
