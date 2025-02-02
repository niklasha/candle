#![allow(dead_code)]

mod device;
mod storage;

use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};

#[derive(Debug, Clone)]
pub struct VulkanDevice {
    pub(crate) gpu_id: usize,
}

#[derive(Debug)]
pub struct VulkanStorage;

#[derive(thiserror::Error, Debug)]
pub enum VulkanError {
    #[error("{0}")]
    Message(String),
}

impl From<String> for VulkanError {
    fn from(e: String) -> Self {
        VulkanError::Message(e)
    }
}

macro_rules! fail {
    () => {
        unimplemented!("vulkan support is incomplete, this function is not yet implemented")
    };
}

impl crate::backend::BackendStorage for VulkanStorage {
    type Device = VulkanDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        // Simple clone of storage (could be implemented as buffer copy later)
        Ok(Self)
    }

    fn dtype(&self) -> DType {
        fail!()
    }

    fn device(&self) -> &Self::Device {
        &VulkanDevice
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        fail!()
    }

    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self> {
        fail!()
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        fail!()
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        fail!()
    }

    fn reduce_op(&self, _: ReduceOp, _: &Layout, _: &[usize]) -> Result<Self> {
        fail!()
    }

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        fail!()
    }

    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
        fail!()
    }

    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self> {
        fail!()
    }

    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        fail!()
    }

    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        fail!()
    }

    fn conv1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        fail!()
    }

    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        fail!()
    }

    fn conv2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        fail!()
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        fail!()
    }

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self> {
        fail!()
    }

    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        fail!()
    }

    fn scatter_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        fail!()
    }

    fn index_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        fail!()
    }

    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        fail!()
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()> {
        fail!()
    }

    fn copy2d(
        &self,
        _: &mut Self,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
    ) -> Result<()> {
        fail!()
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        fail!()
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        fail!()
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        fail!()
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        fail!()
    }
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
