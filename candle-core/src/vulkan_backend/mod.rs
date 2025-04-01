#![allow(dead_code)]

mod device;

use candle_vulkan_kernels::VulkanKernelError;
pub use device::VulkanDevice;
use std::sync::{PoisonError, TryLockError};

mod storage;
pub use storage::VulkanStorage;

/// Simple way to catch lock error without
/// depending on T
#[derive(thiserror::Error, Debug)]
pub enum LockError {
    #[error("{0}")]
    Poisoned(String),
    #[error("Would block")]
    WouldBlock,
}

impl<T> From<TryLockError<T>> for VulkanError {
    fn from(value: TryLockError<T>) -> Self {
        match value {
            TryLockError::Poisoned(p) => VulkanError::LockError(LockError::Poisoned(p.to_string())),
            TryLockError::WouldBlock => VulkanError::LockError(LockError::WouldBlock),
        }
    }
}

impl<T> From<PoisonError<T>> for VulkanError {
    fn from(p: PoisonError<T>) -> Self {
        VulkanError::LockError(LockError::Poisoned(p.to_string()))
    }
}

#[derive(thiserror::Error, Debug)]
pub enum VulkanError {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    LoadingError(#[from] vulkano::LoadingError),
    #[error("{0:?}")]
    LockError(#[from] LockError),
    #[error("{0:?}")]
    ValidatedVulkanError(#[from] vulkano::Validated<vulkano::VulkanError>),
    #[error("{0:?}")]
    VulkanError(#[from] vulkano::VulkanError),
    #[error("{0:?}")]
    ValidationError(#[from] Box<vulkano::ValidationError>),
    #[error("{0:?}")]
    ValidatedAllocateBufferError(#[from] vulkano::Validated<vulkano::buffer::AllocateBufferError>),
    #[error("{0:?}")]
    MemoryAllocatorError(#[from] vulkano::memory::allocator::MemoryAllocatorError),
    #[error("{0:?}")]
    CommandBufferExecError(#[from] vulkano::command_buffer::CommandBufferExecError),
    #[error("{0:?}")]
    IntoPipelineLayoutCreateInfoError(
        #[from] vulkano::pipeline::layout::IntoPipelineLayoutCreateInfoError,
    ),
}

impl From<String> for VulkanError {
    fn from(e: String) -> Self {
        VulkanError::Message(e)
    }
}

impl From<&str> for VulkanError {
    fn from(e: &str) -> Self {
        VulkanError::Message(e.to_string())
    }
}

impl From<VulkanKernelError> for VulkanError {
    fn from(e: VulkanKernelError) -> Self {
        // XXX map each VulkanKernelError enum to a VulkanError enum
        VulkanError::Message(e.to_string())
    }
}
