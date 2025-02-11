#![allow(dead_code)]

mod device;

pub use device::VulkanDevice;
use std::fmt;
mod storage;
use crate::op::CmpOp;
pub use storage::VulkanStorage;

#[derive(thiserror::Error, Debug)]
pub enum VulkanError {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    LoadingError(#[from] vulkano::LoadingError),
    #[error(transparent)]
    ValidatedVulkanError(#[from] vulkano::Validated<vulkano::VulkanError>),
    #[error(transparent)]
    VulkanError(#[from] vulkano::VulkanError),
    #[error(transparent)]
    ValidationError(#[from] Box<vulkano::ValidationError>),
    #[error(transparent)]
    ValidatedAllocateBufferError(#[from] vulkano::Validated<vulkano::buffer::AllocateBufferError>),
    #[error(transparent)]
    MemoryAllocatorError(#[from] vulkano::memory::allocator::MemoryAllocatorError),
    #[error(transparent)]
    CommandBufferExecError(#[from] vulkano::command_buffer::CommandBufferExecError),
    #[error(transparent)]
    IntoPipelineLayoutCreateInfoError(
        #[from] vulkano::pipeline::layout::IntoPipelineLayoutCreateInfoError,
    ),
}

impl From<String> for VulkanError {
    fn from(e: String) -> Self {
        VulkanError::Message(e)
    }
}
