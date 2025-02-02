#![allow(dead_code)]

mod device;
pub use device::VulkanDevice;
mod storage;
pub use storage::VulkanStorage;

use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};

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
