//! Buffers. `gpu-allocator` owns the memory; this is the handle plus upload/download.

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;

use crate::device::Gpu;
use crate::error::GpuError;

/// What the buffer is for. Determines both the Vulkan usage flags and where the memory
/// lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Usage {
    /// Device-local storage buffer: kernel input/output.
    Storage,
    /// Device-local uniform buffer: small read-only parameters.
    Uniform,
    /// Host-visible staging buffer: uploads, downloads and readback.
    Staging,
}

impl Usage {
    fn flags(self) -> vk::BufferUsageFlags {
        let copy = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        match self {
            Self::Storage => vk::BufferUsageFlags::STORAGE_BUFFER | copy,
            Self::Uniform => vk::BufferUsageFlags::UNIFORM_BUFFER | copy,
            Self::Staging => copy,
        }
    }

    fn location(self) -> MemoryLocation {
        match self {
            Self::Storage | Self::Uniform => MemoryLocation::GpuOnly,
            Self::Staging => MemoryLocation::CpuToGpu,
        }
    }
}

/// A buffer and its allocation. Freed on drop.
pub struct Buffer<'gpu> {
    gpu: &'gpu Gpu,
    handle: vk::Buffer,
    allocation: Option<Allocation>,
    /// Bytes actually allocated: `len.max(4)`, then whatever `gpu-allocator` padded it to.
    size: u64,
    /// Bytes the caller asked for. `download` returns exactly this many, never the padding
    /// (review `docs/reviews/M4.md` S-12).
    len: u64,
    usage: Usage,
}

impl std::fmt::Debug for Buffer<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("len", &self.len)
            .field("size", &self.size)
            .field("usage", &self.usage)
            .finish()
    }
}

impl<'gpu> Buffer<'gpu> {
    /// Allocate `bytes` bytes.
    pub fn new(gpu: &'gpu Gpu, bytes: u64, usage: Usage) -> Result<Self, GpuError> {
        let info = vk::BufferCreateInfo::default()
            .size(bytes.max(4))
            .usage(usage.flags())
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: `info` outlives the call; the handle is destroyed exactly once in `Drop`.
        let handle = unsafe { gpu.device().create_buffer(&info, None) }?;
        // SAFETY: `handle` was just created on this device.
        let requirements = unsafe { gpu.device().get_buffer_memory_requirements(handle) };
        let allocation = gpu
            .allocator()
            .borrow_mut()
            .allocate(&AllocationCreateDesc {
                name: "es-gpu buffer",
                requirements,
                location: usage.location(),
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;
        // SAFETY: the allocation was made for exactly this buffer's requirements and is
        // bound once.
        unsafe {
            gpu.device()
                .bind_buffer_memory(handle, allocation.memory(), allocation.offset())
        }?;
        Ok(Self {
            gpu,
            handle,
            allocation: Some(allocation),
            size: bytes.max(4),
            len: bytes,
            usage,
        })
    }

    /// Allocate and fill in one step.
    pub fn from_f32(gpu: &'gpu Gpu, data: &[f32], usage: Usage) -> Result<Self, GpuError> {
        let mut buffer = Self::new(gpu, (std::mem::size_of_val(data)) as u64, usage)?;
        buffer.upload(bytemuck_f32(data))?;
        Ok(buffer)
    }

    pub fn handle(&self) -> vk::Buffer {
        self.handle
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn usage(&self) -> Usage {
        self.usage
    }

    fn mapped(&mut self) -> Option<&mut [u8]> {
        let allocation = self.allocation.as_mut()?;
        let len = allocation.size() as usize;
        let ptr = allocation.mapped_ptr()?;
        // SAFETY: `gpu-allocator` mapped this range for the lifetime of the allocation and
        // reports its length; `&mut self` makes the borrow exclusive.
        Some(unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr().cast::<u8>(), len) })
    }

    /// Write `data` at offset 0. Host-visible memory is written directly; device-local
    /// memory goes through a temporary staging buffer and a one-shot copy.
    pub fn upload(&mut self, data: &[u8]) -> Result<(), GpuError> {
        if data.len() as u64 > self.size {
            return Err(GpuError::Exhausted(format!(
                "upload of {} bytes into a {}-byte buffer",
                data.len(),
                self.size
            )));
        }
        if let Some(dst) = self.mapped() {
            dst[..data.len()].copy_from_slice(data);
            return Ok(());
        }
        let mut staging = Buffer::new(self.gpu, self.size, Usage::Staging)?;
        staging.upload(data)?;
        copy(self.gpu, staging.handle, self.handle, data.len() as u64)
    }

    /// Read the buffer back: exactly the `bytes` it was created with.
    ///
    /// The mapped slice covers the whole allocation, which `gpu-allocator` pads to the
    /// device's alignment, so the tail is trimmed here rather than handed to the caller as
    /// if it were data.
    pub fn download(&mut self) -> Result<Vec<u8>, GpuError> {
        let size = self.size;
        let mut bytes = if let Some(src) = self.mapped() {
            src.to_vec()
        } else {
            let mut staging = Buffer::new(self.gpu, size, Usage::Staging)?;
            copy(self.gpu, self.handle, staging.handle, size)?;
            staging.download()?
        };
        bytes.truncate(self.len as usize);
        Ok(bytes)
    }

    /// Read the buffer back as `f32`.
    pub fn download_f32(&mut self) -> Result<Vec<f32>, GpuError> {
        let bytes = self.download()?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}

impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        if let Some(allocation) = self.allocation.take() {
            let _ = self.gpu.allocator().borrow_mut().free(allocation);
        }
        // SAFETY: the buffer belongs to this device, nothing is in flight (every submission
        // path waits), and this is the only destroy.
        unsafe { self.gpu.device().destroy_buffer(self.handle, None) };
    }
}

fn copy(gpu: &Gpu, src: vk::Buffer, dst: vk::Buffer, size: u64) -> Result<(), GpuError> {
    gpu.one_shot(|device, cmd| {
        let region = vk::BufferCopy::default().size(size);
        // SAFETY: `cmd` is recording on `device`; both buffers are at least `size` bytes and
        // were created with TRANSFER_SRC/TRANSFER_DST.
        unsafe { device.cmd_copy_buffer(cmd, src, dst, &[region]) };
    })
}

/// `&[f32]` as bytes. Little-endian hosts only, which is every platform in spec §2.1.
fn bytemuck_f32(data: &[f32]) -> &[u8] {
    // SAFETY: `f32` has no padding and no invalid bit patterns; the byte slice covers
    // exactly the same allocation and is read-only.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) }
}
