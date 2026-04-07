pub mod checkpoint;
pub mod cli;
pub mod logger;

pub const WINDOW_TITLE: &str = "UFO 50";
pub const OBS_W: u32 = 128;
pub const OBS_H: u32 = 128;

/// Pre-load `torch_cuda.dll` so tch's lazy CUDA init succeeds. No-op on failure.
/// Windows-only — required because tch loads CUDA libraries lazily and Windows
/// won't search the libtorch directory unless something forces it onto the loader path.
#[cfg(target_os = "windows")]
pub fn preload_torch_cuda() {
    unsafe {
        let _ = windows::Win32::System::LibraryLoader::LoadLibraryA(windows::core::s!(
            "torch_cuda.dll"
        ));
    }
}
