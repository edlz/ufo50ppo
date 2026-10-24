pub mod checkpoint;
pub mod cli;
pub mod logger;
pub mod shutdown;

/// Forces `torch_cuda.dll` onto the loader path. tch loads CUDA libraries lazily
/// and Windows won't search the libtorch directory without this nudge.
#[cfg(windows)]
pub fn preload_torch_cuda() {
    unsafe {
        let _ = windows::Win32::System::LibraryLoader::LoadLibraryA(windows::core::s!(
            "torch_cuda.dll"
        ));
    }
}

#[cfg(not(windows))]
pub fn preload_torch_cuda() {}
