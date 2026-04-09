// Reads Ninpek's score and lives from process memory via pointer chains.
// Both fields live on the same Game Maker object — the two chains share their base and
// first 4 offsets before diverging at offset 5 (0xD0 = score, 0x4C0 = lives) and then
// converging again to a final f64 (Game Maker RValue double).
//
//   "ufo50.exe"+0x00742230 -> 0x488 -> 0x18 -> 0x8 -> 0x10 -> [0xD0 | 0x4C0] -> 0x0 -> 0x8 -> 0x0

use crate::platform::win32::find_windows_by_title;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::System::Diagnostics::Debug::ReadProcessMemory;
use windows::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, MODULEENTRY32W, Module32FirstW, Module32NextW, TH32CS_SNAPMODULE,
    TH32CS_SNAPMODULE32,
};
use windows::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_INFORMATION, PROCESS_VM_READ};

const BASE_OFFSET: usize = 0x00742230;
const SHARED_PREFIX: &[usize] = &[0x488, 0x18, 0x8, 0x10];
const SCORE_TAIL: &[usize] = &[0xD0, 0x0, 0x8, 0x0];
const LIVES_TAIL: &[usize] = &[0x4C0, 0x0, 0x8, 0x0];
const MODULE_NAME: &str = "ufo50.exe";

pub struct MemReader {
    process: HANDLE,
    module_base: usize,
}

// HANDLE is only used from the worker thread that owns the tracker.
unsafe impl Send for MemReader {}

impl MemReader {
    pub fn for_pid(pid: u32) -> Result<Self, String> {
        let module_base = find_module_base(pid, MODULE_NAME)?;
        let process = unsafe {
            OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, false, pid)
                .map_err(|e| format!("OpenProcess failed: {:?}", e))?
        };
        Ok(Self {
            process,
            module_base,
        })
    }

    pub fn new(window_title: &str) -> Result<Self, String> {
        let windows = find_windows_by_title(window_title);
        let info = windows
            .first()
            .ok_or_else(|| format!("window '{}' not found", window_title))?;
        Self::for_pid(info.pid)
    }

    /// Read score and lives in one pass, sharing the common prefix derefs.
    pub fn read_both(&self) -> (Option<f64>, Option<f64>) {
        let Some(shared) = self.walk_prefix() else {
            return (None, None);
        };
        (
            self.walk_tail(shared, SCORE_TAIL),
            self.walk_tail(shared, LIVES_TAIL),
        )
    }

    fn walk_prefix(&self) -> Option<usize> {
        let mut addr = self.read_ptr(self.module_base + BASE_OFFSET)?;
        for &off in SHARED_PREFIX {
            addr = self.read_ptr(addr + off)?;
        }
        Some(addr)
    }

    fn walk_tail(&self, start: usize, tail: &[usize]) -> Option<f64> {
        let mut addr = start;
        let last = tail.len() - 1;
        for &off in &tail[..last] {
            addr = self.read_ptr(addr + off)?;
        }
        self.read_f64(addr + tail[last])
    }

    fn read_ptr(&self, addr: usize) -> Option<usize> {
        let mut buf = [0u8; 8];
        let mut got = 0usize;
        let ok = unsafe {
            ReadProcessMemory(
                self.process,
                addr as *const _,
                buf.as_mut_ptr() as *mut _,
                8,
                Some(&mut got),
            )
            .is_ok()
        };
        (ok && got == 8).then(|| usize::from_le_bytes(buf))
    }

    fn read_f64(&self, addr: usize) -> Option<f64> {
        let mut buf = [0u8; 8];
        let mut got = 0usize;
        let ok = unsafe {
            ReadProcessMemory(
                self.process,
                addr as *const _,
                buf.as_mut_ptr() as *mut _,
                8,
                Some(&mut got),
            )
            .is_ok()
        };
        (ok && got == 8).then(|| f64::from_le_bytes(buf))
    }
}

impl Drop for MemReader {
    fn drop(&mut self) {
        unsafe {
            let _ = CloseHandle(self.process);
        }
    }
}

fn find_module_base(pid: u32, module_name: &str) -> Result<usize, String> {
    let snapshot = unsafe {
        CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid)
            .map_err(|e| format!("CreateToolhelp32Snapshot failed: {:?}", e))?
    };
    let mut entry = MODULEENTRY32W {
        dwSize: std::mem::size_of::<MODULEENTRY32W>() as u32,
        ..Default::default()
    };

    let result = (|| -> Result<usize, String> {
        unsafe {
            Module32FirstW(snapshot, &mut entry)
                .map_err(|e| format!("Module32FirstW failed: {:?}", e))?
        };
        loop {
            let len = entry
                .szModule
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(entry.szModule.len());
            let name = String::from_utf16_lossy(&entry.szModule[..len]);
            if name.eq_ignore_ascii_case(module_name) {
                return Ok(entry.modBaseAddr as usize);
            }
            if unsafe { Module32NextW(snapshot, &mut entry) }.is_err() {
                return Err(format!("module '{}' not found in pid {}", module_name, pid));
            }
        }
    })();

    unsafe {
        let _ = CloseHandle(snapshot);
    }
    result
}
