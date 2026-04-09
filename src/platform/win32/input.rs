use windows::{
    Win32::Foundation::*, Win32::UI::Input::KeyboardAndMouse::*, Win32::UI::WindowsAndMessaging::*,
    core::*,
};

// Bit 16 distinguishes a wait sentinel from a real VK code in `reset_game` sequences.
const NOOP_FLAG: usize = 1 << 16;

pub const DEFAULT_RESET_TAP_MS: u64 = 25;

pub const fn vk_noop(ms: u64) -> usize {
    NOOP_FLAG | ms as usize
}

pub const VK_UP: usize = 0x26;
pub const VK_DOWN: usize = 0x28;
pub const VK_LEFT: usize = 0x25;
pub const VK_RIGHT: usize = 0x27;
pub const VK_Z: usize = 0x5A; // A button
pub const VK_X: usize = 0x58; // B button
pub const VK_ESCAPE: usize = 0x1B; // Start (reset only, not used in training)

pub(crate) const ACTION_MAP: &[&[usize]] = &[
    &[],                        // 0: NOOP
    &[VK_UP],                   // 1: Up
    &[VK_DOWN],                 // 2: Down
    &[VK_LEFT],                 // 3: Left
    &[VK_RIGHT],                // 4: Right
    &[VK_UP, VK_RIGHT],         // 5: Up-Right
    &[VK_UP, VK_LEFT],          // 6: Up-Left
    &[VK_DOWN, VK_RIGHT],       // 7: Down-Right
    &[VK_DOWN, VK_LEFT],        // 8: Down-Left
    &[VK_Z],                    // 9: A
    &[VK_X],                    // 10: B
    &[VK_UP, VK_Z],             // 11: Up + A
    &[VK_UP, VK_X],             // 12: Up + B
    &[VK_DOWN, VK_Z],           // 13: Down + A
    &[VK_LEFT, VK_Z],           // 14: Left + A
    &[VK_RIGHT, VK_Z],          // 15: Right + A
    &[VK_LEFT, VK_X],           // 16: Left + B
    &[VK_RIGHT, VK_X],          // 17: Right + B
    &[VK_UP, VK_RIGHT, VK_Z],   // 18: Up-Right + A
    &[VK_UP, VK_RIGHT, VK_X],   // 19: Up-Right + B
    &[VK_UP, VK_LEFT, VK_Z],    // 20: Up-Left + A
    &[VK_UP, VK_LEFT, VK_X],    // 21: Up-Left + B
    &[VK_DOWN, VK_RIGHT, VK_Z], // 22: Down-Right + A
    &[VK_DOWN, VK_RIGHT, VK_X], // 23: Down-Right + B
    &[VK_DOWN, VK_LEFT, VK_Z],  // 24: Down-Left + A
    &[VK_DOWN, VK_LEFT, VK_X],  // 25: Down-Left + B
];

use super::super::NUM_ACTIONS;

pub struct Input {
    pub(crate) hwnd: HWND,
    pub(crate) held: [bool; 256],
}

unsafe impl Send for Input {}

impl Input {
    pub fn for_hwnd(hwnd: HWND) -> Self {
        Self {
            hwnd,
            held: [false; 256],
        }
    }

    pub fn new(window_title: &str) -> Result<Self> {
        let title = format!("{}\0", window_title);
        let hwnd = unsafe { FindWindowA(None, PCSTR(title.as_ptr()))? };
        Ok(Self::for_hwnd(hwnd))
    }

    pub fn execute_action(&mut self, action: usize) {
        let keys = ACTION_MAP[action.min(NUM_ACTIONS - 1)];
        for vk in 0..256 {
            if self.held[vk] && !keys.contains(&vk) {
                self.post_key(vk, true);
                self.held[vk] = false;
            }
        }
        for &vk in keys {
            if !self.held[vk] {
                self.post_key(vk, false);
                self.held[vk] = true;
            }
        }
    }

    pub fn release_all(&mut self) {
        for vk in 0..256 {
            if self.held[vk] {
                self.post_key(vk, true);
                self.held[vk] = false;
            }
        }
    }

    pub fn reset_game(&mut self, sequence: &[usize], tap_ms: u64) {
        self.release_all();
        for &vk in sequence {
            if vk & NOOP_FLAG != 0 {
                let ms = (vk & 0xFFFF) as u64;
                std::thread::sleep(std::time::Duration::from_millis(ms));
            } else {
                self.tap_key(vk, tap_ms);
            }
        }
    }

    pub(crate) fn tap_key(&self, vk: usize, delay_ms: u64) {
        std::thread::sleep(std::time::Duration::from_millis(delay_ms));
        self.post_key(vk, false);
        std::thread::sleep(std::time::Duration::from_millis(delay_ms));
        self.post_key(vk, true);
    }

    pub(crate) fn post_key(&self, vk: usize, up: bool) {
        let msg = if up { WM_KEYUP } else { WM_KEYDOWN };
        let scan = unsafe { MapVirtualKeyA(vk as u32, MAPVK_VK_TO_VSC) };
        let mut lparam = 1u32 | (scan << 16);
        const KEYUP_LPARAM: u32 = 0xC0000000;
        if up {
            lparam |= KEYUP_LPARAM;
        }
        let _ = unsafe { PostMessageA(self.hwnd, msg, WPARAM(vk), LPARAM(lparam as isize)) };
    }
}

impl Drop for Input {
    fn drop(&mut self) {
        self.release_all();
    }
}
