use windows::{
    core::*,
    Win32::UI::WindowsAndMessaging::*,
    Win32::Foundation::*,
};

const VK_UP: usize = 0x26;
const VK_DOWN: usize = 0x28;
const VK_LEFT: usize = 0x25;
const VK_RIGHT: usize = 0x27;
const VK_Z: usize = 0x5A; // A button
const VK_X: usize = 0x58; // B button

const ACTION_MAP: &[&[usize]] = &[
    &[],                        // 0: NOOP
    &[VK_UP],                   // 1: Up
    &[VK_DOWN],                 // 2: Down
    &[VK_LEFT],                 // 3: Left
    &[VK_RIGHT],                // 4: Right
    &[VK_Z],                    // 5: A
    &[VK_X],                    // 6: B
    &[VK_UP, VK_Z],             // 7: Up + A
    &[VK_UP, VK_X],             // 8: Up + B
    &[VK_DOWN, VK_Z],           // 9: Down + A
    &[VK_LEFT, VK_Z],           // 10: Left + A
    &[VK_RIGHT, VK_Z],          // 11: Right + A
    &[VK_LEFT, VK_X],           // 12: Left + B
    &[VK_RIGHT, VK_X],          // 13: Right + B
    &[VK_Z, VK_X],              // 14: A + B
];

pub const NUM_ACTIONS: usize = 15;

pub struct Input {
    hwnd: HWND,
    held: [bool; 256],
}

impl Input {
    pub fn new(window_title: &str) -> Result<Self> {
        let title = format!("{}\0", window_title);
        let hwnd = unsafe { FindWindowA(None, PCSTR(title.as_ptr()))? };
        Ok(Self { hwnd, held: [false; 256] })
    }

    pub fn execute_action(&mut self, action: usize) {
        let keys = ACTION_MAP[action.min(NUM_ACTIONS - 1)];

        // Release keys not in the new action
        for vk in 0..256 {
            if self.held[vk] && !keys.contains(&vk) {
                self.post_key(vk, true);
                self.held[vk] = false;
            }
        }
        // Press keys not already held
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

    fn post_key(&self, vk: usize, up: bool) {
        let msg = if up { WM_KEYUP } else { WM_KEYDOWN };
        // lParam: repeat=1, scancode, extended flag, previous state
        let scan = unsafe { MapVirtualKeyA(vk as u32, MAP_VIRTUAL_KEY_TYPE(0)) };
        let mut lparam = 1u32 | (scan << 16);
        if up {
            lparam |= 0xC0000000; // transition + previous state bits
        }
        let _ = unsafe {
            PostMessageA(Some(self.hwnd), msg, WPARAM(vk), LPARAM(lparam as isize))
        };
    }
}
