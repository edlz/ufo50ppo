use windows::{
    Win32::Foundation::*, Win32::UI::Input::KeyboardAndMouse::*, Win32::UI::WindowsAndMessaging::*,
    core::*,
};

pub const VK_UP: usize = 0x26;
pub const VK_DOWN: usize = 0x28;
pub const VK_LEFT: usize = 0x25;
pub const VK_RIGHT: usize = 0x27;
pub const VK_Z: usize = 0x5A; // A button
pub const VK_X: usize = 0x58; // B button
pub const VK_ESCAPE: usize = 0x1B; // Start (reset only, not used in training)

const ACTION_MAP: &[&[usize]] = &[
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

pub const NUM_ACTIONS: usize = 26;

pub const ACTION_NAMES: &[&str] = &[
    "NOOP",
    "Up",
    "Down",
    "Left",
    "Right",
    "Up-Right",
    "Up-Left",
    "Down-Right",
    "Down-Left",
    "A",
    "B",
    "Up+A",
    "Up+B",
    "Down+A",
    "Left+A",
    "Right+A",
    "Left+B",
    "Right+B",
    "Up-Right+A",
    "Up-Right+B",
    "Up-Left+A",
    "Up-Left+B",
    "Down-Right+A",
    "Down-Right+B",
    "Down-Left+A",
    "Down-Left+B",
];

pub struct Input {
    hwnd: HWND,
    held: [bool; 256],
}

// SAFETY: HWND is a Win32 handle (opaque integer), safe to send across threads.
unsafe impl Send for Input {}

impl Input {
    pub fn new(window_title: &str) -> Result<Self> {
        let title = format!("{}\0", window_title);
        let hwnd = unsafe { FindWindowA(None, PCSTR(title.as_ptr()))? };
        Ok(Self {
            hwnd,
            held: [false; 256],
        })
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

    /// Tap a key: press, wait, release, wait.
    fn tap_key(&self, vk: usize, delay_ms: u64) {
        std::thread::sleep(std::time::Duration::from_millis(delay_ms));
        self.post_key(vk, false);
        std::thread::sleep(std::time::Duration::from_millis(delay_ms));
        self.post_key(vk, true);
    }

    /// Reset the game: Start (Esc) to open menu, Down to select, A to confirm,
    /// then tap any additional keys provided.
    pub fn reset_game(&mut self, extra_keys: &[usize]) {
        self.release_all();
        self.tap_key(VK_ESCAPE, 50);
        self.tap_key(VK_DOWN, 50);
        self.tap_key(VK_Z, 50);
        self.tap_key(VK_Z, 50);
        for &vk in extra_keys {
            self.tap_key(vk, 100);
        }
    }

    fn vk_name(vk: usize) -> &'static str {
        match vk {
            VK_UP => "Up",
            VK_DOWN => "Down",
            VK_LEFT => "Left",
            VK_RIGHT => "Right",
            VK_Z => "A(Z)",
            VK_X => "B(X)",
            VK_ESCAPE => "Start(Esc)",
            _ => "???",
        }
    }

    fn post_key(&self, vk: usize, up: bool) {
        let action = if up { "release" } else { "press" };
        println!("  key: {} {}", action, Self::vk_name(vk));
        let msg = if up { WM_KEYUP } else { WM_KEYDOWN };
        // lParam: repeat=1, scancode, extended flag, previous state
        let scan = unsafe { MapVirtualKeyA(vk as u32, MAPVK_VK_TO_VSC) };
        let mut lparam = 1u32 | (scan << 16);
        const KEYUP_LPARAM: u32 = 0xC0000000; // WM_KEYUP: transition + previous-state bits
        if up {
            lparam |= KEYUP_LPARAM;
        }
        let _ = unsafe { PostMessageA(self.hwnd, msg, WPARAM(vk), LPARAM(lparam as isize)) };
    }
}
