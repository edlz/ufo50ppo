pub mod win32;

#[cfg(target_os = "windows")]
pub use win32::host;

pub trait GameRunner: Send {
    fn next_frame(&mut self) -> Result<Vec<u8>, String>;

    /// Default impl ignores the timeout — multi-env trainers override to detect wedges.
    fn next_frame_timeout(&mut self, _timeout: std::time::Duration) -> Result<Vec<u8>, String> {
        self.next_frame()
    }

    fn execute_action(&mut self, action: usize);

    fn release_all(&mut self);

    /// Replay `sequence` (real VK codes interleaved with `vk_noop(ms)` waits).
    fn reset_game(&mut self, sequence: &[usize], tap_ms: u64);

    fn obs_width(&self) -> u32;

    fn obs_height(&self) -> u32;

    /// 0 when not applicable; trackers reading process memory need a real PID.
    fn pid(&self) -> u32 {
        0
    }
}

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
