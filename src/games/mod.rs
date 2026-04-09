pub mod ninpek;

pub struct Region {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

pub struct FrameResult {
    pub reward: f64,
    pub done: bool,
    pub lives: u32,
    pub event_name: &'static str,
    pub is_event: bool,
    pub is_menu: bool,
}

pub trait GameTracker: Send {
    fn process_frame(&mut self, pixels: &[u8]) -> FrameResult;

    /// Default returns true so games without an idle-state check skip the drain.
    fn observe_idle(&mut self, _pixels: &[u8]) -> bool {
        true
    }

    fn reset_episode(&mut self) {}

    /// Reset key sequence; `vk_noop(ms)` encodes inline waits between presses.
    fn reset_sequence(&self) -> &[usize];

    fn reset_tap_ms(&self) -> u64 {
        crate::platform::win32::input::DEFAULT_RESET_TAP_MS
    }

    fn game_name(&self) -> &str;

    fn obs_width(&self) -> u32;

    fn obs_height(&self) -> u32;

    fn num_actions(&self) -> usize;

    fn episode_breakdown(&self) -> String {
        String::new()
    }
}
