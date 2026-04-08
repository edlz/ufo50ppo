pub mod ninpek;

/// Pixel region for game-specific screen area detection.
/// Coordinates are calibrated for the game's capture resolution.
pub struct Region {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Result from processing a single frame.
pub struct FrameResult {
    pub reward: f64,
    pub done: bool,
    pub lives: u32,
    pub event_name: &'static str,
    pub is_event: bool,
    pub is_menu: bool,
}

/// Game-specific tracker trait. Each game implements this to define
/// reward detection, menu screen identification, and game configuration.
pub trait GameTracker: Send {
    /// Process a frame and return reward/done/event info.
    fn process_frame(&mut self, pixels: &[u8]) -> FrameResult;

    /// Returns true if this frame is a non-gameplay screen (menu, leaderboard, loading, etc).
    fn is_menu_screen(&self, pixels: &[u8]) -> bool;

    /// Full reset sequence: keys to tap, with `vk_noop(ms)` for waits between presses.
    fn reset_sequence(&self) -> &[usize];

    /// Per-tap delay (sleep before keydown and after, before keyup) for the reset sequence.
    fn reset_tap_ms(&self) -> u64 {
        crate::platform::win32::input::DEFAULT_RESET_TAP_MS
    }

    /// Game name for logging and checkpoint paths.
    fn game_name(&self) -> &str;

    /// Observation width this tracker is calibrated for.
    fn obs_width(&self) -> u32;

    /// Observation height this tracker is calibrated for.
    fn obs_height(&self) -> u32;

    /// Number of discrete actions for this game.
    fn num_actions(&self) -> usize;

    /// Format a per-episode breakdown of game-specific events for debug output.
    /// Trackers should accumulate counters in `process_frame` and reset them when the
    /// tracker is rebuilt at episode boundary (via `GameDefinition::make_tracker`).
    /// Default returns empty string (no breakdown).
    fn episode_breakdown(&self) -> String {
        String::new()
    }
}
