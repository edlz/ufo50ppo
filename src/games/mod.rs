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
}

/// Game-specific tracker trait. Each game implements this to define
/// reward detection, menu screen identification, and game configuration.
pub trait GameTracker: Send {
    /// Process a frame and return reward/done/event info.
    fn process_frame(&mut self, pixels: &[u8]) -> FrameResult;

    /// Returns true if this frame is a non-gameplay screen (menu, leaderboard, loading, etc).
    fn is_menu_screen(&self, pixels: &[u8]) -> bool;

    /// Extra keys to press after the standard reset sequence.
    fn extra_reset_keys(&self) -> &[usize];

    /// Game name for logging and checkpoint paths.
    fn game_name(&self) -> &str;

    /// Observation width this tracker is calibrated for.
    fn obs_width(&self) -> u32;

    /// Observation height this tracker is calibrated for.
    fn obs_height(&self) -> u32;

    /// Number of discrete actions for this game.
    fn num_actions(&self) -> usize;
}
