/// Platform implementations live in submodules (e.g., win32/, linux/).
/// Each must implement GameRunner: frame capture, input execution, and game lifecycle.
pub mod win32;

/// Re-export the platform-appropriate `host` function. Each platform exposes the same
/// signature: `host(window_title, obs_w, obs_h, train_fn) -> Result<()>`. This lets binaries
/// stay platform-agnostic — the host owns the threading model (worker thread on Win32,
/// main thread on Linux).
#[cfg(target_os = "windows")]
pub use win32::host;

/// Platform-agnostic interface for running a game.
/// Combines frame capture, input, and game lifecycle.
/// On Windows: wraps capture::run callback + Input via channels.
/// On Linux: would be a single-threaded X11/PipeWire + uinput loop.
pub trait GameRunner: Send {
    /// Block until next frame is available. Returns BGRA pixels at the configured resolution.
    fn next_frame(&mut self) -> Result<Vec<u8>, String>;

    /// Execute a discrete action by index.
    fn execute_action(&mut self, action: usize);

    /// Release all held keys/buttons.
    fn release_all(&mut self);

    /// Reset the game by playing the given key sequence (with `vk_noop(ms)` for waits).
    /// `tap_ms` controls the per-tap delay (sleep before keydown and after, before keyup).
    fn reset_game(&mut self, sequence: &[usize], tap_ms: u64);

    /// Observation width.
    fn obs_width(&self) -> u32;

    /// Observation height.
    fn obs_height(&self) -> u32;
}

/// Number of discrete actions available.
pub const NUM_ACTIONS: usize = 26;

/// Human-readable action names.
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
