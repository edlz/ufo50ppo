pub mod lockstep;

pub use lockstep::GameState;

/// One synced game frame: what the policy sees plus the game state read inside the game.
pub struct Frame {
    /// OBS_W x OBS_H BGRA.
    pub pixels: Vec<u8>,
    pub state: GameState,
}

pub trait GameRunner: Send {
    fn next_frame_timeout(&mut self, timeout: std::time::Duration) -> Result<Frame, String>;

    /// Answers the frame last returned by `next_frame_timeout`; extra calls before the next frame are ignored.
    fn execute_action(&mut self, action: usize);

    /// Relaunches the game; takes effect with the next `execute_action`.
    fn reset_game(&mut self);

    fn obs_width(&self) -> u32;

    fn obs_height(&self) -> u32;
}

pub const VK_UP: u8 = 0x26;
pub const VK_DOWN: u8 = 0x28;
pub const VK_LEFT: u8 = 0x25;
pub const VK_RIGHT: u8 = 0x27;
pub const VK_Z: u8 = 0x5A; // A button
pub const VK_X: u8 = 0x58; // B button

pub const ACTION_MAP: &[&[u8]] = &[
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
