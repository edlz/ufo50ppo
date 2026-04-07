//! Event name constants returned by `NinpekTracker::process_frame` in `FrameResult.event_name`.
//! Centralized here so producers (the tracker) and consumers (the wrapper match,
//! `ninpek_debug_suffix`) reference the same string. A typo at any site is now a compile error.

pub const SCORE: &str = "SCORE";
pub const LIFE_GAINED: &str = "LIFE+";
pub const LIFE_LOST: &str = "LIFE-";
pub const STAGE: &str = "STAGE";
pub const WIN: &str = "WIN";
pub const GAME_OVER: &str = "GAME OVER";
