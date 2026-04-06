// All pixel regions (score, lives, game_over) are calibrated for 128x128 capture resolution.
// If the resolution changes, all coordinates must be recalibrated.
mod game_over;
mod lives;
pub mod rewards;
pub mod score;
mod tracker;

pub use game_over::*;
pub use lives::*;
pub use score::NINPEK_SCORE;
pub use tracker::NinpekTracker;
