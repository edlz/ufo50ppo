// All pixel regions (score, lives, game_over) are calibrated for 128x128 capture resolution.
// If the resolution changes, all coordinates must be recalibrated.
pub mod events;
mod game_over;
mod lives;
pub mod rewards;
pub mod score;
mod tracker;

pub use game_over::*;
pub use lives::*;
pub use score::NINPEK_SCORE;
pub use tracker::NinpekTracker;

use crate::train::runner::{GameDefinition, Hyperparams};

fn ninpek_debug_suffix(event: &str, reward: f64) -> String {
    if event == events::SCORE {
        format!("_+{}", reward as i64)
    } else {
        String::new()
    }
}

fn make_ninpek_tracker(w: u32) -> Box<dyn crate::games::GameTracker> {
    Box::new(NinpekTracker::new(w))
}

pub fn definition() -> GameDefinition {
    GameDefinition {
        name: "ninpek",
        window_title: "UFO 50",
        obs_width: 128,
        obs_height: 128,
        num_actions: crate::platform::NUM_ACTIONS,
        make_tracker: make_ninpek_tracker,
        hyperparams: Hyperparams::default(),
        debug_frame_suffix: ninpek_debug_suffix,
    }
}
