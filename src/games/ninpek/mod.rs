pub mod events;
pub mod rewards;
mod tracker;

pub use tracker::NinpekTracker;

use crate::train::runner::{GameDefinition, Hyperparams};

/// Internal game ID (o34_*/rm34_*); the library lists Ninpek third.
pub const GAME_ID: u8 = 34;

fn make_ninpek_tracker() -> Box<dyn crate::games::GameTracker> {
    Box::new(NinpekTracker::new())
}

pub fn definition() -> GameDefinition {
    GameDefinition {
        name: "ninpek",
        game_id: GAME_ID,
        obs_width: crate::platform::lockstep::OBS_W as u32,
        obs_height: crate::platform::lockstep::OBS_H as u32,
        num_actions: crate::platform::NUM_ACTIONS,
        make_tracker: make_ninpek_tracker,
        hyperparams: Hyperparams::default(),
    }
}
