pub mod ninpek;

use crate::platform::{Frame, GameRunner};
use std::time::Duration;

pub struct FrameResult {
    pub reward: f64,
    pub done: bool,
    pub lives: u32,
    pub event_name: &'static str,
    pub is_event: bool,
    pub is_menu: bool,
}

pub trait GameTracker: Send {
    fn process_frame(&mut self, frame: &Frame) -> FrameResult;

    /// Called each frame after a reset until a fresh run is playable. Returns `None` once ready,
    /// otherwise the action to send this frame (e.g. to get past a title screen).
    fn observe_idle(&mut self, frame: &Frame) -> Option<usize>;

    /// The action to send on a frame `process_frame` marked `is_menu` (e.g. to confirm a screen
    /// that waits for a button mid-run).
    fn menu_action(&mut self) -> usize {
        0
    }

    fn reset_episode(&mut self) {}

    fn game_name(&self) -> &str;

    fn num_actions(&self) -> usize;

    fn episode_breakdown(&self) -> String {
        String::new()
    }

    /// Per-episode scalars logged to TensorBoard under `game/<name>`.
    fn episode_metrics(&self) -> Vec<(&'static str, f64)> {
        Vec::new()
    }
}

/// Relaunches the game and plays `tracker`'s idle actions until a fresh run starts. The next
/// frame is the run's first for `process_frame`.
pub fn start_run(
    runner: &mut dyn GameRunner,
    tracker: &mut dyn GameTracker,
    timeout: Duration,
) -> Result<(), String> {
    runner.reset_game();
    tracker.reset_episode();
    loop {
        let frame = runner.next_frame_timeout(timeout)?;
        match tracker.observe_idle(&frame) {
            Some(action) => runner.execute_action(action),
            None => {
                runner.execute_action(0);
                return Ok(());
            }
        }
    }
}
