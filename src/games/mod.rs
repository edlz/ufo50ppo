pub mod ninpek;

use crate::platform::Frame;

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

    fn reset_episode(&mut self) {}

    fn game_name(&self) -> &str;

    fn num_actions(&self) -> usize;

    fn episode_breakdown(&self) -> String {
        String::new()
    }
}
