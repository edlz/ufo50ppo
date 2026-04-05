use std::sync::mpsc::{Receiver, Sender};
use tch::{Device, Tensor};

use crate::preprocess::FrameStack;

pub struct FrameData {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Reward function: receives previous and current RGBA pixels.
/// Returns (reward, done).
pub type RewardFn = fn(prev: Option<&[u8]>, curr: &[u8], width: u32, height: u32) -> (f64, bool);

/// Stub reward — always returns 0 and never terminates.
pub fn stub_reward(_prev: Option<&[u8]>, _curr: &[u8], _w: u32, _h: u32) -> (f64, bool) {
    (0.0, false)
}

pub struct GameEnv {
    frame_rx: Receiver<FrameData>,
    action_tx: Sender<usize>,
    frame_stack: FrameStack,
    prev_rgba: Option<Vec<u8>>,
    reward_fn: RewardFn,
}

impl GameEnv {
    pub fn new(
        frame_rx: Receiver<FrameData>,
        action_tx: Sender<usize>,
        device: Device,
        reward_fn: RewardFn,
    ) -> Self {
        Self {
            frame_rx,
            action_tx,
            frame_stack: FrameStack::new(device),
            prev_rgba: None,
            reward_fn,
        }
    }

    /// Wait for first frame, return initial observation.
    pub fn reset(&mut self) -> Tensor {
        self.frame_stack.reset();
        self.prev_rgba = None;
        // Send NOOP to unblock the capture callback
        let _ = self.action_tx.send(0);
        let frame = self.frame_rx.recv().expect("capture thread died");
        let obs = self.frame_stack.push(&frame.rgba, frame.width, frame.height);
        self.prev_rgba = Some(frame.rgba);
        obs
    }

    /// Send action, wait for next frame, return (obs, reward, done).
    pub fn step(&mut self, action: usize) -> (Tensor, f64, bool) {
        let _ = self.action_tx.send(action);
        let frame = self.frame_rx.recv().expect("capture thread died");
        let (reward, done) = (self.reward_fn)(
            self.prev_rgba.as_deref(),
            &frame.rgba,
            frame.width,
            frame.height,
        );
        let obs = self.frame_stack.push(&frame.rgba, frame.width, frame.height);
        self.prev_rgba = Some(frame.rgba);
        (obs, reward, done)
    }
}
