use std::sync::mpsc::{Receiver, Sender};
use tch::Device;

use super::model::ActorCritic;
use super::ppo::{self, PpoConfig, RolloutBuffer};
use crate::game::env::{self, FrameData, GameEnv};
use crate::game::input::Input;

const ROLLOUT_LENGTH: usize = 128;
const GAMMA: f64 = 0.99;
const GAE_LAMBDA: f64 = 0.95;
const LEARNING_RATE: f64 = 2.5e-4;
const SAVE_INTERVAL: u64 = 10_000;

pub fn training_loop(frame_rx: Receiver<FrameData>, action_tx: Sender<usize>, mut input: Input) {
    let device = Device::cuda_if_available();
    println!("Training on: {:?}", device);

    let mut model = ActorCritic::new(device);
    let mut opt = model.optimizer(LEARNING_RATE);
    let mut env = GameEnv::new(frame_rx, action_tx, device, env::stub_reward);
    let mut buffer = RolloutBuffer::new(ROLLOUT_LENGTH);
    let ppo_cfg = PpoConfig::default();

    let mut obs = env.reset();
    let mut episode_reward = 0.0;
    let mut episode_count = 0u64;
    let mut total_steps = 0u64;

    loop {
        for _ in 0..ROLLOUT_LENGTH {
            let (action, log_prob, value) = model.act(&obs);
            input.execute_action(action as usize);

            let (next_obs, reward, done) = env.step(action as usize);
            buffer.push(obs, action, log_prob, reward, value, done);

            episode_reward += reward;
            total_steps += 1;

            if done {
                episode_count += 1;
                println!(
                    "Episode {} | reward: {:.2} | steps: {}",
                    episode_count, episode_reward, total_steps
                );
                episode_reward = 0.0;
                input.release_all();
                obs = env.reset();
            } else {
                obs = next_obs;
            }
        }

        let (_, _, last_value) = model.act(&obs);
        let (advantages, returns) = ppo::compute_gae(
            &buffer.rewards,
            &buffer.values,
            &buffer.dones,
            last_value,
            GAMMA,
            GAE_LAMBDA,
        );

        input.release_all();
        ppo::update(
            &mut model,
            &mut opt,
            &buffer,
            &advantages,
            &returns,
            &ppo_cfg,
        );
        buffer.clear();

        if total_steps % SAVE_INTERVAL == 0 {
            model.vs.save("model.safetensors").unwrap_or_else(|e| {
                eprintln!("Failed to save model: {}", e);
            });
        }
    }
}
