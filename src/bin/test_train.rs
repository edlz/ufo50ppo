//! Smoke-test for the PPO training framework on a trivial problem.
//!
//! Environment: random 128×128 observations, action 0 always gives +1 reward,
//! all other actions give -1. The agent should learn to pick action 0.
//!
//! Usage: cargo run --release --bin test_train

use tch::{Device, Kind, Tensor};
use ufo50ppo::platform::NUM_ACTIONS;
use ufo50ppo::train::model::ActorCritic;
use ufo50ppo::train::ppo::{self, PpoConfig, RolloutBuffer};

const ROLLOUT_LENGTH: usize = 512;
const GAMMA: f64 = 0.0; // no temporal credit — each step is independent
const GAE_LAMBDA: f64 = 1.0;
const LEARNING_RATE: f64 = 1e-4;
const NUM_UPDATES: usize = 100;
const TARGET_ACTION: i64 = 0;

fn random_obs(device: Device) -> Tensor {
    Tensor::rand([1, 4, 128, 128], (Kind::Float, device))
}

fn step_reward(action: i64) -> (f64, bool) {
    let reward = if action == TARGET_ACTION { 1.0 } else { -1.0 };
    (reward, false)
}

fn main() {
    ufo50ppo::util::preload_torch_cuda();

    let device = Device::cuda_if_available();
    println!("Device: {:?}", device);

    let mut model = ActorCritic::new(device, 128, 128, NUM_ACTIONS);
    let mut opt = model.optimizer(LEARNING_RATE);
    let mut buffer = RolloutBuffer::new(ROLLOUT_LENGTH);
    let ppo_cfg = PpoConfig {
        entropy_coeff: 0.05, // extra entropy to prevent policy collapse
        num_epochs: 2,       // fewer epochs per update for stability
        ..PpoConfig::default()
    };

    let mut obs = random_obs(device);

    for update in 1..=NUM_UPDATES {
        let mut total_reward = 0.0;
        let mut action_0_count = 0u64;

        for _ in 0..ROLLOUT_LENGTH {
            let (action, log_prob, value) = model.act(&obs);
            let (reward, done) = step_reward(action);

            total_reward += reward;
            if action == TARGET_ACTION {
                action_0_count += 1;
            }

            let next_obs = random_obs(device);
            buffer.push(obs, action, log_prob, reward, value, done);
            obs = next_obs;
        }

        let (_, _, last_value) = model.act(&obs);
        let (advantages, returns) = ppo::compute_gae(
            &buffer.rewards,
            &buffer.values,
            &buffer.dones,
            last_value,
            GAMMA,
            GAE_LAMBDA,
            1.0,
        );

        ppo::update(
            &mut model,
            &mut opt,
            &buffer,
            &advantages,
            &returns,
            &ppo_cfg,
        );
        buffer.clear();

        let pct = action_0_count as f64 / ROLLOUT_LENGTH as f64 * 100.0;
        println!(
            "Update {:>3}/{} | reward: {:>7.1} | action 0: {:>5.1}%",
            update, NUM_UPDATES, total_reward, pct
        );
    }

    // Check that the agent learned
    let mut final_action_0 = 0u64;
    let eval_steps = 1000;
    tch::no_grad(|| {
        for _ in 0..eval_steps {
            let o = random_obs(device);
            let (action, _, _) = model.act(&o);
            if action == TARGET_ACTION {
                final_action_0 += 1;
            }
        }
    });

    let final_pct = final_action_0 as f64 / eval_steps as f64 * 100.0;
    println!(
        "\nEval: action 0 chosen {:.1}% of the time ({}/{})",
        final_pct, final_action_0, eval_steps
    );

    if final_pct > 50.0 {
        println!("PASS — agent learned to prefer the rewarded action");
    } else {
        println!("FAIL — agent did not converge (may need more updates)");
        std::process::exit(1);
    }
}
