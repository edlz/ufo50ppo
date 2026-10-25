use crate::games::GameTracker;

pub struct Hyperparams {
    /// Transitions each env collects per PPO update.
    pub rollout_len: usize,
    pub minibatch_size: usize,
    pub learning_rate: f64,
    /// Frames over which the learning rate falls linearly to `learning_rate * lr_final_frac`,
    /// where it then stays; 0 keeps it constant.
    pub lr_anneal_frames: u64,
    pub lr_final_frac: f64,
    pub gamma: f64,
    pub gae_lambda: f64,
    pub latest_save_interval: u64,
    pub versioned_save_interval: u64,
}

impl Default for Hyperparams {
    fn default() -> Self {
        Self {
            rollout_len: 256,
            minibatch_size: 128,
            learning_rate: 1.5e-4,
            lr_anneal_frames: 15_000_000,
            lr_final_frac: 0.1,
            gamma: 0.99,
            gae_lambda: 0.95,
            latest_save_interval: 10_000,
            versioned_save_interval: 250_000,
        }
    }
}

impl Hyperparams {
    pub fn learning_rate_at(&self, frames: u64) -> f64 {
        if self.lr_anneal_frames == 0 {
            return self.learning_rate;
        }
        let t = (frames as f64 / self.lr_anneal_frames as f64).min(1.0);
        self.learning_rate * (1.0 - t * (1.0 - self.lr_final_frac))
    }
}

/// What the trainer needs to know about a game; each game module's `definition()` returns one.
pub struct GameDefinition {
    pub name: &'static str,
    /// Internal game ID the bridge launches on reset.
    pub game_id: u8,
    pub obs_width: u32,
    pub obs_height: u32,
    pub num_actions: usize,
    /// Furthest start position `--random-start` may pick (0 = the game only starts at the beginning).
    pub max_start: u32,
    pub make_tracker: fn() -> Box<dyn GameTracker>,
    pub hyperparams: Hyperparams,
}

/// Runtime training configuration: training limit flags and the resolved checkpoint/runs paths.
pub struct TrainingConfig {
    pub max_episodes: Option<u32>,
    pub max_frames: Option<u64>,
    pub max_minutes: Option<u64>,
    pub debug: bool,
    pub checkpoint_dir: String,
    pub runs_dir: String,
}

impl TrainingConfig {
    /// Derives the checkpoint and runs directories from the game name and namespace.
    pub fn from_args(args: &crate::util::cli::TrainArgs, game_name: &str) -> Self {
        Self {
            max_episodes: args.max_episodes,
            max_frames: args.max_frames,
            max_minutes: args.max_minutes,
            debug: args.debug,
            checkpoint_dir: format!("checkpoints/{}/{}", game_name, args.namespace),
            runs_dir: format!("runs/{}/{}", game_name, args.namespace),
        }
    }
}

/// CUDA if available, else Apple's Metal backend, else CPU.
pub fn pick_device() -> tch::Device {
    if tch::Cuda::is_available() {
        tch::Device::cuda_if_available()
    } else if tch::utils::has_mps() {
        tch::Device::Mps
    } else {
        tch::Device::Cpu
    }
}

/// Prints the training device, warning loudly on CPU.
pub fn log_device(device: tch::Device) {
    match device {
        tch::Device::Cuda(_) | tch::Device::Mps => println!("Training device: {:?} ✓", device),
        _ => eprintln!(
            "WARNING: training on {:?}. No GPU was detected — training will be slow. \
             Check that LIBTORCH points to a CUDA-enabled (or macOS MPS-enabled) libtorch.",
            device
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::Hyperparams;

    #[test]
    fn learning_rate_anneals_linearly_then_holds() {
        let hp = Hyperparams::default();
        let lr = hp.learning_rate;
        assert_eq!(hp.learning_rate_at(0), lr);
        let half = hp.learning_rate_at(hp.lr_anneal_frames / 2);
        assert!((half - lr * (1.0 + hp.lr_final_frac) / 2.0).abs() < 1e-12);
        let end = lr * hp.lr_final_frac;
        assert!((hp.learning_rate_at(hp.lr_anneal_frames) - end).abs() < 1e-12);
        assert!((hp.learning_rate_at(hp.lr_anneal_frames * 3) - end).abs() < 1e-12);
        let constant = Hyperparams {
            lr_anneal_frames: 0,
            ..Hyperparams::default()
        };
        assert_eq!(constant.learning_rate_at(1_000_000_000), lr);
    }
}
