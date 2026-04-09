use crate::games::GameTracker;
use crate::platform::GameRunner;
use crate::train;
use crate::util::checkpoint::{self, CheckpointMeta};

pub struct Hyperparams {
    pub rollout_len: usize,
    pub minibatch_size: usize,
    pub learning_rate: f64,
    pub gamma: f64,
    pub gae_lambda: f64,
    pub latest_save_interval: u64,
    pub versioned_save_interval: u64,
}

impl Default for Hyperparams {
    fn default() -> Self {
        Self {
            rollout_len: 1024,
            minibatch_size: 128,
            learning_rate: 1.5e-4,
            gamma: 0.99,
            gae_lambda: 0.95,
            latest_save_interval: 10_000,
            versioned_save_interval: 250_000,
        }
    }
}

/// Static description of a game: dimensions, tracker factory, hyperparameters.
/// Each game module exposes one of these via a `definition()` function.
pub struct GameDefinition {
    pub name: &'static str,
    pub window_title: &'static str,
    pub obs_width: u32,
    pub obs_height: u32,
    pub num_actions: usize,
    pub make_tracker: fn(u32, u32) -> Box<dyn GameTracker>,
    pub hyperparams: Hyperparams,
    pub debug_frame_suffix: fn(event: &str, reward: f64) -> String,
}

/// Runtime training configuration: training limit flags and the resolved checkpoint/runs paths.
pub struct TrainingConfig {
    pub max_episodes: Option<u32>,
    pub max_frames: Option<u64>,
    pub max_minutes: Option<u64>,
    pub auto_resume: bool,
    pub debug: bool,
    pub checkpoint_dir: String,
    pub runs_dir: String,
}

impl TrainingConfig {
    /// Build a TrainingConfig from parsed CLI args, deriving checkpoint and runs directories
    /// from the game name and the namespace flag. Drops `args.namespace` (already baked into paths).
    pub fn from_args(args: crate::util::cli::TrainArgs, game_name: &str) -> Self {
        Self {
            max_episodes: args.max_episodes,
            max_frames: args.max_frames,
            max_minutes: args.max_minutes,
            auto_resume: args.auto_resume,
            debug: args.debug,
            checkpoint_dir: format!("checkpoints/{}/{}", game_name, args.namespace),
            runs_dir: format!("runs/{}/{}", game_name, args.namespace),
        }
    }
}

/// Print the training device with a loud warning when it's not CUDA. The user shouldn't
/// have to grep startup logs to find out their GPU isn't being picked up.
pub fn log_device(device: tch::Device) {
    match device {
        tch::Device::Cuda(_) => println!("Training device: {:?} ✓", device),
        _ => eprintln!(
            "WARNING: training on {:?}. CUDA was not detected — multi-env throughput will be limited and training will be slow. \
             Check that LIBTORCH points to a CUDA-enabled libtorch and torch_cuda.dll is on PATH.",
            device
        ),
    }
}

/// Sweep `dir` for stale `.tmp` files left behind by a previous run that died mid-save.
/// Atomic checkpoint writes go through `{name}.tmp → rename` so a kill -9 / power loss
/// during a save leaves orphaned tmp files. Cleaned at startup so they don't accumulate.
pub(super) fn cleanup_stale_tmp(dir: &str) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("tmp")
            && let Err(e) = std::fs::remove_file(&path)
        {
            eprintln!("cleanup_stale_tmp: failed to remove {:?}: {}", path, e);
        }
    }
}

/// Tracks consecutive checkpoint save failures across the lifetime of the process.
/// `save_checkpoint` increments on any failure path and resets on success. After
/// `MAX_CONSECUTIVE_SAVE_FAILURES` in a row, the trainer signals shutdown via the
/// global flag so it can exit gracefully on its next loop iteration.
static CONSECUTIVE_SAVE_FAILURES: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);
const MAX_CONSECUTIVE_SAVE_FAILURES: u32 = 3;

pub(crate) fn reset_save_failure_counter() {
    CONSECUTIVE_SAVE_FAILURES.store(0, std::sync::atomic::Ordering::Relaxed);
}

fn record_save_failure(context: &str) {
    let n = CONSECUTIVE_SAVE_FAILURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    eprintln!(
        "save_checkpoint: failure #{} ({}). Threshold = {}.",
        n, context, MAX_CONSECUTIVE_SAVE_FAILURES
    );
    if n >= MAX_CONSECUTIVE_SAVE_FAILURES {
        eprintln!(
            "save_checkpoint: {} consecutive failures — triggering shutdown",
            n
        );
        crate::util::shutdown::trigger();
    }
}

fn record_save_success() {
    CONSECUTIVE_SAVE_FAILURES.store(0, std::sync::atomic::Ordering::Relaxed);
}

#[allow(clippy::too_many_arguments)]
pub(super) fn save_checkpoint(
    dir: &str,
    name: &str,
    model: &train::model::ActorCritic,
    game: &GameDefinition,
    episode: u32,
    frames: u64,
    updates: u64,
    best: f64,
    reward_norm: &train::normalize::RunningMeanStd,
    opt: &train::adam::Adam,
) {
    // Atomic write: save weights to a tmp file then rename. Without this, a crash mid-
    // write (or ctrl-C during a save) leaves a truncated safetensors file that fails to
    // load on the next resume. Rename safetensors first then metadata so any crash window
    // leaves new-weights-with-old-metadata (recoverable) rather than the inverse
    // (metadata claims progress that the weights don't reflect).
    let path = format!("{}/{}.safetensors", dir, name);
    let tmp = format!("{}.tmp", path);
    if let Err(e) = model.vs.save(&tmp) {
        eprintln!("save_checkpoint: write {} failed: {}", tmp, e);
        let _ = std::fs::remove_file(&tmp);
        record_save_failure("safetensors write");
        return;
    }
    if let Err(e) = std::fs::rename(&tmp, &path) {
        eprintln!("save_checkpoint: rename {} -> {} failed: {}", tmp, path, e);
        let _ = std::fs::remove_file(&tmp);
        record_save_failure("safetensors rename");
        return;
    }

    // Adam state — written atomically alongside weights so resume gets consistent
    // (weights, optimizer) snapshots.
    let adam_path = format!("{}/{}.adam", dir, name);
    let adam_tmp = format!("{}.tmp", adam_path);
    let mut adam_ok = true;
    if let Err(e) = opt.save_state(&adam_tmp) {
        eprintln!("save_checkpoint: adam save failed: {}", e);
        adam_ok = false;
    } else if let Err(e) = std::fs::rename(&adam_tmp, &adam_path) {
        eprintln!("save_checkpoint: adam rename failed: {}", e);
        let _ = std::fs::remove_file(&adam_tmp);
        adam_ok = false;
    }

    checkpoint::save_metadata(
        dir,
        name,
        &CheckpointMeta {
            game: game.name,
            resolution: (game.obs_width, game.obs_height),
            episode,
            total_frames: frames,
            ppo_updates: updates,
            best_reward: best,
            rollout_len: game.hyperparams.rollout_len,
            learning_rate: game.hyperparams.learning_rate,
            gamma: game.hyperparams.gamma,
            gae_lambda: game.hyperparams.gae_lambda,
            reward_norm_mean: reward_norm.mean,
            reward_norm_var_sum: reward_norm.var_sum,
            reward_norm_count: reward_norm.count,
        },
    );

    if adam_ok {
        record_save_success();
    } else {
        record_save_failure("adam save");
    }
}

pub fn write_frame_bmp(path: &str, pixels: &[u8], obs_w: u32, obs_h: u32) {
    let _ = crate::util::bmp::write_bgra(path, pixels, obs_w, obs_h);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn report_episode_end(
    episode: u32,
    env_idx: Option<usize>,
    event_name: &str,
    episode_reward: f64,
    episode_frames: u32,
    total_frames: u64,
    update_count: u64,
    debug: bool,
    tracker: &dyn GameTracker,
    logger: &mut crate::util::logger::TbLogger,
    best_reward: &mut f64,
    model: &train::model::ActorCritic,
    game: &GameDefinition,
    checkpoint_dir: &str,
    reward_norm: &train::normalize::RunningMeanStd,
    opt: &train::adam::Adam,
) {
    let reason = if event_name.is_empty() {
        "DONE"
    } else {
        event_name
    };
    if let Some(i) = env_idx {
        println!(
            "\rEpisode {:5} (env {}) | {} | reward: {:+.1} | frames: {} | total: {} | updates: {}",
            episode, i, reason, episode_reward, episode_frames, total_frames, update_count
        );
    } else {
        println!(
            "\rEpisode {:4} | {} | reward: {:+.1} | frames: {} | total: {} | updates: {}          ",
            episode, reason, episode_reward, episode_frames, total_frames, update_count
        );
    }
    if debug {
        println!("{}", tracker.episode_breakdown());
    }
    logger.log_episode(total_frames as usize, episode_reward, episode_frames);
    if let Some(i) = env_idx {
        logger.log_episode_for_env(total_frames as usize, i, episode_reward, episode_frames);
    }
    if episode_reward > *best_reward {
        *best_reward = episode_reward;
        save_checkpoint(
            checkpoint_dir,
            "best",
            model,
            game,
            episode,
            total_frames,
            update_count,
            *best_reward,
            reward_norm,
            opt,
        );
        println!(
            "  New best: {:+.1}\n{}",
            *best_reward,
            tracker.episode_breakdown()
        );
    }
}

pub(super) const DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(12);

pub(super) fn drain_until_idle(
    runner: &mut dyn GameRunner,
    tracker: &mut dyn GameTracker,
    total_frames: &mut u64,
    timeout: std::time::Duration,
) -> bool {
    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        match runner.next_frame() {
            Ok(pixels) => {
                runner.execute_action(0);
                *total_frames += 1;
                if tracker.observe_idle(&pixels) {
                    return true;
                }
            }
            Err(e) => {
                eprintln!("Frame source closed during idle drain: {}", e);
                return false;
            }
        }
    }
    eprintln!(
        "Idle drain exceeded {:?} — reset key sequence likely missed the game window",
        timeout
    );
    false
}
