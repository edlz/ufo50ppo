// Multi-env PPO trainer. Each tick reads one frame from every game (each blocks until
// answered), runs one batched inference pass and answers them all. PPO updates once every
// env has a full rollout, with GAE computed per env.

use crate::games::{FrameResult, GameTracker};
use crate::platform::{Frame, GameRunner};
use crate::train::adam::Adam;
use crate::train::config::{self, GameDefinition, TrainingConfig};
use crate::train::model::ActorCritic;
use crate::train::normalize::RunningMeanStd;
use crate::train::ppo::{self, PpoConfig, RolloutBuffer, compute_gae};
use crate::train::preprocess::FrameStack;
use crate::util::checkpoint::{self, CheckpointMeta};
use crate::util::logger::TbLogger;
use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::{Duration, Instant};
use tch::Tensor;

// Generous: a relaunch loads the game's textures and audio while the env is blocked.
const FRAME_TIMEOUT: Duration = Duration::from_secs(10);
/// Wall-clock budget for a reset (relaunch + title screen) before it is retried.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(30);
const LOG_INTERVAL: Duration = Duration::from_secs(5);

/// The inputs of the last step, completed into a transition when its reward arrives.
struct PrevStep {
    obs: Tensor,
    action: i64,
    log_prob: f64,
    value: f64,
}

/// One game copy and everything the trainer tracks for it.
struct Env {
    runner: Box<dyn GameRunner>,
    tracker: Box<dyn GameTracker>,
    stack: FrameStack,
    buffer: RolloutBuffer,
    prev: Option<PrevStep>,
    /// When the env started waiting for a fresh run; None while playing.
    draining_since: Option<Instant>,
    /// Only false once the game connection drops or times out.
    alive: bool,
    /// Discounted return so far, fed to the shared reward normalizer.
    return_acc: f64,
    ep_reward: f64,
    ep_frames: u32,
    /// Most recent finished episode's reward, NaN until the first one ends.
    last_ep_reward: f64,
}

impl Env {
    fn new(runner: Box<dyn GameRunner>, game: &GameDefinition, device: tch::Device) -> Self {
        Self {
            runner,
            tracker: (game.make_tracker)(),
            stack: FrameStack::new(device, game.obs_width, game.obs_height),
            buffer: RolloutBuffer::new(game.hyperparams.rollout_len * 2),
            prev: None,
            draining_since: None,
            alive: true,
            return_acc: 0.0,
            ep_reward: 0.0,
            ep_frames: 0,
            last_ep_reward: f64::NAN,
        }
    }

    /// Relaunches the game; the env drains until `observe_idle` reports a fresh run.
    fn restart(&mut self) {
        self.runner.reset_game();
        self.tracker.reset_episode();
        self.stack.reset();
        self.prev = None;
        self.return_acc = 0.0;
        self.draining_since = Some(Instant::now());
    }

    fn recover(&mut self, game: &GameDefinition, i: usize, during: &str) {
        eprintln!(
            "[multi] env {i} tracker.{during} panicked — rebuilding tracker and retrying reset"
        );
        self.tracker = (game.make_tracker)();
        self.restart();
        self.runner.execute_action(0);
    }

    /// Gives `result`'s reward to the previous step's transition. Menu frames have no
    /// transition of their own, so an event there is added to the last one instead.
    fn record(&mut self, result: &FrameResult) {
        if let Some(p) = self.prev.take() {
            self.buffer.push(
                p.obs,
                p.action,
                p.log_prob,
                result.reward,
                p.value,
                result.done,
            );
            self.ep_reward += result.reward;
        } else if result.is_menu
            && let Some(last) = self.buffer.rewards.last_mut()
        {
            *last += result.reward;
            if result.done
                && let Some(done) = self.buffer.dones.last_mut()
            {
                *done = true;
            }
            self.ep_reward += result.reward;
        }
    }
}

/// The model, its optimizer and the run's counters, which together make up a checkpoint.
struct Learner {
    game: GameDefinition,
    model: ActorCritic,
    opt: Adam,
    ppo_cfg: PpoConfig,
    reward_norm: RunningMeanStd,
    logger: TbLogger,
    checkpoint_dir: String,
    debug: bool,
    episode: u32,
    total_frames: u64,
    updates: u64,
    best_reward: f64,
}

struct UpdateStats {
    entropy: f64,
    explained_variance: f64,
    grad_norm: f64,
}

impl Learner {
    fn save(&self, name: &str) {
        let hp = &self.game.hyperparams;
        let meta = CheckpointMeta {
            game: self.game.name,
            resolution: (self.game.obs_width, self.game.obs_height),
            episode: self.episode,
            total_frames: self.total_frames,
            ppo_updates: self.updates,
            best_reward: self.best_reward,
            rollout_len: hp.rollout_len,
            learning_rate: self.opt.lr,
            gamma: hp.gamma,
            gae_lambda: hp.gae_lambda,
            reward_norm_mean: self.reward_norm.mean,
            reward_norm_var_sum: self.reward_norm.var_sum,
            reward_norm_count: self.reward_norm.count,
        };
        checkpoint::save(&self.checkpoint_dir, name, &self.model.vs, &self.opt, &meta);
    }

    fn resume(&mut self) {
        match checkpoint::try_load(&self.checkpoint_dir, "latest", &mut self.model.vs) {
            Ok(Some(state)) => {
                self.episode = state.episode;
                self.total_frames = state.total_frames;
                self.updates = state.ppo_updates;
                self.best_reward = state.best_reward;
                self.reward_norm.mean = state.reward_norm_mean;
                self.reward_norm.var_sum = state.reward_norm_var_sum;
                self.reward_norm.count = state.reward_norm_count;
                let adam_path = format!("{}/latest.adam", self.checkpoint_dir);
                if std::path::Path::new(&adam_path).exists()
                    && let Err(e) = self.opt.load_state(&adam_path)
                {
                    eprintln!("[multi] Adam state load failed: {e} (continuing with fresh Adam)");
                }
                println!(
                    "[multi] resumed: ep={} frames={} updates={} best={:.1} | reward_std={:.3} | adam_step={}",
                    self.episode,
                    self.total_frames,
                    self.updates,
                    self.best_reward,
                    self.reward_norm.std(),
                    self.opt.step_count
                );
            }
            Ok(None) => {}
            Err(e) => eprintln!("[multi] failed to load latest checkpoint: {e}"),
        }
    }

    /// Logs env `i`'s finished episode, saves `best` on a new record and clears its stats.
    fn end_episode(&mut self, i: usize, env: &mut Env, event_name: &str) {
        self.episode += 1;
        let (reward, frames) = (env.ep_reward, env.ep_frames);
        env.last_ep_reward = reward;
        env.ep_reward = 0.0;
        env.ep_frames = 0;

        let reason = if event_name.is_empty() {
            "DONE"
        } else {
            event_name
        };
        println!(
            "\rEpisode {:5} (env {}) | {} | reward: {:+.1} | frames: {} | total: {} | updates: {}",
            self.episode, i, reason, reward, frames, self.total_frames, self.updates
        );
        if self.debug {
            println!("{}", env.tracker.episode_breakdown());
        }
        let step = self.total_frames as usize;
        self.logger.log_episode(step, reward, frames);
        self.logger.log_episode_for_env(step, i, reward, frames);
        for (name, value) in env.tracker.episode_metrics() {
            self.logger.log_scalar(&format!("game/{name}"), value, step);
        }
        if reward > self.best_reward {
            self.best_reward = reward;
            self.save("best");
            println!(
                "  New best: {:+.1}\n{}",
                reward,
                env.tracker.episode_breakdown()
            );
        }
    }

    /// One PPO update over the given envs' buffers, with GAE computed per env.
    fn update(&mut self, envs: &[&Env]) -> UpdateStats {
        let hp = &self.game.hyperparams;
        self.opt.set_lr(hp.learning_rate_at(self.total_frames));
        let total_size: usize = envs.iter().map(|e| e.buffer.len()).sum();
        let mut merged = RolloutBuffer::new(total_size);
        let mut advantages = Vec::with_capacity(total_size);
        let mut returns = Vec::with_capacity(total_size);
        for env in envs {
            let buf = &env.buffer;
            let last_value = env.prev.as_ref().map_or(0.0, |p| p.value);
            let (adv, ret) = compute_gae(
                &buf.rewards,
                &buf.values,
                &buf.dones,
                last_value,
                hp.gamma,
                hp.gae_lambda,
                self.reward_norm.std(),
            );
            for j in 0..buf.len() {
                merged.push(
                    buf.obs[j].shallow_clone(),
                    buf.actions[j],
                    buf.log_probs[j],
                    buf.rewards[j],
                    buf.values[j],
                    buf.dones[j],
                );
            }
            advantages.extend(adv);
            returns.extend(ret);
        }

        let stats = ppo::update(
            &mut self.model,
            &mut self.opt,
            &merged,
            &advantages,
            &returns,
            &self.ppo_cfg,
        );
        self.updates += 1;
        let step = self.total_frames as usize;
        self.logger.log_update(
            step,
            stats.policy_loss,
            stats.value_loss,
            stats.entropy,
            stats.total_loss,
            stats.grad_norm,
        );
        self.logger.log_learning_rate(step, self.opt.lr);
        let ev = ppo::explained_variance(&merged.values, &returns);
        self.logger.log_explained_variance(step, ev);
        UpdateStats {
            entropy: stats.entropy,
            explained_variance: ev,
            grad_norm: stats.grad_norm,
        }
    }
}

pub fn run_training_multi(
    runners: Vec<Box<dyn GameRunner>>,
    game: GameDefinition,
    cfg: TrainingConfig,
) {
    let n = runners.len();
    if n == 0 {
        eprintln!("run_training_multi: no envs provided");
        return;
    }
    let TrainingConfig {
        max_episodes,
        max_frames,
        max_minutes,
        debug,
        checkpoint_dir,
        runs_dir,
    } = cfg;
    let training_start = Instant::now();
    let max_duration = max_minutes.map(|m| Duration::from_secs(m * 60));

    let device = config::pick_device();
    println!("[multi] N={} envs", n);
    config::log_device(device);

    std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create checkpoint directory");
    checkpoint::cleanup_stale_tmp(&checkpoint_dir);
    checkpoint::write_run_metadata(&checkpoint_dir, game.name, n, &game.hyperparams);

    let mut envs: Vec<Env> = runners
        .into_iter()
        .map(|r| Env::new(r, &game, device))
        .collect();
    let model = ActorCritic::new(device, game.obs_width, game.obs_height, game.num_actions);
    let opt = model.optimizer(game.hyperparams.learning_rate);
    let mut learner = Learner {
        ppo_cfg: PpoConfig {
            minibatch_size: game.hyperparams.minibatch_size,
            ..PpoConfig::default()
        },
        game,
        model,
        opt,
        reward_norm: RunningMeanStd::new(),
        logger: TbLogger::new(&runs_dir, 0),
        checkpoint_dir,
        debug,
        episode: 0,
        total_frames: 0,
        updates: 0,
        best_reward: f64::NEG_INFINITY,
    };
    learner.resume();
    for env in &mut envs {
        env.restart();
    }

    let mut last_stats = UpdateStats {
        entropy: f64::NAN,
        explained_variance: f64::NAN,
        grad_norm: f64::NAN,
    };
    let mut grad_norm_history: VecDeque<f64> = VecDeque::with_capacity(100);
    let mut last_log = Instant::now();
    let mut last_log_frames = learner.total_frames;

    'outer: loop {
        if crate::util::shutdown::requested() {
            println!("\n[multi] shutdown requested, saving checkpoint...");
            break;
        }

        // 1. Read a frame from every alive env. A disconnect marks the env dead; training
        // continues with the rest until none remain.
        let frames_before_tick = learner.total_frames;
        let mut frames: Vec<Option<Frame>> = Vec::with_capacity(n);
        for (i, env) in envs.iter_mut().enumerate() {
            if !env.alive {
                frames.push(None);
                continue;
            }
            match env.runner.next_frame_timeout(FRAME_TIMEOUT) {
                Ok(f) => {
                    frames.push(Some(f));
                    learner.total_frames += 1;
                }
                Err(e) => {
                    eprintln!("[multi] env {i} disconnected: {e} — env is dead");
                    env.alive = false;
                    frames.push(None);
                }
            }
        }
        if envs.iter().all(|e| !e.alive) {
            eprintln!("[multi] all envs dead, exiting");
            break;
        }

        // 2. Advance each env's tracker; collect the envs that need an action from the model.
        let mut active: Vec<(usize, Tensor)> = Vec::with_capacity(n);
        for (i, (env, frame)) in envs.iter_mut().zip(&frames).enumerate() {
            let Some(frame) = frame else { continue };

            if let Some(since) = env.draining_since {
                if since.elapsed() > DRAIN_TIMEOUT {
                    eprintln!(
                        "[multi] env {i} drain timeout after {:?} — retrying reset",
                        since.elapsed()
                    );
                    env.restart();
                    env.runner.execute_action(0);
                    continue;
                }
                match catch_unwind(AssertUnwindSafe(|| env.tracker.observe_idle(frame))) {
                    Ok(Some(action)) => env.runner.execute_action(action),
                    Ok(None) => {
                        env.draining_since = None;
                        env.stack.reset();
                        env.runner.execute_action(0);
                    }
                    Err(_) => env.recover(&learner.game, i, "observe_idle"),
                }
                continue;
            }

            let Ok(result) = catch_unwind(AssertUnwindSafe(|| env.tracker.process_frame(frame)))
            else {
                env.recover(&learner.game, i, "process_frame");
                continue;
            };
            env.return_acc = learner.game.hyperparams.gamma * env.return_acc + result.reward;
            learner.reward_norm.update(env.return_acc);

            if result.is_menu {
                env.runner.execute_action(env.tracker.menu_action());
            }
            // Menu frames only matter for their events: they have no transition of their own.
            if !result.is_menu || result.is_event || result.done {
                env.record(&result);
            }
            if result.done {
                if !result.is_menu {
                    env.runner.execute_action(0);
                }
                learner.end_episode(i, env, result.event_name);
                env.restart();
                if let Some(max_ep) = max_episodes
                    && learner.episode >= max_ep
                {
                    println!("[multi] reached {} episodes, stopping", max_ep);
                    break 'outer;
                }
                continue;
            }
            if result.is_menu {
                continue;
            }
            active.push((i, env.stack.push(&frame.pixels)));
            env.ep_frames += 1;
        }

        // 3. One batched inference pass for every env that is playing.
        if !active.is_empty() {
            let obs: Vec<Tensor> = active.iter().map(|(_, o)| o.shallow_clone()).collect();
            let (actions, log_probs, values) = learner.model.act_batch(&Tensor::cat(&obs, 0));
            for (idx, (i, obs)) in active.into_iter().enumerate() {
                envs[i].runner.execute_action(actions[idx] as usize);
                envs[i].prev = Some(PrevStep {
                    obs,
                    action: actions[idx],
                    log_prob: log_probs[idx],
                    value: values[idx],
                });
            }
        }

        // 4. PPO update once every alive env has a full rollout; dead envs are left out.
        let rollout_len = learner.game.hyperparams.rollout_len;
        if envs
            .iter()
            .all(|e| !e.alive || e.buffer.len() >= rollout_len)
        {
            let alive: Vec<&Env> = envs.iter().filter(|e| e.alive).collect();
            let stats = learner.update(&alive);
            if grad_norm_history.len() >= 20 {
                let trailing_mean =
                    grad_norm_history.iter().sum::<f64>() / grad_norm_history.len() as f64;
                if trailing_mean > 0.0 && stats.grad_norm > 5.0 * trailing_mean {
                    eprintln!(
                        "[multi] grad_norm spike: {:.3} (trailing mean {:.3}, ratio {:.1}x) — value function may be diverging",
                        stats.grad_norm,
                        trailing_mean,
                        stats.grad_norm / trailing_mean
                    );
                }
            }
            grad_norm_history.push_back(stats.grad_norm);
            if grad_norm_history.len() > 100 {
                grad_norm_history.pop_front();
            }
            last_stats = stats;
            for env in envs.iter_mut().filter(|e| e.alive) {
                env.buffer.clear();
            }
        }

        // 5. Checkpoints when this tick crossed a save interval.
        let hp = &learner.game.hyperparams;
        let crossed =
            |interval: u64| learner.total_frames / interval > frames_before_tick / interval;
        let (save_latest, save_versioned) = (
            crossed(hp.latest_save_interval),
            crossed(hp.versioned_save_interval),
        );
        if save_latest {
            learner.save("latest");
        }
        if save_versioned {
            learner.save(&format!("frame_{:08}", learner.total_frames));
        }

        // 6. Health log.
        if last_log.elapsed() >= LOG_INTERVAL {
            let fps =
                (learner.total_frames - last_log_frames) as f64 / last_log.elapsed().as_secs_f64();
            let step = learner.total_frames as usize;
            learner.logger.log_fps(step, fps);
            let finished: Vec<f64> = envs
                .iter()
                .map(|e| e.last_ep_reward)
                .filter(|r| !r.is_nan())
                .collect();
            let ep_min = finished.iter().copied().fold(f64::NAN, f64::min);
            let ep_max = finished.iter().copied().fold(f64::NAN, f64::max);
            if !finished.is_empty() {
                learner.logger.log_episode_spread(step, ep_min, ep_max);
            }
            println!(
                "[multi] frames={} ep={} envs={}/{} fps={:.0} ent={:.2} ev={:.2} grad={:.2} lr={:.1e} reward_std={:.2} best={:+.1} ep_range=[{:+.1},{:+.1}]",
                learner.total_frames,
                learner.episode,
                envs.iter().filter(|e| e.alive).count(),
                n,
                fps,
                last_stats.entropy,
                last_stats.explained_variance,
                last_stats.grad_norm,
                learner.opt.lr,
                learner.reward_norm.std(),
                learner.best_reward,
                ep_min,
                ep_max
            );
            last_log = Instant::now();
            last_log_frames = learner.total_frames;
        }

        // 7. Limits.
        if let Some(max_f) = max_frames
            && learner.total_frames >= max_f
        {
            println!("[multi] reached {} frames, stopping", max_f);
            break;
        }
        if let Some(d) = max_duration
            && training_start.elapsed() >= d
        {
            println!("[multi] reached {} minutes, stopping", d.as_secs() / 60);
            break;
        }
    }

    learner.save("latest");
}
