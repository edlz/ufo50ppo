// Synchronous multi-env PPO trainer.
//
// Architecture: N capture pumps (one per game window, spawned by host_multi) feed frames
// into this trainer thread via channels. Each tick the trainer captures from all N envs
// at a sync barrier, runs a single batched inference pass, and dispatches actions back.
//
// Per-env state machine:
//   - Active:   normal training (process_frame, push to rollout buffer, do inference)
//   - Draining: post-reset wait (observe_idle each tick, send noop until fresh state)
//
// PPO update fires when ALL envs have at least `rollout_len` transitions in their buffer.
// GAE is computed independently per env (preserving trajectory boundaries), then
// concatenated into one big batch for the actual gradient update.

use crate::games::GameTracker;
use crate::platform::GameRunner;
use crate::platform::win32::WindowInfo;
use crate::train;
use crate::train::model::ActorCritic;
use crate::train::ppo::{PpoConfig, RolloutBuffer, compute_gae};
use crate::train::preprocess::FrameStack;
use crate::train::runner::{GameDefinition, TrainingConfig, save_checkpoint};
use crate::util::checkpoint;
use tch::Tensor;

#[derive(Clone, Copy, PartialEq, Eq)]
enum EnvState {
    Active,
    Draining,
}

struct PrevStep {
    obs: Tensor,
    action: i64,
    log_prob: f64,
    value: f64,
}

pub fn run_training_multi(
    runners_with_info: Vec<(WindowInfo, Box<dyn GameRunner>)>,
    game: GameDefinition,
    cfg: TrainingConfig,
) {
    train::runner::reset_save_failure_counter();
    let n = runners_with_info.len();
    if n == 0 {
        eprintln!("run_training_multi: no envs provided");
        return;
    }

    let TrainingConfig {
        max_episodes,
        max_frames,
        max_minutes,
        auto_resume,
        debug,
        checkpoint_dir,
        runs_dir,
    } = cfg;

    if auto_resume && n > 1 {
        eprintln!(
            "note: -r/--auto-resume has no effect with N>1. Dead envs are dropped \
             and the trainer continues with survivors."
        );
    }

    let training_start = std::time::Instant::now();
    let max_duration = max_minutes.map(|m| std::time::Duration::from_secs(m * 60));

    let env_pids: Vec<u32> = runners_with_info.iter().map(|(info, _)| info.pid).collect();
    let mut runners: Vec<Box<dyn GameRunner>> =
        runners_with_info.into_iter().map(|(_, r)| r).collect();

    let w = runners[0].obs_width();
    let h = runners[0].obs_height();
    let device = tch::Device::cuda_if_available();
    println!("[multi] N={} envs | pids={:?}", n, env_pids);
    train::runner::log_device(device);

    let hp = &game.hyperparams;
    let mut model = ActorCritic::new(device, game.obs_width, game.obs_height, game.num_actions);
    let mut opt = model.optimizer(hp.learning_rate);
    let ppo_cfg = PpoConfig {
        minibatch_size: hp.minibatch_size,
        ..PpoConfig::default()
    };
    let mut logger = crate::util::logger::TbLogger::new(&runs_dir, 0);

    let mut trackers: Vec<Box<dyn GameTracker>> = (0..n)
        .map(|i| (game.make_tracker)(w, env_pids[i]))
        .collect();
    let mut frame_stacks: Vec<FrameStack> = (0..n)
        .map(|_| FrameStack::new(device, game.obs_width))
        .collect();
    let mut buffers: Vec<RolloutBuffer> = (0..n)
        .map(|_| RolloutBuffer::new(hp.rollout_len * 2))
        .collect();
    let mut prev: Vec<Option<PrevStep>> = (0..n).map(|_| None).collect();
    let mut env_state: Vec<EnvState> = vec![EnvState::Draining; n];
    let mut episode_reward: Vec<f64> = vec![0.0; n];
    let mut episode_frames: Vec<u32> = vec![0; n];
    // Most recent finished episode reward per env — used for the spread metric and the
    // health log. Set to NaN until each env has finished its first episode.
    let mut last_ep_reward: Vec<f64> = vec![f64::NAN; n];
    // Per-env liveness. Only set to false when the capture pump disconnects (game window
    // closed). Drain timeouts and tracker panics retry the reset instead of killing.
    let mut alive: Vec<bool> = vec![true; n];
    // When each env entered the Draining state, for per-env drain timeout enforcement.
    let mut drain_started: Vec<Option<std::time::Instant>> =
        vec![Some(std::time::Instant::now()); n];

    // Global counters
    let mut episode = 0u32;
    let mut total_frames = 0u64;
    let mut update_count = 0u64;
    let mut best_reward = f64::NEG_INFINITY;

    // Latest PPO update stats — captured for the diagnostic health log line.
    let mut last_entropy: f64 = f64::NAN;
    let mut last_ev: f64 = f64::NAN;
    let mut last_grad_norm: f64 = f64::NAN;
    // Ring buffer of recent grad norms for the spike alert.
    let mut grad_norm_history: std::collections::VecDeque<f64> =
        std::collections::VecDeque::with_capacity(100);

    // Reward normalizer (shared across envs, SB3-style). Per-env discounted return
    // accumulators feed into the same RunningMeanStd.
    let mut reward_norm = train::normalize::RunningMeanStd::new();
    let mut return_acc: Vec<f64> = vec![0.0; n];

    std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create checkpoint directory");
    train::runner::cleanup_stale_tmp(&checkpoint_dir);

    checkpoint::write_run_metadata(
        &checkpoint_dir,
        game.name,
        n as u32,
        hp.rollout_len,
        hp.learning_rate,
        hp.gamma,
        hp.gae_lambda,
    );

    match checkpoint::try_load(&checkpoint_dir, "latest", &mut model.vs) {
        Ok(Some(state)) => {
            episode = state.episode;
            total_frames = state.total_frames;
            update_count = state.ppo_updates;
            best_reward = state.best_reward;
            reward_norm.mean = state.reward_norm_mean;
            reward_norm.var_sum = state.reward_norm_var_sum;
            reward_norm.count = state.reward_norm_count;
            let adam_path = format!("{}/latest.adam", checkpoint_dir);
            if std::path::Path::new(&adam_path).exists()
                && let Err(e) = opt.load_state(&adam_path)
            {
                eprintln!(
                    "[multi] Adam state load failed: {} (continuing with fresh Adam)",
                    e
                );
            }
            println!(
                "[multi] resumed: ep={} frames={} updates={} best={:.1} | reward_std={:.3} | adam_step={}",
                episode,
                total_frames,
                update_count,
                best_reward,
                reward_norm.std(),
                opt.step_count
            );
        }
        Ok(None) => {}
        Err(e) => eprintln!("[multi] failed to load latest checkpoint: {}", e),
    }

    // Initial reset for all envs — kicks them into Draining state.
    for i in 0..n {
        runners[i].reset_game(trackers[i].reset_sequence(), trackers[i].reset_tap_ms());
        trackers[i].reset_episode();
    }

    let mut last_log = std::time::Instant::now();
    let mut last_log_frames = total_frames;

    const FRAME_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

    // ─── main loop ────────────────────────────────────────────────────────────
    'outer: loop {
        if crate::util::shutdown::requested() {
            println!("\n[multi] shutdown requested, saving checkpoint...");
            break 'outer;
        }

        // 1. Capture from all alive envs (sync barrier with per-env timeout). A disconnect
        // marks the env dead; the loop continues with N-1 envs until none remain.
        let mut pixels: Vec<Vec<u8>> = vec![Vec::new(); n];
        let frames_before_tick = total_frames;
        for i in 0..n {
            if !alive[i] {
                continue;
            }
            match runners[i].next_frame_timeout(FRAME_TIMEOUT) {
                Ok(p) => {
                    pixels[i] = p;
                    total_frames += 1;
                }
                Err(e) => {
                    eprintln!(
                        "[multi] env {} capture disconnected: {} — env is dead",
                        i, e
                    );
                    alive[i] = false;
                }
            }
        }
        if alive.iter().all(|a| !a) {
            eprintln!("[multi] all envs dead, exiting");
            break 'outer;
        }

        // 2. Per-env processing — collect indices of envs that need inference this tick
        let mut active_envs: Vec<usize> = Vec::with_capacity(n);
        let mut active_obs: Vec<Tensor> = Vec::with_capacity(n);

        for i in 0..n {
            if !alive[i] {
                continue;
            }
            match env_state[i] {
                EnvState::Draining => {
                    runners[i].execute_action(0);
                    // Per-env drain timeout: retry the reset sequence if observe_idle
                    // hasn't returned true within DRAIN_TIMEOUT.
                    if let Some(start) = drain_started[i]
                        && start.elapsed() > crate::train::runner::DRAIN_TIMEOUT
                    {
                        eprintln!(
                            "[multi] env {} drain timeout after {:?} — retrying reset",
                            i,
                            start.elapsed()
                        );
                        runners[i]
                            .reset_game(trackers[i].reset_sequence(), trackers[i].reset_tap_ms());
                        trackers[i].reset_episode();
                        drain_started[i] = Some(std::time::Instant::now());
                        continue;
                    }
                    let pixels_i = &pixels[i];
                    let observe_result =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            trackers[i].observe_idle(pixels_i)
                        }));
                    match observe_result {
                        Ok(true) => {
                            env_state[i] = EnvState::Active;
                            drain_started[i] = None;
                            frame_stacks[i].reset();
                        }
                        Ok(false) => {}
                        Err(_) => {
                            eprintln!(
                                "[multi] env {} tracker.observe_idle panicked — rebuilding tracker and retrying reset",
                                i
                            );
                            trackers[i] = (game.make_tracker)(w, env_pids[i]);
                            frame_stacks[i].reset();
                            runners[i].reset_game(
                                trackers[i].reset_sequence(),
                                trackers[i].reset_tap_ms(),
                            );
                            trackers[i].reset_episode();
                            prev[i] = None;
                            return_acc[i] = 0.0;
                            drain_started[i] = Some(std::time::Instant::now());
                            continue;
                        }
                    }
                }
                EnvState::Active => {
                    let pixels_i = &pixels[i];
                    let process_result =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            trackers[i].process_frame(pixels_i)
                        }));
                    let result = match process_result {
                        Ok(r) => r,
                        Err(_) => {
                            eprintln!(
                                "[multi] env {} tracker.process_frame panicked — rebuilding tracker and retrying reset",
                                i
                            );
                            trackers[i] = (game.make_tracker)(w, env_pids[i]);
                            frame_stacks[i].reset();
                            runners[i].reset_game(
                                trackers[i].reset_sequence(),
                                trackers[i].reset_tap_ms(),
                            );
                            trackers[i].reset_episode();
                            prev[i] = None;
                            return_acc[i] = 0.0;
                            env_state[i] = EnvState::Draining;
                            drain_started[i] = Some(std::time::Instant::now());
                            continue;
                        }
                    };

                    // Feed reward into the per-env discounted return accumulator and the
                    // shared normalizer. Reset on done is done after end_episode below.
                    return_acc[i] = hp.gamma * return_acc[i] + result.reward;
                    reward_norm.update(return_acc[i]);

                    if result.is_menu {
                        runners[i].execute_action(0);
                        // Propagate event reward across menu transitions, same as single-env runner
                        if result.is_event || result.done {
                            if let Some(p) = prev[i].take() {
                                buffers[i].push(
                                    p.obs,
                                    p.action,
                                    p.log_prob,
                                    result.reward,
                                    p.value,
                                    result.done,
                                );
                                episode_reward[i] += result.reward;
                            } else if let Some(last_reward) = buffers[i].rewards.last_mut() {
                                *last_reward += result.reward;
                                if result.done
                                    && let Some(last_done) = buffers[i].dones.last_mut()
                                {
                                    *last_done = true;
                                }
                                episode_reward[i] += result.reward;
                            }
                        }
                        if result.done {
                            end_episode_for_env(
                                i,
                                result.event_name,
                                &mut episode,
                                &mut episode_reward,
                                &mut episode_frames,
                                &mut last_ep_reward,
                                total_frames,
                                update_count,
                                debug,
                                &*trackers[i],
                                &mut logger,
                                &mut best_reward,
                                &model,
                                &game,
                                &checkpoint_dir,
                                &reward_norm,
                                &opt,
                            );
                            runners[i].reset_game(
                                trackers[i].reset_sequence(),
                                trackers[i].reset_tap_ms(),
                            );
                            trackers[i].reset_episode();
                            prev[i] = None;
                            return_acc[i] = 0.0;
                            env_state[i] = EnvState::Draining;
                            drain_started[i] = Some(std::time::Instant::now());
                            if let Some(max_ep) = max_episodes
                                && episode >= max_ep
                            {
                                println!("[multi] reached {} episodes, stopping", max_ep);
                                break 'outer;
                            }
                        }
                        continue;
                    }

                    // Normal gameplay frame: pair prev with this frame's reward, push.
                    if let Some(p) = prev[i].take() {
                        buffers[i].push(
                            p.obs,
                            p.action,
                            p.log_prob,
                            result.reward,
                            p.value,
                            result.done,
                        );
                        episode_reward[i] += result.reward;
                    }

                    if result.done {
                        runners[i].execute_action(0);
                        end_episode_for_env(
                            i,
                            result.event_name,
                            &mut episode,
                            &mut episode_reward,
                            &mut episode_frames,
                            &mut last_ep_reward,
                            total_frames,
                            update_count,
                            debug,
                            &*trackers[i],
                            &mut logger,
                            &mut best_reward,
                            &model,
                            &game,
                            &checkpoint_dir,
                            &reward_norm,
                            &opt,
                        );
                        runners[i]
                            .reset_game(trackers[i].reset_sequence(), trackers[i].reset_tap_ms());
                        trackers[i].reset_episode();
                        prev[i] = None;
                        return_acc[i] = 0.0;
                        env_state[i] = EnvState::Draining;
                        drain_started[i] = Some(std::time::Instant::now());
                        if let Some(max_ep) = max_episodes
                            && episode >= max_ep
                        {
                            println!("[multi] reached {} episodes, stopping", max_ep);
                            break 'outer;
                        }
                        continue;
                    }

                    let obs = frame_stacks[i].push(&pixels[i], w, h);
                    active_envs.push(i);
                    active_obs.push(obs);
                    episode_frames[i] += 1;
                }
            }
        }

        // 3. Batched inference for active envs only
        if !active_envs.is_empty() {
            let obs_batch = Tensor::cat(&active_obs, 0);
            let (actions, log_probs, values) = model.act_batch(&obs_batch);

            for (idx, &i) in active_envs.iter().enumerate() {
                let action = actions[idx];
                runners[i].execute_action(action as usize);
                prev[i] = Some(PrevStep {
                    obs: active_obs[idx].shallow_clone(),
                    action,
                    log_prob: log_probs[idx],
                    value: values[idx],
                });
            }
        }

        // 4. PPO update fires when all ALIVE envs have filled their per-env buffer.
        // Dead envs are excluded from both the gate and the merge.
        let all_full = (0..n).all(|i| !alive[i] || buffers[i].len() >= hp.rollout_len);
        if all_full && alive.iter().any(|a| *a) {
            let alive_idx: Vec<usize> = (0..n).filter(|&i| alive[i]).collect();
            let alive_buffers: Vec<&RolloutBuffer> =
                alive_idx.iter().map(|&i| &buffers[i]).collect();
            let last_values: Vec<f64> = alive_idx
                .iter()
                .map(|&i| prev[i].as_ref().map(|p| p.value).unwrap_or(0.0))
                .collect();

            let stats = run_multi_ppo_update(
                &mut model,
                &mut opt,
                &alive_buffers,
                &last_values,
                hp,
                &ppo_cfg,
                reward_norm.std(),
                &mut update_count,
                total_frames,
                &mut logger,
            );
            last_entropy = stats.entropy;
            last_ev = stats.explained_variance;
            last_grad_norm = stats.grad_norm;

            // grad_norm spike alert: warn when current is >5x the trailing mean.
            if grad_norm_history.len() >= 20 {
                let trailing_mean: f64 =
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

            for &i in &alive_idx {
                buffers[i].clear();
            }
        }

        let prev_total = frames_before_tick;
        let crossed_latest = total_frames > 0
            && (total_frames / hp.latest_save_interval) > (prev_total / hp.latest_save_interval);
        if crossed_latest {
            save_checkpoint(
                &checkpoint_dir,
                "latest",
                &model,
                &game,
                episode,
                total_frames,
                update_count,
                best_reward,
                &reward_norm,
                &opt,
            );
        }
        let crossed_versioned = total_frames > 0
            && (total_frames / hp.versioned_save_interval)
                > (prev_total / hp.versioned_save_interval);
        if crossed_versioned {
            let name = format!("frame_{:08}", total_frames);
            save_checkpoint(
                &checkpoint_dir,
                &name,
                &model,
                &game,
                episode,
                total_frames,
                update_count,
                best_reward,
                &reward_norm,
                &opt,
            );
        }

        // 6. Periodic health log + tensorboard spread metric.
        if last_log.elapsed() >= std::time::Duration::from_secs(5) {
            let dt = last_log.elapsed().as_secs_f64();
            let frames_delta = total_frames - last_log_frames;
            let fps = frames_delta as f64 / dt;
            let alive_count = alive.iter().filter(|a| **a).count();
            let valid_eps: Vec<f64> = last_ep_reward
                .iter()
                .copied()
                .filter(|r| !r.is_nan())
                .collect();
            let (ep_min, ep_max) = if valid_eps.is_empty() {
                (f64::NAN, f64::NAN)
            } else {
                (
                    valid_eps.iter().cloned().fold(f64::INFINITY, f64::min),
                    valid_eps.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                )
            };
            if !ep_min.is_nan() {
                logger.log_episode_spread(total_frames as usize, ep_min, ep_max);
            }
            println!(
                "[multi] frames={} ep={} envs={}/{} fps={:.0} ent={:.2} ev={:.2} grad={:.2} reward_std={:.2} best={:+.1} ep_range=[{:+.1},{:+.1}]",
                total_frames,
                episode,
                alive_count,
                n,
                fps,
                last_entropy,
                last_ev,
                last_grad_norm,
                reward_norm.std(),
                best_reward,
                ep_min,
                ep_max
            );
            last_log = std::time::Instant::now();
            last_log_frames = total_frames;
        }

        // 7. Limit checks
        if let Some(max_f) = max_frames
            && total_frames >= max_f
        {
            println!("[multi] reached {} frames, stopping", max_f);
            break;
        }
        if let Some(d) = max_duration
            && training_start.elapsed() >= d
        {
            println!("[multi] reached {} minutes, stopping", max_minutes.unwrap());
            break;
        }
    }

    // Final checkpoint save
    save_checkpoint(
        &checkpoint_dir,
        "latest",
        &model,
        &game,
        episode,
        total_frames,
        update_count,
        best_reward,
        &reward_norm,
        &opt,
    );
}

#[allow(clippy::too_many_arguments)]
fn end_episode_for_env(
    env_idx: usize,
    event_name: &str,
    episode: &mut u32,
    episode_reward: &mut [f64],
    episode_frames: &mut [u32],
    last_ep_reward: &mut [f64],
    total_frames: u64,
    update_count: u64,
    debug: bool,
    tracker: &dyn GameTracker,
    logger: &mut crate::util::logger::TbLogger,
    best_reward: &mut f64,
    model: &ActorCritic,
    game: &GameDefinition,
    checkpoint_dir: &str,
    reward_norm: &train::normalize::RunningMeanStd,
    opt: &train::adam::Adam,
) {
    *episode += 1;
    last_ep_reward[env_idx] = episode_reward[env_idx];
    train::runner::report_episode_end(
        *episode,
        Some(env_idx),
        event_name,
        episode_reward[env_idx],
        episode_frames[env_idx],
        total_frames,
        update_count,
        debug,
        tracker,
        logger,
        best_reward,
        model,
        game,
        checkpoint_dir,
        reward_norm,
        opt,
    );
    episode_reward[env_idx] = 0.0;
    episode_frames[env_idx] = 0;
}

struct MultiUpdateStats {
    entropy: f64,
    explained_variance: f64,
    grad_norm: f64,
}

#[allow(clippy::too_many_arguments)]
fn run_multi_ppo_update(
    model: &mut ActorCritic,
    opt: &mut train::adam::Adam,
    buffers: &[&RolloutBuffer],
    last_values: &[f64],
    hp: &train::runner::Hyperparams,
    ppo_cfg: &PpoConfig,
    reward_std: f64,
    update_count: &mut u64,
    total_frames: u64,
    logger: &mut crate::util::logger::TbLogger,
) -> MultiUpdateStats {
    // GAE per env, then concatenate everything into one big buffer for the update.
    let total_size: usize = buffers.iter().map(|b| b.len()).sum();
    let mut merged = RolloutBuffer::new(total_size);
    let mut all_advantages: Vec<f64> = Vec::with_capacity(total_size);
    let mut all_returns: Vec<f64> = Vec::with_capacity(total_size);

    for (i, buf) in buffers.iter().enumerate() {
        let (adv, ret) = compute_gae(
            &buf.rewards,
            &buf.values,
            &buf.dones,
            last_values[i],
            hp.gamma,
            hp.gae_lambda,
            reward_std,
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
        all_advantages.extend(adv);
        all_returns.extend(ret);
    }

    let stats = train::ppo::update(model, opt, &merged, &all_advantages, &all_returns, ppo_cfg);
    *update_count += 1;
    logger.log_update(
        total_frames as usize,
        stats.policy_loss,
        stats.value_loss,
        stats.entropy,
        stats.total_loss,
        stats.grad_norm,
    );
    logger.log_learning_rate(total_frames as usize, hp.learning_rate);
    let ev = train::ppo::explained_variance(&merged.values, &all_returns);
    logger.log_explained_variance(total_frames as usize, ev);

    MultiUpdateStats {
        entropy: stats.entropy,
        explained_variance: ev,
        grad_norm: stats.grad_norm,
    }
}
