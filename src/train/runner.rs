use crate::games::GameTracker;
use crate::platform::GameRunner;
use crate::train;
use crate::util::checkpoint::{self, CheckpointMeta};
use std::sync::mpsc;

pub struct Hyperparams {
    pub rollout_len: usize,
    pub minibatch_size: usize,
    pub learning_rate: f64,
    pub gamma: f64,
    pub gae_lambda: f64,
    pub episode_timeout_secs: u64,
    pub latest_save_interval: u64,
    pub versioned_save_interval: u64,
}

impl Default for Hyperparams {
    fn default() -> Self {
        Self {
            rollout_len: 1024,
            minibatch_size: 128,
            learning_rate: 3e-4,
            gamma: 0.99,
            gae_lambda: 0.95,
            episode_timeout_secs: 60,
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
    pub make_tracker: fn(u32) -> Box<dyn GameTracker>,
    pub hyperparams: Hyperparams,
    /// Compute a debug filename suffix for a frame given its event name and reward.
    /// Returns empty string for frames that shouldn't be tagged.
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

pub enum DebugMsg {
    Frame(Vec<u8>, String),
    NewEpisode(u32),
}

fn save_checkpoint(
    dir: &str,
    name: &str,
    model: &train::model::ActorCritic,
    game: &GameDefinition,
    episode: u32,
    frames: u64,
    updates: u64,
    best: f64,
) {
    let path = format!("{}/{}.safetensors", dir, name);
    model
        .vs
        .save(&path)
        .unwrap_or_else(|e| eprintln!("Save error: {}", e));
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
        },
    );
}

fn write_frame_bmp(path: &str, pixels: &[u8], obs_w: u32, obs_h: u32) {
    use std::io::Write;
    let row_bytes = (obs_w * 3 + 3) & !3;
    let pixel_size = row_bytes * obs_h;
    let file_size = 54 + pixel_size;
    let Ok(mut f) = std::fs::File::create(path) else {
        return;
    };
    let _ = f.write_all(b"BM");
    let _ = f.write_all(&file_size.to_le_bytes());
    let _ = f.write_all(&0u32.to_le_bytes());
    let _ = f.write_all(&54u32.to_le_bytes());
    let _ = f.write_all(&40u32.to_le_bytes());
    let _ = f.write_all(&(obs_w as i32).to_le_bytes());
    let _ = f.write_all(&(-(obs_h as i32)).to_le_bytes());
    let _ = f.write_all(&1u16.to_le_bytes());
    let _ = f.write_all(&24u16.to_le_bytes());
    let _ = f.write_all(&0u32.to_le_bytes());
    let _ = f.write_all(&pixel_size.to_le_bytes());
    let _ = f.write_all(&[0u8; 16]);
    let mut row_buf = vec![0u8; row_bytes as usize];
    for y in 0..obs_h as usize {
        for x in 0..obs_w as usize {
            let src = y * obs_w as usize * 4 + x * 4;
            let dst = x * 3;
            row_buf[dst] = pixels[src];
            row_buf[dst + 1] = pixels[src + 1];
            row_buf[dst + 2] = pixels[src + 2];
        }
        let _ = f.write_all(&row_buf);
    }
}

fn save_best_frames(
    dir: &str,
    episode: u32,
    episode_reward: f64,
    frames: &[Vec<u8>],
    obs_w: u32,
    obs_h: u32,
) {
    let out_dir = format!(
        "{}/best_ep_{:04}_r{:+}",
        dir, episode, episode_reward as i64
    );
    if std::fs::create_dir_all(&out_dir).is_err() {
        eprintln!("  Failed to create best frames dir {}", out_dir);
        return;
    }
    for (i, pixels) in frames.iter().enumerate() {
        let path = format!("{}/{:03}.bmp", out_dir, i);
        write_frame_bmp(&path, pixels, obs_w, obs_h);
    }
    println!("  Saved {} best frames to {}", frames.len(), out_dir);
}

/// Print the end-of-episode summary line, optionally print the breakdown, log to tensorboard,
/// and save a `best` checkpoint when episode_reward exceeds best_reward.
#[allow(clippy::too_many_arguments)]
fn report_episode_end(
    episode: u32,
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
) {
    let reason = if event_name.is_empty() {
        "DONE"
    } else {
        event_name
    };
    println!(
        "\rEpisode {:4} | {} | reward: {:+.1} | frames: {} | total: {} | updates: {}          ",
        episode, reason, episode_reward, episode_frames, total_frames, update_count
    );
    if debug {
        println!("{}", tracker.episode_breakdown());
    }
    logger.log_episode(total_frames as usize, episode_reward, episode_frames);
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
        );
        println!("  New best: {:+.1}", *best_reward);
    }
}

/// Run one PPO update over the current buffer (computes GAE, calls update, logs to tensorboard).
/// Caller is responsible for clearing the buffer afterward.
fn run_ppo_update(
    model: &mut train::model::ActorCritic,
    opt: &mut tch::nn::Optimizer,
    buffer: &train::ppo::RolloutBuffer,
    hp: &Hyperparams,
    ppo_cfg: &train::ppo::PpoConfig,
    last_value: f64,
    update_count: &mut u64,
    total_frames: u64,
    logger: &mut crate::util::logger::TbLogger,
) {
    let (advantages, returns) = train::ppo::compute_gae(
        &buffer.rewards,
        &buffer.values,
        &buffer.dones,
        last_value,
        hp.gamma,
        hp.gae_lambda,
    );
    let stats = train::ppo::update(model, opt, buffer, &advantages, &returns, ppo_cfg);
    *update_count += 1;
    logger.log_update(
        total_frames as usize,
        stats.policy_loss,
        stats.value_loss,
        stats.entropy,
        stats.total_loss,
    );
    logger.log_learning_rate(total_frames as usize, hp.learning_rate);
    let ev = train::ppo::explained_variance(&buffer.values, &returns);
    logger.log_explained_variance(total_frames as usize, ev);
}

fn drain_until_gameplay(
    runner: &mut dyn GameRunner,
    tracker: &dyn GameTracker,
    total_frames: &mut u64,
) -> bool {
    loop {
        match runner.next_frame() {
            Ok(pixels) => {
                runner.execute_action(0);
                *total_frames += 1;
                if !tracker.is_menu_screen(&pixels) {
                    return true;
                }
            }
            Err(e) => {
                eprintln!("Frame source closed during drain: {}", e);
                return false;
            }
        }
    }
}

pub fn run_training(
    mut runner: Box<dyn GameRunner>,
    game: GameDefinition,
    cfg: TrainingConfig,
    debug_tx: Option<mpsc::SyncSender<DebugMsg>>,
) {
    let TrainingConfig {
        max_episodes,
        max_frames,
        max_minutes,
        auto_resume,
        debug,
        checkpoint_dir,
        runs_dir,
    } = cfg;
    let training_start = std::time::Instant::now();
    let max_duration = max_minutes.map(|m| std::time::Duration::from_secs(m * 60));
    let w = runner.obs_width();
    let h = runner.obs_height();
    let device = tch::Device::cuda_if_available();
    println!("Training on: {:?}", device);

    let hp = &game.hyperparams;
    let episode_timeout = std::time::Duration::from_secs(hp.episode_timeout_secs);

    let mut model =
        train::model::ActorCritic::new(device, game.obs_width, game.obs_height, game.num_actions);
    let mut opt = model.optimizer(hp.learning_rate);
    let mut frame_stack = train::preprocess::FrameStack::new(device);
    let mut buffer = train::ppo::RolloutBuffer::new(hp.rollout_len);
    let ppo_cfg = train::ppo::PpoConfig {
        minibatch_size: hp.minibatch_size,
        ..train::ppo::PpoConfig::default()
    };
    let mut logger = crate::util::logger::TbLogger::new(&runs_dir, 0);
    let mut tracker: Box<dyn GameTracker> = (game.make_tracker)(w);

    let mut episode = 0u32;
    let mut episode_reward = 0.0f64;
    let mut episode_frames = 0u32;
    let mut total_frames = 0u64;
    let mut update_count = 0u64;
    let mut best_reward = f64::NEG_INFINITY;
    // Pending step (obs, action, log_prob, value) waiting to be paired with the next frame's
    // reward. Standard PPO needs r_{t+1} paired with (s_t, a_t), not r_t.
    let mut prev: Option<(tch::Tensor, i64, f64, f64)> = None;
    // Periodic frame snapshots (every 20 frames) of the current episode.
    // Saved as BMPs alongside best.safetensors when a new best episode is reached.
    let mut best_frames_buf: Vec<Vec<u8>> = Vec::new();

    std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create checkpoint directory");

    match checkpoint::try_load(&checkpoint_dir, "latest", &mut model.vs) {
        Ok(Some(state)) => {
            episode = state.episode;
            total_frames = state.total_frames;
            update_count = state.ppo_updates;
            best_reward = state.best_reward;
            println!(
                "Resuming: ep={} frames={} updates={} best={:.1}",
                episode, total_frames, update_count, best_reward
            );
        }
        Ok(None) => {}
        Err(e) => eprintln!("Failed to load latest checkpoint: {}", e),
    }

    runner.execute_action(0);

    let mut t_recv = std::time::Duration::ZERO;
    let mut t_preprocess = std::time::Duration::ZERO;
    let mut t_act = std::time::Duration::ZERO;
    let mut t_tracker = std::time::Duration::ZERO;
    let mut t_ppo = std::time::Duration::ZERO;
    let mut timing_frames = 0u32;
    let mut last_timing_print = std::time::Instant::now();
    let mut episode_start = std::time::Instant::now();

    macro_rules! reset_episode_state {
        () => {{
            runner.reset_game(tracker.reset_sequence(), tracker.reset_tap_ms());
            if !drain_until_gameplay(&mut *runner, &*tracker, &mut total_frames) {
                return;
            }
            frame_stack.reset();
            tracker = (game.make_tracker)(w);
            episode += 1;
            if let Some(ref dtx) = debug_tx {
                let _ = dtx.try_send(DebugMsg::NewEpisode(episode));
            }
            episode_reward = 0.0;
            episode_frames = 0;
            episode_start = std::time::Instant::now();
            prev = None;
            best_frames_buf.clear();
        }};
    }

    macro_rules! maybe_save_best_frames {
        () => {{
            if episode_reward > best_reward {
                save_best_frames(
                    &checkpoint_dir,
                    episode,
                    episode_reward,
                    &best_frames_buf,
                    game.obs_width,
                    game.obs_height,
                );
            }
        }};
    }

    macro_rules! check_max_episodes_and_break {
        () => {{
            if let Some(max) = max_episodes {
                if episode >= max {
                    println!("Reached {} episodes, stopping.", max);
                    if buffer.len() >= hp.minibatch_size {
                        run_ppo_update(
                            &mut model,
                            &mut opt,
                            &buffer,
                            hp,
                            &ppo_cfg,
                            0.0,
                            &mut update_count,
                            total_frames,
                            &mut logger,
                        );
                    }
                    save_checkpoint(
                        &checkpoint_dir,
                        "latest",
                        &model,
                        &game,
                        episode,
                        total_frames,
                        update_count,
                        best_reward,
                    );
                    break;
                }
            }
        }};
    }

    macro_rules! end_episode {
        ($event:expr) => {{
            report_episode_end(
                episode,
                $event,
                episode_reward,
                episode_frames,
                total_frames,
                update_count,
                debug,
                &*tracker,
                &mut logger,
                &mut best_reward,
                &model,
                &game,
                &checkpoint_dir,
            );
            reset_episode_state!();
            check_max_episodes_and_break!();
        }};
    }

    runner.reset_game(tracker.reset_sequence(), tracker.reset_tap_ms());
    if !drain_until_gameplay(&mut *runner, &*tracker, &mut total_frames) {
        return;
    }

    loop {
        let t0 = std::time::Instant::now();
        let pixels = match runner.next_frame() {
            Ok(p) => p,
            Err(_) => break,
        };
        t_recv += t0.elapsed();

        let track_start = std::time::Instant::now();
        let result = tracker.process_frame(&pixels);
        t_tracker += track_start.elapsed();
        total_frames += 1;

        if result.is_menu {
            runner.execute_action(0);
            if result.is_event || result.done {
                let pushed = if let Some((p_obs, p_action, p_log_prob, p_value)) = prev.take() {
                    buffer.push(
                        p_obs,
                        p_action,
                        p_log_prob,
                        result.reward,
                        p_value,
                        result.done,
                    );
                    true
                } else if let Some(last_reward) = buffer.rewards.last_mut() {
                    *last_reward += result.reward;
                    if result.done {
                        if let Some(last_done) = buffer.dones.last_mut() {
                            *last_done = true;
                        }
                    }
                    true
                } else {
                    false
                };
                if pushed {
                    episode_reward += result.reward;
                }
            }
            if result.done {
                maybe_save_best_frames!();
                end_episode!(result.event_name);
            }
            continue;
        }

        let reward = result.reward;
        let done = result.done;
        episode_frames += 1;

        // Pair the previous step's (s, a, log_prob, V) with the reward observed at this frame.
        if let Some((p_obs, p_action, p_log_prob, p_value)) = prev.take() {
            buffer.push(p_obs, p_action, p_log_prob, reward, p_value, done);
            episode_reward += reward;
        }

        // Terminal frame: skip the wasted forward pass and go straight to episode reset.
        if done {
            runner.execute_action(0);
            maybe_save_best_frames!();
            end_episode!(result.event_name);
            continue;
        }

        let pre_start = std::time::Instant::now();
        let obs = frame_stack.push(&pixels, w, h);
        t_preprocess += pre_start.elapsed();

        let act_start = std::time::Instant::now();
        let (action, log_prob, value) = model.act(&obs);
        t_act += act_start.elapsed();

        runner.execute_action(action as usize);

        let time_up =
            max_duration.is_some_and(|d| total_frames % 100 == 0 && training_start.elapsed() >= d);
        if time_up || max_frames.is_some_and(|max| total_frames >= max) {
            if time_up {
                println!("\rReached {} minutes, stopping.", max_minutes.unwrap());
            } else {
                println!("\rReached {} frames, stopping.", max_frames.unwrap());
            }
            if buffer.len() >= hp.minibatch_size {
                run_ppo_update(
                    &mut model,
                    &mut opt,
                    &buffer,
                    hp,
                    &ppo_cfg,
                    value,
                    &mut update_count,
                    total_frames,
                    &mut logger,
                );
            }
            save_checkpoint(
                &checkpoint_dir,
                "latest",
                &model,
                &game,
                episode,
                total_frames,
                update_count,
                best_reward,
            );
            break;
        }

        if episode_start.elapsed() > episode_timeout {
            eprintln!(
                "\rEpisode {:4} TIMEOUT after {:?}",
                episode,
                episode_start.elapsed()
            );
            if !auto_resume {
                eprintln!("Exiting due to timeout.");
                return;
            }
            match checkpoint::try_load(&checkpoint_dir, "latest", &mut model.vs) {
                Ok(Some(state)) => {
                    episode = state.episode;
                    total_frames = state.total_frames;
                    update_count = state.ppo_updates;
                    best_reward = state.best_reward;
                    eprintln!(
                        "  Reloaded latest: ep={} frames={} updates={}",
                        episode, total_frames, update_count
                    );
                }
                Ok(None) => {
                    eprintln!("  No latest checkpoint to reload, exiting.");
                    return;
                }
                Err(e) => {
                    eprintln!("  Failed to reload latest: {}", e);
                    return;
                }
            }
            // Adam moments are stale after vs.load — rebuild to match the freshly loaded weights.
            opt = model.optimizer(hp.learning_rate);
            buffer.clear();
            reset_episode_state!();
            continue;
        }

        timing_frames += 1;

        if buffer.len() >= hp.rollout_len {
            let ppo_start = std::time::Instant::now();
            run_ppo_update(
                &mut model,
                &mut opt,
                &buffer,
                hp,
                &ppo_cfg,
                value,
                &mut update_count,
                total_frames,
                &mut logger,
            );
            t_ppo += ppo_start.elapsed();
            buffer.clear();
        }

        prev = Some((obs, action, log_prob, value));
        if episode_frames % 20 == 1 {
            best_frames_buf.push(pixels.clone());
        }

        if let Some(ref dtx) = debug_tx {
            let suffix = (game.debug_frame_suffix)(result.event_name, reward);
            let _ = dtx.try_send(DebugMsg::Frame(pixels, suffix));
        }

        if timing_frames > 0 && last_timing_print.elapsed() >= std::time::Duration::from_secs(300) {
            let n = timing_frames as f64;
            println!(
                "\r[timing] recv: {:.1}ms  preprocess: {:.1}ms  act: {:.1}ms  tracker: {:.1}ms  ppo: {:.1}ms (per frame avg)     ",
                t_recv.as_secs_f64() / n * 1000.0,
                t_preprocess.as_secs_f64() / n * 1000.0,
                t_act.as_secs_f64() / n * 1000.0,
                t_tracker.as_secs_f64() / n * 1000.0,
                t_ppo.as_secs_f64() / n * 1000.0,
            );
            let fps = n / (t_recv + t_preprocess + t_act + t_tracker + t_ppo).as_secs_f64();
            logger.log_fps(total_frames as usize, fps);
            t_recv = std::time::Duration::ZERO;
            t_preprocess = std::time::Duration::ZERO;
            t_act = std::time::Duration::ZERO;
            t_tracker = std::time::Duration::ZERO;
            t_ppo = std::time::Duration::ZERO;
            timing_frames = 0;
            last_timing_print = std::time::Instant::now();
        }

        if episode_frames % 10 == 0 {
            print!(
                "\rEp {:3} | reward: {:+7.1} | lives: {} | frame: {} | updates: {} | {}     ",
                episode,
                episode_reward,
                result.lives,
                episode_frames,
                update_count,
                result.event_name
            );
        }

        if total_frames > 0 && total_frames % hp.latest_save_interval == 0 {
            save_checkpoint(
                &checkpoint_dir,
                "latest",
                &model,
                &game,
                episode,
                total_frames,
                update_count,
                best_reward,
            );
        }
        if total_frames > 0 && total_frames % hp.versioned_save_interval == 0 {
            save_checkpoint(
                &checkpoint_dir,
                &format!("frame_{:08}", total_frames),
                &model,
                &game,
                episode,
                total_frames,
                update_count,
                best_reward,
            );
        }
    }
}

pub fn spawn_debug_thread(obs_w: u32, obs_h: u32) -> mpsc::SyncSender<DebugMsg> {
    let (tx, rx) = mpsc::sync_channel::<DebugMsg>(1);
    std::thread::spawn(move || {
        let mut ep = 0u32;
        let mut frame = 0u32;
        let row_bytes = (obs_w * 3 + 3) & !3;
        let pixel_size = row_bytes * obs_h;
        let file_size = 54 + pixel_size;
        let mut row_buf = vec![0u8; row_bytes as usize];

        while let Ok(msg) = rx.recv() {
            match msg {
                DebugMsg::NewEpisode(n) => {
                    ep = n;
                    frame = 0;
                    std::fs::create_dir_all(format!("debug_frames/ep_{:04}", ep)).ok();
                }
                DebugMsg::Frame(pixels, suffix) => {
                    let path = format!("debug_frames/ep_{:04}/{:05}{}.bmp", ep, frame, suffix);
                    if let Ok(mut f) = std::fs::File::create(&path) {
                        use std::io::Write;
                        let _ = f.write_all(b"BM");
                        let _ = f.write_all(&file_size.to_le_bytes());
                        let _ = f.write_all(&0u32.to_le_bytes());
                        let _ = f.write_all(&54u32.to_le_bytes());
                        let _ = f.write_all(&40u32.to_le_bytes());
                        let _ = f.write_all(&(obs_w as i32).to_le_bytes());
                        let _ = f.write_all(&(-(obs_h as i32)).to_le_bytes());
                        let _ = f.write_all(&1u16.to_le_bytes());
                        let _ = f.write_all(&24u16.to_le_bytes());
                        let _ = f.write_all(&0u32.to_le_bytes());
                        let _ = f.write_all(&pixel_size.to_le_bytes());
                        let _ = f.write_all(&[0u8; 16]);
                        for y in 0..obs_h as usize {
                            for x in 0..obs_w as usize {
                                let src = y * obs_w as usize * 4 + x * 4;
                                let dst = x * 3;
                                row_buf[dst] = pixels[src];
                                row_buf[dst + 1] = pixels[src + 1];
                                row_buf[dst + 2] = pixels[src + 2];
                            }
                            let _ = f.write_all(&row_buf);
                        }
                    }
                    frame += 1;
                }
            }
        }
    });
    let _ = tx.send(DebugMsg::NewEpisode(0));
    tx
}
