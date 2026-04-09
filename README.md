# ufo50ppo

PPO reinforcement learning agent that learns to play [UFO 50](https://50games.fun/) games by capturing the screen and sending keyboard inputs. Windows-only (Linux-portable architecture). The training pipeline is game-agnostic; per-game logic (reward signals, reset sequences, scene detection) lives in `src/games/<name>/`.

## Requirements

- **Windows 10/11**, ≥16 GB RAM recommended for multi-env training
- **Rust** ≥ 1.85 (edition 2024)
- **libtorch** — set `LIBTORCH` env var to your libtorch path, add `lib/` to `PATH`
  - Set `LIBTORCH_BYPASS_VERSION_CHECK=1` if version mismatch
  - CUDA strongly recommended; the trainer prints a loud warning at startup if it falls back to CPU
- **UFO 50** running with window title "UFO 50" (one window per env for multi-env)

## Quick Start

```bash
# Single-env training (default namespace)
cargo run --release

# Multi-env training: launch N UFO 50 windows first, then
cargo run --release -- -N 4 -n exp4 -r

# Custom namespace
cargo run --release -- -n experiment1

# Train with limits (whichever fires first)
cargo run --release -- -e 100 -f 500000
cargo run --release -- -m 60

# Debug mode (saves frames + reward breakdown — single-env only)
cargo run --release -- -d

# View training progress
tensorboard --logdir runs/<game>
```

`cargo run` (no `--bin`) is the training entry point; the `main` binary lives in `src/main.rs` and dispatches to either the single-env or multi-env trainer based on `-N`.

## CLI Flags

| Flag | Short | Description |
|------|-------|-------------|
| `--namespace` | `-n` | Training namespace (default: "default") |
| `--num-envs` | `-N` | Number of parallel game instances (default: 1). Requires N existing UFO 50 windows. |
| `--episodes` | `-e` | Max episodes before stopping |
| `--frames` | `-f` | Max frames before stopping |
| `--minutes` | `-m` | Max training time in minutes |
| `--auto-resume` | `-r` | Single-env: reload `latest` on drain timeout. Multi-env: dead-env detection takes the place of reload-and-retry. |
| `--debug` | `-d` | Save frames to `debug_frames/ep_NNNN/`, print per-episode reward breakdown (single-env only) |

Press **Ctrl+C** at any time to save a final checkpoint and exit cleanly.

## Training Output

```
checkpoints/<game>/<namespace>/
  latest.safetensors          # weights — written every 10k frames + on clean exit
  latest.adam                 # Adam optimizer state (m, v, step) — same cadence
  latest.json                 # CheckpointMeta + reward normalizer state
  best.safetensors            # highest episode reward
  frame_00250000.safetensors  # versioned archive every 250k frames
  run_metadata.jsonl          # one line per training run start (git hash, hyperparams)

runs/<game>/<namespace>/
  20260405_143022/            # tensorboard logs (timestamped per run)
```

Resuming reads `latest.{safetensors,adam,json}` and continues from the saved episode/frame/update count, **with full optimizer momentum and reward normalizer state preserved** — there's no Adam warmup or normalizer relearning across restarts.

## Architecture

The codebase has two trainers that share everything except the per-env loop:

- **Single-env** — `run_training` in `src/train/runner.rs`. Two-thread model: capture pump on the main thread (Win32 message pump requirement), training on a worker thread, mpsc channels between.
- **Multi-env (sync)** — `run_training_multi` in `src/train/multi.rs`. N capture pump threads + 1 trainer thread. The trainer uses `next_frame_timeout(2s)` as a sync barrier across all alive envs each tick, runs **one** batched inference call (`act_batch`), dispatches actions, and runs PPO updates when all alive envs have filled their per-env buffer. Dead envs (frame timeout, drain timeout, tracker panic) are dropped from the inference batch and the PPO gate. Trainer exits when all envs are dead.

```
┌─────────────────────────────────────┐
│  Platform Layer (src/platform/)     │
│  GameRunner trait + Win32 impl      │
│  Capture (D3D11) + Input (PostMsg)  │
└──────────────┬──────────────────────┘
               │ GameRunner::next_frame() / execute_action() / pid()
┌──────────────┴──────────────────────┐
│  Training Layer (game-agnostic)     │
│  FrameStack → ActorCritic → PPO    │
│  Custom Adam (persistent state)     │
│  RunningMeanStd reward normalization│
│  GameTracker (per-game rewards)     │
│  Atomic checkpoints + TensorBoard   │
└─────────────────────────────────────┘
```

In multi-env, `host_multi` enumerates all matching game windows, spawns one capture pump per window on its own thread, and passes a `Vec<(WindowInfo, Box<dyn GameRunner>)>` to the trainer. Each tracker is constructed with its env's specific PID so any process-memory readers attach to the right game process.

## Reward Pipeline

Per-game `GameTracker` impls emit `FrameResult { reward, done, event_name, is_menu, ... }` per frame. The trainer adds **reward normalization** (`RunningMeanStd`, Welford) on top: each reward is divided by the running std of the discounted return before GAE, stabilizing training across reward distribution shifts. The normalizer state persists across resumes via `latest.json`.

Trackers can read game state via pixel detectors (`game_over.rs`-style scene matching) or via process memory (`MemReader` + Cheat Engine pointer chains, see `ninpek/mem.rs` for an example). `observe_idle` lets a tracker block the post-reset drain until the game is back in a fresh playable state. `is_menu` marks frames that should be skipped from the PPO rollout.

## TensorBoard Metrics

| Metric | Description |
|--------|-------------|
| `rollout/ep_rew_mean` | Episode reward (aggregate across envs in multi-env) |
| `rollout/ep_len_mean` | Episode length (frames) |
| `rollout/ep_rew_min` / `ep_rew_max` | Per-tick spread across envs (multi-env only) |
| `rollout/env_{i}/ep_rew_mean` | Per-env episode reward (multi-env only) |
| `train/policy_loss` | PPO clipped surrogate loss |
| `train/value_loss` | Clipped value function MSE |
| `train/entropy` | Policy entropy |
| `train/grad_norm` | Pre-clip gradient L2 norm — diagnostic for value function divergence |
| `train/explained_variance` | Fraction of return variance explained by V — should climb above 0.5 |
| `train/learning_rate` | LR (constant unless you change it) |
| `time/fps` | Training throughput |

## Production-readiness features

- **Atomic checkpoint writes** — weights, Adam state, and metadata each go through `.tmp` + rename. Crash mid-save can't corrupt the existing checkpoint. Stale `.tmp` files from prior crashes are swept at startup.
- **Persistent Adam state** — custom Adam optimizer (`src/train/adam.rs`) with `save_state` / `load_state`. Resumes preserve `m`, `v`, and step counter exactly. **9 unit tests** verify the math.
- **Reward normalizer persistence** — running mean/var/count saved in `latest.json`, restored on resume. **6 unit tests** including the count<2 edge case.
- **Per-env panic safety** (multi-env) — tracker panics in `process_frame` / `observe_idle` are caught with `catch_unwind`. The offending env is marked dead and the trainer continues with survivors.
- **NaN guards** — PPO update skips minibatches with non-finite loss or gradient instead of propagating to the optimizer.
- **CUDA detection warning** — startup prints a loud warning if `cuda_if_available` returns CPU.
- **Ctrl+C graceful shutdown** — Win32 console handler flips a static flag; trainer saves a final checkpoint and exits.
- **Save-failure escalation** — `save_checkpoint` tracks consecutive failures across the process; after 3 in a row it triggers shutdown so unattended runs don't silently lose hours of work.
- **Run metadata** — every trainer startup appends a JSON line to `run_metadata.jsonl` with git hash, dirty flag, hyperparam snapshot, and start time.

## Binaries

| Binary | Description |
|--------|-------------|
| `ufo50ppo` (default, `cargo run --release`) | PPO training entry — single or multi-env via `-N` |
| `test_ninpek` | Live reward testing for Ninpek (single-env diagnostic) |
| `test_reset` | Reset-sequence cycle test for tuning input timings |
| `test_model` | Model sanity check (no game needed) |
| `test_train` | PPO smoke test on a synthetic environment |
| `test_multi_capture` | Multi-env capture-only smoke test (no training) |
| `bench_capture` | Capture+downscale FPS benchmark |

## Adding a New Game

1. Create `src/games/yourgame/` with at minimum: `mod.rs`, `tracker.rs`, `rewards.rs`, `events.rs` (event-name constants). Add `mem.rs` if you have process-memory pointer chains; add `game_over.rs` (or similar) if you need pixel-based scene detection.
2. Implement `GameTracker` for `YourGameTracker`. Required: `process_frame`, `reset_sequence`, `game_name`, `obs_width`, `obs_height`, `num_actions`. Optional overrides: `observe_idle` (default `true` — set this if you have a fresh-state check), `reset_episode` (default no-op), `reset_tap_ms` (default 25 ms), `episode_breakdown` (default empty).
3. In `src/games/yourgame/mod.rs`, expose `pub const WINDOW_TITLE: &str = "..."` and `pub fn definition() -> GameDefinition` with the title, obs dims, action count, tracker factory `fn(width: u32, pid: u32) -> Box<dyn GameTracker>`, hyperparameters, and `debug_frame_suffix`. The factory's `pid` argument is the OS process ID of the env's game window — pass it through to `MemReader::for_pid(pid)` if you read process memory; ignore it otherwise.
4. Edit `src/main.rs` to use `games::yourgame::definition()` instead of `games::ninpek::definition()` (or add a CLI flag to switch games).
5. Pixel detectors should be calibrated against a reference obs resolution and scaled at runtime — see `ninpek/game_over.rs` for the `s_x` / `s_y` / `s_count` pattern. The helpers are private to each game.

No edits to `train::runner`, `train::multi`, `platform::host`, or any other shared code.
