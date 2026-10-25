# ufo50ppo

PPO reinforcement learning agent that learns to play [UFO 50](https://50games.fun/) games. The game runs in **lockstep** with the trainer: a patched copy of the game hands each frame (native 384×216 pixels plus score, lives and state read inside the game) to the trainer over a socket and waits for the next action, so there is no screen capture, no OS-level key injection, and no dependency on window focus. Runs on Windows, macOS (Wine) and headless Linux (Wine + Xvfb). The training pipeline is game-agnostic; per-game logic (rewards, reset handling) lives in `src/games/<name>/`.

## Requirements

- **A copy of UFO 50** (the Steam install folder: `ufo50.exe`, `data.win`, audio groups, ...)
- **Rust** ≥ 1.85 (edition 2024)
- **libtorch** — set `LIBTORCH` to your libtorch path (`LIBTORCH_BYPASS_VERSION_CHECK=1` on version mismatch). CUDA is used when available, Apple MPS on macOS, otherwise CPU (with a loud warning). `build.rs` embeds an rpath on macOS/Linux; on Windows add `lib/` to `PATH`.
- **To build the patched game** (`bridge/build.sh`): .NET 10 SDK, [UndertaleModTool](https://github.com/UnderminersTeam/UndertaleModTool) built from source (`dotnet build UndertaleModCli -c Release`, after `git submodule update --init`), and `mingw-w64` (`x86_64-w64-mingw32-gcc`). On Windows, run it from Git Bash/MSYS2.
- **macOS**: Wine (`brew install --cask wine-stable`, needs Rosetta 2) and [DXVK-macOS](https://github.com/Gcenx/DXVK-macOS/releases) — Wine's own Direct3D 11 draws nothing for this game on macOS.
- **Linux**: `wine`, `xvfb`, Mesa (`libgl1-mesa-dri`) for headless runs.

## Setup

```bash
# Build the lockstep copy of the game into build/game (your game folder is not modified;
# its files are symlinked). UTMT_CLI defaults to ~/dev/UndertaleModTool/.../UndertaleModCli.dll.
bridge/build.sh "/path/to/UFO 50" build/game

# macOS: also drop in DXVK
DXVK_DIR=/path/to/dxvk-macOS-async-v1.10.3-20230507-repack bridge/build.sh "/path/to/UFO 50" build/game
```

The patched game behaves like the original unless `UFO50RL_PORT` is set, which the trainer does when it launches it. In lockstep mode it also runs uncapped (no vsync), mutes audio, never writes your save file, high scores or achievements (`global.cheatedNoSave`), and suppresses the title screen's attract-mode demo.

## Quick Start

```bash
# Train (launches the game copies itself)
cargo run --release -- -N 4 -n exp4

# Headless Linux: same command wrapped in a virtual display
bridge/headless.sh cargo run --release -- -N 16 -n exp16

# Check the bridge + tracker without a model: 4 envs, 60 s, random actions
cargo run --release --bin test_lockstep -- build/game 4 60

# Train with limits (whichever fires first)
cargo run --release -- -e 100 -f 500000
cargo run --release -- -m 60

# View training progress
tensorboard --logdir runs/<game>
```

Throughput: each game copy runs uncapped where it can. On macOS a visible window is held to the display refresh (120 game frames/s per copy on a 120 Hz display), so scale with `-N`; the game's own logic and drawing take ~1 ms per frame.

## CLI Flags

| Flag | Short | Description |
|------|-------|-------------|
| `--namespace` | `-n` | Training namespace (default: "default") |
| `--num-envs` | `-N` | Number of game copies to launch (default: 1) |
| `--game-dir` | `-g` | Lockstep game copy from `bridge/build.sh` (default: `$UFO50_GAME_DIR` or `build/game`) |
| `--frame-skip` | `-k` | Game frames per training step / action repeat (default: 4) |
| `--random-start` | `-s` | Fraction of runs that start at a random point in the level, entering as the respawn ghost (default: 0.5) |
| `--episodes` | `-e` | Max episodes before stopping |
| `--frames` | `-f` | Max training steps before stopping |
| `--minutes` | `-m` | Max training time in minutes |
| `--debug` | `-d` | Print per-episode reward breakdown |

Press **Ctrl+C** at any time to save a final checkpoint, close the game copies and exit cleanly.

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

```
┌──────────── game process (×N) ─────────────┐        ┌──────── trainer ─────────┐
│ ufo50.exe + patched data.win               │        │ LockstepGame (GameRunner)│
│  oController Post Draw, every k frames:    │ frame  │ GameTracker (state-based │
│   384×216 frame + state ─────────────────► │ ─────► │  rewards / done / reset) │
│   blocks in ufo50rl.dll until reply ◄───── │ ◄───── │ FrameStack → ActorCritic │
│  scrGetInput reads the replied keys        │ action │  → PPO (batched over N)  │
└────────────────────────────────────────────┘ (TCP)  └──────────────────────────┘
```

- **`bridge/`** — `ufo50rl.c` (GameMaker extension DLL: TCP client, blocking `rl_step`), `patch.csx` (UndertaleModTool script that registers the DLL and injects the sync point, state export and input override), `build.sh`, `headless.sh`. The wire layout is documented at the top of `patch.csx` and mirrored in `src/platform/lockstep.rs`.
- **`src/platform/`** — `GameRunner` trait, the lockstep implementation (launches the game, under Wine off Windows, and averages each frame down 3× to the 128×72 colour observation), and the action → key table.
- **`src/train/multi.rs`** — the trainer. Each tick it reads one frame from every alive env (each game is blocked waiting for its action), runs **one** batched inference call, and answers every env; the games then advance concurrently until the next tick. PPO updates run when all alive envs have filled their per-env buffer. Dead envs (connection lost or timed out) are dropped; the trainer exits when none remain.

Input goes through the game's own `scrGetInput`, not the OS keyboard, because GameMaker ignores keyboard state for unfocused windows — with several game windows only one would receive input.

## Reward Pipeline

Per-game `GameTracker` impls turn each `Frame` (pixels + `GameState` from inside the game) into `FrameResult { reward, done, event_name, is_menu, ... }`. The trainer adds **reward normalization** (`RunningMeanStd`, Welford) on top: each reward is divided by the running std of the discounted return before GAE, stabilizing training across reward distribution shifts. The normalizer state persists across resumes via `latest.json`.

After a reset the trainer calls `observe_idle` each frame; it returns the action to send (e.g. to get through the title screen) until a fresh run is playable. `is_menu` marks frames that are skipped from the PPO rollout; `menu_action` picks the button to send on them (Ninpek uses it to confirm the win screen and start the second loop).

Ninpek (internal game ID 34 — internal IDs don't follow library order) reads `o34_Mas`: `points[0]`, `life[0]` (-1 = game over), `state` (20 playing, 50 game over, 60–70 intro, 80+ win), and `iWater` (stage section) for the stage reward.

## TensorBoard Metrics

| Metric | Description |
|--------|-------------|
| `rollout/ep_rew_mean` | Episode reward (aggregate across envs) |
| `rollout/ep_len_mean` | Episode length (frames) |
| `rollout/ep_rew_min` / `ep_rew_max` | Spread of the latest episode reward across envs |
| `rollout/env_{i}/ep_rew_mean` | Per-env episode reward |
| `train/policy_loss` | PPO clipped surrogate loss |
| `train/value_loss` | Clipped value function MSE |
| `train/entropy` | Policy entropy |
| `train/grad_norm` | Pre-clip gradient L2 norm — diagnostic for value function divergence |
| `train/explained_variance` | Fraction of return variance explained by V — should climb above 0.5 |
| `train/learning_rate` | LR: falls linearly to 10% over the first 15M frames, then holds |
| `time/fps` | Training throughput |

## Production-readiness features

- **Atomic checkpoint writes** — weights, Adam state, and metadata each go through `.tmp` + rename. Crash mid-save can't corrupt the existing checkpoint. Stale `.tmp` files from prior crashes are swept at startup.
- **Persistent Adam state** — custom Adam optimizer (`src/train/adam.rs`) with `save_state` / `load_state`. Resumes preserve `m`, `v`, and step counter exactly. **9 unit tests** verify the math.
- **Reward normalizer persistence** — running mean/var/count saved in `latest.json`, restored on resume. **6 unit tests** including the count<2 edge case.
- **Per-env panic safety** — tracker panics in `process_frame` / `observe_idle` are caught with `catch_unwind`; the tracker is rebuilt and the game relaunched.
- **NaN guards** — PPO update skips minibatches with non-finite loss or gradient instead of propagating to the optimizer.
- **GPU detection warning** — startup prints a loud warning when neither CUDA nor MPS is available.
- **Ctrl+C graceful shutdown** — a signal handler (Ctrl+C / SIGTERM) flips a static flag; trainer saves a final checkpoint, quits the game copies and exits.
- **Save-failure escalation** — `checkpoint::save` (`src/util/checkpoint.rs`) tracks consecutive failures across the process; after 3 in a row it triggers shutdown so unattended runs don't silently lose hours of work.
- **Run metadata** — every trainer startup appends a JSON line to `run_metadata.jsonl` with git hash, dirty flag, hyperparam snapshot, and start time.

## Binaries

| Binary | Description |
|--------|-------------|
| `ufo50ppo` (default, `cargo run --release`) | PPO training entry |
| `test_lockstep` | Bridge + tracker check with random actions: resets, events, throughput |
| `record_deaths` | Plays a checkpoint and saves the last 16 frames before each lost life (`runs/ninpek/deaths`) |
| `record_ghosts` | Replays every saved checkpoint from the level start and logs player positions for `scripts/render_ghosts.py` |
| `test_model` | Model sanity check (no game needed) |
| `test_train` | PPO smoke test on a synthetic environment |

## Ghost video

Overlays many games from each saved checkpoint as translucent ninjas, tinted from blue (oldest) to yellow (newest), on the latest checkpoint's longest game. Needs a game copy built from the current bridge (it reports the player's position) and ffmpeg.

```sh
dotnet <UndertaleModCli.dll> load "$HOME/dev/UFO 50/data.win" -s scripts/export_ninja_sprite.csx < /dev/null  # once
cargo run --release --bin record_ghosts -- build/game checkpoints/ninpek/<run> 32 3  # games per checkpoint, parallel games
python3 scripts/render_ghosts.py                    # runs/ninpek/ghosts/ghosts.mp4, all checkpoints at once
python3 scripts/render_ghosts.py --per-checkpoint   # one segment per checkpoint
```

## Adding a New Game

1. Find the game's internal ID and controller object in a decompile of `data.win` (UndertaleModCli `dump -c UMT_DUMP_ALL`). Library slot → ID comes from the order of the `scrNN_Meta(nGame)` calls; `scrNN_Meta` also names the game's room.
2. Export the state you need from `bridge/patch.csx` (each slot is one f64; adding one means bumping `STATE_LEN` and the buffer size and frame offset in `patch.csx`) and add matching fields to `GameState` in `src/platform/lockstep.rs`.
3. Create `src/games/yourgame/` with `mod.rs`, `tracker.rs`, `rewards.rs`, `events.rs`. Implement `GameTracker`: `process_frame`, `observe_idle` (actions to reach a fresh run, `None` once playable), `game_name`, `num_actions`; optionally `menu_action`, `reset_episode`, `episode_breakdown` and `episode_metrics`.
4. Expose `pub fn definition() -> GameDefinition` with the game ID, obs dims, action count, tracker factory and hyperparameters, and point `src/main.rs` at it.
