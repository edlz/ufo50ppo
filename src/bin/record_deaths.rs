// Plays a checkpoint in its own game copy and saves the frames leading up to each lost life,
// as the policy saw them, to see what kills the agent. Doesn't touch a running trainer.
// Usage: cargo run --release --bin record_deaths -- <game dir> <checkpoint dir> [deaths] [out dir]

use std::collections::VecDeque;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;
use ufo50ppo::games::ninpek::{GAME_ID, NinpekTracker, events};
use ufo50ppo::games::{GameTracker, start_run};
use ufo50ppo::platform::lockstep::{LockstepGame, OBS_H, OBS_W};
use ufo50ppo::platform::{ACTION_NAMES, GameRunner, GameState, NUM_ACTIONS};
use ufo50ppo::train::model::ActorCritic;
use ufo50ppo::train::preprocess::FrameStack;
use ufo50ppo::util::checkpoint;

const HISTORY: usize = 16;
const COLS: usize = 4;
const SCALE: usize = 3;
const TIMEOUT: Duration = Duration::from_secs(10);

struct Step {
    pixels: Vec<u8>,
    state: GameState,
    action: usize,
}

/// Oldest frame top-left, the frame that registered the death bottom-right.
fn save_sheet(path: &Path, steps: &VecDeque<Step>) -> std::io::Result<()> {
    let rows = steps.len().div_ceil(COLS);
    let (tw, th) = (OBS_W * SCALE, OBS_H * SCALE);
    let (w, h) = (tw * COLS, th * rows);
    let mut rgb = vec![0u8; w * h * 3];
    for (n, step) in steps.iter().enumerate() {
        let (ox, oy) = ((n % COLS) * tw, (n / COLS) * th);
        for y in 0..th {
            for x in 0..tw {
                let src = ((y / SCALE) * OBS_W + x / SCALE) * 4;
                let dst = ((oy + y) * w + ox + x) * 3;
                let px = &step.pixels[src..src + 4];
                rgb[dst..dst + 3].copy_from_slice(&[px[2], px[1], px[0]]);
            }
        }
    }
    let mut f = std::fs::File::create(path)?;
    write!(f, "P6\n{w} {h}\n255\n")?;
    f.write_all(&rgb)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let usage = "usage: record_deaths <game dir> <checkpoint dir> [deaths] [out dir]";
    let mut args = std::env::args().skip(1);
    let game_dir = PathBuf::from(args.next().ok_or(usage)?);
    let ckpt_dir = args.next().ok_or(usage)?;
    let deaths: usize = args.next().map_or(Ok(12), |s| s.parse())?;
    let out_dir = PathBuf::from(args.next().unwrap_or_else(|| "runs/ninpek/deaths".into()));
    std::fs::create_dir_all(&out_dir)?;

    // CPU keeps this off the GPU a running trainer is using.
    let device = tch::Device::Cpu;
    let mut model = ActorCritic::new(device, OBS_W as u32, OBS_H as u32, NUM_ACTIONS);
    let ckpt = checkpoint::try_load(&ckpt_dir, "latest", &mut model.vs)?
        .ok_or_else(|| format!("no checkpoint in {ckpt_dir}"))?;
    println!(
        "loaded {ckpt_dir} at {} steps; launching game...",
        ckpt.total_frames
    );

    let mut game = LockstepGame::launch(&game_dir, 4, GAME_ID)?;
    let mut tracker = NinpekTracker::new();
    let mut stack = FrameStack::new(device, OBS_W as u32, OBS_H as u32);
    let mut history: VecDeque<Step> = VecDeque::with_capacity(HISTORY);
    let mut summary = std::fs::File::create(out_dir.join("summary.txt"))?;
    let (mut recorded, mut episode, mut step) = (0, 0, 0u32);

    while recorded < deaths {
        if step == 0 {
            start_run(&mut game, &mut tracker, TIMEOUT)?;
            stack.reset();
            history.clear();
            episode += 1;
        }
        let frame = game.next_frame_timeout(TIMEOUT)?;
        let r = tracker.process_frame(&frame);
        step += 1;
        let action = if r.is_menu {
            tracker.menu_action()
        } else {
            model.act(&stack.push(&frame.pixels)).0 as usize
        };
        if history.len() == HISTORY {
            history.pop_front();
        }
        history.push_back(Step {
            pixels: frame.pixels,
            state: frame.state,
            action,
        });

        if r.event_name == events::LIFE_LOST || r.event_name == events::GAME_OVER {
            recorded += 1;
            let name = format!("death{recorded:02}_ep{episode}_x{:.0}", frame.state.view_x);
            save_sheet(&out_dir.join(format!("{name}.ppm")), &history)?;
            writeln!(
                summary,
                "{name}: episode {episode} step {step}, lives left {}",
                frame.state.lives
            )?;
            for (n, s) in history.iter().enumerate() {
                writeln!(
                    summary,
                    "  {n:2}: x={:<5.0} action={}",
                    s.state.view_x, ACTION_NAMES[s.action]
                )?;
            }
            println!("{name}");
        }

        if r.done {
            game.execute_action(0);
            step = 0;
        } else {
            game.execute_action(action);
        }
    }
    println!("{recorded} deaths written to {}", out_dir.display());
    Ok(())
}
