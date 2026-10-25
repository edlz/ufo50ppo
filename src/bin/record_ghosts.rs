// Replays every saved checkpoint from the level start, logging player positions for
// scripts/render_ghosts.py; the latest checkpoint's longest game becomes the background.
// Usage: cargo run --release --bin record_ghosts -- <game dir> <checkpoint dir> [games per checkpoint] [parallel games] [out dir]

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;
use ufo50ppo::games::ninpek::{GAME_ID, NinpekTracker};
use ufo50ppo::games::{GameTracker, start_run};
use ufo50ppo::platform::lockstep::{LockstepGame, OBS_H, OBS_W, RAW_H, RAW_W};
use ufo50ppo::platform::{GameRunner, NUM_ACTIONS};
use ufo50ppo::train::model::ActorCritic;
use ufo50ppo::train::preprocess::FrameStack;
use ufo50ppo::util::checkpoint;

const MAX_STEPS: usize = 5000;
const TIMEOUT: Duration = Duration::from_secs(10);
const LATEST: &str = "latest";

/// (view_x, player_x, player_y) per step.
type Track = Vec<(f64, f64, f64)>;

struct Shared {
    out_dir: PathBuf,
    csv: Mutex<BufWriter<File>>,
    /// Steps in the background written so far.
    bg_len: Mutex<usize>,
}

/// Versioned checkpoints in step order, then the latest.
fn checkpoints(dir: &str) -> std::io::Result<Vec<String>> {
    let mut names: Vec<String> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter_map(|f| Some(f.strip_suffix(".safetensors")?.to_string()))
        .filter(|n| n.starts_with("frame_"))
        .collect();
    names.sort();
    names.push(LATEST.into());
    Ok(names)
}

fn bgr_to_rgb(bgra: &[u8], out: &mut Vec<u8>) {
    out.clear();
    for px in bgra.chunks_exact(4) {
        out.extend_from_slice(&[px[2], px[1], px[0]]);
    }
}

fn play(
    game: &mut LockstepGame,
    model: &ActorCritic,
    mut bg: Option<&mut BufWriter<File>>,
) -> Result<Track, Box<dyn std::error::Error>> {
    let mut tracker = NinpekTracker::new();
    let mut stack = FrameStack::new(tch::Device::Cpu, OBS_W as u32, OBS_H as u32);
    let mut path = Vec::new();
    let mut rgb = Vec::with_capacity(RAW_W * RAW_H * 3);
    start_run(game, &mut tracker, TIMEOUT)?;
    while path.len() < MAX_STEPS {
        let frame = game.next_frame_timeout(TIMEOUT)?;
        let r = tracker.process_frame(&frame);
        let s = frame.state;
        path.push((s.view_x, s.player_x, s.player_y));
        if let Some(w) = bg.as_mut() {
            bgr_to_rgb(game.raw(), &mut rgb);
            w.write_all(&rgb)?;
        }
        if r.done {
            break;
        }
        let action = if r.is_menu {
            tracker.menu_action()
        } else {
            model.act(&stack.push(&frame.pixels)).0 as usize
        };
        game.execute_action(action);
    }
    game.execute_action(0);
    Ok(path)
}

fn worker(
    id: usize,
    workers: usize,
    game_dir: &Path,
    ckpt_dir: &str,
    names: &[String],
    runs: usize,
    shared: &Shared,
) -> Result<(), String> {
    let mut game = LockstepGame::launch(game_dir, 4, GAME_ID).map_err(|e| e.to_string())?;
    let tmp = shared.out_dir.join(format!("bg_{id}.tmp"));
    for name in names {
        let mut model = ActorCritic::new(tch::Device::Cpu, OBS_W as u32, OBS_H as u32, NUM_ACTIONS);
        let dir = if name == LATEST {
            shared.out_dir.to_str().ok_or("non-UTF-8 out dir")?
        } else {
            ckpt_dir
        };
        let ckpt = checkpoint::try_load(dir, name, &mut model.vs)?
            .ok_or_else(|| format!("no checkpoint {name} in {dir}"))?;
        for run in (id..runs).step_by(workers) {
            let mut bg = if name == LATEST {
                Some(BufWriter::new(
                    File::create(&tmp).map_err(|e| e.to_string())?,
                ))
            } else {
                None
            };
            let path = play(&mut game, &model, bg.as_mut()).map_err(|e| e.to_string())?;
            let furthest = path.iter().map(|p| p.0).fold(0.0, f64::max);
            println!(
                "{name} ({} steps) game {run}: {} steps, furthest x {furthest:.0}",
                ckpt.total_frames,
                path.len()
            );
            {
                let mut csv = shared.csv.lock().unwrap();
                for (step, (vx, x, y)) in path.iter().enumerate() {
                    writeln!(csv, "{},{run},{step},{vx},{x},{y}", ckpt.total_frames)
                        .map_err(|e| e.to_string())?;
                }
                csv.flush().map_err(|e| e.to_string())?;
            }
            if let Some(bg) = bg {
                drop(bg);
                let mut bg_len = shared.bg_len.lock().unwrap();
                if path.len() > *bg_len {
                    *bg_len = path.len();
                    std::fs::rename(&tmp, shared.out_dir.join("bg.rgb"))
                        .map_err(|e| e.to_string())?;
                    let views: Vec<String> = path.iter().map(|p| p.0.to_string()).collect();
                    std::fs::write(shared.out_dir.join("bg_view_x.txt"), views.join("\n"))
                        .map_err(|e| e.to_string())?;
                }
            }
        }
    }
    let _ = std::fs::remove_file(tmp);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let usage = "usage: record_ghosts <game dir> <checkpoint dir> [games per checkpoint] [parallel games] [out dir]";
    let mut args = std::env::args().skip(1);
    let game_dir = PathBuf::from(args.next().ok_or(usage)?);
    let ckpt_dir = args.next().ok_or(usage)?;
    let runs: usize = args.next().map_or(Ok(32), |s| s.parse())?;
    let workers: usize = args.next().map_or(Ok(3), |s| s.parse())?;
    let out_dir = PathBuf::from(args.next().unwrap_or_else(|| "runs/ninpek/ghosts".into()));
    std::fs::create_dir_all(&out_dir)?;

    // Each game's model is tiny; one thread apiece leaves the cores to a running trainer.
    tch::set_num_threads(1);
    let names = checkpoints(&ckpt_dir)?;
    // A running trainer rewrites `latest`; every game copy must play the same one.
    for ext in ["safetensors", "json"] {
        std::fs::copy(
            format!("{ckpt_dir}/{LATEST}.{ext}"),
            out_dir.join(format!("{LATEST}.{ext}")),
        )?;
    }
    println!(
        "{} checkpoints x {runs} games on {workers} game copies",
        names.len()
    );
    let mut csv = BufWriter::new(File::create(out_dir.join("ghosts.csv"))?);
    writeln!(csv, "checkpoint_steps,game,step,view_x,x,y")?;
    let shared = Shared {
        out_dir: out_dir.clone(),
        csv: Mutex::new(csv),
        bg_len: Mutex::new(0),
    };

    std::thread::scope(|s| {
        let handles: Vec<_> = (0..workers)
            .map(|id| {
                let (game_dir, ckpt_dir, names, shared) = (&game_dir, &ckpt_dir, &names, &shared);
                s.spawn(move || {
                    // Staggered so the game copies don't all boot at once.
                    std::thread::sleep(Duration::from_secs(id as u64 * 2));
                    worker(id, workers, game_dir, ckpt_dir, names, runs, shared)
                })
            })
            .collect();
        for h in handles {
            if let Err(e) = h.join().unwrap() {
                eprintln!("worker failed: {e}");
            }
        }
    });
    println!(
        "wrote {} (background: {} steps)",
        out_dir.display(),
        shared.bg_len.lock().unwrap()
    );
    Ok(())
}
