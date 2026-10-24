// End-to-end check of the lockstep bridge + Ninpek tracker without a model: launches N game
// copies, plays random actions through the same reset/drain flow as the trainer, and prints
// events, episode ends, and throughput.
// Usage: cargo run --release --bin test_lockstep -- <game dir> [envs] [seconds] [frame skip]

use std::path::PathBuf;
use std::time::{Duration, Instant};
use ufo50ppo::games::GameTracker;
use ufo50ppo::games::ninpek::{GAME_ID, NinpekTracker};
use ufo50ppo::platform::lockstep::{LockstepGame, OBS_H, OBS_W};
use ufo50ppo::platform::{GameRunner, NUM_ACTIONS};

fn save_ppm(path: &std::path::Path, bgra: &[u8]) {
    let mut out = format!("P6\n{OBS_W} {OBS_H}\n255\n").into_bytes();
    for px in bgra.chunks_exact(4) {
        out.extend_from_slice(&[px[2], px[1], px[0]]);
    }
    std::fs::write(path, out).unwrap();
}

fn main() -> std::io::Result<()> {
    let mut args = std::env::args().skip(1);
    let game_dir = PathBuf::from(
        args.next()
            .expect("usage: test_lockstep <game dir> [envs] [seconds] [frame skip]"),
    );
    let n: usize = args.next().map_or(1, |s| s.parse().unwrap());
    let seconds: u64 = args.next().map_or(60, |s| s.parse().unwrap());
    let frame_skip: u32 = args.next().map_or(4, |s| s.parse().unwrap());

    let t = Instant::now();
    let mut runners = LockstepGame::launch_many(&game_dir, n, frame_skip, GAME_ID)?;
    println!(
        "{n} game(s) connected after {:.1}s",
        t.elapsed().as_secs_f64()
    );

    let mut trackers: Vec<NinpekTracker> = (0..n).map(|_| NinpekTracker::new()).collect();
    let mut draining = vec![true; n];
    let mut drain_start = vec![Instant::now(); n];
    for r in &mut runners {
        r.reset_game();
    }

    let mut rng = 0x2545F4914F6CDD1Du64;
    let mut steps = 0u64;
    let mut first_frame = vec![None; n];
    let mut last_frame = vec![0u64; n];
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(seconds) {
        for i in 0..n {
            let frame = runners[i]
                .next_frame_timeout(Duration::from_secs(10))
                .map_err(std::io::Error::other)?;
            steps += 1;
            first_frame[i].get_or_insert(frame.state.frame);
            last_frame[i] = frame.state.frame;
            if draining[i] {
                if drain_start[i].elapsed() > Duration::from_secs(5) && frame.state.frame % 200 < 4
                {
                    println!("env {i}: still resetting: {:?}", frame.state);
                }
                match trackers[i].observe_idle(&frame) {
                    Some(a) => runners[i].execute_action(a),
                    None => {
                        println!(
                            "env {i}: run started after {:.1}s (game frame {})",
                            drain_start[i].elapsed().as_secs_f64(),
                            frame.state.frame
                        );
                        save_ppm(
                            &std::env::temp_dir().join(format!("lockstep_env{i}_start.ppm")),
                            &frame.pixels,
                        );
                        draining[i] = false;
                        runners[i].execute_action(0);
                    }
                }
                continue;
            }
            let r = trackers[i].process_frame(&frame);
            if r.is_event {
                println!(
                    "env {i}: {} {:+.2} lives={} state={:?}",
                    r.event_name, r.reward, r.lives, frame.state
                );
            }
            if r.done {
                println!(
                    "env {i}: episode over ({})\n{}",
                    r.event_name,
                    trackers[i].episode_breakdown()
                );
                runners[i].execute_action(0);
                runners[i].reset_game();
                trackers[i].reset_episode();
                draining[i] = true;
                drain_start[i] = Instant::now();
                continue;
            }
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            runners[i].execute_action((rng % NUM_ACTIONS as u64) as usize);
        }
    }
    let secs = start.elapsed().as_secs_f64();
    let game_frames: u64 = (0..n)
        .map(|i| last_frame[i] - first_frame[i].unwrap_or(0))
        .sum();
    println!(
        "{n} env(s), frame skip {frame_skip}: {:.0} steps/s, {:.0} game frames/s total",
        steps as f64 / secs,
        game_frames as f64 / secs
    );
    Ok(())
}
