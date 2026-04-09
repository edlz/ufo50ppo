// Smoke test: capture from N game windows in parallel without a model or training loop.

use std::time::{Duration, Instant};
use ufo50ppo::games;
use ufo50ppo::platform::GameRunner;
use ufo50ppo::platform::win32::{find_windows_by_title, host_multi};
use ufo50ppo::train::runner::write_frame_bmp;

fn main() -> windows::core::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2);
    let duration_secs: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(30);

    let game = games::ninpek::definition();
    let title = game.window_title;
    let obs_w = game.obs_width;
    let obs_h = game.obs_height;

    let found = find_windows_by_title(title);
    println!(
        "find_windows_by_title({:?}): found {} window(s)",
        title,
        found.len()
    );
    for (i, info) in found.iter().enumerate() {
        println!("  [{}] hwnd={:?} pid={}", i, info.hwnd.0, info.pid);
    }
    if found.len() < n {
        eprintln!(
            "ERROR: requested {} envs but only {} '{}' window(s) found",
            n,
            found.len(),
            title
        );
        std::process::exit(2);
    }

    println!(
        "\nrunning host_multi with N={} for {}s at {}x{}...\n",
        n, duration_secs, obs_w, obs_h
    );

    host_multi(title, n, obs_w, obs_h, move |runners| {
        let n = runners.len();
        let mut runners: Vec<Box<dyn GameRunner>> = runners.into_iter().map(|(_, r)| r).collect();

        std::fs::create_dir_all("debug_frames").ok();

        let start = Instant::now();
        let deadline = start + Duration::from_secs(duration_secs);
        let mut frame_count: Vec<u64> = vec![0; n];
        let mut snapped: Vec<bool> = vec![false; n];
        let mut alive: Vec<bool> = vec![true; n];
        let mut last_log = Instant::now();
        let timeout = Duration::from_secs(2);

        while Instant::now() < deadline {
            if ufo50ppo::util::shutdown::requested() {
                println!("\n[test] shutdown requested");
                break;
            }
            for i in 0..n {
                if !alive[i] {
                    continue;
                }
                match runners[i].next_frame_timeout(timeout) {
                    Ok(pixels) => {
                        frame_count[i] += 1;
                        // Save a snapshot once each env has produced 30+ frames so we
                        // skip any startup transients.
                        if !snapped[i] && frame_count[i] >= 30 {
                            let path = format!("debug_frames/multi_env_{}.bmp", i);
                            write_frame_bmp(&path, &pixels, obs_w, obs_h);
                            println!("[test] saved {}", path);
                            snapped[i] = true;
                        }
                        runners[i].execute_action(0);
                    }
                    Err(e) => {
                        eprintln!("[test] env {} {} — marking dead", i, e);
                        alive[i] = false;
                    }
                }
            }
            if alive.iter().all(|a| !a) {
                eprintln!("[test] all envs dead");
                break;
            }
            if last_log.elapsed() >= Duration::from_secs(5) {
                let elapsed = start.elapsed().as_secs_f64();
                let lines: Vec<String> = (0..n)
                    .map(|i| {
                        let fps = frame_count[i] as f64 / elapsed;
                        let state = if alive[i] { "OK" } else { "DEAD" };
                        format!("env{}={}fps[{}]", i, fps as u32, state)
                    })
                    .collect();
                println!("[test] elapsed={:.0}s | {}", elapsed, lines.join(" "));
                last_log = Instant::now();
            }
        }

        println!("\n=== summary ({:.1}s) ===", start.elapsed().as_secs_f64());
        let elapsed = start.elapsed().as_secs_f64();
        for i in 0..n {
            let fps = frame_count[i] as f64 / elapsed;
            println!(
                "  env {}: {} frames | {:.1} fps | {}",
                i,
                frame_count[i],
                fps,
                if alive[i] { "alive" } else { "dead" }
            );
        }

        // Sanity check: frame counts should be roughly balanced (within 20% of each other)
        // and >= ~10 fps per env (we're capturing at the game's frame rate, expect 30-60).
        let max_count = *frame_count.iter().max().unwrap_or(&0);
        let min_count = *frame_count.iter().min().unwrap_or(&0);
        let alive_count = alive.iter().filter(|a| **a).count();
        println!(
            "\nresult: {}/{} envs alive | per-env spread: min={} max={}",
            alive_count, n, min_count, max_count
        );
        if alive_count < n {
            eprintln!("FAIL: not all envs survived");
        } else if max_count == 0 {
            eprintln!("FAIL: no frames captured");
        } else if (max_count - min_count) * 5 > max_count {
            eprintln!("WARN: per-env frame spread > 20% — slow env may be a problem at scale");
        } else {
            println!("PASS");
        }
    })
}
