use ufo50ppo::games;
use ufo50ppo::games::GameTracker;
use ufo50ppo::platform;

const SNAP_INTERVAL: u32 = 100;

fn log_mem_change(name: &str, frame: u32, prev: &mut Option<u64>, curr: Option<u64>) {
    if *prev == curr {
        return;
    }
    let fmt = |v: Option<u64>| v.map(|x| x.to_string()).unwrap_or_else(|| "n/a".into());
    let delta = match (*prev, curr) {
        (Some(p), Some(c)) => format!(" (Δ{:+})", c as i64 - p as i64),
        _ => String::new(),
    };
    println!(
        "Frame {:5} | mem {} {} -> {}{}",
        frame,
        name,
        fmt(*prev),
        fmt(curr),
        delta
    );
    *prev = curr;
}

fn main() -> windows::core::Result<()> {
    platform::win32::capture::init()?;

    let game = games::ninpek::definition();
    let obs_w = game.obs_width;
    let obs_h = game.obs_height;
    let window_title: String = std::env::args()
        .nth(1)
        .unwrap_or_else(|| game.window_title.to_string());
    let mut tracker = games::ninpek::NinpekTracker::new(obs_w, 0);
    let mut total_reward = 0.0;
    let mut frame_count = 0u32;
    let mut last_mem_score: Option<u64> = None;
    let mut last_mem_lives: Option<u64> = None;

    std::fs::create_dir_all("debug_frames").ok();

    println!(
        "Testing reward for: {} | window: {}",
        game.name, window_title
    );
    println!("Press Ctrl+C to stop.\n");

    platform::win32::capture::run(
        &window_title,
        move |crop, frame, reader: &mut platform::win32::capture::FrameReader| {
            let pixels = match reader.read_cropped(frame, crop, obs_w, obs_h) {
                Ok(p) => p.to_vec(),
                Err(e) => {
                    eprintln!("read error: {}", e);
                    return true;
                }
            };

            if frame_count.is_multiple_of(SNAP_INTERVAL) {
                let path = format!("debug_frames/snap_{:03}.bmp", frame_count / SNAP_INTERVAL);
                reader.save_debug_bmp(&path).ok();
            }

            let result = tracker.process_frame(&pixels);
            total_reward += result.reward;

            if !result.event_name.is_empty() {
                println!(
                    "Frame {:5} | {} {:+.1} | total: {:.1} | lives:{}",
                    frame_count, result.event_name, result.reward, total_reward, result.lives
                );
            }

            log_mem_change(
                "score",
                frame_count,
                &mut last_mem_score,
                tracker.mem_score(),
            );
            log_mem_change(
                "lives",
                frame_count,
                &mut last_mem_lives,
                tracker.mem_lives(),
            );

            frame_count += 1;
            true
        },
    )
}
