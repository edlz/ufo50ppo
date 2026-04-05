use ufo50ppo::game;

const WINDOW_TITLE: &str = "UFO 50";
const BENCH_FRAMES: u32 = 300;
const SIZES: &[(u32, u32)] = &[
    (84, 84),
    (224, 224),
    (320, 320),
    (480, 480),
    (640, 640),
    (800, 800),
    (1024, 1024),
];

fn main() -> windows::core::Result<()> {
    game::capture::init()?;

    let mut size_idx = 0;
    let mut frame_count = 0u32;
    let mut start = std::time::Instant::now();

    std::fs::create_dir_all("debug_frames").ok();

    println!(
        "Benchmarking capture+downscale at {} frames per size\n",
        BENCH_FRAMES
    );
    println!(
        "{:>9} {:>10} {:>10} {:>10}",
        "Size", "Total(ms)", "Avg(us)", "FPS"
    );

    game::capture::run(
        WINDOW_TITLE,
        move |crop, frame, reader: &mut game::capture::FrameReader| {
            if size_idx >= SIZES.len() {
                return false;
            }

            let (w, h) = SIZES[size_idx];

            if frame_count == 0 {
                start = std::time::Instant::now();
            }

            let _ = reader.read_cropped(frame, crop, w, h);
            if frame_count == 0 {
                let path = format!("debug_frames/bench_{}x{}.bmp", w, h);
                reader.save_debug_bmp(&path).ok();
            }
            frame_count += 1;

            if frame_count >= BENCH_FRAMES {
                let elapsed = start.elapsed();
                let avg_us = elapsed.as_micros() as f64 / BENCH_FRAMES as f64;
                let fps = BENCH_FRAMES as f64 / elapsed.as_secs_f64();
                println!(
                    "{:>4}x{:<4} {:>10.1} {:>10.1} {:>10.1}",
                    w,
                    h,
                    elapsed.as_millis(),
                    avg_us,
                    fps,
                );
                size_idx += 1;
                frame_count = 0;
            }

            true
        },
    )
}
