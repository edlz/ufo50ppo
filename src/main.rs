use ufo50ppo::game;

const OBS_W: u32 = 128;
const OBS_H: u32 = 128;
const WINDOW_TITLE: &str = "UFO 50";

fn main() -> windows::core::Result<()> {
    game::capture::init()?;

    let mut input = game::input::Input::new(WINDOW_TITLE)?;

    let mut frame_count = 0u32;
    let mut snap_count = 0u32;
    let max_snaps = 10;
    let snap_interval = 180;

    std::fs::create_dir_all("debug_frames").ok();

    game::capture::run(
        WINDOW_TITLE,
        move |crop, frame, reader: &mut game::capture::FrameReader| {
            if let Err(e) = reader.read_cropped(frame, crop, OBS_W, OBS_H) {
                eprintln!("read error: {}", e);
                return true;
            }

            if frame_count % snap_interval == 0 {
                if snap_count >= max_snaps {
                    println!("Captured {} frames, done.", max_snaps);
                    input.release_all();
                    return false;
                }
                let path = format!("debug_frames/frame_{:03}.bmp", snap_count);
                match reader.save_debug_bmp(&path) {
                    Ok(()) => println!("Saved {} (frame {})", path, frame_count),
                    Err(e) => eprintln!("Save error: {}", e),
                }
                snap_count += 1;
            }

            frame_count += 1;
            true
        },
    )
}
