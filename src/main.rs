mod agent;
mod capture;
mod env;
mod input;
mod model;
mod ppo;
mod preprocess;

const OBS_W: u32 = 84;
const OBS_H: u32 = 84;
const WINDOW_TITLE: &str = "UFO 50";

fn main() -> windows::core::Result<()> {
    // Pre-load torch_cuda.dll so PyTorch's lazy CUDA init can find it.
    unsafe {
        windows::Win32::System::LibraryLoader::LoadLibraryA(windows::core::s!("torch_cuda.dll"))
            .ok();
    }

    capture::init()?;

    let input = input::Input::new(WINDOW_TITLE)?;

    let (frame_tx, frame_rx) = std::sync::mpsc::sync_channel::<env::FrameData>(2);
    let (action_tx, action_rx) = std::sync::mpsc::channel::<usize>();

    // Spawn training thread
    std::thread::spawn(move || {
        agent::training_loop(frame_rx, action_tx, input);
    });

    // Capture loop on main thread — sends frames to training thread
    capture::run(WINDOW_TITLE, move |frame, reader| {
        // Wait for action from training thread (blocks until agent decides)
        let _action = match action_rx.recv() {
            Ok(a) => a,
            Err(_) => return false,
        };

        // Read pixels at observation resolution
        let pixels = match reader.read(frame, OBS_W, OBS_H) {
            Ok(p) => p,
            Err(_) => return true,
        };

        let frame_data = env::FrameData {
            width: OBS_W,
            height: OBS_H,
            rgba: pixels.to_vec(),
        };

        // Send frame to training thread
        frame_tx.send(frame_data).is_ok()
    })
}
