pub mod capture;
pub mod input;

use self::input::Input;
use super::GameRunner;
use std::sync::mpsc;

/// Run `train_fn` against a Win32-hosted runner. Spawns the training closure on a worker
/// thread (so the Win32 message pump can own the main thread) and blocks until both finish.
pub fn host<F>(window_title: &str, obs_w: u32, obs_h: u32, train_fn: F) -> windows::core::Result<()>
where
    F: FnOnce(Box<dyn GameRunner>) + Send + 'static,
{
    let (runner, main_loop) = Win32Runner::new(window_title, obs_w, obs_h)?;
    std::thread::spawn(move || train_fn(Box::new(runner)));
    main_loop()
}

/// Win32 GameRunner: spawns a capture thread with a message pump,
/// communicates via channels. Training thread calls next_frame/execute_action.
pub struct Win32Runner {
    frame_rx: mpsc::Receiver<Vec<u8>>,
    action_tx: mpsc::SyncSender<usize>,
    reset_tx: mpsc::Sender<(Vec<usize>, u64)>,
    width: u32,
    height: u32,
}

impl Win32Runner {
    /// Create runner + main loop closure. Runner goes to training thread,
    /// closure must be called on the main thread (Win32 message pump).
    pub fn new(
        window_title: &str,
        width: u32,
        height: u32,
    ) -> windows::core::Result<(Self, impl FnOnce() -> windows::core::Result<()>)> {
        capture::init()?;

        let mut input = Input::new(window_title)?;

        let (frame_tx, frame_rx) = mpsc::sync_channel::<Vec<u8>>(1);
        let (action_tx, action_rx) = mpsc::sync_channel::<usize>(1);
        let (reset_tx, reset_rx) = mpsc::channel::<(Vec<usize>, u64)>();

        let title = window_title.to_string();
        let w = width;
        let h = height;

        // Send initial NOOP to unblock first frame read
        let action_tx_clone = action_tx.clone();
        let _ = action_tx_clone.send(0);

        let main_loop = move || {
            capture::run(
                &title,
                move |crop, frame, reader: &mut capture::FrameReader| {
                    // Check for reset signal (non-blocking). Empty sequence = release-only.
                    if let Ok((sequence, tap_ms)) = reset_rx.try_recv() {
                        input.reset_game(&sequence, tap_ms);
                    }

                    // Wait for action from training thread
                    let action = match action_rx.recv() {
                        Ok(a) => a,
                        Err(_) => return false,
                    };

                    input.execute_action(action);

                    let pixels = match reader.read_cropped(frame, crop, w, h) {
                        Ok(p) => p.to_vec(),
                        Err(e) => {
                            eprintln!("FATAL read error: {}", e);
                            return false;
                        }
                    };

                    frame_tx.send(pixels).is_ok()
                },
            )
        };

        Ok((
            Self {
                frame_rx,
                action_tx,
                reset_tx,
                width,
                height,
            },
            main_loop,
        ))
    }
}

impl GameRunner for Win32Runner {
    fn next_frame(&mut self) -> Result<Vec<u8>, String> {
        self.frame_rx
            .recv()
            .map_err(|_| "Capture thread disconnected".to_string())
    }

    fn execute_action(&mut self, action: usize) {
        let _ = self.action_tx.send(action);
    }

    fn release_all(&mut self) {
        let _ = self.reset_tx.send((vec![], 0));
    }

    fn reset_game(&mut self, sequence: &[usize], tap_ms: u64) {
        let _ = self.reset_tx.send((sequence.to_vec(), tap_ms));
    }

    fn obs_width(&self) -> u32 {
        self.width
    }

    fn obs_height(&self) -> u32 {
        self.height
    }
}
