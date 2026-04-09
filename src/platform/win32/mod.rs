pub mod capture;
pub mod input;

use self::input::Input;
use super::GameRunner;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use windows::Win32::Foundation::{BOOL, HWND, LPARAM, TRUE};
use windows::Win32::UI::HiDpi::{
    DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, SetProcessDpiAwarenessContext,
};
use windows::Win32::UI::WindowsAndMessaging::{
    EnumWindows, GetWindowTextW, GetWindowThreadProcessId,
};

static DPI_AWARE: AtomicBool = AtomicBool::new(false);

/// Per-monitor DPI awareness so Win32 window/client metrics return physical pixels
/// matching captured frames. Must run before any HWND query — DPI scaling otherwise
/// gives off-by-1.5× crops on high-DPI monitors. Idempotent.
pub fn ensure_dpi_aware() {
    if DPI_AWARE.swap(true, Ordering::Relaxed) {
        return;
    }
    unsafe {
        let _ = SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    }
}

#[derive(Clone, Copy)]
pub struct WindowInfo {
    pub hwnd: HWND,
    pub pid: u32,
}

unsafe impl Send for WindowInfo {}

pub fn find_windows_by_title(title: &str) -> Vec<WindowInfo> {
    ensure_dpi_aware();
    struct EnumState {
        target: String,
        results: Vec<WindowInfo>,
    }
    unsafe extern "system" fn enum_proc(hwnd: HWND, lparam: LPARAM) -> BOOL {
        let state = unsafe { &mut *(lparam.0 as *mut EnumState) };
        let mut buf = [0u16; 256];
        let len = unsafe { GetWindowTextW(hwnd, &mut buf) };
        if len > 0 {
            let actual = String::from_utf16_lossy(&buf[..len as usize]);
            if actual == state.target {
                let mut pid = 0u32;
                unsafe { GetWindowThreadProcessId(hwnd, Some(&mut pid)) };
                if pid != 0 {
                    state.results.push(WindowInfo { hwnd, pid });
                }
            }
        }
        TRUE
    }
    let mut state = EnumState {
        target: title.to_string(),
        results: Vec::new(),
    };
    let lparam = LPARAM(&mut state as *mut EnumState as isize);
    unsafe {
        let _ = EnumWindows(Some(enum_proc), lparam);
    }
    state.results
}

pub fn host<F>(window_title: &str, obs_w: u32, obs_h: u32, train_fn: F) -> windows::core::Result<()>
where
    F: FnOnce(Box<dyn GameRunner>) + Send + 'static,
{
    let (runner, main_loop) = Win32Runner::new(window_title, obs_w, obs_h)?;
    std::thread::spawn(move || train_fn(Box::new(runner)));
    main_loop()
}

pub fn host_multi<F>(
    window_title: &str,
    expected_count: usize,
    obs_w: u32,
    obs_h: u32,
    train_fn: F,
) -> windows::core::Result<()>
where
    F: FnOnce(Vec<(WindowInfo, Box<dyn GameRunner>)>) + Send + 'static,
{
    let windows = find_windows_by_title(window_title);
    if windows.len() < expected_count {
        eprintln!(
            "host_multi: expected {} windows titled '{}', found {}",
            expected_count,
            window_title,
            windows.len()
        );
        return Err(windows::core::Error::from_win32());
    }
    let windows = windows.into_iter().take(expected_count).collect::<Vec<_>>();

    let mut runners: Vec<(WindowInfo, Box<dyn GameRunner>)> = Vec::with_capacity(windows.len());
    let mut pump_handles: Vec<std::thread::JoinHandle<windows::core::Result<()>>> =
        Vec::with_capacity(windows.len());

    for info in windows.iter().copied() {
        let (runner, main_loop) = Win32Runner::for_window(info, obs_w, obs_h)?;
        runners.push((info, Box::new(runner)));
        pump_handles.push(std::thread::spawn(main_loop));
    }

    let train_handle = std::thread::spawn(move || train_fn(runners));

    let _ = train_handle.join();
    for h in pump_handles {
        let _ = h.join();
    }
    Ok(())
}

pub struct Win32Runner {
    frame_rx: mpsc::Receiver<Vec<u8>>,
    action_tx: mpsc::SyncSender<usize>,
    reset_tx: mpsc::Sender<(Vec<usize>, u64)>,
    width: u32,
    height: u32,
    pid: u32,
}

impl Win32Runner {
    pub fn for_window(
        info: WindowInfo,
        width: u32,
        height: u32,
    ) -> windows::core::Result<(Self, impl FnOnce() -> windows::core::Result<()>)> {
        // capture::init creates a per-thread DispatcherQueueController, so it must run
        // inside main_loop on the pump thread itself, not here.
        let mut input = Input::for_hwnd(info.hwnd);

        let (frame_tx, frame_rx) = mpsc::sync_channel::<Vec<u8>>(1);
        let (action_tx, action_rx) = mpsc::sync_channel::<usize>(1);
        let (reset_tx, reset_rx) = mpsc::channel::<(Vec<usize>, u64)>();

        let w = width;
        let h = height;
        let hwnd_addr = info.hwnd.0 as usize;

        // Initial NOOP unblocks the first frame read.
        let action_tx_clone = action_tx.clone();
        let _ = action_tx_clone.send(0);

        let main_loop = move || {
            capture::init()?;
            capture::run_for_hwnd(
                HWND(hwnd_addr as *mut _),
                move |crop, frame, reader: &mut capture::FrameReader| {
                    if let Ok((sequence, tap_ms)) = reset_rx.try_recv() {
                        input.reset_game(&sequence, tap_ms);
                    }

                    let action = loop {
                        match action_rx.recv_timeout(std::time::Duration::from_millis(100)) {
                            Ok(a) => break a,
                            Err(mpsc::RecvTimeoutError::Timeout) => {
                                if crate::util::shutdown::requested() {
                                    return false;
                                }
                            }
                            Err(mpsc::RecvTimeoutError::Disconnected) => return false,
                        }
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
                pid: info.pid,
            },
            main_loop,
        ))
    }

    pub fn new(
        window_title: &str,
        width: u32,
        height: u32,
    ) -> windows::core::Result<(Self, impl FnOnce() -> windows::core::Result<()>)> {
        let info = find_windows_by_title(window_title)
            .into_iter()
            .next()
            .ok_or_else(windows::core::Error::from_win32)?;
        Self::for_window(info, width, height)
    }
}

impl GameRunner for Win32Runner {
    fn next_frame(&mut self) -> Result<Vec<u8>, String> {
        self.frame_rx
            .recv()
            .map_err(|_| "Capture thread disconnected".to_string())
    }

    fn next_frame_timeout(&mut self, timeout: std::time::Duration) -> Result<Vec<u8>, String> {
        match self.frame_rx.recv_timeout(timeout) {
            Ok(p) => Ok(p),
            Err(mpsc::RecvTimeoutError::Timeout) => Err("timeout".to_string()),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                Err("Capture thread disconnected".to_string())
            }
        }
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

    fn pid(&self) -> u32 {
        self.pid
    }
}
