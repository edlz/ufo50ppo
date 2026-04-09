// Cooperative shutdown signal. The console control handler flips a static AtomicBool
// when Ctrl+C / Ctrl+Break / window close fires. The trainer loops poll `requested()`
// each iteration and exit gracefully (save checkpoint, return) instead of being killed
// mid-write.

use std::sync::atomic::{AtomicBool, Ordering};

static SHUTDOWN: AtomicBool = AtomicBool::new(false);

/// Install the console control handler. Call once at process startup.
pub fn install() -> windows::core::Result<()> {
    use windows::Win32::Foundation::BOOL;
    use windows::Win32::System::Console::SetConsoleCtrlHandler;

    unsafe extern "system" fn handler(_ctrl_type: u32) -> BOOL {
        // CTRL_C_EVENT=0, CTRL_BREAK_EVENT=1, CTRL_CLOSE_EVENT=2,
        // CTRL_LOGOFF_EVENT=5, CTRL_SHUTDOWN_EVENT=6 — flag shutdown for all of them.
        SHUTDOWN.store(true, Ordering::SeqCst);
        BOOL(1) // handled — caller's default handler should not run
    }

    unsafe { SetConsoleCtrlHandler(Some(handler), true) }
}

/// Returns true once a shutdown signal has been received.
pub fn requested() -> bool {
    SHUTDOWN.load(Ordering::Relaxed)
}

/// Programmatic shutdown trigger. Used by error-handling paths (e.g. checkpoint save
/// failure threshold exceeded) to flag the trainer for graceful exit on its next loop
/// iteration without going through a real signal.
pub fn trigger() {
    SHUTDOWN.store(true, Ordering::SeqCst);
}
