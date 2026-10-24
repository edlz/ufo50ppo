// Cooperative shutdown signal. Ctrl+C / SIGTERM (and console close on Windows) flip a static
// AtomicBool. The trainer loops poll `requested()` each iteration and exit gracefully (save
// checkpoint, return) instead of being killed mid-write.

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};

static SHUTDOWN: AtomicBool = AtomicBool::new(false);

/// Install the signal handler. Call once at process startup.
pub fn install() -> io::Result<()> {
    ctrlc::set_handler(|| SHUTDOWN.store(true, Ordering::SeqCst)).map_err(io::Error::other)
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
