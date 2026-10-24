#!/bin/sh
# Linux: runs a command (usually the trainer) on a virtual X display, so the Wine game windows
# need no monitor or desktop session.
# Usage: bridge/headless.sh cargo run --release -- -g build/game -N 8
set -eu

if ! command -v Xvfb >/dev/null 2>&1; then
    echo "headless.sh: Xvfb not found (Debian/Ubuntu: apt install xvfb)" >&2
    exit 1
fi

# Started directly rather than via xvfb-run, whose readiness handshake can hang.
display=99
while [ -e "/tmp/.X11-unix/X$display" ] || [ -e "/tmp/.X$display-lock" ]; do
    display=$((display + 1))
done
Xvfb ":$display" -screen 0 1280x720x24 -nolisten tcp >/dev/null 2>&1 &
xvfb_pid=$!
trap 'kill "$xvfb_pid" 2>/dev/null' EXIT INT TERM
while [ ! -e "/tmp/.X11-unix/X$display" ]; do
    kill -0 "$xvfb_pid" 2>/dev/null || { echo "headless.sh: Xvfb failed to start" >&2; exit 1; }
    sleep 0.2
done
export DISPLAY=":$display"
export WINEDEBUG="${WINEDEBUG:--all}"

# Create the Wine prefix up front: first-time setup can outlast the trainer's launch timeout.
if [ ! -d "${WINEPREFIX:-$HOME/.wine}" ]; then
    WINEDLLOVERRIDES="mscoree,mshtml=" wineboot -i
fi

"$@"
