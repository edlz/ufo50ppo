#!/bin/sh
# Builds a lockstep copy of UFO 50 in <out dir>: patched data.win + ufo50rl.dll, other game files symlinked.
# Usage: bridge/build.sh <game dir> <out dir>
# Env: UTMT_CLI (UndertaleModCli.dll path), DXVK_DIR (macOS only: extracted DXVK-macOS release)
set -eu

GAME=${1:?usage: build.sh <game dir> <out dir>}
OUT=${2:?usage: build.sh <game dir> <out dir>}
HERE=$(cd "$(dirname "$0")" && pwd)
UTMT_CLI=${UTMT_CLI:-$HOME/dev/UndertaleModTool/UndertaleModCli/bin/Release/net10.0/UndertaleModCli.dll}
if [ -z "${DOTNET_ROOT:-}" ] && [ -d /opt/homebrew/opt/dotnet/libexec ]; then
    export DOTNET_ROOT=/opt/homebrew/opt/dotnet/libexec
fi

mkdir -p "$OUT"
for f in "$GAME"/*; do
    b=$(basename "$f")
    case "$b" in
        data.win|ufo50.exe) ;;
        *) ln -sfn "$f" "$OUT/$b" ;;
    esac
done
# Wine resolves the game directory from the exe path, so the exe must be a real file.
cp "$GAME/ufo50.exe" "$OUT/ufo50.exe"

x86_64-w64-mingw32-gcc -O2 -shared -static-libgcc -o "$OUT/ufo50rl.dll" "$HERE/ufo50rl.c" -lws2_32 -luser32

# UndertaleModCli hangs reading stdin unless it is closed.
dotnet "$UTMT_CLI" load "$GAME/data.win" -s "$HERE/patch.csx" -o "$OUT/data.win" -f < /dev/null > "$OUT/patch.log" 2>&1
if grep -qiE "error|exception" "$OUT/patch.log"; then
    cat "$OUT/patch.log" >&2
    exit 1
fi

# macOS: Wine's built-in d3d11 draws nothing for this game; DXVK-macOS works.
if [ -n "${DXVK_DIR:-}" ]; then
    cp "$DXVK_DIR/x64/d3d11.dll" "$DXVK_DIR/x64/d3d10core.dll" "$OUT/"
    echo "dxgi.syncInterval = 0" > "$OUT/dxvk.conf"
fi

echo "built $OUT"
