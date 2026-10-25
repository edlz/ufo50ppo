#!/usr/bin/env python3
"""Renders record_ghosts output as a video: every recorded game is a translucent ninja, tinted by
checkpoint (blue = oldest, yellow = newest), over the latest checkpoint's longest game. A fading
cross marks each lost life.

Needs ffmpeg and the ninja sprite from scripts/export_ninja_sprite.csx.
Usage: scripts/render_ghosts.py [ghost dir] [--per-checkpoint] [--fps N] [--scale N] [-o out.mp4]
"""
import argparse
import csv
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 384, 216
STOPS = [(60, 110, 255), (210, 70, 230), (255, 120, 40), (255, 240, 70)]
# Camera px a death cross stays on screen (2 px per step).
DEATH_FADE = 60


def gen_color(t):
    t *= len(STOPS) - 1
    i = min(int(t), len(STOPS) - 2)
    return np.array([a + (b - a) * (t - i) for a, b in zip(STOPS[i], STOPS[i + 1])]) / 255


def steps_label(n):
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


class Game:
    def __init__(self, rows):
        rows.sort(key=lambda r: r[0])
        self.vx = np.array([r[1] for r in rows])
        self.x = np.array([r[2] for r in rows])
        self.y = np.array([r[3] for r in rows])
        self.index = {int(v): i for i, v in enumerate(self.vx)}
        alive = self.x >= 0
        # A life ends where the player disappears, or at the last step on game over.
        ends = np.flatnonzero(alive[:-1] & ~alive[1:]).tolist()
        if alive[-1]:
            ends.append(len(alive) - 1)
        self.deaths = [(self.vx[i], self.vx[i] + self.x[i], self.y[i]) for i in ends]

    def sample(self, v):
        """Index of the step with camera x `v`, or None outside the game."""
        for d in (0, -1, 1):
            i = self.index.get(v + d)
            if i is not None:
                return i
        return None


def load(ghost_dir):
    games = defaultdict(list)
    with open(ghost_dir / "ghosts.csv") as f:
        for r in csv.DictReader(f):
            games[(int(r["checkpoint_steps"]), int(r["game"]))].append(
                (int(r["step"]), float(r["view_x"]), float(r["x"]), float(r["y"]))
            )
    by_ckpt = defaultdict(list)
    for (ckpt, _), rows in sorted(games.items()):
        by_ckpt[ckpt].append(Game([(s, vx, x, y) for s, vx, x, y in rows]))
    bg = np.memmap(ghost_dir / "bg.rgb", dtype=np.uint8, mode="r").reshape(-1, H, W, 3)
    bg_vx = [int(float(v)) for v in (ghost_dir / "bg_view_x.txt").read_text().split()]
    return dict(by_ckpt), bg, bg_vx


def load_sprite(d):
    frames = [
        np.asarray(Image.open(p).convert("RGBA"), dtype=np.float32) / 255
        for p in sorted(d.glob("run_*.png"))
    ]
    ox, oy = map(int, (d / "origin.txt").read_text().split()[:2])
    return frames, ox, oy


def tinted(frames, color, alpha):
    """Sprite frames recoloured to `color`, keeping their shading; [frame][flipped]."""
    out = []
    for f in frames:
        lum = f[..., :3].mean(axis=2, keepdims=True)
        rgba = np.concatenate([color * (0.35 + 0.65 * lum), f[..., 3:] * alpha], axis=2)
        out.append((rgba, rgba[:, ::-1]))
    return out


def blit(canvas, spr, x, y):
    h, w = spr.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    s = spr[y0 - y : y1 - y, x0 - x : x1 - x]
    a = s[..., 3:]
    region = canvas[y0:y1, x0:x1]
    region *= 1 - a
    region += s[..., :3] * a


def cross(canvas, x, y, color, alpha):
    for d in range(-2, 3):
        for px, py in ((x + d, y + d), (x + d, y - d)):
            if 0 <= px < W and 0 <= py < H:
                canvas[py, px] = canvas[py, px] * (1 - alpha) + color * alpha


def draw_frame(bg_frame, v, layers, ox, oy):
    """`layers`: (games, sprite frames, colour) drawn in order, so later ones end up on top."""
    canvas = bg_frame.astype(np.float32) / 255 * 0.55
    alive = 0
    for games, sprites, color in layers:
        for g in games:
            for dv, wx, y in g.deaths:
                if 0 <= v - dv < DEATH_FADE:
                    cross(canvas, int(wx - v), int(y), color, 1 - (v - dv) / DEATH_FADE)
            i = g.sample(v)
            if i is None or g.x[i] < 0:
                continue
            alive += 1
            back = max(i - 2, 0)
            flip = int(g.vx[i] + g.x[i] < g.vx[back] + g.x[back])
            spr = sprites[(i // 4) % len(sprites)][flip]
            sx = int(round(g.x[i] + g.vx[i] - v))
            blit(canvas, spr, sx - ox, int(round(g.y[i])) - oy)
    return canvas, alive


def hud(canvas, lines, font):
    img = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for i, (text, color) in enumerate(lines):
        d.text((W - 4, 4 + 12 * i), text, fill=tuple(int(c * 255) for c in color), font=font, anchor="ra")
    return img.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ghost_dir", nargs="?", default="runs/ninpek/ghosts", type=Path)
    ap.add_argument("--sprite", type=Path, help="default: <ghost dir>/sprite")
    ap.add_argument("--per-checkpoint", action="store_true", help="one segment per checkpoint instead of all at once")
    ap.add_argument("--fps", type=int, default=30, help="video fps; the game runs 15 steps/s")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("-o", "--out", type=Path)
    args = ap.parse_args()

    by_ckpt, bg, bg_vx = load(args.ghost_dir)
    frames, ox, oy = load_sprite(args.sprite or args.ghost_dir / "sprite")
    ckpts = sorted(by_ckpt)
    colors = {c: gen_color(i / max(len(ckpts) - 1, 1)) for i, c in enumerate(ckpts)}
    out = args.out or args.ghost_dir / ("ghosts_per_checkpoint.mp4" if args.per_checkpoint else "ghosts.mp4")
    font = ImageFont.load_default()

    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-",
         "-vf", f"scale=iw*{args.scale}:ih*{args.scale}:flags=neighbor",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)],
        stdin=subprocess.PIPE,
    )
    total = sum(len(g) for g in by_ckpt.values())
    if args.per_checkpoint:
        for c in ckpts:
            layer = [(by_ckpt[c], tinted(frames, colors[c], 0.6), colors[c])]
            end = max(g.vx[-1] for g in by_ckpt[c]) + DEATH_FADE
            for f, v in enumerate(bg_vx):
                if v > end:
                    break
                canvas, alive = draw_frame(bg[f], v, layer, ox, oy)
                ffmpeg.stdin.write(hud(canvas, [
                    (f"{steps_label(c)} training steps", colors[c]),
                    (f"alive {alive}/{len(by_ckpt[c])}", (1, 1, 1)),
                ], font))
    else:
        layers = [(by_ckpt[c], tinted(frames, colors[c], 0.45), colors[c]) for c in ckpts]
        legend = [
            (f"oldest: {steps_label(ckpts[0])} steps", colors[ckpts[0]]),
            (f"newest: {steps_label(ckpts[-1])} steps", colors[ckpts[-1]]),
        ]
        for f, v in enumerate(bg_vx):
            canvas, alive = draw_frame(bg[f], v, layers, ox, oy)
            ffmpeg.stdin.write(hud(canvas, [(f"alive {alive}/{total}", (1, 1, 1))] + legend, font))
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
