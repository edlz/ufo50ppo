use super::*;
use crate::game::input::*;

pub const NINPEK_SCORE: ScoreRegion = ScoreRegion {
    x: 0,
    y: 21,
    w: 16,
    h: 3,
    bw_threshold: 128,
};

/// Check that the 1px border around the score region has no colored pixels in prev frame.
fn border_clean_in_prev(prev: &[u8], width: u32, region: &ScoreRegion) -> bool {
    let top = if region.y > 0 { region.y - 1 } else { region.y };
    let bottom = region.y + region.h;
    let left = if region.x > 0 { region.x - 1 } else { region.x };
    let right = region.x + region.w;

    for y in top..=bottom {
        for x in left..=right {
            if y >= region.y && y < region.y + region.h && x >= region.x && x < region.x + region.w
            {
                continue;
            }
            let i = (y * width + x) as usize * 4;
            let b = prev[i];
            let g = prev[i + 1];
            let r = prev[i + 2];
            let sat = r.max(g).max(b) - r.min(g).min(b);
            if sat > super::MAX_SATURATION {
                return false;
            }
        }
    }
    true
}

// Character colors: blue + green
const CHAR_COLORS: &[(u8, u8, u8, u8)] = &[
    (59, 42, 163, 20),  // purple-blue
    (68, 48, 186, 20),  // deep-blue
    (88, 245, 177, 20), // green
];

// Life icon region
const LIFE_REGION_X: u32 = 0;
const LIFE_REGION_Y: u32 = 14;
const LIFE_REGION_W: u32 = 16;
const LIFE_REGION_H: u32 = 7;

// Play area (below HUD)
const PLAY_REGION_Y: u32 = 25;
const PLAY_REGION_H: u32 = 103; // 128 - 25

// Thresholds
const LIFE_ICON_MIN_PIXELS: i32 = 20;
const DEATH_TOTAL_DROP: i32 = -60;

const BLUE_ONLY: &[(u8, u8, u8, u8)] = &[(59, 42, 163, 20), (68, 48, 186, 20)];

fn count_colors_in_rect(
    bgra: &[u8],
    width: u32,
    rx: u32,
    ry: u32,
    rw: u32,
    rh: u32,
    colors: &[(u8, u8, u8, u8)],
) -> u32 {
    let mut count = 0;
    for y in ry..ry + rh {
        for x in rx..rx + rw {
            let i = (y * width + x) as usize * 4;
            let b = bgra[i];
            let g = bgra[i + 1];
            let r = bgra[i + 2];
            for &(tr, tg, tb, tol) in colors {
                if r.abs_diff(tr) <= tol && g.abs_diff(tg) <= tol && b.abs_diff(tb) <= tol {
                    count += 1;
                    break;
                }
            }
        }
    }
    count
}

fn ninpek_reward(prev: Option<&[u8]>, curr: &[u8], width: u32, _height: u32) -> (f64, bool, bool) {
    let Some(prev) = prev else {
        return (0.0, false, true);
    };

    // Life region: blue-only (green alone is not a life icon)
    // Play region: all char colors (blue+green) to detect the character
    let prev_life = count_colors_in_rect(
        prev,
        width,
        LIFE_REGION_X,
        LIFE_REGION_Y,
        LIFE_REGION_W,
        LIFE_REGION_H,
        BLUE_ONLY,
    ) as i32;
    let curr_life = count_colors_in_rect(
        curr,
        width,
        LIFE_REGION_X,
        LIFE_REGION_Y,
        LIFE_REGION_W,
        LIFE_REGION_H,
        BLUE_ONLY,
    ) as i32;
    let prev_play = count_colors_in_rect(
        prev,
        width,
        0,
        PLAY_REGION_Y,
        width,
        PLAY_REGION_H,
        CHAR_COLORS,
    ) as i32;
    let curr_play = count_colors_in_rect(
        curr,
        width,
        0,
        PLAY_REGION_Y,
        width,
        PLAY_REGION_H,
        CHAR_COLORS,
    ) as i32;

    let life_delta = curr_life - prev_life;
    let total_delta = (curr_life + curr_play) - (prev_life + prev_play);

    // Game over: blue drops to 0 in life region
    if curr_life == 0 && prev_life > 0 {
        return (-2.0, true, true);
    }

    // Death: character + life icon disappear → large total drop
    if total_delta < DEATH_TOTAL_DROP && life_delta < -LIFE_ICON_MIN_PIXELS {
        return (-1.0, false, true);
    }

    // Life gain: life region grew and total also grew (not character moving in)
    if life_delta > LIFE_ICON_MIN_PIXELS && total_delta > LIFE_ICON_MIN_PIXELS {
        return (0.5, false, true);
    }

    // Score change check
    if !is_score_clean(curr, width, &NINPEK_SCORE) {
        return (0.0, false, false);
    }
    if !border_clean_in_prev(prev, width, &NINPEK_SCORE) {
        return (0.0, false, true);
    }
    let flips = count_score_flips(prev, curr, width, &NINPEK_SCORE);
    let reward = if flips > 0 { 1.0 } else { 0.0 };
    (reward, false, true)
}

pub const CONFIG: GameConfig = GameConfig {
    name: "Ninpek",
    extra_reset_keys: &[VK_Z],
    reward_fn: ninpek_reward,
};
