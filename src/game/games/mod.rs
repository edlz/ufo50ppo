pub mod ninpek;

use super::env::RewardFn;
use super::input::*;

pub struct ScoreRegion {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub bw_threshold: u8,
}

pub struct GameConfig {
    pub name: &'static str,
    pub extra_reset_keys: &'static [usize],
    pub reward_fn: RewardFn,
}

/// Classify a pixel as black (Some(false)), white (Some(true)), or neither (None).
const MAX_SATURATION: u8 = 30;

fn classify_bw(bgra: &[u8], i: usize, threshold: u8) -> Option<bool> {
    let b = bgra[i];
    let g = bgra[i + 1];
    let r = bgra[i + 2];
    let sat = r.max(g).max(b) - r.min(g).min(b);
    if sat > MAX_SATURATION {
        return None;
    }
    let gray = ((r as u16 + g as u16 + b as u16) / 3) as u8;
    if gray <= threshold {
        Some(false)
    } else if gray >= 255 - threshold {
        Some(true)
    } else {
        None
    }
}

/// Returns true if every pixel in the score region is pure B/W.
pub fn is_score_clean(bgra: &[u8], width: u32, region: &ScoreRegion) -> bool {
    for y in region.y..region.y + region.h {
        for x in region.x..region.x + region.w {
            let i = (y * width + x) as usize * 4;
            if classify_bw(bgra, i, region.bw_threshold).is_none() {
                return false;
            }
        }
    }
    true
}

/// Count B/W pixel flips in the score region.
pub fn count_score_flips(prev: &[u8], curr: &[u8], width: u32, region: &ScoreRegion) -> u32 {
    let mut flips = 0;
    for y in region.y..region.y + region.h {
        for x in region.x..region.x + region.w {
            let i = (y * width + x) as usize * 4;
            if let (Some(p), Some(c)) = (
                classify_bw(prev, i, region.bw_threshold),
                classify_bw(curr, i, region.bw_threshold),
            ) {
                if p != c {
                    flips += 1;
                }
            }
        }
    }
    flips
}

pub const GAME_1: GameConfig = GameConfig {
    name: "Barbuta",
    extra_reset_keys: &[],
    reward_fn: super::env::stub_reward,
};

pub const GAME_3: GameConfig = ninpek::CONFIG;

pub const GAME_21: GameConfig = GameConfig {
    name: "Waldorfs Journey",
    extra_reset_keys: &[VK_Z],
    reward_fn: super::env::stub_reward,
};
