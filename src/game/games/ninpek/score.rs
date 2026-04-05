use super::super::Region;

pub struct ScoreRegion {
    pub region: Region,
    pub bw_threshold: u8,
}

pub const NINPEK_SCORE: ScoreRegion = ScoreRegion {
    region: Region {
        x: 6,
        y: 9,
        w: 7,
        h: 4,
    },
    bw_threshold: 128,
};

const MAX_SATURATION: u8 = 30;

/// Round to nearest 25% (0, 64, 128, 192, 255) to reduce aliasing flicker.
pub fn quantize(v: u8) -> u8 {
    match v {
        0..=31 => 0,
        32..=95 => 64,
        96..=159 => 128,
        160..=223 => 192,
        224..=255 => 255,
    }
}

fn classify_bw(bgra: &[u8], i: usize, threshold: u8) -> Option<bool> {
    let b = bgra[i];
    let g = bgra[i + 1];
    let r = bgra[i + 2];
    let sat = r.max(g).max(b) - r.min(g).min(b);
    if sat > MAX_SATURATION {
        return None;
    }
    let gray = quantize(((r as u16 + g as u16 + b as u16) / 3) as u8);
    if gray <= threshold {
        Some(false)
    } else if gray >= 255 - threshold {
        Some(true)
    } else {
        None
    }
}

pub fn is_score_clean(bgra: &[u8], width: u32, sr: &ScoreRegion) -> bool {
    let r = &sr.region;
    for y in r.y..r.y + r.h {
        for x in r.x..r.x + r.w {
            let i = (y * width + x) as usize * 4;
            if classify_bw(bgra, i, sr.bw_threshold).is_none() {
                return false;
            }
        }
    }
    true
}

pub fn count_score_flips(prev: &[u8], curr: &[u8], width: u32, sr: &ScoreRegion) -> u32 {
    let r = &sr.region;
    let mut flips = 0;
    for y in r.y..r.y + r.h {
        for x in r.x..r.x + r.w {
            let i = (y * width + x) as usize * 4;
            if let (Some(p), Some(c)) = (
                classify_bw(prev, i, sr.bw_threshold),
                classify_bw(curr, i, sr.bw_threshold),
            ) {
                if p != c {
                    flips += 1;
                }
            }
        }
    }
    flips
}
