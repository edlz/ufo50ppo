use super::super::Region;

const BLUE_ONLY: &[(u8, u8, u8, u8)] = &[(59, 42, 163, 20), (68, 48, 186, 20)];

pub const LIFE_REGION: Region = Region {
    x: 0,
    y: 2,
    w: 32,
    h: 6,
};
const MAX_LIVES: u32 = 6;
const SLOT_W: u32 = LIFE_REGION.w / MAX_LIVES;
const SLOT_BLUE_MIN: u32 = 4;

fn count_blue_in_slot(bgra: &[u8], width: u32, slot: u32) -> u32 {
    let sx = LIFE_REGION.x + slot * SLOT_W;
    let mut count = 0;
    for y in LIFE_REGION.y..LIFE_REGION.y + LIFE_REGION.h {
        for x in sx..sx + SLOT_W {
            let i = (y * width + x) as usize * 4;
            let b = bgra[i];
            let g = bgra[i + 1];
            let r = bgra[i + 2];
            for &(tr, tg, tb, tol) in BLUE_ONLY {
                if r.abs_diff(tr) <= tol && g.abs_diff(tg) <= tol && b.abs_diff(tb) <= tol {
                    count += 1;
                    break;
                }
            }
        }
    }
    count
}

pub fn count_lives(bgra: &[u8], width: u32) -> u32 {
    let mut lives = 0;
    for slot in 0..MAX_LIVES {
        if count_blue_in_slot(bgra, width, slot) >= SLOT_BLUE_MIN {
            lives += 1;
        }
    }
    lives
}

/// Returns: -1 (lost), 0 (no change), +1 (gained)
pub fn check_life_change(bgra: &[u8], width: u32, committed: u32) -> i32 {
    if committed > 0 {
        let last_slot = committed - 1;
        if count_blue_in_slot(bgra, width, last_slot) < SLOT_BLUE_MIN {
            return -1;
        }
    }
    if committed < MAX_LIVES {
        let next_slot = committed;
        if count_blue_in_slot(bgra, width, next_slot) >= SLOT_BLUE_MIN {
            return 1;
        }
    }
    0
}
