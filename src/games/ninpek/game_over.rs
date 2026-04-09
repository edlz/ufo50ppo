// All pixel detectors here were originally calibrated for a 128x128 capture. Coordinates
// and thresholds scale from that reference at runtime via `s_x` / `s_y` / `s_count`, so the
// detectors work at any obs resolution close to 128 (84x84 has been verified).

const REF: u32 = 128;

#[inline]
fn s_x(v: u32, width: u32) -> u32 {
    v * width / REF
}

#[inline]
fn s_y(v: u32, height: u32) -> u32 {
    v * height / REF
}

#[inline]
fn s_count(v: u32, width: u32, height: u32) -> u32 {
    // Pixel-count thresholds scale with capture area.
    let scaled = (v as u64) * (width as u64) * (height as u64) / (REF as u64 * REF as u64);
    scaled as u32
}

fn valid_buf(bgra: &[u8], width: u32) -> bool {
    bgra.len() >= (width as usize * width as usize * 4)
}

/// Detect leaderboard by checking for its structural pattern.
pub fn is_leaderboard(bgra: &[u8], width: u32) -> bool {
    if !valid_buf(bgra, width) {
        return false;
    }
    let height = width; // square obs
    const REF_ROWS: &[u32] = &[5, 6, 19, 20, 28, 29, 38, 39, 47, 48, 57, 58, 66, 67];
    let mut total_white = 0u32;
    let mut rows_with_white = 0u32;

    let x_start = s_x(30, width);
    let x_end = s_x(100, width);
    let row_white_threshold = s_x(5, width).max(2);

    for &ref_y in REF_ROWS {
        let y = s_y(ref_y, height);
        let mut row_white = 0;
        for x in x_start..x_end {
            let i = (y * width + x) as usize * 4;
            if bgra[i] > 200 && bgra[i + 1] > 200 && bgra[i + 2] > 200 {
                row_white += 1;
            }
        }
        if row_white > row_white_threshold {
            rows_with_white += 1;
        }
        total_white += row_white;
    }

    rows_with_white >= 10 && total_white > s_count(150, width, height)
}

/// Check for the shared completion screen pattern: blue+orange icons + white center text + no HUD.
fn is_completion_screen(bgra: &[u8], width: u32) -> bool {
    if !valid_buf(bgra, width) {
        return false;
    }
    let height = width;
    let mut blue_count = 0u32;
    let mut orange_count = 0u32;
    let mut center_white = 0u32;

    let icon_y0 = s_y(44, height);
    let icon_y1 = s_y(56, height);
    let blue_x0 = s_x(54, width);
    let blue_x1 = s_x(62, width);
    let orange_x0 = s_x(65, width);
    let orange_x1 = s_x(73, width);

    for y in icon_y0..icon_y1 {
        for x in blue_x0..blue_x1 {
            let i = (y * width + x) as usize * 4;
            if bgra[i] > 100 && bgra[i + 2] < 100 && bgra[i + 1] < 80 {
                blue_count += 1;
            }
        }
    }

    for y in icon_y0..icon_y1 {
        for x in orange_x0..orange_x1 {
            let i = (y * width + x) as usize * 4;
            if bgra[i + 2] > 80 && bgra[i] < 60 {
                orange_count += 1;
            }
        }
    }

    let text_y0 = s_y(67, height);
    let text_y1 = s_y(75, height);
    let text_x0 = s_x(40, width);
    let text_x1 = s_x(85, width);
    for y in text_y0..text_y1 {
        for x in text_x0..text_x1 {
            let i = (y * width + x) as usize * 4;
            if bgra[i] > 200 && bgra[i + 1] > 200 && bgra[i + 2] > 200 {
                center_white += 1;
            }
        }
    }

    let blue_min = s_count(3, width, height).max(1);
    let orange_min = s_count(3, width, height).max(1);
    let white_min = s_count(50, width, height);
    if blue_count < blue_min || orange_count < orange_min || center_white < white_min {
        return false;
    }

    // No white in top area (HUD would show if still in gameplay)
    let top_y = s_y(30, height);
    for y in 0..top_y {
        for x in 0..width {
            let i = (y * width + x) as usize * 4;
            if bgra[i] > 200 && bgra[i + 1] > 200 && bgra[i + 2] > 200 {
                return false;
            }
        }
    }
    true
}

/// Count non-black pixels in the upper area (y=8-36 in 128px ref) to distinguish stage 1 vs game end.
fn upper_content(bgra: &[u8], width: u32) -> u32 {
    let height = width;
    let y0 = s_y(8, height);
    let y1 = s_y(36, height);
    let mut count = 0;
    for y in y0..y1 {
        for x in 0..width {
            let i = (y * width + x) as usize * 4;
            if bgra[i] > 30 || bgra[i + 1] > 30 || bgra[i + 2] > 30 {
                count += 1;
            }
        }
    }
    count
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CompletionKind {
    Stage,
    Game,
}

/// Single-pass classification — runs `is_completion_screen` + `upper_content` once.
pub fn classify_completion(bgra: &[u8], width: u32) -> Option<CompletionKind> {
    if !is_completion_screen(bgra, width) {
        return None;
    }
    let height = width;
    let upper = upper_content(bgra, width);
    if upper < s_count(50, width, height) {
        Some(CompletionKind::Stage)
    } else if upper > s_count(80, width, height) {
        Some(CompletionKind::Game)
    } else {
        None
    }
}
