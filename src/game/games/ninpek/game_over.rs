/// Detect leaderboard by checking for its structural pattern:
/// Dense white header at top + repeating white text rows at regular intervals.
pub fn is_leaderboard(bgra: &[u8], width: u32) -> bool {
    let leaderboard_rows: &[u32] = &[5, 6, 19, 20, 28, 29, 38, 39, 47, 48, 57, 58, 66, 67];
    let mut total_white = 0u32;
    let mut rows_with_white = 0u32;

    for &y in leaderboard_rows {
        let mut row_white = 0;
        for x in 30..100 {
            let i = (y * width + x) as usize * 4;
            if bgra[i] > 200 && bgra[i + 1] > 200 && bgra[i + 2] > 200 {
                row_white += 1;
            }
        }
        if row_white > 5 {
            rows_with_white += 1;
        }
        total_white += row_white;
    }

    rows_with_white >= 10 && total_white > 150
}

/// Check for the shared completion screen pattern: blue+orange icons + white center text + no HUD.
fn is_completion_screen(bgra: &[u8], width: u32) -> bool {
    let mut blue_count = 0u32;
    let mut orange_count = 0u32;
    let mut center_white = 0u32;

    for y in 44..56 {
        for x in 54..62 {
            let i = (y * width + x) as usize * 4;
            if bgra[i] > 100 && bgra[i + 2] < 100 && bgra[i + 1] < 80 {
                blue_count += 1;
            }
        }
    }

    for y in 44..56 {
        for x in 65..73 {
            let i = (y * width + x) as usize * 4;
            if bgra[i + 2] > 80 && bgra[i] < 60 {
                orange_count += 1;
            }
        }
    }

    for y in 67..75 {
        for x in 40..85 {
            let i = (y * width + x) as usize * 4;
            if bgra[i] > 200 && bgra[i + 1] > 200 && bgra[i + 2] > 200 {
                center_white += 1;
            }
        }
    }

    if blue_count < 3 || orange_count < 3 || center_white < 50 {
        return false;
    }

    // No white in top area (HUD would show if still in gameplay)
    for y in 0..30 {
        for x in 0..width {
            let i = (y * width + x) as usize * 4;
            if bgra[i] > 200 && bgra[i + 1] > 200 && bgra[i + 2] > 200 {
                return false;
            }
        }
    }
    true
}

/// Count non-black pixels in the upper area (y=8-36) to distinguish stage 1 vs game end.
fn upper_content(bgra: &[u8], width: u32) -> u32 {
    let mut count = 0;
    for y in 8..36 {
        for x in 0..width {
            let i = (y * width + x) as usize * 4;
            if bgra[i] > 30 || bgra[i + 1] > 30 || bgra[i + 2] > 30 {
                count += 1;
            }
        }
    }
    count
}

/// Stage 1 complete: completion screen with minimal upper content (<50 pixels)
pub fn is_stage_complete(bgra: &[u8], width: u32) -> bool {
    is_completion_screen(bgra, width) && upper_content(bgra, width) < 50
}

/// Game complete (final stage): completion screen with lots of upper content (>80 pixels)
pub fn is_game_complete(bgra: &[u8], width: u32) -> bool {
    is_completion_screen(bgra, width) && upper_content(bgra, width) > 80
}
