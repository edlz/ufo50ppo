use std::sync::{Arc, Mutex};
use ufo50ppo::game;
use windows::Win32::Foundation::*;
use windows::Win32::Graphics::Gdi::*;
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::*;

const OBS_W: u32 = 128;
const OBS_H: u32 = 128;
const WINDOW_TITLE: &str = "UFO 50";

// Score region from ninpek config
const SCORE_X: u32 = game::games::ninpek::NINPEK_SCORE.x;
const SCORE_Y: u32 = game::games::ninpek::NINPEK_SCORE.y;
const SCORE_W: u32 = game::games::ninpek::NINPEK_SCORE.w;
const SCORE_H: u32 = game::games::ninpek::NINPEK_SCORE.h;

const PREVIEW_SCALE: u32 = 16;
const BORDER: u32 = 1;
const REGION_X: u32 = if SCORE_X > BORDER {
    SCORE_X - BORDER
} else {
    0
};
const REGION_Y: u32 = if SCORE_Y > BORDER {
    SCORE_Y - BORDER
} else {
    0
};
const REGION_W: u32 = SCORE_W + BORDER * 2;
const REGION_H: u32 = SCORE_H + BORDER * 2;
const WIN_W: u32 = REGION_W * PREVIEW_SCALE;
const WIN_H: u32 = REGION_H * PREVIEW_SCALE;

// Blue-only colors for tracking
const TRACKED_COLORS: &[(u8, u8, u8, u8, &str)] = &[
    (59, 42, 163, 20, "purple-blue"),
    (68, 48, 186, 20, "deep-blue"),
];

fn count_color(pixels: &[u8], w: u32, h: u32, tr: u8, tg: u8, tb: u8, tol: u8) -> u32 {
    let mut count = 0;
    for i in 0..(w * h) as usize {
        let b = pixels[i * 4];
        let g = pixels[i * 4 + 1];
        let r = pixels[i * 4 + 2];
        if r.abs_diff(tr) <= tol && g.abs_diff(tg) <= tol && b.abs_diff(tb) <= tol {
            count += 1;
        }
    }
    count
}

static PREVIEW_BUF: std::sync::LazyLock<Arc<Mutex<Vec<u8>>>> =
    std::sync::LazyLock::new(|| Arc::new(Mutex::new(vec![0u8; (WIN_W * WIN_H * 4) as usize])));

unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    match msg {
        WM_PAINT => {
            let mut ps = PAINTSTRUCT::default();
            let hdc = unsafe { BeginPaint(hwnd, &mut ps) };
            let buf = PREVIEW_BUF.lock().unwrap();
            let bmi = BITMAPINFO {
                bmiHeader: BITMAPINFOHEADER {
                    biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                    biWidth: WIN_W as i32,
                    biHeight: -(WIN_H as i32),
                    biPlanes: 1,
                    biBitCount: 32,
                    biCompression: BI_RGB.0,
                    ..Default::default()
                },
                ..Default::default()
            };
            unsafe {
                StretchDIBits(
                    hdc,
                    0,
                    0,
                    WIN_W as i32,
                    WIN_H as i32,
                    0,
                    0,
                    WIN_W as i32,
                    WIN_H as i32,
                    Some(buf.as_ptr() as _),
                    &bmi,
                    DIB_RGB_COLORS,
                    SRCCOPY,
                );
                EndPaint(hwnd, &ps);
            }
            LRESULT(0)
        }
        _ => unsafe { DefWindowProcA(hwnd, msg, wparam, lparam) },
    }
}

fn create_preview_window() -> Result<HWND> {
    let class_name = s!("ScorePreview");
    let wc = WNDCLASSA {
        lpfnWndProc: Some(wndproc),
        lpszClassName: PCSTR(class_name.as_ptr() as _),
        hCursor: unsafe { LoadCursorW(None, IDC_ARROW)? },
        ..Default::default()
    };
    unsafe { RegisterClassA(&wc) };
    let hwnd = unsafe {
        CreateWindowExA(
            WINDOW_EX_STYLE(0),
            class_name,
            s!("Score Preview"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            WIN_W as i32 + 16,
            WIN_H as i32 + 39,
            None,
            None,
            None,
            None,
        )?
    };
    Ok(hwnd)
}

fn update_preview(pixels: &[u8], src_w: u32, hwnd: HWND) {
    let mut buf = PREVIEW_BUF.lock().unwrap();
    let sl = SCORE_X.saturating_sub(REGION_X);
    let st = SCORE_Y.saturating_sub(REGION_Y);
    let sr = sl + SCORE_W;
    let sb = st + SCORE_H;

    for dy in 0..REGION_H {
        for dx in 0..REGION_W {
            let sx = REGION_X + dx;
            let sy = REGION_Y + dy;
            let si = (sy * src_w + sx) as usize * 4;
            let (b, g, r, a) = (pixels[si], pixels[si + 1], pixels[si + 2], pixels[si + 3]);

            let on_score = (dx == sl.wrapping_sub(1) || dx == sr) && dy >= st && dy < sb
                || (dy == st.wrapping_sub(1) || dy == sb) && dx >= sl.wrapping_sub(1) && dx <= sr;

            for py in 0..PREVIEW_SCALE {
                for px in 0..PREVIEW_SCALE {
                    let di =
                        ((dy * PREVIEW_SCALE + py) * WIN_W + dx * PREVIEW_SCALE + px) as usize * 4;
                    let edge = px == 0 || py == 0;
                    if on_score && edge {
                        buf[di] = 0;
                        buf[di + 1] = 0;
                        buf[di + 2] = 255;
                        buf[di + 3] = 255;
                    } else {
                        buf[di] = b;
                        buf[di + 1] = g;
                        buf[di + 2] = r;
                        buf[di + 3] = a;
                    }
                }
            }
        }
    }
    drop(buf);
    unsafe { InvalidateRect(hwnd, None, false) };
}

fn main() -> windows::core::Result<()> {
    game::capture::init()?;

    let preview_hwnd = create_preview_window()?.0 as usize;

    let config = &game::games::GAME_3;
    let reward_fn = config.reward_fn;

    let mut prev_pixels: Option<Vec<u8>> = None;
    let mut total_reward = 0.0;
    let mut frame_count = 0u32;
    let mut reward_count = 0u32;
    let mut prev_blue: u32 = 0;

    std::fs::create_dir_all("debug_frames").ok();

    println!("Testing reward for: {}", config.name);
    println!("Score Preview window open.");
    println!("Press Ctrl+C to stop.\n");

    game::capture::run(
        WINDOW_TITLE,
        move |frame, reader: &mut game::capture::FrameReader| {
            let pixels = match reader.read(frame, OBS_W, OBS_H) {
                Ok(p) => p.to_vec(),
                Err(e) => {
                    eprintln!("read error: {}", e);
                    return true;
                }
            };

            update_preview(&pixels, OBS_W, HWND(preview_hwnd as *mut _));

            let (reward, _done, clean) = reward_fn(prev_pixels.as_deref(), &pixels, OBS_W, OBS_H);

            let blue_count: u32 = TRACKED_COLORS
                .iter()
                .map(|&(r, g, b, tol, _)| count_color(&pixels, OBS_W, OBS_H, r, g, b, tol))
                .sum();

            if reward != 0.0 {
                total_reward += reward;
                let path = format!("debug_frames/reward_{:03}.bmp", reward_count);
                reader.save_debug_bmp(&path).ok();
                let kind = if reward > 0.5 {
                    "SCORE"
                } else if reward > 0.0 {
                    "LIFE+"
                } else {
                    "LIFE-"
                };
                println!(
                    "Frame {:5} | {} {:+.1} | total: {:.1} | blue:{} | {}",
                    frame_count, kind, reward, total_reward, blue_count, path
                );
                reward_count += 1;
            }

            if prev_blue > blue_count && prev_blue - blue_count >= 20 {
                println!(
                    "Frame {:5} | BLUE DROP   {} -> {} (-{})",
                    frame_count,
                    prev_blue,
                    blue_count,
                    prev_blue - blue_count
                );
            } else if blue_count > prev_blue && blue_count - prev_blue >= 20 {
                println!(
                    "Frame {:5} | BLUE GAIN   {} -> {} (+{})",
                    frame_count,
                    prev_blue,
                    blue_count,
                    blue_count - prev_blue
                );
            }

            prev_blue = blue_count;

            if clean {
                prev_pixels = Some(pixels);
            }
            frame_count += 1;
            true
        },
    )
}
