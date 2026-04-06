use std::sync::{Arc, Mutex};
use ufo50ppo::games;
use ufo50ppo::games::GameTracker;
use ufo50ppo::platform;
use windows::Win32::Foundation::*;
use windows::Win32::Graphics::Gdi::*;
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::*;

use ufo50ppo::util::{OBS_H, OBS_W, WINDOW_TITLE};
const SNAP_INTERVAL: u32 = 100;

// Regions from ninpek config
const SCORE_X: u32 = games::ninpek::NINPEK_SCORE.region.x;
const SCORE_Y: u32 = games::ninpek::NINPEK_SCORE.region.y;
const SCORE_W: u32 = games::ninpek::NINPEK_SCORE.region.w;
const SCORE_H: u32 = games::ninpek::NINPEK_SCORE.region.h;

const LIFE_X: u32 = games::ninpek::LIFE_REGION.x;
const LIFE_Y: u32 = games::ninpek::LIFE_REGION.y;
const LIFE_W: u32 = games::ninpek::LIFE_REGION.w;
const LIFE_H: u32 = games::ninpek::LIFE_REGION.h;

// Preview window
const PREVIEW_SCALE: u32 = 10;
const PREVIEW_X: u32 = 0;
const PREVIEW_Y: u32 = if LIFE_Y > 1 { LIFE_Y - 1 } else { 0 };
const PREVIEW_W: u32 = 34;
const PREVIEW_H: u32 = SCORE_Y + SCORE_H + 2 - PREVIEW_Y;
const WIN_W: u32 = PREVIEW_W * PREVIEW_SCALE;
const WIN_H: u32 = PREVIEW_H * PREVIEW_SCALE;

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
                let _ = EndPaint(hwnd, &ps);
            }
            LRESULT(0)
        }
        _ => unsafe { DefWindowProcA(hwnd, msg, wparam, lparam) },
    }
}

fn create_preview_window() -> Result<HWND> {
    let class_name = s!("NinpekPreview");
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
            s!("Ninpek Preview"),
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

    for dy in 0..PREVIEW_H {
        for dx in 0..PREVIEW_W {
            let sx = PREVIEW_X + dx;
            let sy = PREVIEW_Y + dy;
            let si = (sy * src_w + sx) as usize * 4;
            let (b, g, r, a) = if si + 3 < pixels.len() {
                (pixels[si], pixels[si + 1], pixels[si + 2], pixels[si + 3])
            } else {
                (0, 0, 0, 255)
            };
            for py in 0..PREVIEW_SCALE {
                for px in 0..PREVIEW_SCALE {
                    let di =
                        ((dy * PREVIEW_SCALE + py) * WIN_W + dx * PREVIEW_SCALE + px) as usize * 4;
                    buf[di] = b;
                    buf[di + 1] = g;
                    buf[di + 2] = r;
                    buf[di + 3] = a;
                }
            }
        }
    }

    fn draw_box(
        buf: &mut [u8],
        win_w: u32,
        rx: u32,
        ry: u32,
        rw: u32,
        rh: u32,
        px: u32,
        py: u32,
        scale: u32,
        cr: u8,
        cg: u8,
        cb: u8,
    ) {
        let x1 = (rx - px) * scale;
        let y1 = (ry - py) * scale;
        let x2 = (rx + rw - px) * scale;
        let y2 = (ry + rh - py) * scale;
        for x in x1..x2 {
            for t in 0..2 {
                let top = ((y1 + t) * win_w + x) as usize * 4;
                let bot = ((y2 - 1 - t) * win_w + x) as usize * 4;
                if top + 3 < buf.len() {
                    buf[top] = cb;
                    buf[top + 1] = cg;
                    buf[top + 2] = cr;
                    buf[top + 3] = 255;
                }
                if bot + 3 < buf.len() {
                    buf[bot] = cb;
                    buf[bot + 1] = cg;
                    buf[bot + 2] = cr;
                    buf[bot + 3] = 255;
                }
            }
        }
        for y in y1..y2 {
            for t in 0..2 {
                let left = (y * win_w + x1 + t) as usize * 4;
                let right = (y * win_w + x2 - 1 - t) as usize * 4;
                if left + 3 < buf.len() {
                    buf[left] = cb;
                    buf[left + 1] = cg;
                    buf[left + 2] = cr;
                    buf[left + 3] = 255;
                }
                if right + 3 < buf.len() {
                    buf[right] = cb;
                    buf[right + 1] = cg;
                    buf[right + 2] = cr;
                    buf[right + 3] = 255;
                }
            }
        }
    }

    draw_box(
        &mut buf,
        WIN_W,
        SCORE_X,
        SCORE_Y,
        SCORE_W,
        SCORE_H,
        PREVIEW_X,
        PREVIEW_Y,
        PREVIEW_SCALE,
        255,
        0,
        0,
    );
    draw_box(
        &mut buf,
        WIN_W,
        LIFE_X,
        LIFE_Y,
        LIFE_W,
        LIFE_H,
        PREVIEW_X,
        PREVIEW_Y,
        PREVIEW_SCALE,
        255,
        255,
        0,
    );

    drop(buf);
    let _ = unsafe { InvalidateRect(hwnd, None, false) };
}

fn main() -> windows::core::Result<()> {
    platform::win32::capture::init()?;

    let preview_hwnd = create_preview_window()?.0 as usize;

    let mut tracker = games::ninpek::NinpekTracker::new(OBS_W);
    let mut total_reward = 0.0;
    let mut frame_count = 0u32;

    std::fs::create_dir_all("debug_frames").ok();

    println!("Testing reward for: Ninpek");
    println!("Preview: red=score, yellow=life");
    println!("Press Ctrl+C to stop.\n");

    platform::win32::capture::run(
        WINDOW_TITLE,
        move |crop, frame, reader: &mut platform::win32::capture::FrameReader| {
            let pixels = match reader.read_cropped(frame, crop, OBS_W, OBS_H) {
                Ok(p) => p.to_vec(),
                Err(e) => {
                    eprintln!("read error: {}", e);
                    return true;
                }
            };

            if frame_count % SNAP_INTERVAL == 0 {
                let path = format!("debug_frames/snap_{:03}.bmp", frame_count / SNAP_INTERVAL);
                reader.save_debug_bmp(&path).ok();
            }

            update_preview(&pixels, OBS_W, HWND(preview_hwnd as *mut _));

            let result = tracker.process_frame(&pixels);
            total_reward += result.reward;

            if !result.event_name.is_empty() {
                println!(
                    "Frame {:5} | {} {:+.1} | total: {:.1} | lives:{}",
                    frame_count, result.event_name, result.reward, total_reward, result.lives
                );
            }

            if result.done {
                reader.save_debug_bmp("debug_frames/game_over.bmp").ok();
                return false;
            }

            frame_count += 1;
            true
        },
    )
}
