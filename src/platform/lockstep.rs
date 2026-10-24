// Lockstep connection to a game copy built by `bridge/build.sh`. The game blocks after each
// sync until we reply, so every observation pairs with exactly the action that produced it.
// Wire layout must match bridge/patch.csx.

use super::{ACTION_MAP, Frame, GameRunner};
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const RAW_W: usize = 384;
const RAW_H: usize = 216;
const SCALE: usize = 3;
pub const OBS_W: usize = RAW_W / SCALE;
pub const OBS_H: usize = RAW_H / SCALE;
const STATE_LEN: usize = 16;
const MSG_LEN: usize = STATE_LEN * 8 + RAW_W * RAW_H * 4;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(90);

pub const FLAG_QUIT: u32 = 1;
pub const FLAG_LAUNCH: u32 = 2;

pub const fn launch_flag(game_id: u8) -> u32 {
    FLAG_LAUNCH | (game_id as u32) << 8
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GameState {
    /// Frames since boot, counted by the game.
    pub frame: u64,
    pub room: i32,
    /// Internal game ID (Ninpek = 34), 0 in the library.
    pub game_id: i32,
    pub paused: bool,
    /// Ninpek: o34_Mas.points[0], -1 outside a run.
    pub points: f64,
    /// Ninpek: o34_Mas.life[0], -1 on game over, -99 outside a run.
    pub lives: f64,
    /// Ninpek: o34_Mas.state, -1 when o34_Mas doesn't exist.
    pub mas_state: i32,
    /// Ninpek: o34_Mas.iWater, the stage section (0-3) by scroll position, -1 outside a run.
    pub zone: i32,
    /// Title screen cursor (0 = first option, e.g. 1 PLAYER), -1 off the title screen.
    pub menu_sel: i32,
    pub num_players: i32,
    /// A recorded attract-mode demo is playing; input is ignored.
    pub demo: bool,
    /// Ninpek: camera x, i.e. how far the run has scrolled; -1 outside a run.
    pub view_x: f64,
    /// Ninpek: global.g34_totalKills, enemies killed this run; -1 outside a run.
    pub kills: f64,
    /// Ninpek: o34_BossEye.life (35 at full health), -1 while the boss doesn't exist.
    pub boss_life: f64,
}

pub struct LockstepGame {
    child: Child,
    stream: TcpStream,
    msg: Vec<u8>,
    obs: Vec<u8>,
    frame_skip: u32,
    game_id: u8,
    pending_flags: u32,
    awaiting_reply: bool,
}

impl LockstepGame {
    /// Starts `ufo50.exe` in `game_dir` (through Wine off Windows) and waits for its first frame.
    /// Each training step then runs `frame_skip` game frames; `reset_game` launches `game_id`.
    pub fn launch(game_dir: &Path, frame_skip: u32, game_id: u8) -> io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();

        let mut cmd = if cfg!(windows) {
            Command::new(game_dir.join("ufo50.exe"))
        } else {
            let mut c = Command::new("wine");
            c.arg("ufo50.exe").env("WINEDEBUG", "-all");
            // Skip Wine's Mono/Gecko install prompts, which would block a headless run forever.
            let mut overrides = std::env::var("WINEDLLOVERRIDES").unwrap_or_default();
            overrides.push_str(";mscoree,mshtml=");
            if game_dir.join("d3d11.dll").exists() {
                overrides.push_str(";d3d11,d3d10core=n,b");
                // Copies share the game dir; concurrent writers of DXVK's state cache can hang startup.
                c.env("DXVK_STATE_CACHE", "0");
            }
            c.env("WINEDLLOVERRIDES", overrides.trim_start_matches(';'));
            c
        };
        let log = std::fs::File::create(std::env::temp_dir().join(format!("ufo50rl_{port}.log")))?;
        let mut child = cmd
            .current_dir(game_dir)
            .env("UFO50RL_PORT", port.to_string())
            .stdin(Stdio::null())
            .stdout(log.try_clone()?)
            .stderr(log)
            .spawn()?;

        listener.set_nonblocking(true)?;
        let deadline = Instant::now() + CONNECT_TIMEOUT;
        let stream = loop {
            match listener.accept() {
                Ok((s, _)) => break s,
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    if let Some(status) = child.try_wait()? {
                        return Err(io::Error::other(format!(
                            "game exited before connecting: {status}"
                        )));
                    }
                    if Instant::now() > deadline {
                        let _ = child.kill();
                        return Err(io::Error::new(
                            io::ErrorKind::TimedOut,
                            "game never connected",
                        ));
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(e) => return Err(e),
            }
        };
        stream.set_nonblocking(false)?;
        stream.set_nodelay(true)?;

        let mut game = Self {
            child,
            stream,
            msg: vec![0; MSG_LEN],
            obs: vec![0; OBS_W * OBS_H * 4],
            frame_skip: frame_skip.max(1),
            game_id,
            pending_flags: 0,
            awaiting_reply: false,
        };
        game.recv()?;
        Ok(game)
    }

    /// Launches `n` copies concurrently (each takes several seconds to boot), staggered and
    /// with retries so one slow start doesn't fail the batch.
    pub fn launch_many(
        game_dir: &Path,
        n: usize,
        frame_skip: u32,
        game_id: u8,
    ) -> io::Result<Vec<Self>> {
        const ATTEMPTS: usize = 3;
        let handles: Vec<_> = (0..n)
            .map(|i| {
                let dir: PathBuf = game_dir.to_path_buf();
                std::thread::spawn(move || {
                    std::thread::sleep(Duration::from_secs(i as u64));
                    let mut result = Self::launch(&dir, frame_skip, game_id);
                    for attempt in 2..=ATTEMPTS {
                        let Err(e) = &result else { break };
                        eprintln!(
                            "[lockstep] env {i} launch failed ({e}), attempt {attempt}/{ATTEMPTS}"
                        );
                        result = Self::launch(&dir, frame_skip, game_id);
                    }
                    result
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err(io::Error::other("launch thread panicked")))
            })
            .collect()
    }

    /// Holds `keys` (VK codes, at most 8) for `frames` frames, then returns once the next frame arrives.
    pub fn step(&mut self, keys: &[u8], frames: u32, flags: u32) -> io::Result<()> {
        self.send(keys, frames, flags)?;
        self.recv()
    }

    /// OBS_W x OBS_H BGRA: the game's native frame averaged over SCALE x SCALE blocks, so
    /// nothing smaller than the scale factor is skipped.
    pub fn obs(&self) -> &[u8] {
        &self.obs
    }

    fn downscale(&mut self) {
        let raw = &self.msg[STATE_LEN * 8..];
        for oy in 0..OBS_H {
            for ox in 0..OBS_W {
                let mut sum = [0u32; 4];
                for y in oy * SCALE..(oy + 1) * SCALE {
                    let row = (y * RAW_W + ox * SCALE) * 4;
                    for px in raw[row..row + SCALE * 4].chunks_exact(4) {
                        for (s, &v) in sum.iter_mut().zip(px) {
                            *s += v as u32;
                        }
                    }
                }
                let dst = (oy * OBS_W + ox) * 4;
                for (d, s) in self.obs[dst..dst + 4].iter_mut().zip(sum) {
                    *d = (s / (SCALE * SCALE) as u32) as u8;
                }
            }
        }
    }

    pub fn state(&self) -> GameState {
        let f = |i: usize| f64::from_le_bytes(self.msg[i * 8..i * 8 + 8].try_into().unwrap());
        GameState {
            frame: f(0) as u64,
            room: f(1) as i32,
            game_id: f(2) as i32,
            paused: f(3) != 0.0,
            points: f(4),
            lives: f(5),
            mas_state: f(6) as i32,
            zone: f(7) as i32,
            menu_sel: f(8) as i32,
            num_players: f(9) as i32,
            demo: f(10) != 0.0,
            view_x: f(11),
            kills: f(12),
            boss_life: f(13),
        }
    }

    fn send(&mut self, keys: &[u8], frames: u32, flags: u32) -> io::Result<()> {
        let mut reply = [0u8; 16];
        for (dst, &k) in reply[..8].iter_mut().zip(keys) {
            *dst = k;
        }
        reply[8..12].copy_from_slice(&frames.max(1).to_le_bytes());
        reply[12..16].copy_from_slice(&flags.to_le_bytes());
        self.awaiting_reply = false;
        self.stream.write_all(&reply)
    }

    fn recv(&mut self) -> io::Result<()> {
        let mut len = [0u8; 4];
        self.stream.read_exact(&mut len)?;
        let len = u32::from_le_bytes(len) as usize;
        if len != MSG_LEN {
            return Err(io::Error::other(format!(
                "expected {MSG_LEN}-byte frame, got {len}; rebuild with bridge/build.sh"
            )));
        }
        self.stream.read_exact(&mut self.msg)?;
        self.downscale();
        self.awaiting_reply = true;
        Ok(())
    }
}

impl GameRunner for LockstepGame {
    fn next_frame_timeout(&mut self, timeout: Duration) -> Result<Frame, String> {
        // The game blocks until answered, so an unanswered frame would deadlock this read.
        if self.awaiting_reply {
            self.execute_action(0);
        }
        self.stream
            .set_read_timeout(Some(timeout))
            .map_err(|e| e.to_string())?;
        self.recv().map_err(|e| e.to_string())?;
        Ok(Frame {
            pixels: self.obs().to_vec(),
            state: self.state(),
        })
    }

    fn execute_action(&mut self, action: usize) {
        if !self.awaiting_reply {
            return;
        }
        let flags = std::mem::take(&mut self.pending_flags);
        let frames = if flags == 0 { self.frame_skip } else { 1 };
        // A failed write surfaces as an error on the next read.
        let _ = self.send(ACTION_MAP[action.min(ACTION_MAP.len() - 1)], frames, flags);
    }

    fn reset_game(&mut self) {
        self.pending_flags = launch_flag(self.game_id);
    }

    fn obs_width(&self) -> u32 {
        OBS_W as u32
    }

    fn obs_height(&self) -> u32 {
        OBS_H as u32
    }
}

impl Drop for LockstepGame {
    fn drop(&mut self) {
        if !self.awaiting_reply {
            let _ = self.stream.set_read_timeout(Some(Duration::from_secs(1)));
            let _ = self.recv();
        }
        if self.awaiting_reply {
            let _ = self.send(&[], 1, FLAG_QUIT);
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if let Ok(Some(_)) = self.child.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = self.child.kill();
    }
}
