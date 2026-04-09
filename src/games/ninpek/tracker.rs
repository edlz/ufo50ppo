use super::super::{FrameResult, GameTracker};
use super::events;
use super::game_over;
use super::mem::MemReader;
use super::rewards;
use crate::platform::NUM_ACTIONS;
use crate::platform::win32::input::{VK_DOWN, VK_ESCAPE, VK_Z, vk_noop};

fn update_field(field: &mut Option<u64>, new: Option<f64>) {
    if let Some(v) = new {
        *field = Some(v as u64);
    }
}

pub struct NinpekTracker {
    mem_reader: MemReader,
    last_mem_score: Option<u64>,
    last_mem_lives: Option<u64>,
    game_over_pending: bool,
    stage_complete_pending: bool,
    stage_complete_rewarded: bool,
    game_complete_pending: bool,
    width: u32,
    ep_scores: u32,
    ep_life_gained: u32,
    ep_life_lost: u32,
    ep_survival: f64,
}

impl NinpekTracker {
    pub fn mem_score(&self) -> Option<u64> {
        self.last_mem_score
    }

    pub fn mem_lives(&self) -> Option<u64> {
        self.last_mem_lives
    }

    pub fn new(width: u32, pid: u32) -> Self {
        let mem_reader = if pid != 0 {
            MemReader::for_pid(pid)
                .unwrap_or_else(|e| panic!("MemReader::for_pid({}) failed: {}", pid, e))
        } else {
            MemReader::new(super::WINDOW_TITLE)
                .unwrap_or_else(|e| panic!("MemReader::new failed: {}", e))
        };
        println!("[mem] attached to ufo50.exe (pid={})", pid);
        Self {
            mem_reader,
            last_mem_score: None,
            last_mem_lives: None,
            game_over_pending: false,
            stage_complete_pending: false,
            stage_complete_rewarded: false,
            game_complete_pending: false,
            width,
            ep_scores: 0,
            ep_life_gained: 0,
            ep_life_lost: 0,
            ep_survival: 0.0,
        }
    }

    fn refresh_mem_state(&mut self) -> (Option<u64>, Option<u64>) {
        let prev = (self.last_mem_score, self.last_mem_lives);
        let (score, lives) = self.mem_reader.read_both();
        update_field(&mut self.last_mem_score, score);
        update_field(&mut self.last_mem_lives, lives);
        prev
    }

    fn process_frame_inner(&mut self, pixels: &[u8]) -> FrameResult {
        let w = self.width;
        let (prev_score, prev_lives) = self.refresh_mem_state();

        let leaderboard = game_over::is_leaderboard(pixels, w);
        let completion_kind = if leaderboard {
            None
        } else {
            game_over::classify_completion(pixels, w)
        };
        let completion = completion_kind == Some(game_over::CompletionKind::Stage);
        let game_win = completion_kind == Some(game_over::CompletionKind::Game);
        let is_menu = leaderboard || completion || game_win;

        let score_delta = match (prev_score, self.last_mem_score) {
            (Some(p), Some(c)) => c as i64 - p as i64,
            _ => 0,
        };
        let lives_delta = match (prev_lives, self.last_mem_lives) {
            (Some(p), Some(c)) => c as i64 - p as i64,
            _ => 0,
        };

        let curr_lives = self.last_mem_lives.unwrap_or(0) as u32;

        if leaderboard {
            if self.game_over_pending {
                return FrameResult {
                    reward: rewards::GAME_OVER,
                    event_name: events::GAME_OVER,
                    lives: 0,
                    done: true,
                    is_event: true,
                    is_menu,
                };
            }
            self.game_over_pending = true;
        } else {
            self.game_over_pending = false;
        }

        if game_win {
            if self.game_complete_pending {
                return FrameResult {
                    reward: rewards::STAGE_COMPLETE,
                    event_name: events::WIN,
                    lives: curr_lives,
                    done: true,
                    is_event: true,
                    is_menu,
                };
            }
            self.game_complete_pending = true;
        } else {
            self.game_complete_pending = false;
        }

        if completion {
            if self.stage_complete_pending && !self.stage_complete_rewarded {
                self.stage_complete_rewarded = true;
                return FrameResult {
                    reward: rewards::STAGE_COMPLETE,
                    event_name: events::STAGE,
                    lives: curr_lives,
                    done: false,
                    is_event: true,
                    is_menu,
                };
            }
            self.stage_complete_pending = true;
        } else {
            self.stage_complete_pending = false;
            self.stage_complete_rewarded = false;
        }

        if lives_delta == -1 {
            return FrameResult {
                reward: rewards::LIFE_LOST,
                event_name: events::LIFE_LOST,
                lives: curr_lives,
                done: false,
                is_event: true,
                is_menu,
            };
        }
        if lives_delta == 1 {
            return FrameResult {
                reward: rewards::LIFE_GAINED,
                event_name: events::LIFE_GAINED,
                lives: curr_lives,
                done: false,
                is_event: true,
                is_menu,
            };
        }
        if score_delta > 0 {
            return FrameResult {
                reward: rewards::SCORE_UP * (score_delta as f64) / rewards::POINTS_PER_PICKUP,
                event_name: events::SCORE,
                lives: curr_lives,
                done: false,
                is_event: true,
                is_menu,
            };
        }

        FrameResult {
            reward: rewards::SURVIVAL,
            event_name: "",
            lives: curr_lives,
            done: false,
            is_event: false,
            is_menu,
        }
    }
}

impl GameTracker for NinpekTracker {
    fn process_frame(&mut self, pixels: &[u8]) -> FrameResult {
        let result = self.process_frame_inner(pixels);
        match result.event_name {
            events::SCORE => self.ep_scores += 1,
            events::LIFE_GAINED => self.ep_life_gained += 1,
            events::LIFE_LOST => self.ep_life_lost += 1,
            "" => self.ep_survival += result.reward,
            _ => {}
        }
        result
    }

    fn episode_breakdown(&self) -> String {
        format!(
            "  scores: {} | life+: {} | life-: {} | survival: {:.1}",
            self.ep_scores, self.ep_life_gained, self.ep_life_lost, self.ep_survival,
        )
    }

    fn observe_idle(&mut self, _pixels: &[u8]) -> bool {
        self.refresh_mem_state();
        self.last_mem_score.is_some() && self.last_mem_lives.is_some()
    }

    fn reset_episode(&mut self) {
        self.last_mem_score = None;
        self.last_mem_lives = None;
        self.game_over_pending = false;
        self.stage_complete_pending = false;
        self.stage_complete_rewarded = false;
        self.game_complete_pending = false;
        self.ep_scores = 0;
        self.ep_life_gained = 0;
        self.ep_life_lost = 0;
        self.ep_survival = 0.0;
    }

    fn reset_sequence(&self) -> &[usize] {
        // Wait values are empirically tuned: shorter durations race the menu transitions.
        const SEQ: &[usize] = &[
            VK_ESCAPE,
            VK_DOWN,
            VK_Z,
            VK_Z,
            vk_noop(800),
            VK_Z,
            vk_noop(700),
            VK_Z,
        ];
        SEQ
    }

    fn game_name(&self) -> &str {
        "ninpek"
    }

    fn obs_width(&self) -> u32 {
        self.width
    }

    fn obs_height(&self) -> u32 {
        // game_over.rs assumes height == width.
        self.width
    }

    fn num_actions(&self) -> usize {
        NUM_ACTIONS
    }
}
