use super::NINPEK_SCORE;
use super::game_over;
use super::lives;
use super::rewards;
use super::score;

pub enum RewardEvent {
    Survival,
    ScoreUp,
    LifeGained,
    LifeLost,
    StageComplete,
    GameComplete,
    GameOver,
}

pub struct FrameResult {
    pub reward: f64,
    pub event: RewardEvent,
    pub lives: u32,
    pub done: bool,
}

pub struct NinpekTracker {
    prev_pixels: Vec<u8>,
    prev_valid: bool,
    committed_lives: Option<u32>,
    pending_lives: Option<(u32, Vec<u8>)>,
    committed_score: Option<Vec<u8>>,
    pending_score: Option<Vec<u8>>,
    game_over_pending: bool,
    stage_complete_pending: bool,
    stage_complete_rewarded: bool,
    game_complete_pending: bool,
    life_pixels: Vec<u8>,
    score_pixels: Vec<u8>,
    width: u32,
}

impl NinpekTracker {
    pub fn new(width: u32) -> Self {
        Self {
            prev_pixels: Vec::new(),
            prev_valid: false,
            committed_lives: None,
            pending_lives: None,
            committed_score: None,
            pending_score: None,
            game_over_pending: false,
            stage_complete_pending: false,
            stage_complete_rewarded: false,
            game_complete_pending: false,
            life_pixels: Vec::with_capacity(
                lives::LIFE_REGION.w as usize * lives::LIFE_REGION.h as usize * 4,
            ),
            score_pixels: Vec::with_capacity(
                NINPEK_SCORE.region.w as usize * NINPEK_SCORE.region.h as usize * 3,
            ),
            width,
        }
    }

    pub fn process_frame(&mut self, pixels: &[u8]) -> FrameResult {
        let w = self.width;

        // Game over: leaderboard (check first — cheap early exit on most frames)
        if game_over::is_leaderboard(pixels, w) {
            if self.game_over_pending {
                return FrameResult {
                    reward: rewards::GAME_OVER,
                    event: RewardEvent::GameOver,
                    lives: 0,
                    done: true,
                };
            }
            self.game_over_pending = true;
        } else {
            self.game_over_pending = false;
        }

        // Stage/game complete (shares is_completion_screen, only check once)
        let completion = game_over::is_stage_complete(pixels, w);
        let game_win = game_over::is_game_complete(pixels, w);

        if game_win {
            if self.game_complete_pending {
                return FrameResult {
                    reward: rewards::STAGE_COMPLETE,
                    event: RewardEvent::GameComplete,
                    lives: self.committed_lives.unwrap_or(0),
                    done: true,
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
                    event: RewardEvent::StageComplete,
                    lives: self.committed_lives.unwrap_or(0),
                    done: false,
                };
            }
            self.stage_complete_pending = true;
        } else {
            self.stage_complete_pending = false;
            self.stage_complete_rewarded = false;
        }

        // Life tracking
        self.life_pixels.clear();
        let lr = &lives::LIFE_REGION;
        for y in lr.y as usize..(lr.y + lr.h) as usize {
            let start = (y * w as usize + lr.x as usize) * 4;
            let end = start + lr.w as usize * 4;
            self.life_pixels.extend_from_slice(&pixels[start..end]);
        }

        let mut life_reward = 0.0;
        let curr_lives = if let Some(committed) = self.committed_lives {
            let change = lives::check_life_change(pixels, w, committed);
            let detected = (committed as i32 + change) as u32;

            if change != 0 {
                match &self.pending_lives {
                    Some((pend_count, pend_pixels))
                        if *pend_count == detected && *pend_pixels == self.life_pixels =>
                    {
                        if change == 1 {
                            life_reward = rewards::LIFE_GAINED;
                        } else if change == -1 {
                            life_reward = rewards::LIFE_LOST;
                        }
                        self.committed_lives = Some(detected);
                        self.pending_lives = None;
                    }
                    _ => {
                        self.pending_lives = Some((detected, self.life_pixels.clone()));
                    }
                }
            } else {
                self.pending_lives = None;
            }
            committed
        } else {
            let l = lives::count_lives(pixels, w);
            self.committed_lives = Some(l);
            l
        };

        if life_reward != 0.0 {
            return FrameResult {
                reward: life_reward,
                event: if life_reward > 0.0 {
                    RewardEvent::LifeGained
                } else {
                    RewardEvent::LifeLost
                },
                lives: curr_lives,
                done: false,
            };
        }

        // Score tracking with 2-frame stability
        let curr_clean = score::is_score_clean(pixels, w, &NINPEK_SCORE);

        // prev_valid means prev was clean when stored — no need to re-check
        let both_clean = curr_clean && self.prev_valid;

        let mut score_reward = 0.0;
        if curr_clean {
            // Only extract score pixels when current frame is clean
            self.score_pixels.clear();
            for y in NINPEK_SCORE.region.y..NINPEK_SCORE.region.y + NINPEK_SCORE.region.h {
                for x in NINPEK_SCORE.region.x..NINPEK_SCORE.region.x + NINPEK_SCORE.region.w {
                    let i = (y * w + x) as usize * 4;
                    self.score_pixels.push(score::quantize(pixels[i]));
                    self.score_pixels.push(score::quantize(pixels[i + 1]));
                    self.score_pixels.push(score::quantize(pixels[i + 2]));
                }
            }

            if both_clean {
                match &self.pending_score {
                    Some(pend) if *pend == self.score_pixels => {
                        if let Some(committed) = &self.committed_score {
                            if *committed != self.score_pixels {
                                score_reward = rewards::SCORE_UP;
                            }
                        }
                        self.committed_score = Some(self.score_pixels.clone());
                        self.pending_score = None;
                    }
                    _ => {
                        self.pending_score = Some(self.score_pixels.clone());
                    }
                }
            }

            // Reuse prev buffer instead of allocating
            self.prev_pixels.clear();
            self.prev_pixels.extend_from_slice(pixels);
            self.prev_valid = true;
        } else {
            self.pending_score = None;
            self.prev_valid = false;
        }

        if score_reward != 0.0 {
            return FrameResult {
                reward: score_reward,
                event: RewardEvent::ScoreUp,
                lives: curr_lives,
                done: false,
            };
        }

        FrameResult {
            reward: rewards::SURVIVAL,
            event: RewardEvent::Survival,
            lives: curr_lives,
            done: false,
        }
    }
}
