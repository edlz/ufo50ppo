use super::super::{FrameResult, GameTracker};
use super::events;
use super::rewards;
use crate::platform::{Frame, GameState, NUM_ACTIONS};

// o34_Mas.state values (gml_Object_o34_Mas_Step_0).
const PLAYING: i32 = 20;
const GAME_OVER: i32 = 50;
const WIN: i32 = 80; // 80..=84: boss dead, ending sequence
const NINPEK_ID: i32 = 34;

const ACTION_UP: usize = 1;
const ACTION_A: usize = 9;

pub struct NinpekTracker {
    prev: Option<GameState>,
    left_old_run: bool,
    idle_ticks: u32,
    ep_scores: u32,
    ep_life_gained: u32,
    ep_life_lost: u32,
    ep_stages: u32,
    ep_survival: f64,
    ep_max_x: f64,
    ep_max_zone: i32,
}

impl NinpekTracker {
    pub fn new() -> Self {
        Self {
            prev: None,
            left_old_run: false,
            idle_ticks: 0,
            ep_scores: 0,
            ep_life_gained: 0,
            ep_life_lost: 0,
            ep_stages: 0,
            ep_survival: 0.0,
            ep_max_x: 0.0,
            ep_max_zone: 0,
        }
    }

    fn process_frame_inner(&mut self, s: GameState) -> FrameResult {
        let prev = self.prev.replace(s);
        let lives = s.lives.max(0.0) as u32;
        let result = |reward, event_name, done, is_menu| FrameResult {
            reward,
            done,
            lives,
            event_name,
            is_event: !event_name.is_empty(),
            is_menu,
        };

        // The final death sets life[0] = -1 and state = 50 in the same frame.
        if s.mas_state == GAME_OVER || s.lives < 0.0 {
            return result(
                rewards::LIFE_LOST + rewards::GAME_OVER,
                events::GAME_OVER,
                true,
                false,
            );
        }
        if s.mas_state >= WIN {
            return result(rewards::WIN, events::WIN, true, false);
        }
        if s.game_id != NINPEK_ID || s.mas_state < 0 || s.demo || s.num_players != 1 {
            return result(0.0, events::EXITED, true, true);
        }
        if s.mas_state != PLAYING || s.paused {
            return result(0.0, "", false, true);
        }

        let Some(p) = prev.filter(|p| p.mas_state == PLAYING) else {
            return result(rewards::SURVIVAL, "", false, false);
        };
        let score_delta = s.points - p.points;
        let lives_delta = s.lives - p.lives;
        let zone_delta = s.zone - p.zone;

        let mut reward = 0.0;
        let mut event = "";
        if score_delta > 0.0 {
            reward += rewards::SCORE_UP * score_delta / rewards::POINTS_PER_PICKUP;
            event = events::SCORE;
        }
        if zone_delta > 0 {
            reward += rewards::STAGE_COMPLETE;
            event = events::STAGE;
        }
        if lives_delta > 0.0 {
            reward += rewards::LIFE_GAINED;
            event = events::LIFE_GAINED;
        }
        if lives_delta < 0.0 {
            reward += rewards::LIFE_LOST;
            event = events::LIFE_LOST;
        }
        if event.is_empty() {
            reward = rewards::SURVIVAL;
        }
        result(reward, event, false, false)
    }
}

impl Default for NinpekTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl GameTracker for NinpekTracker {
    fn process_frame(&mut self, frame: &Frame) -> FrameResult {
        let result = self.process_frame_inner(frame.state);
        if frame.state.mas_state == PLAYING {
            self.ep_max_x = self.ep_max_x.max(frame.state.view_x);
            self.ep_max_zone = self.ep_max_zone.max(frame.state.zone);
        }
        match result.event_name {
            events::SCORE => self.ep_scores += 1,
            events::LIFE_GAINED => self.ep_life_gained += 1,
            events::LIFE_LOST | events::GAME_OVER => self.ep_life_lost += 1,
            events::STAGE => self.ep_stages += 1,
            "" if !result.is_menu => self.ep_survival += result.reward,
            _ => {}
        }
        result
    }

    fn observe_idle(&mut self, frame: &Frame) -> Option<usize> {
        let s = frame.state;
        // Until the relaunch lands, the previous run may still report PLAYING.
        if s.game_id != NINPEK_ID || s.mas_state != PLAYING {
            self.left_old_run = true;
        }
        let fresh_run = s.mas_state == PLAYING && !s.paused && !s.demo && s.num_players == 1;
        if self.left_old_run && s.game_id == NINPEK_ID && fresh_run {
            self.prev = Some(s);
            return None;
        }
        // Tap through the title screen (with the cursor on 1 PLAYER) and skip the intro.
        self.idle_ticks += 1;
        let tap = if s.menu_sel > 0 { ACTION_UP } else { ACTION_A };
        Some(if self.idle_ticks.is_multiple_of(2) {
            tap
        } else {
            0
        })
    }

    fn episode_breakdown(&self) -> String {
        format!(
            "  furthest x: {:.0} (section {}) | scores: {} | stages: {} | life+: {} | life-: {} | survival: {:.1}",
            self.ep_max_x,
            self.ep_max_zone,
            self.ep_scores,
            self.ep_stages,
            self.ep_life_gained,
            self.ep_life_lost,
            self.ep_survival,
        )
    }

    fn episode_metrics(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("furthest_x", self.ep_max_x),
            ("furthest_section", self.ep_max_zone as f64),
        ]
    }

    fn reset_episode(&mut self) {
        *self = Self::new();
    }

    fn game_name(&self) -> &str {
        "ninpek"
    }

    fn num_actions(&self) -> usize {
        NUM_ACTIONS
    }
}
