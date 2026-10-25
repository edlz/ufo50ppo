use super::super::{FrameResult, GameTracker};
use super::events;
use super::rewards;
use crate::platform::{Frame, GameState, NUM_ACTIONS};

// o34_Mas.state values (gml_Object_o34_Mas_Step_0).
const PLAYING: i32 = 20;
const GAME_OVER: i32 = 50;
const WIN: i32 = 80; // 80..=84: boss dead, ending sequence
const WIN_SCREEN: i32 = 84; // waits for fire2; on the first loop that starts the second
const NINPEK_ID: i32 = 34;
/// Right edge of the camera at the end of the room; the second loop is laid after the first.
const LOOP_W: f64 = 14976.0 + 384.0;
/// Steps the win sequence and loop restart take is ~150; past this, give up on the run.
const MAX_LOOPING_STEPS: u32 = 1000;

const ACTION_UP: usize = 1;
const ACTION_A: usize = 9;

pub struct NinpekTracker {
    prev: Option<GameState>,
    left_old_run: bool,
    idle_ticks: u32,
    /// Between the first boss kill and the second loop starting.
    looping: bool,
    looping_steps: u32,
    /// Ninja's position in the level last step, None while dead or not playing.
    last_reach: Option<f64>,
    ep_scores: u32,
    ep_kills: u32,
    ep_boss_damage: u32,
    ep_life_gained: u32,
    ep_life_lost: u32,
    ep_stages: u32,
    ep_progress: f64,
    ep_start_x: f64,
    ep_max_x: f64,
    ep_max_reach: f64,
    ep_max_zone: i32,
    ep_loop: i32,
}

impl NinpekTracker {
    pub fn new() -> Self {
        Self {
            prev: None,
            left_old_run: false,
            idle_ticks: 0,
            looping: false,
            looping_steps: 0,
            last_reach: None,
            ep_scores: 0,
            ep_kills: 0,
            ep_boss_damage: 0,
            ep_life_gained: 0,
            ep_life_lost: 0,
            ep_stages: 0,
            ep_progress: 0.0,
            ep_start_x: 0.0,
            ep_max_x: -1.0,
            ep_max_reach: -1.0,
            ep_max_zone: 0,
            ep_loop: 0,
        }
    }

    fn process_frame_inner(&mut self, s: GameState) -> FrameResult {
        let prev = self.prev.replace(s);
        let last_reach = self.last_reach.take();
        let lives = s.lives.max(0.0) as u32;
        let result = |reward, event_name, done, is_menu| FrameResult {
            reward,
            done,
            lives,
            event_name,
            is_event: !event_name.is_empty(),
            is_menu,
        };

        if s.mas_state >= WIN && s.game_loop == 0 && !self.looping {
            self.looping = true;
            return result(rewards::WIN, events::WIN, false, true);
        }
        // o34_Mas restarts the room, so its run state is briefly missing.
        if self.looping {
            if s.mas_state != PLAYING || s.game_loop == 0 {
                self.looping_steps += 1;
                let stuck = self.looping_steps > MAX_LOOPING_STEPS;
                return result(0.0, if stuck { events::EXITED } else { "" }, stuck, true);
            }
            self.looping = false;
            self.ep_loop = s.game_loop;
        }
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
        // Several kills can land in one step, so they are counted here rather than per event.
        let kills = if p.kills >= 0.0 {
            (s.kills - p.kills).max(0.0)
        } else {
            0.0
        };
        let boss_damage = if p.boss_life >= 0.0 && s.boss_life >= 0.0 {
            (p.boss_life - s.boss_life).max(0.0)
        } else {
            0.0
        };

        let mut reward = 0.0;
        let mut event = "";
        if score_delta > 0.0 {
            reward += rewards::SCORE_UP * score_delta / rewards::POINTS_PER_PICKUP;
            event = events::SCORE;
        }
        if kills > 0.0 {
            reward += rewards::KILL * kills;
            event = events::KILL;
            self.ep_kills += kills as u32;
        }
        if boss_damage > 0.0 {
            reward += rewards::BOSS_HIT * boss_damage;
            event = events::BOSS_HIT;
            self.ep_boss_damage += boss_damage as u32;
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
        let offset = s.game_loop as f64 * LOOP_W;
        self.ep_max_x = self.ep_max_x.max(offset + s.view_x);
        // Only steady movement pays: respawning and the loop restart jump the ninja ahead.
        let mut progress = 0.0;
        if s.player_x >= 0.0 {
            let reach = offset + s.view_x + s.player_x;
            if last_reach.is_some() {
                progress = (reach - self.ep_max_reach).max(0.0) * rewards::PROGRESS;
            }
            self.ep_max_reach = self.ep_max_reach.max(reach);
            self.last_reach = Some(reach);
        }
        self.ep_progress += progress;
        result(reward + progress, event, false, false)
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
            self.ep_max_zone = self.ep_max_zone.max(frame.state.zone);
        }
        match result.event_name {
            events::SCORE => self.ep_scores += 1,
            events::LIFE_GAINED => self.ep_life_gained += 1,
            events::LIFE_LOST | events::GAME_OVER => self.ep_life_lost += 1,
            events::STAGE => self.ep_stages += 1,
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
            self.ep_start_x = s.view_x;
            self.ep_max_x = s.view_x;
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
            "  furthest x: {:.0} (section {}) | loop: {} | start x: {:.0} | scores: {} | kills: {} | boss dmg: {} | stages: {} | life+: {} | life-: {} | progress: {:.1}",
            self.ep_max_x.max(0.0),
            self.ep_max_zone,
            self.ep_loop,
            self.ep_start_x,
            self.ep_scores,
            self.ep_kills,
            self.ep_boss_damage,
            self.ep_stages,
            self.ep_life_gained,
            self.ep_life_lost,
            self.ep_progress,
        )
    }

    fn episode_metrics(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("furthest_x", self.ep_max_x.max(0.0)),
            ("start_x", self.ep_start_x),
            ("distance", (self.ep_max_x - self.ep_start_x).max(0.0)),
            ("furthest_section", self.ep_max_zone as f64),
            ("loop", self.ep_loop as f64),
            ("kills", self.ep_kills as f64),
            ("boss_damage", self.ep_boss_damage as f64),
        ]
    }

    fn menu_action(&mut self) -> usize {
        let win_screen = self.looping && self.prev.is_some_and(|p| p.mas_state == WIN_SCREEN);
        self.idle_ticks += 1;
        // fire2 is a press, so it must be released in between.
        if win_screen && self.idle_ticks.is_multiple_of(2) {
            ACTION_A
        } else {
            0
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn playing(view_x: f64, player_x: f64) -> GameState {
        GameState {
            game_id: NINPEK_ID,
            mas_state: PLAYING,
            num_players: 1,
            lives: 3.0,
            view_x,
            player_x,
            player_y: 100.0,
            boss_life: -1.0,
            ..Default::default()
        }
    }

    fn step(t: &mut NinpekTracker, s: GameState) -> FrameResult {
        t.process_frame(&Frame {
            pixels: Vec::new(),
            state: s,
        })
    }

    #[test]
    fn progress_follows_the_ninja_not_the_camera() {
        let mut t = NinpekTracker::new();
        step(&mut t, playing(0.0, 50.0));
        step(&mut t, playing(2.0, 50.0));
        // Camping: the camera carries the ninja 2 px a step.
        let camp = step(&mut t, playing(4.0, 50.0)).reward;
        // Running: 2 px of scroll plus 6 px of its own.
        let run = step(&mut t, playing(6.0, 56.0)).reward;
        assert!((camp - (rewards::SURVIVAL + 2.0 * rewards::PROGRESS)).abs() < 1e-9);
        assert!((run - (rewards::SURVIVAL + 8.0 * rewards::PROGRESS)).abs() < 1e-9);
        // Walking back and forth again doesn't pay twice.
        step(&mut t, playing(8.0, 40.0));
        let back = step(&mut t, playing(10.0, 52.0)).reward;
        assert!((back - rewards::SURVIVAL).abs() < 1e-9);
    }

    #[test]
    fn respawning_ahead_does_not_pay() {
        let mut t = NinpekTracker::new();
        step(&mut t, playing(0.0, 10.0));
        step(&mut t, playing(2.0, 10.0));
        let mut dead = playing(4.0, -1.0);
        dead.lives = 2.0;
        assert_eq!(step(&mut t, dead).event_name, events::LIFE_LOST);
        let mut respawn = playing(40.0, 192.0);
        respawn.lives = 2.0;
        assert!((step(&mut t, respawn).reward - rewards::SURVIVAL).abs() < 1e-9);
        respawn.view_x = 42.0;
        let r = step(&mut t, respawn).reward;
        assert!((r - (rewards::SURVIVAL + 2.0 * rewards::PROGRESS)).abs() < 1e-9);
    }

    #[test]
    fn first_win_continues_into_the_second_loop() {
        let mut t = NinpekTracker::new();
        step(&mut t, playing(14976.0, 100.0));
        let mut win = playing(14976.0, 100.0);
        win.mas_state = WIN;
        let r = step(&mut t, win);
        assert!(!r.done && r.is_menu && r.event_name == events::WIN);
        assert_eq!(r.reward, rewards::WIN);

        // Win screen: fire2 is tapped, not held.
        win.mas_state = WIN_SCREEN;
        assert!(step(&mut t, win).is_menu);
        let taps: Vec<usize> = (0..4).map(|_| t.menu_action()).collect();
        assert!(taps.contains(&ACTION_A) && taps.contains(&0));

        // The room restarts with no run state, which must not read as a game over.
        let mut restart = playing(0.0, -1.0);
        restart.mas_state = 70;
        restart.lives = -99.0;
        restart.game_loop = 1;
        let r = step(&mut t, restart);
        assert!(!r.done && r.is_menu);

        // The jump from the end of the first loop to the start of the second doesn't pay either.
        let mut loop2 = playing(0.0, 100.0);
        loop2.game_loop = 1;
        step(&mut t, loop2);
        loop2.view_x = 2.0;
        assert!((step(&mut t, loop2).reward - rewards::SURVIVAL).abs() < 1e-9);
        loop2.view_x = 4.0;
        let r = step(&mut t, loop2);
        assert!(!r.done);
        assert!((r.reward - (rewards::SURVIVAL + 2.0 * rewards::PROGRESS)).abs() < 1e-9);
        assert_eq!(t.ep_loop, 1);

        loop2.mas_state = WIN;
        let r = step(&mut t, loop2);
        assert!(r.done && r.event_name == events::WIN);
    }
}
