use tensorboard_rs::summary_writer::SummaryWriter;

pub struct TbLogger {
    writer: SummaryWriter,
    step_offset: usize,
}

impl TbLogger {
    pub fn new(base_dir: &str, step_offset: u64) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let secs_per_day = 86400u64;
        let days = now / secs_per_day;
        let time_of_day = now % secs_per_day;
        let hours = time_of_day / 3600;
        let minutes = (time_of_day % 3600) / 60;
        let seconds = time_of_day % 60;
        let (year, month, day) = days_to_date(days);
        let run_name = format!(
            "{:04}{:02}{:02}_{:02}{:02}{:02}",
            year, month, day, hours, minutes, seconds
        );
        let log_dir = format!("{}/{}", base_dir, run_name);
        println!("TensorBoard: {}", log_dir);
        Self {
            writer: SummaryWriter::new(&log_dir),
            step_offset: step_offset as usize,
        }
    }

    fn step(&self, step: usize) -> usize {
        step + self.step_offset
    }

    pub fn log_episode(&mut self, step: usize, reward: f64, length: u32) {
        let s = self.step(step);
        self.writer
            .add_scalar("rollout/ep_rew_mean", reward as f32, s);
        self.writer
            .add_scalar("rollout/ep_len_mean", length as f32, s);
    }

    pub fn log_episode_for_env(&mut self, step: usize, env_id: usize, reward: f64, length: u32) {
        let s = self.step(step);
        self.writer.add_scalar(
            &format!("rollout/env_{}/ep_rew_mean", env_id),
            reward as f32,
            s,
        );
        self.writer.add_scalar(
            &format!("rollout/env_{}/ep_len_mean", env_id),
            length as f32,
            s,
        );
    }

    pub fn log_episode_spread(&mut self, step: usize, min: f64, max: f64) {
        let s = self.step(step);
        self.writer.add_scalar("rollout/ep_rew_min", min as f32, s);
        self.writer.add_scalar("rollout/ep_rew_max", max as f32, s);
    }

    pub fn log_update(
        &mut self,
        step: usize,
        policy_loss: f64,
        value_loss: f64,
        entropy: f64,
        total_loss: f64,
        grad_norm: f64,
    ) {
        let s = self.step(step);
        self.writer
            .add_scalar("train/policy_loss", policy_loss as f32, s);
        self.writer
            .add_scalar("train/value_loss", value_loss as f32, s);
        self.writer.add_scalar("train/entropy", entropy as f32, s);
        self.writer
            .add_scalar("train/total_loss", total_loss as f32, s);
        self.writer
            .add_scalar("train/grad_norm", grad_norm as f32, s);
    }

    pub fn log_explained_variance(&mut self, step: usize, ev: f64) {
        self.writer
            .add_scalar("train/explained_variance", ev as f32, self.step(step));
    }

    pub fn log_learning_rate(&mut self, step: usize, lr: f64) {
        self.writer
            .add_scalar("train/learning_rate", lr as f32, self.step(step));
    }

    pub fn log_fps(&mut self, step: usize, fps: f64) {
        self.writer
            .add_scalar("time/fps", fps as f32, self.step(step));
    }

    pub fn log_scalar(&mut self, tag: &str, value: f64, step: usize) {
        self.writer.add_scalar(tag, value as f32, self.step(step));
    }
}

fn days_to_date(days_since_epoch: u64) -> (u64, u64, u64) {
    // Simplified — good enough for folder names
    let mut y = 1970;
    let mut remaining = days_since_epoch as i64;
    loop {
        let days_in_year = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
            366
        } else {
            365
        };
        if remaining < days_in_year {
            break;
        }
        remaining -= days_in_year;
        y += 1;
    }
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let days_in_months: [i64; 12] = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut m = 0;
    for (i, &dim) in days_in_months.iter().enumerate() {
        if remaining < dim {
            m = i;
            break;
        }
        remaining -= dim;
    }
    (y, (m + 1) as u64, (remaining + 1) as u64)
}
