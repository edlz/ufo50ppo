use tch::{Kind, Tensor};

use super::model::ActorCritic;

pub struct RolloutBuffer {
    pub obs: Vec<Tensor>,
    pub actions: Vec<i64>,
    pub log_probs: Vec<f64>,
    pub rewards: Vec<f64>,
    pub values: Vec<f64>,
    pub dones: Vec<bool>,
    capacity: usize,
}

impl RolloutBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            obs: Vec::with_capacity(capacity),
            actions: Vec::with_capacity(capacity),
            log_probs: Vec::with_capacity(capacity),
            rewards: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
            dones: Vec::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(
        &mut self,
        obs: Tensor,
        action: i64,
        log_prob: f64,
        reward: f64,
        value: f64,
        done: bool,
    ) {
        self.obs.push(obs);
        self.actions.push(action);
        self.log_probs.push(log_prob);
        self.rewards.push(reward);
        self.values.push(value);
        self.dones.push(done);
    }

    pub fn len(&self) -> usize {
        self.obs.len()
    }

    pub fn clear(&mut self) {
        self.obs.clear();
        self.actions.clear();
        self.log_probs.clear();
        self.rewards.clear();
        self.values.clear();
        self.dones.clear();
    }
}

/// Compute GAE (Generalized Advantage Estimation).
/// Returns (advantages, returns) as Vec<f64>.
pub fn compute_gae(
    rewards: &[f64],
    values: &[f64],
    dones: &[bool],
    last_value: f64,
    gamma: f64,
    lambda: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = rewards.len();
    let mut advantages = vec![0.0; n];
    let mut gae = 0.0;

    for t in (0..n).rev() {
        let next_value = if t == n - 1 {
            last_value
        } else {
            values[t + 1]
        };
        let next_non_terminal = if dones[t] { 0.0 } else { 1.0 };
        let delta = rewards[t] + gamma * next_value * next_non_terminal - values[t];
        gae = delta + gamma * lambda * next_non_terminal * gae;
        advantages[t] = gae;
    }

    let returns: Vec<f64> = advantages.iter().zip(values).map(|(a, v)| a + v).collect();
    (advantages, returns)
}

/// Fraction of the variance in `returns` explained by the value function predictions.
/// Returns 0.0 when there's not enough data or when returns are constant.
/// Range typically (-inf, 1.0]; closer to 1.0 means the value head is well-calibrated.
pub fn explained_variance(values: &[f64], returns: &[f64]) -> f64 {
    let n = values.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean_r = returns.iter().sum::<f64>() / n;
    let var_r = returns.iter().map(|r| (r - mean_r).powi(2)).sum::<f64>() / n;
    if var_r < 1e-8 {
        return 0.0;
    }
    let var_diff = values
        .iter()
        .zip(returns)
        .map(|(v, r)| (r - v).powi(2))
        .sum::<f64>()
        / n;
    1.0 - var_diff / var_r
}

pub struct PpoConfig {
    pub clip_epsilon: f64,
    pub value_coeff: f64,
    pub entropy_coeff: f64,
    pub num_epochs: usize,
    pub minibatch_size: usize,
    pub max_grad_norm: f64,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            clip_epsilon: 0.2,
            value_coeff: 0.5,
            entropy_coeff: 0.01,
            num_epochs: 4,
            minibatch_size: 32,
            max_grad_norm: 0.5,
        }
    }
}

pub struct UpdateStats {
    pub policy_loss: f64,
    pub value_loss: f64,
    pub entropy: f64,
    pub total_loss: f64,
}

/// Run one PPO update over the rollout buffer.
pub fn update(
    model: &mut ActorCritic,
    opt: &mut tch::nn::Optimizer,
    buffer: &RolloutBuffer,
    advantages: &[f64],
    returns: &[f64],
    cfg: &PpoConfig,
) -> UpdateStats {
    let clip_epsilon = cfg.clip_epsilon;
    let value_coeff = cfg.value_coeff;
    let entropy_coeff = cfg.entropy_coeff;
    let num_epochs = cfg.num_epochs;
    let minibatch_size = cfg.minibatch_size;
    let max_grad_norm = cfg.max_grad_norm;
    let device = model.vs.device();
    let n = buffer.len() as i64;

    let f32_tensor = |slice: &[f64]| {
        Tensor::from_slice(slice)
            .to_device(device)
            .to_kind(Kind::Float)
            .detach()
    };

    let obs_batch = Tensor::cat(&buffer.obs, 0);
    let actions_t = Tensor::from_slice(&buffer.actions).to_device(device);
    // Old log probs must not carry gradients — PPO compares new vs old, not optimizes old
    let old_log_probs_t = Tensor::from_slice(&buffer.log_probs)
        .to_device(device)
        .detach();
    let old_values_t = f32_tensor(&buffer.values);
    let advantages_t = f32_tensor(advantages);
    let returns_t = f32_tensor(returns);

    // Normalize advantages
    let adv_mean = advantages_t.mean(Kind::Float);
    let adv_std = advantages_t.std(true).clamp_min(1e-5);
    let advantages_t = (&advantages_t - &adv_mean) / &adv_std;

    let mut total_policy_loss = 0.0;
    let mut total_value_loss = 0.0;
    let mut total_entropy = 0.0;
    let mut total_loss_sum = 0.0;
    let mut num_batches = 0;

    for _epoch in 0..num_epochs {
        let perm = Tensor::randperm(n, (Kind::Int64, device));

        let mut start = 0i64;
        while start < n {
            let end = (start + minibatch_size as i64).min(n);
            let idx = perm.narrow(0, start, end - start);

            let mb_obs = obs_batch.index_select(0, &idx);
            let mb_actions = actions_t.index_select(0, &idx);
            let mb_old_log_probs = old_log_probs_t.index_select(0, &idx);
            let mb_old_values = old_values_t.index_select(0, &idx);
            let mb_advantages = advantages_t.index_select(0, &idx);
            let mb_returns = returns_t.index_select(0, &idx);

            let (new_log_probs, values) = model.forward(&mb_obs);
            let new_values = values.squeeze_dim(1);

            // Gather log probs for the taken actions
            let new_log_probs_a = new_log_probs
                .gather(1, &mb_actions.unsqueeze(1), false)
                .squeeze_dim(1);

            // Policy loss (clipped surrogate)
            let ratio = (&new_log_probs_a - &mb_old_log_probs).exp();
            let surr1 = &ratio * &mb_advantages;
            let surr2 = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * &mb_advantages;
            let policy_loss = -surr1.min_other(&surr2).mean(Kind::Float);

            // Clipped value loss: prevents large value-function moves between updates
            let v_clipped =
                &mb_old_values + (&new_values - &mb_old_values).clamp(-clip_epsilon, clip_epsilon);
            let unclipped = (&new_values - &mb_returns).pow_tensor_scalar(2);
            let clipped = (&v_clipped - &mb_returns).pow_tensor_scalar(2);
            let value_loss = unclipped.max_other(&clipped).mean(Kind::Float);

            // Entropy bonus
            let entropy = -(new_log_probs.exp() * &new_log_probs)
                .sum_dim_intlist(-1, false, Kind::Float)
                .mean(Kind::Float);

            let loss = &policy_loss + value_coeff * &value_loss - entropy_coeff * &entropy;

            total_policy_loss += f64::try_from(&policy_loss).unwrap_or(0.0);
            total_value_loss += f64::try_from(&value_loss).unwrap_or(0.0);
            total_entropy += f64::try_from(&entropy).unwrap_or(0.0);
            total_loss_sum += f64::try_from(&loss).unwrap_or(0.0);
            num_batches += 1;

            opt.zero_grad();
            loss.backward();
            opt.clip_grad_norm(max_grad_norm);
            opt.step();

            start = end;
        }
    }

    let n = num_batches.max(1) as f64;
    UpdateStats {
        policy_loss: total_policy_loss / n,
        value_loss: total_value_loss / n,
        entropy: total_entropy / n,
        total_loss: total_loss_sum / n,
    }
}
