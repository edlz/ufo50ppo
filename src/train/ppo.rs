use tch::{Kind, Tensor};

use super::adam::Adam;
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

    pub fn is_empty(&self) -> bool {
        self.obs.is_empty()
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
/// `reward_std` divides each reward before the delta computation — keeps the gradient
/// signal stable across reward distribution shifts. Pass 1.0 to disable normalization.
/// Returns (advantages, returns) as Vec<f64>.
pub fn compute_gae(
    rewards: &[f64],
    values: &[f64],
    dones: &[bool],
    last_value: f64,
    gamma: f64,
    lambda: f64,
    reward_std: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = rewards.len();
    let mut advantages = vec![0.0; n];
    let mut gae = 0.0;
    let inv_std = if reward_std > 1e-8 {
        1.0 / reward_std
    } else {
        1.0
    };

    for t in (0..n).rev() {
        let next_value = if t == n - 1 {
            last_value
        } else {
            values[t + 1]
        };
        let next_non_terminal = if dones[t] { 0.0 } else { 1.0 };
        let r_norm = rewards[t] * inv_std;
        let delta = r_norm + gamma * next_value * next_non_terminal - values[t];
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
    pub grad_norm: f64,
}

/// Run one PPO update over the rollout buffer.
pub fn update(
    model: &mut ActorCritic,
    opt: &mut Adam,
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
    let old_log_probs_t = Tensor::from_slice(&buffer.log_probs)
        .to_device(device)
        .detach();
    let old_values_t = f32_tensor(&buffer.values);
    let advantages_t = f32_tensor(advantages);
    let returns_t = f32_tensor(returns);

    let adv_mean = advantages_t.mean(Kind::Float);
    let adv_std = advantages_t.std(true).clamp_min(1e-5);
    let advantages_t = (&advantages_t - &adv_mean) / &adv_std;

    let mut total_policy_loss = 0.0;
    let mut total_value_loss = 0.0;
    let mut total_entropy = 0.0;
    let mut total_loss_sum = 0.0;
    let mut total_grad_norm = 0.0;
    let mut grad_norm_batches = 0;
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

            // Refuse to backward a non-finite loss. Without this, NaN/Inf would propagate
            // into Adam's m and v, then into the model parameters, silently corrupting the
            // run for the rest of training while the process keeps printing healthy-looking
            // numbers. Skip the minibatch and let the next one try.
            let loss_val = f64::try_from(&loss).unwrap_or(f64::NAN);
            if !loss_val.is_finite() {
                eprintln!(
                    "[ppo] non-finite loss ({}) on minibatch — skipping backward+step",
                    loss_val
                );
                start = end;
                continue;
            }

            total_policy_loss += f64::try_from(&policy_loss).unwrap_or(0.0);
            total_value_loss += f64::try_from(&value_loss).unwrap_or(0.0);
            total_entropy += f64::try_from(&entropy).unwrap_or(0.0);
            total_loss_sum += loss_val;
            num_batches += 1;

            opt.zero_grad();
            loss.backward();

            // Measure pre-clip grad norm so we can see when clipping is firing.
            let mut sq_sum = 0.0f64;
            for var in model.vs.trainable_variables() {
                let g = var.grad();
                if g.defined() {
                    let nrm = f64::try_from(g.norm()).unwrap_or(0.0);
                    sq_sum += nrm * nrm;
                }
            }
            let mb_grad_norm = sq_sum.sqrt();

            // Same idea for grads — non-finite gradient means backward propagated NaN/Inf.
            // Don't let it reach the optimizer.
            if !mb_grad_norm.is_finite() {
                eprintln!(
                    "[ppo] non-finite grad_norm ({}) — skipping optimizer step",
                    mb_grad_norm
                );
                start = end;
                continue;
            }
            total_grad_norm += mb_grad_norm;
            grad_norm_batches += 1;

            opt.clip_grad_norm(max_grad_norm);
            opt.step();

            start = end;
        }
    }

    let n = num_batches.max(1) as f64;
    let gn = grad_norm_batches.max(1) as f64;
    UpdateStats {
        policy_loss: total_policy_loss / n,
        value_loss: total_value_loss / n,
        entropy: total_entropy / n,
        total_loss: total_loss_sum / n,
        grad_norm: total_grad_norm / gn,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn gae_zero_rewards_zero_values_returns_zero() {
        let rewards = vec![0.0; 5];
        let values = vec![0.0; 5];
        let dones = vec![false; 5];
        let (adv, ret) = compute_gae(&rewards, &values, &dones, 0.0, 0.99, 0.95, 1.0);
        for &a in &adv {
            assert!(approx_eq(a, 0.0, 1e-12));
        }
        for &r in &ret {
            assert!(approx_eq(r, 0.0, 1e-12));
        }
    }

    #[test]
    fn gae_terminal_truncates_bootstrap() {
        // Single step, immediately done. Advantage should be reward - value.
        // No bootstrap from next_value because done=true zeroes the term.
        let rewards = vec![1.0];
        let values = vec![0.5];
        let dones = vec![true];
        let (adv, ret) = compute_gae(&rewards, &values, &dones, 999.0, 0.99, 0.95, 1.0);
        // delta = r + γ * V_next * (1-done) - V_t = 1.0 + 0 - 0.5 = 0.5
        assert!(approx_eq(adv[0], 0.5, 1e-12), "adv = {}", adv[0]);
        assert!(approx_eq(ret[0], 1.0, 1e-12), "ret = {}", ret[0]);
    }

    #[test]
    fn gae_uses_last_value_when_not_done() {
        // Single step, NOT done — last_value used as the bootstrap.
        let rewards = vec![0.0];
        let values = vec![0.0];
        let dones = vec![false];
        let (adv, _) = compute_gae(&rewards, &values, &dones, 10.0, 0.99, 0.95, 1.0);
        // delta = 0 + 0.99 * 10 * 1 - 0 = 9.9
        assert!(approx_eq(adv[0], 9.9, 1e-12), "adv = {}", adv[0]);
    }

    #[test]
    fn gae_returns_equal_advantages_plus_values() {
        // returns_t = advantages_t + values_t — basic invariant
        let rewards = vec![1.0, 0.5, -0.3, 2.0];
        let values = vec![0.1, 0.2, 0.3, 0.4];
        let dones = vec![false, false, false, false];
        let (adv, ret) = compute_gae(&rewards, &values, &dones, 0.5, 0.99, 0.95, 1.0);
        for i in 0..rewards.len() {
            assert!(
                approx_eq(ret[i], adv[i] + values[i], 1e-12),
                "ret[{}] != adv + V",
                i
            );
        }
    }

    #[test]
    fn gae_done_in_middle_resets_propagation() {
        // 4-step trajectory with done at index 1. The advantage at index 0 should NOT
        // include any contribution from indices 2-3 (the done at index 1 cuts the chain).
        let rewards = vec![1.0, 1.0, 1.0, 1.0];
        let values = vec![0.0, 0.0, 0.0, 0.0];
        let dones_truncated = vec![false, true, false, false];
        let dones_continuous = vec![false, false, false, false];
        let (adv_t, _) = compute_gae(&rewards, &values, &dones_truncated, 0.0, 0.99, 0.95, 1.0);
        let (adv_c, _) = compute_gae(&rewards, &values, &dones_continuous, 0.0, 0.99, 0.95, 1.0);
        // adv_t[0] should depend only on r[0] and r[1]; adv_c[0] should depend on all 4.
        // So adv_c[0] > adv_t[0] (more positive contributions accumulated).
        assert!(
            adv_c[0] > adv_t[0],
            "continuous {} should exceed truncated {}",
            adv_c[0],
            adv_t[0]
        );
    }

    #[test]
    fn reward_std_divides_correctly() {
        // Same trajectory with reward_std = 1.0 vs 2.0 should give exactly half the
        // advantage magnitudes (because rewards are pre-divided by std before delta).
        let rewards = vec![1.0, 1.0, 1.0];
        let values = vec![0.0, 0.0, 0.0];
        let dones = vec![false, false, false];
        let (adv1, _) = compute_gae(&rewards, &values, &dones, 0.0, 0.99, 0.95, 1.0);
        let (adv2, _) = compute_gae(&rewards, &values, &dones, 0.0, 0.99, 0.95, 2.0);
        for i in 0..3 {
            assert!(
                approx_eq(adv1[i], 2.0 * adv2[i], 1e-12),
                "adv1[{}]={}, adv2[{}]={}",
                i,
                adv1[i],
                i,
                adv2[i]
            );
        }
    }

    #[test]
    fn reward_std_zero_disables_normalization() {
        // reward_std = 0 (or very small) should fall back to inv_std=1.0, not divide-by-zero
        let rewards = vec![1.0];
        let values = vec![0.0];
        let dones = vec![true];
        let (adv, _) = compute_gae(&rewards, &values, &dones, 0.0, 0.99, 0.95, 0.0);
        // Should behave as if reward_std = 1.0
        let (adv_ref, _) = compute_gae(&rewards, &values, &dones, 0.0, 0.99, 0.95, 1.0);
        assert!(approx_eq(adv[0], adv_ref[0], 1e-12));
    }

    #[test]
    fn explained_variance_perfect_predictions() {
        // V exactly matches returns → EV = 1.0
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let returns = vec![1.0, 2.0, 3.0, 4.0];
        let ev = explained_variance(&values, &returns);
        assert!(approx_eq(ev, 1.0, 1e-12), "ev = {}", ev);
    }

    #[test]
    fn explained_variance_mean_predictions_returns_zero() {
        // Predicting the mean of returns gives EV = 0 (you've explained none of the
        // variance, but you also haven't done worse than uninformed).
        let returns = vec![1.0, 2.0, 3.0, 4.0];
        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let values = vec![mean; 4];
        let ev = explained_variance(&values, &returns);
        assert!(approx_eq(ev, 0.0, 1e-12), "ev = {}", ev);
    }

    #[test]
    fn explained_variance_zero_predictions_can_be_negative() {
        // V always zero, returns far from zero → EV is highly negative (worse than mean).
        let values = vec![0.0, 0.0, 0.0, 0.0];
        let returns = vec![1.0, 2.0, 3.0, 4.0];
        let ev = explained_variance(&values, &returns);
        assert!(ev < 0.0, "ev should be < 0, got {}", ev);
    }

    #[test]
    fn explained_variance_constant_returns_returns_zero() {
        // var(returns) = 0 — undefined ratio. Spec says return 0.0.
        let values = vec![1.0, 2.0, 3.0];
        let returns = vec![5.0, 5.0, 5.0];
        let ev = explained_variance(&values, &returns);
        assert_eq!(ev, 0.0);
    }
}
