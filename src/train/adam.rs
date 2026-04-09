// Custom Adam optimizer with persistent state.
//
// `tch::nn::Optimizer` is opaque — there's no public way to extract its internal Adam
// moments. This implementation gives us full control over `m`, `v`, and the step counter
// so we can save them alongside the model weights and resume training without losing
// optimizer momentum.
//
// Algorithm matches PyTorch's torch.optim.Adam (no AMSGrad, no weight decay):
//   m_t = β₁ * m_{t-1} + (1-β₁) * g_t
//   v_t = β₂ * v_{t-1} + (1-β₂) * g_t²
//   m̂_t = m_t / (1 - β₁^t)
//   v̂_t = v_t / (1 - β₂^t)
//   θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)

use std::collections::HashMap;
use tch::nn::VarStore;
use tch::{Device, Tensor};

const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const EPS: f64 = 1e-8;

pub struct Adam {
    /// (parameter name, parameter tensor) — name is the key used in save/load.
    params: Vec<(String, Tensor)>,
    /// First moment estimates per parameter, parallel to `params`.
    m: Vec<Tensor>,
    /// Second moment estimates per parameter, parallel to `params`.
    v: Vec<Tensor>,
    pub step_count: i64,
    pub lr: f64,
    device: Device,
}

impl Adam {
    pub fn new(vs: &VarStore, lr: f64) -> Self {
        let device = vs.device();
        let mut params: Vec<(String, Tensor)> = vs.variables().into_iter().collect();
        params.retain(|(_, t)| t.requires_grad());
        // Sort by name so save/load order is deterministic regardless of HashMap iteration.
        params.sort_by(|a, b| a.0.cmp(&b.0));

        let m: Vec<Tensor> = params
            .iter()
            .map(|(_, p)| Tensor::zeros(p.size(), (p.kind(), p.device())))
            .collect();
        let v: Vec<Tensor> = params
            .iter()
            .map(|(_, p)| Tensor::zeros(p.size(), (p.kind(), p.device())))
            .collect();

        Self {
            params,
            m,
            v,
            step_count: 0,
            lr,
            device,
        }
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    pub fn zero_grad(&self) {
        for (_, p) in &self.params {
            let mut g = p.grad();
            if g.defined() {
                let _ = g.zero_();
            }
        }
    }

    pub fn clip_grad_norm(&self, max_norm: f64) {
        let mut sq_sum = 0.0f64;
        for (_, p) in &self.params {
            let g = p.grad();
            if g.defined() {
                let nrm = f64::try_from(g.norm()).unwrap_or(0.0);
                sq_sum += nrm * nrm;
            }
        }
        let total = sq_sum.sqrt();
        if total > max_norm && total > 0.0 {
            let scale = max_norm / (total + 1e-6);
            tch::no_grad(|| {
                for (_, p) in &self.params {
                    let mut g = p.grad();
                    if g.defined() {
                        let _ = g.f_mul_scalar_(scale);
                    }
                }
            });
        }
    }

    pub fn step(&mut self) {
        self.step_count += 1;
        let bc1 = 1.0 - BETA1.powi(self.step_count as i32);
        let bc2 = 1.0 - BETA2.powi(self.step_count as i32);
        let step_size = self.lr / bc1;

        tch::no_grad(|| {
            for i in 0..self.params.len() {
                let g = self.params[i].1.grad();
                if !g.defined() {
                    continue;
                }

                // m = β₁m + (1-β₁)g
                let new_m = &self.m[i] * BETA1 + &g * (1.0 - BETA1);
                // v = β₂v + (1-β₂)g²
                let g_sq = &g * &g;
                let new_v = &self.v[i] * BETA2 + g_sq * (1.0 - BETA2);

                // denom = √(v / bc2) + ε    (pre-corrected v)
                let denom = (&new_v / bc2).sqrt() + EPS;
                // update = (lr / bc1) * m / denom    (pre-corrected m via step_size)
                let update = (&new_m / &denom) * step_size;
                // p -= update
                let _ = self.params[i].1.f_sub_(&update);

                self.m[i] = new_m;
                self.v[i] = new_v;
            }
        });
    }

    /// Save Adam state (m, v per parameter, plus the step counter) to a safetensors file.
    pub fn save_state(&self, path: &str) -> Result<(), String> {
        let mut named: Vec<(String, Tensor)> = Vec::with_capacity(self.params.len() * 2 + 1);
        for (i, (name, _)) in self.params.iter().enumerate() {
            named.push((format!("m.{}", name), self.m[i].shallow_clone()));
            named.push((format!("v.{}", name), self.v[i].shallow_clone()));
        }
        named.push((
            "__step__".to_string(),
            Tensor::from_slice(&[self.step_count]),
        ));
        Tensor::save_multi(&named, path).map_err(|e| format!("Adam save_state: {}", e))
    }

    /// Load Adam state from a file written by `save_state`. Missing keys are silently
    /// kept at their current value (zero on a fresh optimizer).
    pub fn load_state(&mut self, path: &str) -> Result<(), String> {
        let loaded = Tensor::load_multi(path).map_err(|e| format!("Adam load_state: {}", e))?;
        let map: HashMap<String, Tensor> = loaded.into_iter().collect();

        for i in 0..self.params.len() {
            let name = self.params[i].0.clone();
            let m_key = format!("m.{}", name);
            let v_key = format!("v.{}", name);
            if let Some(m_loaded) = map.get(&m_key) {
                self.m[i] = m_loaded.shallow_clone().to_device(self.device);
            }
            if let Some(v_loaded) = map.get(&v_key) {
                self.v[i] = v_loaded.shallow_clone().to_device(self.device);
            }
        }
        if let Some(step_t) = map.get("__step__") {
            self.step_count = i64::try_from(step_t).unwrap_or(0);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Kind, nn};

    /// Build a single-parameter VarStore with parameter `p = init` and stash `grad` into
    /// `p.grad()` via a real backward pass on `loss = grad * p`. This way the test exercises
    /// the same code path the trainer does (parameters get gradients through autograd, not
    /// poked manually).
    fn one_param(init: f64, grad: f64) -> (nn::VarStore, Tensor) {
        let vs = nn::VarStore::new(Device::Cpu);
        let p = vs.root().var("p", &[1], nn::Init::Const(init));
        let loss = (&p * grad).sum(Kind::Float);
        loss.backward();
        (vs, p)
    }

    fn val(t: &Tensor) -> f64 {
        f64::try_from(t).unwrap()
    }

    /// After a single Adam step from m=v=0, the parameter moves by exactly `-lr * sign(g)`
    /// (modulo eps). This is the famous gradient-magnitude-normalization property of Adam.
    #[test]
    fn first_step_normalizes_gradient_magnitude() {
        // Three different gradient magnitudes should all produce the same step size.
        for &g in &[0.5_f64, 2.0, 10.0, 0.001] {
            let (vs, p) = one_param(1.0, g);
            let mut adam = Adam::new(&vs, 0.1);
            adam.step();
            let after = val(&p);
            // Expected: 1.0 - 0.1 * sign(g) = 0.9 for positive g
            assert!(
                (after - 0.9).abs() < 1e-5,
                "g={}: expected ~0.9, got {}",
                g,
                after
            );
        }
    }

    #[test]
    fn negative_gradient_increases_parameter() {
        let (vs, p) = one_param(1.0, -0.5);
        let mut adam = Adam::new(&vs, 0.1);
        adam.step();
        let after = val(&p);
        // grad < 0 → param goes UP by lr
        assert!((after - 1.1).abs() < 1e-5, "expected ~1.1, got {}", after);
    }

    #[test]
    fn step_count_increments() {
        let (vs, _p) = one_param(1.0, 0.5);
        let mut adam = Adam::new(&vs, 0.1);
        assert_eq!(adam.step_count, 0);
        adam.step();
        assert_eq!(adam.step_count, 1);
        adam.step();
        assert_eq!(adam.step_count, 2);
    }

    /// Constant-gradient training: each step should move the parameter by ~lr (because
    /// Adam's bias correction makes m_hat ≈ g and v_hat ≈ g², so the update normalizes
    /// to lr * sign(g) regardless of gradient magnitude).
    #[test]
    fn constant_gradient_steps_by_approximately_lr() {
        let (vs, p) = one_param(1.0, 0.5);
        let mut adam = Adam::new(&vs, 0.1);
        let initial = val(&p);

        // Take 5 steps. Re-set the gradient before each step (otherwise it'd accumulate
        // because tch grads are accumulators).
        for _ in 0..5 {
            // Zero current grad and re-run backward
            adam.zero_grad();
            let loss = (&p * 0.5_f64).sum(Kind::Float);
            loss.backward();
            adam.step();
        }
        let after = val(&p);
        // ~5 * 0.1 = 0.5 movement, so p ≈ 0.5. Allow some slack since later steps have
        // slightly imperfect normalization due to the bias-correction tail.
        let movement = initial - after;
        assert!(
            (movement - 0.5).abs() < 0.05,
            "expected ~0.5 total movement after 5 steps, got {}",
            movement
        );
    }

    #[test]
    fn first_moment_matches_formula() {
        let (vs, _p) = one_param(1.0, 0.5);
        let mut adam = Adam::new(&vs, 0.1);
        adam.step();
        // m_1 = (1-β₁) * g = 0.1 * 0.5 = 0.05
        let m_1 = val(&adam.m[0]);
        assert!((m_1 - 0.05).abs() < 1e-7, "m_1 expected 0.05, got {}", m_1);
        // v_1 = (1-β₂) * g² = 0.001 * 0.25 = 0.00025
        let v_1 = val(&adam.v[0]);
        assert!(
            (v_1 - 0.00025).abs() < 1e-9,
            "v_1 expected 0.00025, got {}",
            v_1
        );
    }

    #[test]
    fn zero_grad_clears_gradient() {
        let (vs, p) = one_param(1.0, 0.5);
        let adam = Adam::new(&vs, 0.1);
        // Grad should be 0.5 from the backward pass
        assert!((val(&p.grad()) - 0.5).abs() < 1e-7);
        adam.zero_grad();
        assert!(val(&p.grad()).abs() < 1e-7);
    }

    #[test]
    fn clip_grad_norm_scales_when_above() {
        let (vs, p) = one_param(1.0, 10.0);
        let adam = Adam::new(&vs, 0.1);
        adam.clip_grad_norm(1.0);
        let g_after = val(&p.grad());
        // Clipped to ~1.0 (with the +1e-6 epsilon in the divisor it's slightly less)
        assert!(g_after > 0.0 && g_after <= 1.0 + 1e-5);
        assert!(g_after > 0.99, "expected ~1.0, got {}", g_after);
    }

    #[test]
    fn clip_grad_norm_noop_when_below() {
        let (vs, p) = one_param(1.0, 0.3);
        let adam = Adam::new(&vs, 0.1);
        adam.clip_grad_norm(1.0);
        let g_after = val(&p.grad());
        // Below threshold → unchanged
        assert!(
            (g_after - 0.3).abs() < 1e-7,
            "expected 0.3, got {}",
            g_after
        );
    }

    #[test]
    fn save_load_round_trip_preserves_state() {
        let (vs, _p) = one_param(1.0, 0.5);
        let mut adam = Adam::new(&vs, 0.1);
        // Take a few steps to populate m and v
        for _ in 0..3 {
            adam.zero_grad();
            let loss = (&vs.root().get("p").unwrap() * 0.5_f64).sum(Kind::Float);
            loss.backward();
            adam.step();
        }
        let m_before = val(&adam.m[0]);
        let v_before = val(&adam.v[0]);
        let step_before = adam.step_count;

        // Save to a temp file
        let tmp_path = std::env::temp_dir()
            .join(format!("adam_test_{}.adam", std::process::id()))
            .to_string_lossy()
            .into_owned();
        adam.save_state(&tmp_path).expect("save_state failed");

        // Build a fresh optimizer (m=0, v=0, step=0) and load
        let (vs2, _p2) = one_param(1.0, 0.5);
        let mut adam2 = Adam::new(&vs2, 0.1);
        adam2.load_state(&tmp_path).expect("load_state failed");

        assert_eq!(adam2.step_count, step_before);
        assert!((val(&adam2.m[0]) - m_before).abs() < 1e-9);
        assert!((val(&adam2.v[0]) - v_before).abs() < 1e-9);

        let _ = std::fs::remove_file(&tmp_path);
    }
}
