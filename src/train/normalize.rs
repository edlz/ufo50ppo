// Welford's online algorithm for streaming mean/variance. Used to track the variance
// of the running discounted reward — dividing reward by this std before GAE keeps the
// gradient signal stable across reward distribution shifts (e.g. bumping SCORE_UP mid-
// training, or naturally as the agent visits higher-reward states).
//
// Matches the reward-normalization half of SB3's VecNormalize: divide reward by std,
// do not mean-center (subtracting a constant doesn't affect the optimal policy but does
// change the value function offset, so we leave reward zero-point alone).

#[derive(Clone, Copy)]
pub struct RunningMeanStd {
    pub mean: f64,
    pub var_sum: f64, // sum of squared deltas (Welford); divide by count for variance
    pub count: u64,
}

impl Default for RunningMeanStd {
    fn default() -> Self {
        Self::new()
    }
}

impl RunningMeanStd {
    pub fn new() -> Self {
        Self {
            mean: 0.0,
            var_sum: 0.0,
            count: 0,
        }
    }

    pub fn update(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.var_sum += delta * delta2;
    }

    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            1.0
        } else {
            self.var_sum / self.count as f64
        }
    }

    pub fn std(&self) -> f64 {
        self.variance().sqrt().max(1e-8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_returns_unit_std() {
        let r = RunningMeanStd::new();
        assert_eq!(r.std(), 1.0);
        assert_eq!(r.variance(), 1.0);
    }

    #[test]
    fn single_sample_still_unit_std() {
        // The count<2 special case prevents division-by-something-tiny when only one
        // sample has been seen. If this fails, the first PPO update after a fresh
        // start will divide rewards by ~0 and blow up.
        let mut r = RunningMeanStd::new();
        r.update(42.0);
        assert_eq!(r.variance(), 1.0);
        assert_eq!(r.std(), 1.0);
    }

    #[test]
    fn two_samples_correct_variance() {
        // Welford with n=2: variance = (x1-x2)² / 4 (biased estimator)
        let mut r = RunningMeanStd::new();
        r.update(0.0);
        r.update(2.0);
        // mean = 1.0, var = ((0-1)² + (2-1)²) / 2 = 1.0
        assert!((r.mean - 1.0).abs() < 1e-12);
        assert!((r.variance() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn many_samples_converge_to_true_std() {
        let mut r = RunningMeanStd::new();
        // 1000 pseudo-random samples spread over [-2, 2] around a mean of 10.
        // Wrapping arithmetic to avoid overflow with the LCG-style hash.
        for i in 0u32..1000 {
            let h = i.wrapping_mul(1103515245).wrapping_add(12345);
            let x = 10.0 + 2.0 * ((h & 0xff) as f64 / 128.0 - 1.0);
            r.update(x);
        }
        // Mean should be close to 10
        assert!((r.mean - 10.0).abs() < 0.5, "mean = {}", r.mean);
        // Std should be small but positive (uniform on [-2, 2] has std ≈ 1.155)
        let s = r.std();
        assert!(s > 0.5 && s < 2.5, "std = {}", s);
    }

    #[test]
    fn welford_numerically_stable_for_constant_stream() {
        // Welford should give variance = 0 for a constant stream (and thus std = 1e-8 floor).
        // The naive (sum² / n - mean²) formula would give catastrophic cancellation here.
        let mut r = RunningMeanStd::new();
        for _ in 0..10000 {
            r.update(1_000_000.0);
        }
        assert!((r.mean - 1_000_000.0).abs() < 1e-6);
        assert_eq!(r.std(), 1e-8); // floored
    }

    #[test]
    fn std_never_returns_zero() {
        // GAE divides by std — must never be zero.
        let mut r = RunningMeanStd::new();
        for _ in 0..100 {
            r.update(0.0);
        }
        assert!(r.std() > 0.0);
        assert_eq!(r.std(), 1e-8);
    }
}
