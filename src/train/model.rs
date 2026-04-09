use tch::{Device, Kind, Tensor, nn, nn::Module};

pub struct ActorCritic {
    pub vs: nn::VarStore,
    shared: nn::Sequential,
    policy_head: nn::Linear,
    value_head: nn::Linear,
}

/// Compute the flatten dim for the Nature DQN CNN given the input observation size.
/// Conv1: kernel=8 stride=4, Conv2: kernel=4 stride=2, Conv3: kernel=3 stride=1, 64 out channels.
fn nature_dqn_flatten_dim(obs_w: u32, obs_h: u32) -> i64 {
    let conv_out = |input: u32, kernel: u32, stride: u32| (input - kernel) / stride + 1;
    let w = conv_out(conv_out(conv_out(obs_w, 8, 4), 4, 2), 3, 1);
    let h = conv_out(conv_out(conv_out(obs_h, 8, 4), 4, 2), 3, 1);
    (64 * w * h) as i64
}

impl ActorCritic {
    pub fn new(device: Device, obs_w: u32, obs_h: u32, num_actions: usize) -> Self {
        assert!(
            obs_w >= 36 && obs_h >= 36,
            "Nature DQN CNN requires obs >= 36x36, got {}x{}",
            obs_w,
            obs_h
        );
        let vs = nn::VarStore::new(device);
        let root = &vs.root();
        let flatten_dim = nature_dqn_flatten_dim(obs_w, obs_h);

        let shared = nn::seq()
            .add(nn::conv2d(
                root / "c1",
                4,
                32,
                8,
                nn::ConvConfig {
                    stride: 4,
                    ..Default::default()
                },
            ))
            .add_fn(|x| x.relu())
            .add(nn::conv2d(
                root / "c2",
                32,
                64,
                4,
                nn::ConvConfig {
                    stride: 2,
                    ..Default::default()
                },
            ))
            .add_fn(|x| x.relu())
            .add(nn::conv2d(
                root / "c3",
                64,
                64,
                3,
                nn::ConvConfig {
                    stride: 1,
                    ..Default::default()
                },
            ))
            .add_fn(|x| x.relu())
            .add_fn(|x| x.flat_view())
            .add(nn::linear(
                root / "fc",
                flatten_dim,
                512,
                Default::default(),
            ))
            .add_fn(|x| x.relu());

        let policy_head = nn::linear(root / "policy", 512, num_actions as i64, Default::default());
        let value_head = nn::linear(root / "value", 512, 1, Default::default());

        Self {
            vs,
            shared,
            policy_head,
            value_head,
        }
    }

    pub fn forward(&self, obs: &Tensor) -> (Tensor, Tensor) {
        let features = self.shared.forward(obs);
        let log_probs = self
            .policy_head
            .forward(&features)
            .log_softmax(-1, Kind::Float);
        let values = self.value_head.forward(&features);
        (log_probs, values)
    }

    pub fn act(&self, obs: &Tensor) -> (i64, f64, f64) {
        tch::no_grad(|| {
            let (log_probs, values) = self.forward(obs);
            let probs = log_probs.exp();
            let action = probs.multinomial(1, true).int64_value(&[0, 0]);
            let log_prob = log_probs.double_value(&[0, action]);
            let value = values.double_value(&[0, 0]);
            (action, log_prob, value)
        })
    }

    pub fn act_batch(&self, obs: &Tensor) -> (Vec<i64>, Vec<f64>, Vec<f64>) {
        tch::no_grad(|| {
            let (log_probs, values) = self.forward(obs);
            let probs = log_probs.exp();
            let actions = probs.multinomial(1, true); // [N, 1]
            let n = obs.size()[0];
            let mut a_vec = Vec::with_capacity(n as usize);
            let mut lp_vec = Vec::with_capacity(n as usize);
            let mut v_vec = Vec::with_capacity(n as usize);
            for i in 0..n {
                let a = actions.int64_value(&[i, 0]);
                a_vec.push(a);
                lp_vec.push(log_probs.double_value(&[i, a]));
                v_vec.push(values.double_value(&[i, 0]));
            }
            (a_vec, lp_vec, v_vec)
        })
    }

    pub fn optimizer(&self, lr: f64) -> super::adam::Adam {
        super::adam::Adam::new(&self.vs, lr)
    }
}
