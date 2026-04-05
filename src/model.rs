use tch::{Device, Kind, Tensor, nn, nn::Module, nn::OptimizerConfig};

use crate::input::NUM_ACTIONS;

pub struct ActorCritic {
    pub vs: nn::VarStore,
    shared: nn::Sequential,
    policy_head: nn::Linear,
    value_head: nn::Linear,
}

impl ActorCritic {
    pub fn new(device: Device) -> Self {
        let vs = nn::VarStore::new(device);
        let root = &vs.root();

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
            .add(nn::linear(root / "fc", 3136, 512, Default::default()))
            .add_fn(|x| x.relu());

        let policy_head = nn::linear(root / "policy", 512, NUM_ACTIONS as i64, Default::default());
        let value_head = nn::linear(root / "value", 512, 1, Default::default());

        Self {
            vs,
            shared,
            policy_head,
            value_head,
        }
    }

    /// Returns (log_probs [batch, NUM_ACTIONS], values [batch, 1])
    pub fn forward(&self, obs: &Tensor) -> (Tensor, Tensor) {
        let features = self.shared.forward(obs);
        let log_probs = self
            .policy_head
            .forward(&features)
            .log_softmax(-1, Kind::Float);
        let values = self.value_head.forward(&features);
        (log_probs, values)
    }

    /// Sample an action. Returns (action_index, log_prob, value).
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

    pub fn optimizer(&self, lr: f64) -> nn::Optimizer {
        nn::Adam::default().build(&self.vs, lr).unwrap()
    }
}
