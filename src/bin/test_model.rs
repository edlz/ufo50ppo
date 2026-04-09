use ufo50ppo::train;

fn main() {
    ufo50ppo::util::preload_torch_cuda();

    let device = tch::Device::cuda_if_available();
    println!("Device: {:?}", device);

    let model = train::model::ActorCritic::new(device, 128, 128, ufo50ppo::platform::NUM_ACTIONS);
    println!("Model created");

    let obs = tch::Tensor::randn([1, 4, 128, 128], (tch::Kind::Float, device));
    println!("Obs shape: {:?}", obs.size());

    let (log_probs, values) = model.forward(&obs);
    println!("Log probs shape: {:?}", log_probs.size());
    println!("Values shape: {:?}", values.size());
    println!("Log probs: {:?}", log_probs);

    let (action, log_prob, value) = model.act(&obs);
    println!(
        "Action: {}, log_prob: {:.4}, value: {:.4}",
        action, log_prob, value
    );

    let mut fs = train::preprocess::FrameStack::new(device, 128);
    let fake_bgra = vec![128u8; 128 * 128 * 4];
    let stacked = fs.push(&fake_bgra, 128, 128);
    println!("Frame stack output: {:?}", stacked.size());

    let (action2, _, _) = model.act(&stacked);
    println!("Action from frame stack: {}", action2);

    println!("\nAll checks passed!");
}
