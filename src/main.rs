use ufo50ppo::games;
use ufo50ppo::train::multi::run_training_multi;
use ufo50ppo::train::runner::TrainingConfig;
use ufo50ppo::util::cli;

fn main() -> windows::core::Result<()> {
    let args = cli::parse_train_args("default");
    let mut game = games::ninpek::definition();

    ufo50ppo::util::preload_torch_cuda();
    ufo50ppo::util::shutdown::install()?;

    let num_envs = args.num_envs;

    if num_envs > 1 {
        game.hyperparams.rollout_len = 256;
    }

    println!(
        "Training {} | ns: {} | {}x{} | envs: {}",
        game.name, args.namespace, game.obs_width, game.obs_height, num_envs
    );
    if let Some(n) = args.max_episodes {
        println!("  Max episodes: {}", n);
    }

    let cfg = TrainingConfig::from_args(args, game.name);

    ufo50ppo::platform::win32::host_multi(
        game.window_title,
        num_envs as usize,
        game.obs_width,
        game.obs_height,
        move |runners| {
            run_training_multi(runners, game, cfg);
        },
    )
}
