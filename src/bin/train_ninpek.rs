use ufo50ppo::games;
use ufo50ppo::platform;
use ufo50ppo::train::runner::{TrainingConfig, run_training, spawn_debug_thread};
use ufo50ppo::util::cli;

fn main() -> windows::core::Result<()> {
    let args = cli::parse_train_args("default");
    let game = games::ninpek::definition();

    ufo50ppo::util::preload_torch_cuda();

    println!(
        "Training {} | ns: {} | {}x{}",
        game.name, args.namespace, game.obs_width, game.obs_height
    );
    if let Some(n) = args.max_episodes {
        println!("  Max episodes: {}", n);
    }
    if args.debug {
        println!("Debug mode: saving frames to debug_frames/\n");
    }

    let debug_tx = if args.debug {
        Some(spawn_debug_thread(game.obs_width, game.obs_height))
    } else {
        None
    };
    let cfg = TrainingConfig::from_args(args, game.name);

    platform::host(
        game.window_title,
        game.obs_width,
        game.obs_height,
        move |runner| {
            run_training(runner, game, cfg, debug_tx);
        },
    )
}
