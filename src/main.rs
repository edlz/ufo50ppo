use ufo50ppo::games;
use ufo50ppo::platform::GameRunner;
use ufo50ppo::platform::lockstep::{LockstepGame, Start};
use ufo50ppo::train::config::TrainingConfig;
use ufo50ppo::train::multi::run_training_multi;
use ufo50ppo::util::cli;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = cli::parse_train_args("default");
    let game = games::ninpek::definition();

    ufo50ppo::util::preload_torch_cuda();
    ufo50ppo::util::shutdown::install()?;

    println!(
        "Training {} | ns: {} | {}x{} | envs: {} | frame skip: {} | random start: {} | game: {}",
        game.name,
        args.namespace,
        game.obs_width,
        game.obs_height,
        args.num_envs,
        args.frame_skip,
        args.random_start,
        args.game_dir.display()
    );
    if let Some(n) = args.max_episodes {
        println!("  Max episodes: {}", n);
    }

    let mut games = LockstepGame::launch_many(
        &args.game_dir,
        args.num_envs as usize,
        args.frame_skip,
        game.game_id,
    )
    .map_err(|e| {
        format!(
            "launching {}: {e} (build it with bridge/build.sh)",
            args.game_dir.display()
        )
    })?;
    for g in &mut games {
        g.set_start(Start::Random {
            frac: args.random_start,
            max_x: game.max_start,
        });
    }
    let runners: Vec<Box<dyn GameRunner>> = games
        .into_iter()
        .map(|g| Box::new(g) as Box<dyn GameRunner>)
        .collect();

    let cfg = TrainingConfig::from_args(&args, game.name);
    run_training_multi(runners, game, cfg);
    Ok(())
}
