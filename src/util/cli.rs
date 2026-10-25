pub struct TrainArgs {
    pub namespace: String,
    pub max_episodes: Option<u32>,
    pub max_frames: Option<u64>,
    pub max_minutes: Option<u64>,
    pub auto_resume: bool,
    pub debug: bool,
    pub num_envs: u32,
    /// Lockstep game copy built by bridge/build.sh.
    pub game_dir: std::path::PathBuf,
    /// Game frames per training step (action repeat).
    pub frame_skip: u32,
    /// Fraction of runs that start at a random point in the level instead of the beginning.
    pub random_start: f64,
}

fn parse_or_die<T: std::str::FromStr>(val: &str, flag: &str) -> T {
    val.parse().unwrap_or_else(|_| {
        eprintln!("error: invalid value for {}: '{}'", flag, val);
        std::process::exit(2);
    })
}

fn require_value<'a>(args: &'a [String], i: usize, flag: &str) -> &'a str {
    args.get(i).map(|s| s.as_str()).unwrap_or_else(|| {
        eprintln!("error: {} requires a value", flag);
        std::process::exit(2);
    })
}

pub fn parse_train_args(default_namespace: &str) -> TrainArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut namespace = default_namespace.to_string();
    let mut max_episodes = None;
    let mut max_frames = None;
    let mut max_minutes = None;
    let mut auto_resume = false;
    let mut debug = false;
    let mut num_envs: u32 = 1;
    let mut game_dir = std::env::var_os("UFO50_GAME_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "build/game".into());
    let mut frame_skip: u32 = 4;
    let mut random_start = 0.5;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--namespace" | "-n" => {
                i += 1;
                namespace = require_value(&args, i, "--namespace").to_string();
            }
            "--episodes" | "-e" => {
                i += 1;
                max_episodes = Some(parse_or_die(
                    require_value(&args, i, "--episodes"),
                    "--episodes",
                ));
            }
            "--frames" | "-f" => {
                i += 1;
                max_frames = Some(parse_or_die(
                    require_value(&args, i, "--frames"),
                    "--frames",
                ));
            }
            "--minutes" | "-m" => {
                i += 1;
                max_minutes = Some(parse_or_die(
                    require_value(&args, i, "--minutes"),
                    "--minutes",
                ));
            }
            "--auto-resume" | "-r" => auto_resume = true,
            "--debug" | "-d" => debug = true,
            "--num-envs" | "-N" => {
                i += 1;
                num_envs = parse_or_die(require_value(&args, i, "--num-envs"), "--num-envs");
                if num_envs == 0 {
                    eprintln!("error: --num-envs must be >= 1");
                    std::process::exit(2);
                }
            }
            "--game-dir" | "-g" => {
                i += 1;
                game_dir = require_value(&args, i, "--game-dir").into();
            }
            "--frame-skip" | "-k" => {
                i += 1;
                frame_skip = parse_or_die(require_value(&args, i, "--frame-skip"), "--frame-skip");
                if frame_skip == 0 {
                    eprintln!("error: --frame-skip must be >= 1");
                    std::process::exit(2);
                }
            }
            "--random-start" | "-s" => {
                i += 1;
                random_start =
                    parse_or_die(require_value(&args, i, "--random-start"), "--random-start");
                if !(0.0..=1.0).contains(&random_start) {
                    eprintln!("error: --random-start must be in [0, 1]");
                    std::process::exit(2);
                }
            }
            other => {
                eprintln!("error: unknown argument '{}'", other);
                std::process::exit(2);
            }
        }
        i += 1;
    }
    TrainArgs {
        namespace,
        max_episodes,
        max_frames,
        max_minutes,
        auto_resume,
        debug,
        num_envs,
        game_dir,
        frame_skip,
        random_start,
    }
}
