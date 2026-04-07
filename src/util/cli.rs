pub struct TrainArgs {
    pub namespace: String,
    pub max_episodes: Option<u32>,
    pub max_frames: Option<u64>,
    pub max_minutes: Option<u64>,
    pub auto_resume: bool,
    pub debug: bool,
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
    }
}
