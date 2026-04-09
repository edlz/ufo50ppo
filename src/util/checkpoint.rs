use std::io::{Read, Write};
use tch::nn::VarStore;

/// Write a single-shot run snapshot (git hash, hyperparams, num_envs, start time) to
/// `{dir}/run_metadata.json`. Called once per training process at startup. The file is
/// append-only across runs — if it already exists, we append a new JSON object so a
/// directory shared by multiple resume attempts captures the full history.
pub fn write_run_metadata(
    dir: &str,
    game: &str,
    num_envs: u32,
    rollout_len: usize,
    learning_rate: f64,
    gamma: f64,
    gae_lambda: f64,
) {
    let _ = std::fs::create_dir_all(dir);
    let path = format!("{}/run_metadata.jsonl", dir);

    let git_hash = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into());

    let git_dirty = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let line = format!(
        "{{\"start_unix\":{},\"game\":\"{}\",\"git_hash\":\"{}\",\"git_dirty\":{},\"num_envs\":{},\"rollout_len\":{},\"learning_rate\":{},\"gamma\":{},\"gae_lambda\":{}}}\n",
        now, game, git_hash, git_dirty, num_envs, rollout_len, learning_rate, gamma, gae_lambda
    );

    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        Ok(mut f) => {
            if let Err(e) = f.write_all(line.as_bytes()) {
                eprintln!("write_run_metadata: {}", e);
            }
        }
        Err(e) => eprintln!("write_run_metadata: open {}: {}", path, e),
    }
}

pub struct CheckpointMeta {
    pub game: &'static str,
    pub resolution: (u32, u32),
    pub episode: u32,
    pub total_frames: u64,
    pub ppo_updates: u64,
    pub best_reward: f64,
    pub rollout_len: usize,
    pub learning_rate: f64,
    pub gamma: f64,
    pub gae_lambda: f64,
    // Reward normalizer state — persisted so resumes don't restart with std=1.0 and
    // re-trigger value-function divergence.
    pub reward_norm_mean: f64,
    pub reward_norm_var_sum: f64,
    pub reward_norm_count: u64,
}

pub fn save_metadata(dir: &str, name: &str, meta: &CheckpointMeta) {
    let path = format!("{}/{}.json", dir, name);
    let tmp = format!("{}.tmp", path);
    let Ok(mut f) = std::fs::File::create(&tmp) else {
        eprintln!("save_metadata: failed to create {}", tmp);
        return;
    };
    let result = write!(
        f,
        concat!(
            "{{\n",
            "  \"game\": \"{}\",\n",
            "  \"resolution\": [{}, {}],\n",
            "  \"episode\": {},\n",
            "  \"total_frames\": {},\n",
            "  \"ppo_updates\": {},\n",
            "  \"best_reward\": {:.1},\n",
            "  \"rollout_len\": {},\n",
            "  \"learning_rate\": {},\n",
            "  \"gamma\": {},\n",
            "  \"gae_lambda\": {},\n",
            "  \"reward_norm_mean\": {},\n",
            "  \"reward_norm_var_sum\": {},\n",
            "  \"reward_norm_count\": {}\n",
            "}}"
        ),
        meta.game,
        meta.resolution.0,
        meta.resolution.1,
        meta.episode,
        meta.total_frames,
        meta.ppo_updates,
        meta.best_reward,
        meta.rollout_len,
        meta.learning_rate,
        meta.gamma,
        meta.gae_lambda,
        meta.reward_norm_mean,
        meta.reward_norm_var_sum,
        meta.reward_norm_count
    );
    if let Err(e) = result {
        eprintln!("save_metadata: write error: {}", e);
        let _ = std::fs::remove_file(&tmp);
        return;
    }
    drop(f); // close before rename (Windows needs this)
    if let Err(e) = std::fs::rename(&tmp, &path) {
        eprintln!("save_metadata: rename {} -> {} failed: {}", tmp, path, e);
        let _ = std::fs::remove_file(&tmp);
    }
}

/// State loaded from a previous checkpoint to resume training.
pub struct ResumedState {
    pub episode: u32,
    pub total_frames: u64,
    pub ppo_updates: u64,
    pub best_reward: f64,
    pub reward_norm_mean: f64,
    pub reward_norm_var_sum: f64,
    pub reward_norm_count: u64,
}

/// Try to load both model weights and metadata for a checkpoint.
/// - `Ok(None)` — `.safetensors` file doesn't exist (fresh start)
/// - `Ok(Some(state))` — both files loaded successfully
/// - `Err(msg)` — `.safetensors` exists but couldn't be loaded, or metadata is missing/unparseable
pub fn try_load(dir: &str, name: &str, vs: &mut VarStore) -> Result<Option<ResumedState>, String> {
    let path = format!("{}/{}.safetensors", dir, name);
    if !std::path::Path::new(&path).exists() {
        return Ok(None);
    }
    vs.load(&path).map_err(|e| format!("load weights: {}", e))?;
    let state = load_metadata(dir, name)
        .ok_or_else(|| format!("metadata missing or unparseable: {}/{}.json", dir, name))?;
    Ok(Some(state))
}

/// Try to load metadata from a checkpoint JSON file.
/// Returns None if file doesn't exist or can't be parsed.
pub fn load_metadata(dir: &str, name: &str) -> Option<ResumedState> {
    let path = format!("{}/{}.json", dir, name);
    let mut contents = String::new();
    std::fs::File::open(&path)
        .ok()?
        .read_to_string(&mut contents)
        .ok()?;

    // Simple JSON parsing without serde — extract numeric fields
    fn extract_u32(s: &str, key: &str) -> Option<u32> {
        let pat = format!("\"{}\":", key);
        let idx = s.find(&pat)? + pat.len();
        s[idx..]
            .trim()
            .split(|c: char| !c.is_ascii_digit())
            .next()?
            .parse()
            .ok()
    }
    fn extract_u64(s: &str, key: &str) -> Option<u64> {
        let pat = format!("\"{}\":", key);
        let idx = s.find(&pat)? + pat.len();
        s[idx..]
            .trim()
            .split(|c: char| !c.is_ascii_digit())
            .next()?
            .parse()
            .ok()
    }
    fn extract_f64(s: &str, key: &str) -> Option<f64> {
        let pat = format!("\"{}\":", key);
        let idx = s.find(&pat)? + pat.len();
        s[idx..]
            .trim()
            .split(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
            .next()?
            .parse()
            .ok()
    }

    Some(ResumedState {
        episode: extract_u32(&contents, "episode")?,
        total_frames: extract_u64(&contents, "total_frames")?,
        ppo_updates: extract_u64(&contents, "ppo_updates")?,
        best_reward: extract_f64(&contents, "best_reward").unwrap_or(f64::NEG_INFINITY),
        reward_norm_mean: extract_f64(&contents, "reward_norm_mean").unwrap_or(0.0),
        reward_norm_var_sum: extract_f64(&contents, "reward_norm_var_sum").unwrap_or(0.0),
        reward_norm_count: extract_u64(&contents, "reward_norm_count").unwrap_or(0),
    })
}
