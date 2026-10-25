use crate::train::adam::Adam;
use crate::train::config::Hyperparams;
use std::io::Write;
use std::sync::atomic::{AtomicU32, Ordering};
use tch::nn::VarStore;

/// Appends this process's git hash and hyperparameters to `{dir}/run_metadata.jsonl`.
pub fn write_run_metadata(dir: &str, game: &str, num_envs: usize, hp: &Hyperparams) {
    let _ = std::fs::create_dir_all(dir);
    let path = format!("{}/run_metadata.jsonl", dir);

    let git = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let git_hash = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let git_dirty = git(&["status", "--porcelain"]).is_some_and(|s| !s.is_empty());
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let line = format!(
        "{{\"start_unix\":{},\"game\":\"{}\",\"git_hash\":\"{}\",\"git_dirty\":{},\"num_envs\":{},\"rollout_len\":{},\"learning_rate\":{},\"lr_anneal_frames\":{},\"lr_final_frac\":{},\"gamma\":{},\"gae_lambda\":{}}}\n",
        now,
        game,
        git_hash,
        git_dirty,
        num_envs,
        hp.rollout_len,
        hp.learning_rate,
        hp.lr_anneal_frames,
        hp.lr_final_frac,
        hp.gamma,
        hp.gae_lambda
    );
    let written = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut f| f.write_all(line.as_bytes()));
    if let Err(e) = written {
        eprintln!("write_run_metadata: {}: {}", path, e);
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
    /// The learning rate in effect at the save (it anneals).
    pub learning_rate: f64,
    pub gamma: f64,
    pub gae_lambda: f64,
    // Reward normalizer state — persisted so resumes don't restart with std=1.0 and
    // re-trigger value-function divergence.
    pub reward_norm_mean: f64,
    pub reward_norm_var_sum: f64,
    pub reward_norm_count: u64,
}

/// Consecutive failed saves; after MAX_CONSECUTIVE_SAVE_FAILURES the trainer shuts down so an
/// unattended run doesn't silently lose hours of work.
static CONSECUTIVE_SAVE_FAILURES: AtomicU32 = AtomicU32::new(0);
const MAX_CONSECUTIVE_SAVE_FAILURES: u32 = 3;

fn record_save_failure(context: &str) {
    let n = CONSECUTIVE_SAVE_FAILURES.fetch_add(1, Ordering::Relaxed) + 1;
    eprintln!(
        "checkpoint save: failure #{} ({}). Threshold = {}.",
        n, context, MAX_CONSECUTIVE_SAVE_FAILURES
    );
    if n >= MAX_CONSECUTIVE_SAVE_FAILURES {
        eprintln!(
            "checkpoint save: {} consecutive failures — triggering shutdown",
            n
        );
        crate::util::shutdown::trigger();
    }
}

/// Writes `{dir}/{name}.{safetensors,adam,json}` via tmp files and renames, weights first, so a
/// crash mid-save never leaves metadata ahead of the weights.
pub fn save(dir: &str, name: &str, vs: &VarStore, opt: &Adam, meta: &CheckpointMeta) {
    let path = format!("{}/{}.safetensors", dir, name);
    // tch picks the format from the extension, so the tmp file must still end in .safetensors.
    let tmp = format!("{}/{}.tmp.safetensors", dir, name);
    if let Err(e) = vs.save(&tmp) {
        eprintln!("checkpoint save: write {} failed: {}", tmp, e);
        let _ = std::fs::remove_file(&tmp);
        record_save_failure("safetensors write");
        return;
    }
    if let Err(e) = std::fs::rename(&tmp, &path) {
        eprintln!("checkpoint save: rename {} -> {} failed: {}", tmp, path, e);
        let _ = std::fs::remove_file(&tmp);
        record_save_failure("safetensors rename");
        return;
    }

    let adam_path = format!("{}/{}.adam", dir, name);
    let adam_tmp = format!("{}.tmp", adam_path);
    let adam = opt
        .save_state(&adam_tmp)
        .map_err(|e| e.to_string())
        .and_then(|()| std::fs::rename(&adam_tmp, &adam_path).map_err(|e| e.to_string()));
    if let Err(e) = &adam {
        eprintln!("checkpoint save: adam: {}", e);
        let _ = std::fs::remove_file(&adam_tmp);
    }

    save_metadata(dir, name, meta);

    if adam.is_ok() {
        CONSECUTIVE_SAVE_FAILURES.store(0, Ordering::Relaxed);
    } else {
        record_save_failure("adam save");
    }
}

fn save_metadata(dir: &str, name: &str, meta: &CheckpointMeta) {
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

/// Removes `.tmp` files a previous run left behind by dying mid-save.
pub fn cleanup_stale_tmp(dir: &str) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let is_tmp = name.ends_with(".tmp") || name.ends_with(".tmp.safetensors");
        if is_tmp && let Err(e) = std::fs::remove_file(&path) {
            eprintln!("cleanup_stale_tmp: failed to remove {:?}: {}", path, e);
        }
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
    if let Err(e) = vs.load(&path) {
        // Older saves wrote libtorch's format under a .safetensors name; tch only reads that via another extension.
        let legacy = format!("{}/{}.legacy.ot", dir, name);
        std::fs::copy(&path, &legacy)
            .map_err(|e2| format!("load weights: {e}; copy for legacy load: {e2}"))?;
        let loaded = vs.load(&legacy);
        let _ = std::fs::remove_file(&legacy);
        loaded.map_err(|e2| format!("load weights: {e}; legacy format: {e2}"))?;
        println!("[checkpoint] loaded {path} from the legacy (non-safetensors) format");
    }
    let state = load_metadata(dir, name)
        .ok_or_else(|| format!("metadata missing or unparseable: {}/{}.json", dir, name))?;
    Ok(Some(state))
}

/// Reads the metadata `save` wrote; None if the file is missing or a required field is unparseable.
pub fn load_metadata(dir: &str, name: &str) -> Option<ResumedState> {
    let contents = std::fs::read_to_string(format!("{}/{}.json", dir, name)).ok()?;
    // The writer above puts one field per line, so this needs no JSON parser.
    fn field<T: std::str::FromStr>(s: &str, key: &str) -> Option<T> {
        let pat = format!("\"{}\":", key);
        let rest = &s[s.find(&pat)? + pat.len()..];
        rest.split([',', '\n', '}']).next()?.trim().parse().ok()
    }
    Some(ResumedState {
        episode: field(&contents, "episode")?,
        total_frames: field(&contents, "total_frames")?,
        ppo_updates: field(&contents, "ppo_updates")?,
        best_reward: field(&contents, "best_reward").unwrap_or(f64::NEG_INFINITY),
        reward_norm_mean: field(&contents, "reward_norm_mean").unwrap_or(0.0),
        reward_norm_var_sum: field(&contents, "reward_norm_var_sum").unwrap_or(0.0),
        reward_norm_count: field(&contents, "reward_norm_count").unwrap_or(0),
    })
}
