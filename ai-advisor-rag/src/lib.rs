use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tracing::warn;

pub mod documents;
pub mod embeddings;
pub mod expert_search;
pub mod translate;
pub mod openrouter;
pub mod handler;
pub mod ingestion;
pub mod knowledge;
pub mod models;
pub mod prompt;
pub mod vector_store;

use crate::embeddings::Embedder;
use crate::ingestion::JobState;
use crate::vector_store::VectorStore;

/// Soft per-hour cap on YouTube transcript fetches. Empirically YouTube
/// starts returning `IpBlocked` after ~30-40 rapid fetches from a residential
/// IP and the block lasts hours. We surface this number in the UI so the
/// user can see how close they are before kicking off another ingest.
pub const SAFE_FETCH_BUDGET_PER_HOUR: usize = 30;

/// Rolling window length for fetch-rate tracking.
pub const FETCH_WINDOW: Duration = Duration::from_secs(3600);

/// On-disk path for the persisted rate-limit window. The file is a single
/// JSON array of UNIX-millis timestamps. Path is overridable via the
/// `FETCH_WINDOW_PATH` env var; default is `.fetch-window.json` in CWD.
fn fetch_window_path() -> PathBuf {
    std::env::var("FETCH_WINDOW_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".fetch-window.json"))
}

/// Read the persisted fetch window from disk and prune entries older than
/// FETCH_WINDOW. Returns an empty deque if the file is missing or malformed.
pub fn load_fetch_window() -> VecDeque<SystemTime> {
    let path = fetch_window_path();
    let raw = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(_) => return VecDeque::new(),
    };
    let millis: Vec<u64> = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            warn!("malformed fetch window at {}: {e}", path.display());
            return VecDeque::new();
        }
    };
    let now = SystemTime::now();
    let mut q: VecDeque<SystemTime> = millis
        .into_iter()
        .map(|ms| UNIX_EPOCH + Duration::from_millis(ms))
        .collect();
    AppState::prune(&mut q, now);
    q
}

/// Atomically persist the rolling window to disk as a JSON array of
/// UNIX-millis timestamps. Best-effort — failures are logged, not propagated.
fn persist_fetch_window(q: &VecDeque<SystemTime>) {
    let path = fetch_window_path();
    let millis: Vec<u128> = q
        .iter()
        .filter_map(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_millis())
        .collect();
    let body = match serde_json::to_string(&millis) {
        Ok(s) => s,
        Err(e) => {
            warn!("serialize fetch window: {e}");
            return;
        }
    };
    if let Err(e) = write_atomic(&path, &body) {
        warn!("persist fetch window to {}: {e}", path.display());
    }
}

/// Write `body` to `path` atomically (write to sibling tmp file, then rename).
fn write_atomic(path: &Path, body: &str) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, body)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Shared application state available to all handlers.
pub struct AppState {
    pub experts: HashSet<String>,
    pub data_dir: PathBuf,
    pub openrouter_api_key: String,
    pub http_client: reqwest::Client,
    pub embedder: Arc<Embedder>,
    /// Caps how many chat-query embeddings run at once. Each one is CPU-bound
    /// ONNX inference on a blocking thread, and ONNX already spreads a single
    /// call across cores, so piling on more parallel calls only thrashes the
    /// CPU. Extra requests wait their turn here instead. Sized by
    /// `EMBED_CONCURRENCY` (default 2).
    pub embed_permits: Arc<tokio::sync::Semaphore>,
    pub vector_store: Arc<VectorStore>,
    /// In-memory creator-ingestion job registry. Keyed by job_id.
    /// Not persisted across restarts — see Entry 009 in CLAUDE.md.
    pub jobs: Arc<Mutex<HashMap<String, JobState>>>,
    /// Rolling window of YouTube transcript-fetch timestamps. Used to warn
    /// the user before they trip YouTube's per-IP rate limit. Pruned lazily
    /// when read or written. Persisted to disk so the window survives backend
    /// restarts (because YouTube's actual block does too).
    pub recent_fetches: Arc<Mutex<VecDeque<SystemTime>>>,
}

impl AppState {
    /// Record a fetch attempt happening "now", prune entries older than
    /// FETCH_WINDOW, and write the snapshot to disk. Called once per
    /// video-fetch by the ingestion job.
    pub fn record_fetch(&self) {
        let now = SystemTime::now();
        let mut q = self.recent_fetches.lock().expect("recent_fetches poisoned");
        Self::prune(&mut q, now);
        q.push_back(now);
        persist_fetch_window(&q);
    }

    /// Read the current state of the rolling window.
    /// Returns (count_in_window, optional minutes-until-oldest-falls-out when at/over budget).
    pub fn fetch_rate(&self) -> (usize, Option<u64>) {
        let now = SystemTime::now();
        let mut q = self.recent_fetches.lock().expect("recent_fetches poisoned");
        Self::prune(&mut q, now);
        let count = q.len();
        let mins_until_safe = if count >= SAFE_FETCH_BUDGET_PER_HOUR {
            q.front().copied().map(|first| {
                let age = now
                    .duration_since(first)
                    .unwrap_or(Duration::ZERO);
                FETCH_WINDOW
                    .saturating_sub(age)
                    .as_secs()
                    .div_ceil(60)
            })
        } else {
            None
        };
        (count, mins_until_safe)
    }

    /// Atomically reserve up to `want` fetch slots within the rolling hourly
    /// budget. Returns `Ok(granted)` (>= 1) when at least one slot is free, or
    /// `Err(minutes_until_safe)` when the window is already full.
    ///
    /// Reserved slots are recorded immediately (timestamped `now`) under the
    /// same lock, so concurrent ingest requests cannot each observe the full
    /// budget and collectively overshoot it. Reserving up front (rather than
    /// counting per completed video) is deliberately conservative: it errs
    /// toward fewer real YouTube hits, which is the safe side for rate limits.
    pub fn try_reserve_fetches(&self, want: usize) -> Result<usize, u64> {
        let now = SystemTime::now();
        let mut q = self.recent_fetches.lock().expect("recent_fetches poisoned");
        Self::prune(&mut q, now);
        let remaining = SAFE_FETCH_BUDGET_PER_HOUR.saturating_sub(q.len());
        if remaining == 0 {
            let mins = q
                .front()
                .copied()
                .map(|first| {
                    let age = now.duration_since(first).unwrap_or(Duration::ZERO);
                    FETCH_WINDOW.saturating_sub(age).as_secs().div_ceil(60)
                })
                .unwrap_or(60);
            return Err(mins);
        }
        let granted = want.clamp(1, remaining);
        for _ in 0..granted {
            q.push_back(now);
        }
        persist_fetch_window(&q);
        Ok(granted)
    }

    pub(crate) fn prune(q: &mut VecDeque<SystemTime>, now: SystemTime) {
        while let Some(&front) = q.front() {
            let age = now.duration_since(front).unwrap_or(Duration::ZERO);
            if age > FETCH_WINDOW {
                q.pop_front();
            } else {
                break;
            }
        }
    }
}
