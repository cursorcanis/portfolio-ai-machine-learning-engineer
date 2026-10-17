use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Utc};
use serde::Serialize;
use tokio::process::Command;
use tracing::{info, warn};

use crate::embeddings::Embedder;
use crate::vector_store::{UpsertItem, VectorStore};

const CHUNK_CHARS: usize = 500;
const CHUNK_OVERLAP: usize = 50;
const EMBED_BATCH: usize = 32;
/// Sleep between per-video transcript fetches. YouTube's anti-scraping rate
/// limit kicks in after ~30 fast hits from one IP and lasts hours; a 2-second
/// pace keeps a 10-video ingest under that radar.
const FETCH_THROTTLE_MS: u64 = 2000;

/// A single video discovered on a creator's channel.
#[derive(Debug, Clone)]
pub struct VideoInfo {
    pub id: String,
    pub title: String,
    /// Channel name as reported by yt-dlp; used to decide whether the
    /// title-relevance filter should fire (skipped on personal channels).
    pub channel: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Pending,
    Running,
    Done,
    Failed,
}

/// Progress record for an in-flight or finished creator-index job.
#[derive(Debug, Clone, Serialize)]
pub struct JobState {
    pub job_id: String,
    pub handle: String,
    pub status: JobStatus,
    pub videos_total: usize,
    pub videos_done: usize,
    pub chunks_indexed: usize,
    pub started_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
    /// Human-readable status line for work that isn't video-shaped — document
    /// uploads use it to report the detected language and translation
    /// progress. `None` for jobs that have nothing extra to say.
    #[serde(default)]
    pub note: Option<String>,
}

impl JobState {
    pub fn new(job_id: String, handle: String) -> Self {
        Self {
            job_id,
            handle,
            status: JobStatus::Pending,
            videos_total: 0,
            videos_done: 0,
            chunks_indexed: 0,
            started_at: Utc::now(),
            finished_at: None,
            error: None,
            note: None,
        }
    }
}

/// Turn whatever the user typed into a URL yt-dlp can enumerate.
///
/// Accepts: bare handle ("aiscout"), @-prefixed handle ("@aiscout"), full channel URL
/// (`/channel/UCxxx` or `/@xxx`). Always returns a URL ending in `/videos` so
/// yt-dlp lists the Videos tab only — not shorts, community posts, or the
/// channel home page (which can include all of the above and ignore
/// `--playlist-end`, blowing past the requested max_videos cap).
pub fn normalize_channel_url(handle: &str) -> String {
    let h = handle.trim();
    let base = if h.starts_with("http://") || h.starts_with("https://") {
        h.trim_end_matches('/').to_string()
    } else {
        let stripped = h.strip_prefix('@').unwrap_or(h);
        format!("https://www.youtube.com/@{stripped}")
    };
    if base.ends_with("/videos") || base.contains("/videos?") {
        base
    } else {
        format!("{base}/videos")
    }
}

/// Ask yt-dlp to enumerate the first `max` videos on a channel.
pub async fn list_videos(
    handle: &str,
    max: usize,
) -> Result<Vec<VideoInfo>> {
    let url = normalize_channel_url(handle);
    info!("Listing videos for {url} (max {max})");

    let mut args: Vec<String> = Vec::new();
    args.extend([
        "--flat-playlist".into(),
        "--dump-json".into(),
        "--playlist-end".into(),
        max.to_string(),
        url,
    ]);

    let output = Command::new("yt-dlp")
        .args(&args)
        .output()
        .await
        .with_context(|| "failed to spawn yt-dlp (is it on PATH?)")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!(
            "yt-dlp exited with status {}: {}",
            output.status,
            stderr.lines().last().unwrap_or("").trim()
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut videos = Vec::new();
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let id = v.get("id").and_then(|x| x.as_str()).unwrap_or("").to_string();
        let title = v
            .get("title")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let channel = v
            .get("channel")
            .and_then(|x| x.as_str())
            .map(str::to_string)
            .filter(|s| !s.is_empty());
        if !id.is_empty() {
            videos.push(VideoInfo { id, title, channel });
        }
    }
    Ok(videos)
}

/// Tokenize a name for the relevance filter. Lowercased words ≥3 chars,
/// punctuation stripped. "Dana Whitfield" -> ["dana", "whitfield"].
/// "J. R. R. Tolkien" -> ["tolkien"].
fn name_tokens(name: &str) -> Vec<String> {
    name.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|t| t.chars().count() >= 3)
        .collect()
}

/// True if `haystack` (lowercased) contains any of the given lowercased tokens.
fn contains_any_token(haystack: &str, tokens: &[String]) -> bool {
    let h = haystack.to_lowercase();
    tokens.iter().any(|t| h.contains(t.as_str()))
}

/// Heuristically classify a transcript as a single-narrator monologue
/// (`"single"`) or a multi-speaker conversation (`"multiple"`).
///
/// YouTube transcripts carry no speaker diarization, so this relies on
/// textual evidence that more than one person is talking:
///   - `>>` — the standard closed-caption marker for a speaker change
///   - `NAME:` — all-caps speaker labels some manual captions use
///   - interview/podcast phrases ("thanks for having me", "my guest today")
/// Absent positive evidence of a second voice, we default to `"single"`,
/// which matches the most common ingested shape (a creator's own videos).
pub fn classify_speaker_mode(text: &str) -> &'static str {
    let speaker_arrows = text.matches(">>").count();

    // "JOE:", "INTERVIEWER:" — at least 3 uppercase letters then a colon.
    let caps_labels = text
        .split_whitespace()
        .filter(|w| {
            let Some(stem) = w.strip_suffix(':') else {
                return false;
            };
            stem.len() >= 3 && stem.chars().all(|c| c.is_ascii_uppercase())
        })
        .count();

    let lower = text.to_lowercase();
    const STRONG_CUES: [&str; 6] = [
        "thanks for having me",
        "thank you for having me",
        "for having me on",
        "my guest today",
        "please welcome",
        "joining us today",
    ];
    const WEAK_CUES: [&str; 7] = [
        "my guest",
        "our guest",
        "joining me",
        "joining us",
        "great to be here",
        "good to be here",
        "pleasure to be here",
    ];
    let strong_hits = STRONG_CUES.iter().filter(|p| lower.contains(*p)).count();
    let weak_hits = WEAK_CUES.iter().filter(|p| lower.contains(*p)).count();

    if speaker_arrows >= 3 || caps_labels >= 4 || strong_hits >= 1 || weak_hits >= 2 {
        "multiple"
    } else {
        "single"
    }
}

/// Resolve `scripts/fetch_transcript.py` relative to the running binary.
///
/// Cargo runs from the crate root, so a plain relative path works in dev. In
/// production this can be overridden with `TRANSCRIPT_FETCHER` if the binary
/// is moved away from the source tree.
fn fetch_transcript_script() -> PathBuf {
    if let Ok(p) = std::env::var("TRANSCRIPT_FETCHER") {
        if !p.trim().is_empty() {
            return PathBuf::from(p);
        }
    }
    PathBuf::from("scripts").join("fetch_transcript.py")
}

/// Outcome of a per-video transcript fetch.
///
/// `Missing` carries *why* nothing came back, because the two cases need
/// different advice: captions switched off by the uploader is a property of
/// that video (pick another), while no usable track in any language is rarer
/// and worth reporting separately.
#[derive(Debug)]
pub enum TranscriptOutcome {
    Found(FetchedTranscript),
    /// One of the `no_transcript` categories: `captions_disabled` or
    /// `language_unavailable`.
    Missing(&'static str),
}

#[derive(Debug)]
pub struct FetchedTranscript {
    pub text: String,
    /// Language of the returned text (post-translation when translated).
    pub language: Option<String>,
    /// Language of the source captions before any translation.
    pub original_language: Option<String>,
    /// Whether YouTube translated the track into the corpus language.
    pub translated: bool,
}

/// Fetch a single video's transcript via the Python sidecar
/// (`youtube-transcript-api`).
///
/// Language handling lives in the sidecar: it prefers English, falls back to
/// whatever track the video actually has, and asks YouTube to translate a
/// foreign track into English so the text matches the English-only retrieval
/// embedder. See `scripts/fetch_transcript.py`.
///
/// We use the Python lib instead of yt-dlp + VTT because it hits YouTube's
/// timed-text endpoint directly, which proved more reliable than yt-dlp's
/// subtitle download for plain caption text.
pub async fn fetch_transcript(
    video_id: &str,
) -> Result<TranscriptOutcome> {
    let script = fetch_transcript_script();

    let mut cmd = Command::new("python3");
    cmd.arg(&script).arg(video_id);
    let output = cmd
        .output()
        .await
        .with_context(|| {
            format!(
                "failed to spawn python3 (script={}). Install with: pip install --user --break-system-packages youtube-transcript-api",
                script.display()
            )
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!(
            "transcript fetcher exited with status {}: {}",
            output.status,
            stderr.trim()
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.lines().last().unwrap_or("").trim();
    if line.is_empty() {
        return Err(anyhow!("transcript fetcher produced no output"));
    }
    let v: serde_json::Value = serde_json::from_str(line)
        .with_context(|| format!("transcript fetcher returned invalid JSON: {line}"))?;

    let status = v.get("status").and_then(|x| x.as_str()).unwrap_or("");
    match status {
        "ok" => {
            let text = v.get("text").and_then(|x| x.as_str()).unwrap_or("").to_string();
            if text.trim().is_empty() {
                return Ok(TranscriptOutcome::Missing("empty_transcript"));
            }
            let language = v.get("language").and_then(|x| x.as_str()).map(str::to_string);
            let original_language = v
                .get("original_language")
                .and_then(|x| x.as_str())
                .map(str::to_string);
            let translated = v.get("translated").and_then(|x| x.as_bool()).unwrap_or(false);
            // A failed translation still yields usable text, but the corpus
            // ends up with a foreign-language chunk the English embedder
            // handles poorly - surface it rather than letting it pass silently.
            if let Some(err) = v.get("translation_error").and_then(|x| x.as_str()) {
                warn!(
                    "[{video_id}] translation to English failed ({err}); indexing original {} text",
                    original_language.as_deref().unwrap_or("source-language")
                );
            }
            Ok(TranscriptOutcome::Found(FetchedTranscript {
                text,
                language,
                original_language,
                translated,
            }))
        }
        "no_transcript" => {
            let reason = v.get("reason").and_then(|x| x.as_str()).unwrap_or("");
            warn!("[{video_id}] no transcript: {reason}");
            let category = match reason {
                "captions_disabled" => "captions_disabled",
                "language_unavailable" => "language_unavailable",
                // Older sidecar builds reported the raw exception name.
                "TranscriptsDisabled" => "captions_disabled",
                "NoTranscriptFound" => "language_unavailable",
                _ => "no_transcript",
            };
            Ok(TranscriptOutcome::Missing(category))
        }
        _ => {
            let reason = v.get("reason").and_then(|x| x.as_str()).unwrap_or("unknown");
            Err(anyhow!("transcript fetcher: {reason}"))
        }
    }
}

/// Strip VTT cue timings, tags, and header lines; dedupe consecutive repeats.
///
/// YouTube auto-subs repeat each cue several times with rolling word highlights,
/// so naively concatenating lines produces huge duplicated blocks. We drop any
/// line identical to the previous one after tag-stripping.
pub fn parse_vtt(vtt: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut last_kept: Option<String> = None;

    for raw in vtt.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with("WEBVTT")
            || line.starts_with("Kind:")
            || line.starts_with("Language:")
            || line.starts_with("NOTE")
            || line.starts_with("STYLE")
        {
            continue;
        }
        if line.contains("-->") {
            continue;
        }
        if line.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let cleaned = strip_vtt_tags(line);
        let cleaned = cleaned.trim().to_string();
        if cleaned.is_empty() {
            continue;
        }
        if last_kept.as_deref() == Some(cleaned.as_str()) {
            continue;
        }
        last_kept = Some(cleaned.clone());
        out.push(cleaned);
    }

    out.join(" ")
}

/// Remove inline VTT tags: timestamp anchors (`<00:00:05.220>`), class spans (`<c>...</c>`), etc.
fn strip_vtt_tags(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut in_tag = false;
    for c in line.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out
}

/// Split text into overlapping char windows. Mirrors the chunker in migrate_kb.rs.
pub fn chunk_text(text: &str, size: usize, overlap: usize) -> Vec<String> {
    assert!(overlap < size, "overlap must be smaller than size");
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return Vec::new();
    }
    let step = size - overlap;
    let mut out = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let end = (start + size).min(chars.len());
        let slice: String = chars[start..end].iter().collect();
        let trimmed = slice.trim().to_string();
        if !trimmed.is_empty() {
            out.push(trimmed);
        }
        if end == chars.len() {
            break;
        }
        start += step;
    }
    out
}

/// Outcome of an ingestion run — richer than just a chunk count so the
/// caller can build a useful job-state error when 0 chunks landed.
#[derive(Debug, Default)]
pub struct IndexSummary {
    pub chunks: usize,
    /// Number of videos actually attempted (after the title-relevance
    /// filter, if any). What `videos_done` in the job state counts up to.
    pub videos_total: usize,
    /// Number of videos yt-dlp originally enumerated, before any filtering.
    /// When `videos_total < videos_listed`, the title-relevance filter
    /// dropped the difference; when both are 0, yt-dlp itself returned
    /// nothing (channel doesn't exist, IP blocked at the listing step,
    /// or the channel really has no videos in its Videos tab).
    pub videos_listed: usize,
    /// Whether the Wikipedia baseline was successfully ingested for this
    /// expert. `None` = not attempted; `Some(true)` = chunks indexed;
    /// `Some(false)` = attempted but no usable article found.
    pub wikipedia_ingested: Option<bool>,
    /// Categorized failure reasons across all per-video attempts. Keys are
    /// short stable categories: `rate_limited`, `captions_disabled`,
    /// `language_unavailable`, `empty_transcript`, `no_transcript`, `other`.
    pub errors: BTreeMap<&'static str, usize>,
}

impl IndexSummary {
    /// One-line user-facing reason for a 0-chunk run. Picks the dominant
    /// failure category and translates it into a concrete next step.
    pub fn empty_run_message(&self) -> String {
        let total: usize = self.errors.values().sum();
        // No per-video errors recorded means we never reached the transcript
        // fetcher. Either the channel listing returned nothing, or the
        // title-relevance filter ate everything.
        if total == 0 {
            if self.videos_listed == 0 {
                return "yt-dlp returned 0 videos for this channel. Check the channel URL, try a different candidate, or verify yt-dlp can reach YouTube (VPN/IP block?).".to_string();
            }
            if self.videos_total == 0 && self.videos_listed > 0 {
                return format!(
                    "{} videos listed but the title-relevance filter dropped them all. The channel name didn't match the expert and no video titles contained the expert's name. Try a different channel candidate, or the channel may belong to the expert but use topic-only titles.",
                    self.videos_listed
                );
            }
            return format!(
                "{} videos attempted but none produced chunks (transcripts empty or unavailable).",
                self.videos_total
            );
        }
        let dominant = self
            .errors
            .iter()
            .max_by_key(|(_, count)| **count)
            .map(|(cat, _)| *cat)
            .unwrap_or("other");
        let breakdown = self
            .errors
            .iter()
            .map(|(k, v)| format!("{v} {k}"))
            .collect::<Vec<_>>()
            .join(", ");
        match dominant {
            "rate_limited" => format!(
                "All {total} videos rate-limited by YouTube. Wait a few hours, then retry with a smaller max_videos (3-5). Breakdown: {breakdown}"
            ),
            // The uploader turned captions off. Not a language problem and not
            // a bad channel pick - any other video from anyone would work.
            "captions_disabled" => format!(
                "Captions are switched off for {total} of the videos, so there is nothing to transcribe (any language). Pick a video that shows a CC badge on YouTube. Breakdown: {breakdown}"
            ),
            "language_unavailable" => format!(
                "{total} videos expose no caption track in any language. Non-English videos are supported and auto-translated, so this means the tracks are genuinely absent. Breakdown: {breakdown}"
            ),
            "empty_transcript" => format!(
                "{total} videos returned an empty transcript (captions exist but contain no text). Breakdown: {breakdown}"
            ),
            "no_transcript" => format!(
                "None of the {total} videos have transcripts available — try a different channel candidate. Breakdown: {breakdown}"
            ),
            _ => format!("0 chunks indexed across {total} videos. Breakdown: {breakdown}"),
        }
    }
}

/// Classify a fetch_transcript error string into a stable category for the summary.
fn classify_error(err: &str) -> &'static str {
    if err.contains("IpBlocked") || err.contains("RequestBlocked") {
        "rate_limited"
    } else {
        "other"
    }
}

/// Full ingestion pipeline for one creator: enumerate → fetch subs → chunk →
/// embed → upsert. Progress is reported via `update_progress`, which the caller
/// wires into shared job state. Returns a summary including chunk count and
/// per-category failure breakdown.
///
/// `display_name` is what gets stored as the `advisor` in Qdrant payload
/// (e.g. "Dana Whitfield") — it's what the chat retrieval surfaces and
/// what the prompt uses to group chunks. `handle` is the YouTube identifier
/// (`@whitfield` or similar) used by yt-dlp to enumerate videos. The two are
/// the same when an /api/creators caller doesn't disambiguate.
pub async fn index_creator<F>(
    display_name: &str,
    handle: &str,
    max_videos: usize,
    embedder: Arc<Embedder>,
    vector_store: Arc<VectorStore>,
    mut update_progress: F,
) -> Result<IndexSummary>
where
    F: FnMut(&JobProgress) + Send,
{
    let raw_videos = list_videos(handle, max_videos).await?;
    let videos_listed = raw_videos.len();

    // Title-relevance filter — only fires when we have positive evidence
    // the channel is *not* personal (i.e. an aggregator/archive channel
    // like "Old Tonight Show Clips" indexed under "Lenny Bruce"). When the
    // channel name is unknown (yt-dlp didn't return one), we trust the
    // channel — the user picked it from the candidate list, and false-
    // positive filtering would drop legit videos with topic-only titles
    // (e.g. Mo Bitar's videos on his own channel that don't say his
    // name in the title). Per Beads ai_advisor-4y9.
    let tokens = name_tokens(display_name);
    let channel_name = raw_videos
        .iter()
        .find_map(|v| v.channel.clone())
        .unwrap_or_default();
    let channel_known = !channel_name.is_empty();
    let channel_matches_name =
        channel_known && !tokens.is_empty() && contains_any_token(&channel_name, &tokens);
    // Only filter when we KNOW the channel is foreign to the expert.
    let should_filter = channel_known && !channel_matches_name && !tokens.is_empty();

    let videos: Vec<VideoInfo> = if should_filter {
        let kept: Vec<VideoInfo> = raw_videos
            .into_iter()
            .filter(|v| contains_any_token(&v.title, &tokens))
            .collect();
        info!(
            "[{display_name}] archive-channel detected ('{channel_name}'); title-relevance filter kept {}/{}",
            kept.len(),
            videos_listed
        );
        kept
    } else {
        if !channel_known {
            info!(
                "[{display_name}] channel name unknown from yt-dlp; trusting candidate, no title filter"
            );
        }
        raw_videos
    };

    update_progress(&JobProgress::Listed {
        total: videos.len(),
    });
    let mut summary = IndexSummary {
        videos_total: videos.len(),
        videos_listed,
        ..Default::default()
    };
    if videos.is_empty() {
        return Ok(summary);
    }

    let mut items: Vec<UpsertItem> = Vec::new();
    let mut texts: Vec<String> = Vec::new();

    for (i, v) in videos.iter().enumerate() {
        // Pace requests so we don't trip YouTube's per-IP transcript-API rate limit.
        if i > 0 {
            tokio::time::sleep(Duration::from_millis(FETCH_THROTTLE_MS)).await;
        }
        match fetch_transcript(&v.id).await {
            Ok(TranscriptOutcome::Found(t)) => {
                let transcript = t.text;
                let speaker_mode = classify_speaker_mode(&transcript);
                let chunks = chunk_text(&transcript, CHUNK_CHARS, CHUNK_OVERLAP);
                let lang_note = if t.translated {
                    format!(
                        " [translated {}->{}]",
                        t.original_language.as_deref().unwrap_or("?"),
                        t.language.as_deref().unwrap_or("en")
                    )
                } else {
                    String::new()
                };
                info!(
                    "[{handle}] video {}/{}: {} ({}) -> {} chunks (speakers: {speaker_mode}){lang_note}",
                    i + 1,
                    videos.len(),
                    v.title,
                    v.id,
                    chunks.len()
                );
                for (idx, chunk) in chunks.into_iter().enumerate() {
                    texts.push(chunk.clone());
                    items.push(UpsertItem {
                        key: format!("creator::{handle}::{}::{idx}", v.id),
                        vector: Vec::new(),
                        source_type: "creator_transcript".into(),
                        advisor: display_name.to_string(),
                        topic: v.title.clone(),
                        content: chunk,
                        chunk_idx: Some(idx as u32),
                        video_id: Some(v.id.clone()),
                        keywords: Vec::new(),
                        speaker_mode: Some(speaker_mode.to_string()),
                    });
                }
            }
            Ok(TranscriptOutcome::Missing(reason)) => {
                info!("[{handle}] video {}: {reason}, skipping", v.id);
                *summary.errors.entry(reason).or_insert(0) += 1;
            }
            Err(e) => {
                let msg = e.to_string();
                let category = classify_error(&msg);
                *summary.errors.entry(category).or_insert(0) += 1;
                warn!("[{handle}] video {}: {msg}", v.id);
            }
        }
        update_progress(&JobProgress::VideoDone { done: i + 1 });
    }

    if texts.is_empty() {
        return Ok(summary);
    }

    // Batch embed
    info!("[{handle}] embedding {} chunks...", texts.len());
    let embedder_clone = embedder.clone();
    let texts_for_embed = texts.clone();
    let vectors: Vec<Vec<f32>> = tokio::task::spawn_blocking(move || {
        let mut out = Vec::with_capacity(texts_for_embed.len());
        for batch in texts_for_embed.chunks(EMBED_BATCH) {
            let v = embedder_clone.embed_batch(batch.to_vec())?;
            out.extend(v);
        }
        anyhow::Ok(out)
    })
    .await??;

    for (item, vec) in items.iter_mut().zip(vectors.into_iter()) {
        item.vector = vec;
    }

    let count = items.len();
    vector_store.upsert_batch(items).await?;

    update_progress(&JobProgress::Indexed { chunks: count });

    summary.chunks = count;
    Ok(summary)
}

/// Extract an 11-char YouTube video id from a URL or bare id. Handles
/// `watch?v=`, `youtu.be/`, `/shorts/`, `/embed/`, `/live/`, and a bare id.
/// Returns None for channel/playlist URLs (which go through `index_creator`).
pub fn extract_video_id(input: &str) -> Option<String> {
    let s = input.trim();
    let is_id_char = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '-';
    let take_id = |rest: &str| -> Option<String> {
        let id: String = rest.chars().take_while(|&c| is_id_char(c)).collect();
        if id.len() == 11 {
            Some(id)
        } else {
            None
        }
    };
    if let Some(pos) = s.find("v=") {
        if let Some(id) = take_id(&s[pos + 2..]) {
            return Some(id);
        }
    }
    if let Some(pos) = s.find("youtu.be/") {
        if let Some(id) = take_id(&s[pos + "youtu.be/".len()..]) {
            return Some(id);
        }
    }
    for pat in ["/shorts/", "/embed/", "/live/"] {
        if let Some(pos) = s.find(pat) {
            if let Some(id) = take_id(&s[pos + pat.len()..]) {
                return Some(id);
            }
        }
    }
    if s.len() == 11 && s.chars().all(is_id_char) {
        return Some(s.to_string());
    }
    None
}

/// Best-effort fetch of a single video's title + channel via yt-dlp `--print`.
/// Returns (title, channel); either may be None. Bounded so a slow YouTube
/// can't stall the ingest job.
pub async fn fetch_video_meta(
    video_id: &str,
) -> (Option<String>, Option<String>) {
    let url = format!("https://www.youtube.com/watch?v={video_id}");
    let mut args: Vec<String> = Vec::new();
    args.extend([
        "--skip-download".into(),
        "--no-warnings".into(),
        "--socket-timeout".into(),
        "8".into(),
        "--print".into(),
        "%(title)s\n%(channel)s".into(),
        url,
    ]);
    let fut = Command::new("yt-dlp").args(&args).kill_on_drop(true).output();
    let output = match tokio::time::timeout(Duration::from_secs(20), fut).await {
        Ok(Ok(o)) if o.status.success() => o,
        _ => return (None, None),
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut lines = stdout.lines();
    let clean = |s: Option<&str>| {
        s.map(str::trim)
            .filter(|t| !t.is_empty() && *t != "NA")
            .map(str::to_string)
    };
    (clean(lines.next()), clean(lines.next()))
}

/// Ingest a single YouTube video (by id) under `display_name`. Same
/// transcript → chunk → embed → upsert path as `index_creator`, for the
/// "paste a video URL" flow. `title` becomes the chunk topic.
pub async fn index_single_video<F>(
    display_name: &str,
    video_id: &str,
    title: &str,
    embedder: Arc<Embedder>,
    vector_store: Arc<VectorStore>,
    mut update_progress: F,
) -> Result<IndexSummary>
where
    F: FnMut(&JobProgress) + Send,
{
    update_progress(&JobProgress::Listed { total: 1 });
    let mut summary = IndexSummary {
        videos_total: 1,
        videos_listed: 1,
        ..Default::default()
    };

    let transcript = match fetch_transcript(video_id).await {
        Ok(TranscriptOutcome::Found(t)) => {
            if t.translated {
                info!(
                    "[url] video {video_id}: translated {}->{} for indexing",
                    t.original_language.as_deref().unwrap_or("?"),
                    t.language.as_deref().unwrap_or("en")
                );
            }
            t.text
        }
        Ok(TranscriptOutcome::Missing(reason)) => {
            *summary.errors.entry(reason).or_insert(0) += 1;
            update_progress(&JobProgress::VideoDone { done: 1 });
            return Ok(summary);
        }
        Err(e) => {
            let cat = classify_error(&e.to_string());
            *summary.errors.entry(cat).or_insert(0) += 1;
            update_progress(&JobProgress::VideoDone { done: 1 });
            return Ok(summary);
        }
    };
    update_progress(&JobProgress::VideoDone { done: 1 });

    let speaker_mode = classify_speaker_mode(&transcript);
    let chunks = chunk_text(&transcript, CHUNK_CHARS, CHUNK_OVERLAP);
    if chunks.is_empty() {
        return Ok(summary);
    }
    info!(
        "[url] video {} ({}) -> {} chunks under '{display_name}' (speakers: {speaker_mode})",
        title,
        video_id,
        chunks.len()
    );

    let mut items: Vec<UpsertItem> = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    for (idx, chunk) in chunks.into_iter().enumerate() {
        texts.push(chunk.clone());
        items.push(UpsertItem {
            key: format!("creator::url::{video_id}::{idx}"),
            vector: Vec::new(),
            source_type: "creator_transcript".into(),
            advisor: display_name.to_string(),
            topic: title.to_string(),
            content: chunk,
            chunk_idx: Some(idx as u32),
            video_id: Some(video_id.to_string()),
            keywords: Vec::new(),
            speaker_mode: Some(speaker_mode.to_string()),
        });
    }

    let embedder_clone = embedder.clone();
    let texts_for_embed = texts.clone();
    let vectors: Vec<Vec<f32>> = tokio::task::spawn_blocking(move || {
        let mut out = Vec::with_capacity(texts_for_embed.len());
        for batch in texts_for_embed.chunks(EMBED_BATCH) {
            out.extend(embedder_clone.embed_batch(batch.to_vec())?);
        }
        anyhow::Ok(out)
    })
    .await??;
    for (item, vec) in items.iter_mut().zip(vectors.into_iter()) {
        item.vector = vec;
    }

    let count = items.len();
    vector_store.upsert_batch(items).await?;
    update_progress(&JobProgress::Indexed { chunks: count });
    summary.chunks = count;
    Ok(summary)
}

/// Best-effort metadata for any yt-dlp-supported URL: returns (id, title,
/// channel). Used for non-YouTube sources (e.g. Rumble) that don't go through
/// the YouTube transcript API.
pub async fn fetch_url_meta(
    url: &str,
) -> (Option<String>, Option<String>, Option<String>) {
    let mut args: Vec<String> = Vec::new();
    args.extend([
        "--skip-download".into(),
        "--no-warnings".into(),
        "--socket-timeout".into(),
        "10".into(),
        "--print".into(),
        "%(id)s\n%(title)s\n%(channel)s".into(),
        url.to_string(),
    ]);
    let fut = Command::new("yt-dlp").args(&args).kill_on_drop(true).output();
    let output = match tokio::time::timeout(Duration::from_secs(25), fut).await {
        Ok(Ok(o)) if o.status.success() => o,
        _ => return (None, None, None),
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut lines = stdout.lines();
    let clean = |s: Option<&str>| {
        s.map(str::trim)
            .filter(|t| !t.is_empty() && *t != "NA")
            .map(str::to_string)
    };
    (clean(lines.next()), clean(lines.next()), clean(lines.next()))
}

/// Download available subtitles for a single video via yt-dlp and return the
/// plain text (VTT parsed + deduped). Works for any yt-dlp-supported site;
/// used for Rumble, which has no transcript API. Returns Ok(None) when the
/// video simply has no subtitles (the common Rumble case).
pub async fn fetch_subtitles_via_ytdlp(
    url: &str,
) -> Result<Option<String>> {
    let tmp = std::env::temp_dir().join(format!("aiadv-subs-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&tmp).context("failed to create temp subs dir")?;
    let out_tmpl = tmp.join("%(id)s.%(ext)s");

    let mut args: Vec<String> = Vec::new();
    args.extend([
        "--skip-download".into(),
        "--write-subs".into(),
        "--write-auto-subs".into(),
        // English first, but accept any language the site offers - a
        // foreign-language caption track is far better than none. (No
        // translation is available on this path; yt-dlp only downloads what
        // the site publishes.)
        "--sub-langs".into(),
        "en.*,en,all,-live_chat".into(),
        "--convert-subs".into(),
        "vtt".into(),
        "--no-warnings".into(),
        "--socket-timeout".into(),
        "10".into(),
        "-o".into(),
        out_tmpl.to_string_lossy().into_owned(),
        url.to_string(),
    ]);

    let fut = Command::new("yt-dlp").args(&args).kill_on_drop(true).output();
    let output = match tokio::time::timeout(Duration::from_secs(45), fut).await {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => {
            let _ = std::fs::remove_dir_all(&tmp);
            return Err(anyhow!("yt-dlp subtitle spawn failed: {e}"));
        }
        Err(_) => {
            let _ = std::fs::remove_dir_all(&tmp);
            return Err(anyhow!("yt-dlp subtitle download timed out"));
        }
    };

    // Parse the first non-empty .vtt we find. Now that we ask for every
    // language, sort English files first so a multi-language video still
    // prefers English over an arbitrary directory order.
    let mut vtts: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&tmp) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) == Some("vtt") {
                vtts.push(p);
            }
        }
    }
    vtts.sort_by_key(|p| {
        let name = p.file_name().and_then(|x| x.to_str()).unwrap_or("");
        // yt-dlp names them "<id>.<lang>.vtt", so ".en" marks an English track.
        if name.contains(".en.") || name.contains(".en-") {
            0
        } else {
            1
        }
    });
    let mut text = None;
    for p in vtts {
        if let Ok(vtt) = std::fs::read_to_string(&p) {
            let parsed = parse_vtt(&vtt);
            if !parsed.trim().is_empty() {
                text = Some(parsed);
                break;
            }
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);

    if text.is_none() && !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!(
            "yt-dlp subtitles: {}",
            stderr.lines().last().unwrap_or("").trim()
        ));
    }
    Ok(text)
}

/// Ingest a single non-YouTube video (e.g. Rumble) by URL, using yt-dlp's
/// subtitle download (no transcript API exists for these sites). When the
/// video has no captions, returns a summary with a `no_transcript` error so
/// the caller can tell the user plainly.
pub async fn index_url_via_subtitles<F>(
    display_name: &str,
    url: &str,
    embedder: Arc<Embedder>,
    vector_store: Arc<VectorStore>,
    mut update_progress: F,
) -> Result<IndexSummary>
where
    F: FnMut(&JobProgress) + Send,
{
    update_progress(&JobProgress::Listed { total: 1 });
    let mut summary = IndexSummary {
        videos_total: 1,
        videos_listed: 1,
        ..Default::default()
    };

    let (id, title, _channel) = fetch_url_meta(url).await;
    let video_id = id.unwrap_or_else(|| {
        // Fallback id: a sanitized tail of the URL.
        let tail = url.rsplit('/').next().unwrap_or("video");
        tail.chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .take(40)
            .collect()
    });
    let topic = title.unwrap_or_else(|| "Rumble video".to_string());

    let transcript = match fetch_subtitles_via_ytdlp(url).await {
        Ok(Some(t)) => t,
        Ok(None) => {
            *summary.errors.entry("no_transcript").or_insert(0) += 1;
            update_progress(&JobProgress::VideoDone { done: 1 });
            return Ok(summary);
        }
        Err(e) => {
            let cat = classify_error(&e.to_string());
            *summary.errors.entry(cat).or_insert(0) += 1;
            warn!("[rumble] {url}: {e}");
            update_progress(&JobProgress::VideoDone { done: 1 });
            return Ok(summary);
        }
    };
    update_progress(&JobProgress::VideoDone { done: 1 });

    let speaker_mode = classify_speaker_mode(&transcript);
    let chunks = chunk_text(&transcript, CHUNK_CHARS, CHUNK_OVERLAP);
    if chunks.is_empty() {
        return Ok(summary);
    }
    info!(
        "[rumble] {} ({}) -> {} chunks under '{display_name}' (speakers: {speaker_mode})",
        topic,
        video_id,
        chunks.len()
    );

    let mut items: Vec<UpsertItem> = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    for (idx, chunk) in chunks.into_iter().enumerate() {
        texts.push(chunk.clone());
        items.push(UpsertItem {
            key: format!("creator::rumble::{video_id}::{idx}"),
            vector: Vec::new(),
            source_type: "creator_transcript".into(),
            advisor: display_name.to_string(),
            topic: topic.clone(),
            content: chunk,
            chunk_idx: Some(idx as u32),
            video_id: Some(video_id.clone()),
            keywords: Vec::new(),
            speaker_mode: Some(speaker_mode.to_string()),
        });
    }

    let embedder_clone = embedder.clone();
    let texts_for_embed = texts.clone();
    let vectors: Vec<Vec<f32>> = tokio::task::spawn_blocking(move || {
        let mut out = Vec::with_capacity(texts_for_embed.len());
        for batch in texts_for_embed.chunks(EMBED_BATCH) {
            out.extend(embedder_clone.embed_batch(batch.to_vec())?);
        }
        anyhow::Ok(out)
    })
    .await??;
    for (item, vec) in items.iter_mut().zip(vectors.into_iter()) {
        item.vector = vec;
    }

    let count = items.len();
    vector_store.upsert_batch(items).await?;
    update_progress(&JobProgress::Indexed { chunks: count });
    summary.chunks = count;
    Ok(summary)
}

/// Incremental progress events emitted during `index_creator`.
pub enum JobProgress {
    Listed { total: usize },
    VideoDone { done: usize },
    Indexed { chunks: usize },
}

/// Ingest a Wikipedia article body as Qdrant chunks under
/// `source_type=wikipedia_article`, `advisor=display_name`. Returns the
/// chunk count (0 means article was empty or not found).
///
/// Used as the on-topic baseline for Beads `ai_advisor-4y9`: for figures
/// without a personal YouTube channel, Wikipedia gives us guaranteed
/// expert-relevant content even when archive-channel YouTube videos turn
/// out to be off-topic.
pub async fn index_wikipedia(
    display_name: &str,
    wikipedia_title: &str,
    article_text: &str,
    embedder: Arc<Embedder>,
    vector_store: Arc<VectorStore>,
) -> Result<usize> {
    let chunks = chunk_text(article_text, CHUNK_CHARS, CHUNK_OVERLAP);
    if chunks.is_empty() {
        return Ok(0);
    }
    info!(
        "[{display_name}] Wikipedia article '{wikipedia_title}' -> {} chunks",
        chunks.len()
    );

    let mut items: Vec<UpsertItem> = Vec::with_capacity(chunks.len());
    let texts: Vec<String> = chunks.clone();

    for (idx, chunk) in chunks.into_iter().enumerate() {
        items.push(UpsertItem {
            // Stable per-(advisor, idx) — re-running the ingest overwrites
            // rather than duplicates.
            key: format!("wiki::{display_name}::{idx}"),
            vector: Vec::new(),
            source_type: "wikipedia_article".into(),
            advisor: display_name.to_string(),
            topic: format!("Wikipedia: {wikipedia_title}"),
            content: chunk,
            chunk_idx: Some(idx as u32),
            video_id: None,
            keywords: Vec::new(),
            // Encyclopedia text, not speech — speaker classification n/a.
            speaker_mode: None,
        });
    }

    let embedder_clone = embedder.clone();
    let vectors: Vec<Vec<f32>> = tokio::task::spawn_blocking(move || {
        let mut out = Vec::with_capacity(texts.len());
        for batch in texts.chunks(EMBED_BATCH) {
            let v = embedder_clone.embed_batch(batch.to_vec())?;
            out.extend(v);
        }
        anyhow::Ok(out)
    })
    .await??;

    for (item, vec) in items.iter_mut().zip(vectors.into_iter()) {
        item.vector = vec;
    }

    let count = items.len();
    vector_store.upsert_batch(items).await?;
    Ok(count)
}

/// Index an uploaded document's text under `source_type=document`.
///
/// Mirrors `index_wikipedia`: chunk, embed, upsert. Keys are stable per
/// (advisor, filename, idx), so re-uploading the same file under the same
/// expert overwrites those chunks instead of duplicating them.
pub async fn index_document(
    display_name: &str,
    filename: &str,
    text: &str,
    embedder: Arc<Embedder>,
    vector_store: Arc<VectorStore>,
) -> Result<usize> {
    let chunks = chunk_text(text, CHUNK_CHARS, CHUNK_OVERLAP);
    if chunks.is_empty() {
        return Ok(0);
    }
    info!(
        "[{display_name}] document '{filename}' -> {} chunks",
        chunks.len()
    );

    let texts: Vec<String> = chunks.clone();
    let mut items: Vec<UpsertItem> = Vec::with_capacity(chunks.len());
    for (idx, chunk) in chunks.into_iter().enumerate() {
        items.push(UpsertItem {
            key: format!("doc::{display_name}::{filename}::{idx}"),
            vector: Vec::new(),
            source_type: "document".into(),
            advisor: display_name.to_string(),
            topic: filename.to_string(),
            content: chunk,
            chunk_idx: Some(idx as u32),
            video_id: None,
            keywords: Vec::new(),
            // Written text, not speech - speaker classification does not apply.
            speaker_mode: None,
        });
    }

    let embedder_clone = embedder.clone();
    let vectors: Vec<Vec<f32>> = tokio::task::spawn_blocking(move || {
        let mut out = Vec::with_capacity(texts.len());
        for batch in texts.chunks(EMBED_BATCH) {
            let v = embedder_clone.embed_batch(batch.to_vec())?;
            out.extend(v);
        }
        anyhow::Ok(out)
    })
    .await??;

    for (item, vec) in items.iter_mut().zip(vectors.into_iter()) {
        item.vector = vec;
    }

    let count = items.len();
    vector_store.upsert_batch(items).await?;
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_channel_url_handles_all_shapes() {
        assert_eq!(
            normalize_channel_url("aiscout"),
            "https://www.youtube.com/@aiscout/videos"
        );
        assert_eq!(
            normalize_channel_url("@aiscout"),
            "https://www.youtube.com/@aiscout/videos"
        );
        assert_eq!(
            normalize_channel_url("  @aiscout  "),
            "https://www.youtube.com/@aiscout/videos"
        );
        // Bare full URLs get /videos appended so yt-dlp lists Videos tab only.
        assert_eq!(
            normalize_channel_url("https://www.youtube.com/@custom"),
            "https://www.youtube.com/@custom/videos"
        );
        assert_eq!(
            normalize_channel_url("https://www.youtube.com/channel/UCxxx"),
            "https://www.youtube.com/channel/UCxxx/videos"
        );
        // Already-normalized URLs are left alone (no double-append).
        assert_eq!(
            normalize_channel_url("https://www.youtube.com/@custom/videos"),
            "https://www.youtube.com/@custom/videos"
        );
    }

    #[test]
    fn name_tokens_drops_short_words_and_punctuation() {
        assert_eq!(name_tokens("Dana Whitfield"), vec!["dana", "whitfield"]);
        assert_eq!(name_tokens("Lenny Bruce"), vec!["lenny", "bruce"]);
        // Initials and short words drop out
        assert_eq!(name_tokens("J. R. R. Tolkien"), vec!["tolkien"]);
        // Non-ASCII letters survive (still alphanumeric per is_alphanumeric)
        assert_eq!(name_tokens("Søren Kierkegaard"), vec!["søren", "kierkegaard"]);
    }

    #[test]
    fn contains_any_token_is_case_insensitive_and_substring() {
        let tokens = vec!["lenny".to_string(), "bruce".to_string()];
        assert!(contains_any_token("Lenny Bruce on the Steve Allen Show", &tokens));
        assert!(contains_any_token("RARE: bruce stand-up 1961", &tokens));
        assert!(!contains_any_token("Phyllis Diller's Driving Tips", &tokens));
        // Substring match is intentional — "whitfield" matches "DanaWhitfield".
        assert!(contains_any_token(
            "DanaWhitfield Talks Back",
            &vec!["whitfield".to_string()]
        ));
    }

    #[test]
    fn empty_run_message_classifies_dominant_failure() {
        let mut s = IndexSummary {
            chunks: 0,
            videos_total: 30,
            videos_listed: 30,
            ..Default::default()
        };
        s.errors.insert("rate_limited", 28);
        s.errors.insert("no_transcript", 2);
        let msg = s.empty_run_message();
        assert!(msg.contains("rate-limited"), "got: {msg}");
        assert!(msg.contains("28 rate_limited"), "got: {msg}");
    }

    #[test]
    fn empty_run_message_separates_disabled_captions_from_missing_language() {
        // Captions switched off: must NOT blame the channel choice, and must
        // make clear that no language would have helped.
        let mut disabled = IndexSummary {
            videos_total: 1,
            videos_listed: 1,
            ..Default::default()
        };
        disabled.errors.insert("captions_disabled", 1);
        let msg = disabled.empty_run_message();
        assert!(msg.contains("switched off"), "got: {msg}");
        assert!(
            !msg.contains("different channel candidate"),
            "captions-disabled must not blame the channel: {msg}"
        );

        // No track in any language: should state that translation exists, so
        // the user doesn't think a foreign-language video was the problem.
        let mut missing = IndexSummary {
            videos_total: 2,
            videos_listed: 2,
            ..Default::default()
        };
        missing.errors.insert("language_unavailable", 2);
        let msg = missing.empty_run_message();
        assert!(msg.contains("any language"), "got: {msg}");
        assert!(msg.contains("auto-translated"), "got: {msg}");
    }

    #[test]
    fn empty_run_message_distinguishes_listing_from_filtering() {
        // yt-dlp returned nothing.
        let s = IndexSummary {
            videos_listed: 0,
            videos_total: 0,
            ..Default::default()
        };
        assert!(
            s.empty_run_message().contains("yt-dlp returned 0 videos"),
            "got: {}",
            s.empty_run_message()
        );

        // yt-dlp returned 10, the title filter dropped them all.
        let s = IndexSummary {
            videos_listed: 10,
            videos_total: 0,
            ..Default::default()
        };
        let msg = s.empty_run_message();
        assert!(
            msg.contains("title-relevance filter dropped them all"),
            "got: {msg}"
        );
        assert!(msg.contains("10 videos listed"), "got: {msg}");
    }

    #[test]
    fn strip_vtt_tags_removes_angle_brackets() {
        assert_eq!(
            strip_vtt_tags("<c>hello</c> <00:00:01.000>world"),
            "hello world"
        );
        assert_eq!(strip_vtt_tags("no tags here"), "no tags here");
    }

    #[test]
    fn parse_vtt_strips_header_and_dedupes_rolling_cues() {
        // Mimics the shape of YouTube auto-sub output: repeated lines as the
        // rolling highlight moves across words.
        let vtt = "WEBVTT\nKind: captions\nLanguage: en\n\n\
            00:00:00.000 --> 00:00:02.000\n\
            Hello<00:00:01.000><c> world</c>\n\n\
            00:00:02.000 --> 00:00:04.000\n\
            Hello world\n\n\
            00:00:04.000 --> 00:00:06.000\n\
            <c>Next</c> line\n";
        let out = parse_vtt(vtt);
        // "Hello world" should appear once (dedup); "Next line" follows.
        assert_eq!(out, "Hello world Next line");
    }

    #[test]
    fn classify_speaker_mode_defaults_to_single() {
        let monologue = "welcome back to my channel today I want to talk about \
                         pointer compression in v8 and why it matters for your \
                         node applications let's dive right in";
        assert_eq!(classify_speaker_mode(monologue), "single");
    }

    #[test]
    fn classify_speaker_mode_detects_interview_phrases() {
        let interview = "my guest today is a brilliant researcher thanks for \
                         having me on the show it's great to be here";
        assert_eq!(classify_speaker_mode(interview), "multiple");
    }

    #[test]
    fn classify_speaker_mode_detects_caption_markers() {
        let captioned = ">> So what do you think? >> Well, I disagree. \
                         >> Interesting. >> Let me explain why.";
        assert_eq!(classify_speaker_mode(captioned), "multiple");

        let labeled = "JOE: welcome back ROGAN: thanks man JOE: so tell me \
                       GUEST: sure INTERVIEWER: go ahead";
        assert_eq!(classify_speaker_mode(labeled), "multiple");
    }

    #[test]
    fn chunk_text_overlaps_correctly() {
        let text: String = "a".repeat(1100);
        let chunks = chunk_text(&text, 500, 50);
        // step = 450, so starts at 0, 450, 900 — three chunks.
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].len(), 500);
        assert_eq!(chunks[1].len(), 500);
        assert_eq!(chunks[2].len(), 200);
    }
}
