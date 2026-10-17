use axum::body::Body;
use axum::extract::{Path, Query, State};
use futures::StreamExt;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use tracing::{error, info, warn};

use crate::openrouter::OpenRouterStream;
use crate::ingestion::{self, JobProgress, JobState, JobStatus};
use crate::knowledge::get_transcript_for_message;
use crate::models::*;
use crate::prompt::build_split_prompt;
use crate::AppState;

const TOP_K: u64 = 8;

/// Instruction appended to the final user turn so the answer comes back in
/// the user's own language.
///
/// Naming the language works; asking the model to infer it does not. Measured
/// on GLM-4.6 with an English question against English-only sources: 4 of 12
/// replies came back in Chinese when the prompt said "the same language as
/// this message" (it is a Chinese-trained model, and a long retrieved context
/// lets that prior win), against 0 of 8 when the prompt named English outright.
///
/// Detection is offline and per-request, so it costs no round trip and stays
/// correct for a multilingual corpus - an Italian question still gets an
/// Italian answer.
fn language_reminder(message: &str) -> String {
    // Too short or too ambiguous to name a language safely. Forcing the wrong
    // one is worse than the drift being fixed, so fall back to mirroring.
    const MIN_CHARS: usize = 20;
    const MIN_CONFIDENCE: f64 = 0.5;

    if message.chars().count() >= MIN_CHARS {
        if let Some(info) = whatlang::detect(message) {
            if info.confidence() >= MIN_CONFIDENCE {
                return format!(
                    "

(Reply entirely in {}, regardless of what language your source material is in.)",
                    info.lang().eng_name()
                );
            }
        }
    }
    "

(Reply in the same language as this message, regardless of what language your source material is in.)"
        .to_string()
}

/// Embed a chat query without stalling the async runtime.
///
/// ONNX inference is CPU-bound and synchronous; called directly from a
/// handler it holds one of Tokio's few worker threads (one per core) for the
/// whole call, so a handful of concurrent questions freezes every other
/// request, including responses already streaming. Running it on the
/// blocking pool keeps the workers free, and the semaphore bounds how many
/// run at once. The permit moves into the closure so it is held until the
/// inference actually finishes, even if the client disconnects first.
async fn embed_query_blocking(state: &Arc<AppState>, text: &str) -> anyhow::Result<Vec<f32>> {
    let permit = state.embed_permits.clone().acquire_owned().await?;
    let embedder = state.embedder.clone();
    let text = text.to_string();
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        embedder.embed_query(&text)
    })
    .await?
}

/// POST /api/chat — streams a Gemini response back to the client.
pub async fn chat_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> impl IntoResponse {
    let start = Instant::now();
    let request_id = uuid::Uuid::new_v4().to_string()[..8].to_string();

    // Optional advisor scope: when the frontend's "active expert" selector
    // is set to one expert, restrict retrieval and prompt to that name.
    let scope = req.expert.as_ref().map(|s| s.trim()).filter(|s| !s.is_empty());

    // Semantic retrieval: embed the user message and query Qdrant for the top-k
    // most relevant chunks across the knowledge base and ingested transcripts.
    let retrieved: Vec<RetrievedChunk> = match embed_query_blocking(&state, &req.message).await {
        Ok(query_vec) => match state.vector_store.search(query_vec, TOP_K, scope).await {
            Ok(hits) => hits,
            Err(e) => {
                warn!("[{request_id}] vector search failed: {e}; proceeding with empty context");
                Vec::new()
            }
        },
        Err(e) => {
            warn!("[{request_id}] embedding failed: {e}; proceeding with empty context");
            Vec::new()
        }
    };

    // In scoped mode, derive the expert's overall speaker mode from the
    // retrieved chunks (labeled at ingest time). Any multi-speaker source in
    // the mix wins — that context already blends voices. Drives both the
    // prompt framing and the visible label prepended to the response.
    let scoped_speaker_mode: Option<&'static str> = if scope.is_some() {
        if retrieved
            .iter()
            .any(|c| c.speaker_mode.as_deref() == Some("multiple"))
        {
            Some("multiple")
        } else if retrieved
            .iter()
            .any(|c| c.speaker_mode.as_deref() == Some("single"))
        {
            Some("single")
        } else {
            None
        }
    } else {
        None
    };

    // Keyword-matched transcript fallback (still used until Phase 3b replaces it).
    let transcript = get_transcript_for_message(&req.message, &state.data_dir);

    // Build the experts set passed to the prompt. When scoped, it's just the
    // single chosen expert; otherwise it's the static KB set merged with the
    // dynamic creator-transcript advisors (so newly-ingested experts show up
    // immediately without a server restart).
    let experts: HashSet<String> = if let Some(name) = scope {
        let mut s = HashSet::new();
        s.insert(name.to_string());
        s
    } else {
        let mut s = state.experts.clone();
        match state.vector_store.list_creators().await {
            Ok(dynamic) => s.extend(dynamic),
            Err(e) => warn!("[{request_id}] list_creators failed: {e}; using static experts only"),
        }
        s
    };

    // Model is configurable via OPENROUTER_MODEL env var; empty or unset falls
    // back to GLM-4.6. Read per-request so a backend restart is all that's
    // needed to swap models (no rebuild).
    let model = std::env::var("OPENROUTER_MODEL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "z-ai/glm-4.6".to_string());

    info!("[{request_id}] ── REQUEST ──");
    info!("[{request_id}] Model: {model} (via OpenRouter)");
    info!("[{request_id}] Prompt: \"{}\"", req.message);
    info!(
        "[{request_id}] Scope: {}",
        scope.unwrap_or("all experts (unscoped)")
    );
    info!("[{request_id}] Experts in scope: {}", experts.len());
    info!("[{request_id}] Retrieved chunks: {}", retrieved.len());
    if let Some(top) = retrieved.first() {
        info!(
            "[{request_id}] Top hit: advisor={} source={} score={:.3}",
            top.advisor, top.source_type, top.score
        );
    }
    info!(
        "[{request_id}] Transcript loaded: {}",
        if transcript.is_empty() { "no" } else { "yes" }
    );
    if let Some(mode) = scoped_speaker_mode {
        info!("[{request_id}] Scoped speaker mode: {mode}");
    }
    info!("[{request_id}] History turns: {}", req.history.len());
    info!(
        "[{request_id}] Timestamp: {}",
        chrono::Utc::now().to_rfc3339()
    );

    // OpenRouter is OpenAI-compatible — no system/user split with cache_control.
    // Concatenate the static and dynamic system prompt halves into one
    // role=system message. Prompt caching that Anthropic gave us is gone here.
    let (static_prompt, dynamic_prompt) = build_split_prompt(&experts, &retrieved, &transcript, scope);
    let system_text = match (static_prompt.is_empty(), dynamic_prompt.is_empty()) {
        (true, true) => String::new(),
        (false, true) => static_prompt,
        (true, false) => dynamic_prompt,
        (false, false) => format!("{static_prompt}\n\n{dynamic_prompt}"),
    };

    let mut messages: Vec<OpenRouterMessage> = Vec::new();
    if !system_text.is_empty() {
        messages.push(OpenRouterMessage {
            role: "system".into(),
            content: system_text,
        });
    }

    for msg in &req.history {
        let text = msg.parts.iter().map(|p| p.text.clone()).collect::<Vec<_>>().join("\n");
        let role = if msg.role == "model" { "assistant" } else { "user" };
        messages.push(OpenRouterMessage {
            role: role.into(),
            content: text,
        });
    }

    // The language rule is stated twice in the system prompt and GLM-4.6 still
    // answered an English question in Chinese roughly one time in five (it is a
    // Chinese-trained model, and a long retrieved context lets that prior win).
    // Restating the rule on the final user turn is the strongest lever left: it
    // is the most recent text in the request and carries user-role authority.
    //
    // It deliberately does not name a language. The corpus is multilingual and
    // an Italian question must still get an Italian answer, so the model is told
    // to mirror the message it is attached to rather than pinned to English.
    //
    // Appended only to the outgoing payload, never to req.message, so it never
    // reaches the client or the history the client sends back on the next turn.
    messages.push(OpenRouterMessage {
        role: "user".into(),
        content: format!("{}{}", req.message, language_reminder(&req.message)),
    });

    let openrouter_req = OpenRouterRequest {
        model,
        messages,
        max_tokens: 8192,
        stream: true,
        stream_options: Some(StreamOptions { include_usage: true }),
        reasoning: Some(ReasoningConfig {
            enabled: Some(false),
            exclude: None,
        }),
    };

    let stream = OpenRouterStream::start(
        &state.http_client,
        &state.openrouter_api_key,
        openrouter_req,
        request_id.clone(),
    );

    let elapsed = start.elapsed().as_millis();
    info!("[{request_id}] Stream started after {elapsed}ms setup");

    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", "text/plain; charset=utf-8".parse().unwrap());
    headers.insert("Transfer-Encoding", "chunked".parse().unwrap());
    headers.insert("X-Request-Id", request_id.parse().unwrap());

    // Scoped chats open with a visible label stating whose voice(s) the
    // answer channels — deterministic, not left to the model. Uses the same
    // annotation wording stored at ingest time.
    let body = match scoped_speaker_mode {
        Some(mode) => {
            let note = if mode == "multiple" {
                crate::prompt::MULTI_SPEAKER_NOTE
            } else {
                crate::prompt::SINGLE_SPEAKER_NOTE
            };
            let label = bytes::Bytes::from(format!("[{note}]\n\n"));
            let labeled = futures::stream::once(async move { Ok::<_, axum::Error>(label) })
                .chain(stream);
            Body::from_stream(labeled)
        }
        None => Body::from_stream(stream),
    };
    (StatusCode::OK, headers, body).into_response()
}

/// Fallback error handler.
pub async fn handle_error() -> impl IntoResponse {
    (
        StatusCode::NOT_FOUND,
        Json(ErrorResponse {
            error: "Not found".into(),
        }),
    )
}

/// Request body for `POST /api/creators`.
#[derive(Debug, Deserialize)]
pub struct IndexCreatorRequest {
    pub handle: String,
    #[serde(default)]
    pub max_videos: Option<usize>,
}

/// 429 response when the rolling YouTube-fetch budget is exhausted.
fn budget_exceeded(minutes: u64) -> axum::response::Response {
    (
        StatusCode::TOO_MANY_REQUESTS,
        Json(serde_json::json!({
            "error": format!(
                "YouTube fetch budget reached ({}/hour). Try again in ~{} minute(s).",
                crate::SAFE_FETCH_BUDGET_PER_HOUR, minutes
            ),
            "retry_after_minutes": minutes,
        })),
    )
        .into_response()
}

/// POST /api/creators — kick off an async ingestion job for one creator.
/// Returns `{ job_id, status }` immediately; poll `/api/creators/:job_id` for progress.
pub async fn start_creator_job(
    State(state): State<Arc<AppState>>,
    Json(req): Json<IndexCreatorRequest>,
) -> impl IntoResponse {
    let handle = req.handle.trim().to_string();
    if handle.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "handle is required" })),
        )
            .into_response();
    }
    let max_videos = req.max_videos.unwrap_or(5).clamp(1, 50);

    // Enforce the rolling fetch budget: atomically reserve up to max_videos
    // slots. When the hourly window is full, reject with 429 + retry time so a
    // public guest can't trigger unbounded scraping and get the IP throttled.
    let max_videos = match state.try_reserve_fetches(max_videos) {
        Ok(granted) => granted,
        Err(mins) => return budget_exceeded(mins),
    };

    let job_id = uuid::Uuid::new_v4().to_string();
    let initial = JobState::new(job_id.clone(), handle.clone());
    {
        let mut jobs = state.jobs.lock().expect("jobs mutex poisoned");
        jobs.insert(job_id.clone(), initial);
    }

    let state_cloned = state.clone();
    let job_id_cloned = job_id.clone();
    let handle_cloned = handle.clone();
    tokio::spawn(async move {
        info!(
            "[job {job_id_cloned}] starting ingestion for '{handle_cloned}'"
        );
        set_status(&state_cloned, &job_id_cloned, JobStatus::Running);

        let jobs = state_cloned.jobs.clone();
        let jid = job_id_cloned.clone();
        let update = move |p: &JobProgress| {
            let mut guard = jobs.lock().expect("jobs mutex poisoned");
            if let Some(job) = guard.get_mut(&jid) {
                match p {
                    JobProgress::Listed { total } => job.videos_total = *total,
                    JobProgress::VideoDone { done } => job.videos_done = *done,
                    JobProgress::Indexed { chunks } => job.chunks_indexed = *chunks,
                }
            }
        };

        let result = ingestion::index_creator(
            &handle_cloned,
            &handle_cloned,
            max_videos,
            state_cloned.embedder.clone(),
            state_cloned.vector_store.clone(),
            update,
        )
        .await;

        finalize_job(&state_cloned, &job_id_cloned, result);
    });

    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({ "job_id": job_id, "status": "started" })),
    )
        .into_response()
}

/// Apply the terminal job state. A `done` run with 0 chunks is treated as a
/// failure with a descriptive error message — otherwise the UI shows a green
/// "done" pill while having indexed nothing, which is misleading.
fn finalize_job(
    state: &Arc<AppState>,
    job_id: &str,
    result: anyhow::Result<ingestion::IndexSummary>,
) {
    let mut guard = state.jobs.lock().expect("jobs mutex poisoned");
    let Some(job) = guard.get_mut(job_id) else { return };
    job.finished_at = Some(chrono::Utc::now());
    match result {
        Ok(summary) if summary.chunks > 0 => {
            job.status = JobStatus::Done;
            job.chunks_indexed = summary.chunks;
            info!(
                "[job {job_id}] done: {} chunks from {} videos",
                summary.chunks, summary.videos_total
            );
        }
        Ok(summary) => {
            job.status = JobStatus::Failed;
            let msg = summary.empty_run_message();
            warn!("[job {job_id}] failed (0 chunks): {msg}");
            job.error = Some(msg);
        }
        Err(e) => {
            job.status = JobStatus::Failed;
            job.error = Some(e.to_string());
            error!("[job {job_id}] failed: {e}");
        }
    }
}

fn set_status(state: &Arc<AppState>, job_id: &str, status: JobStatus) {
    let mut guard = state.jobs.lock().expect("jobs mutex poisoned");
    if let Some(job) = guard.get_mut(job_id) {
        job.status = status;
    }
}

/// GET /api/creators/:job_id — fetch progress for one ingestion job.
pub async fn get_creator_job(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    let guard = state.jobs.lock().expect("jobs mutex poisoned");
    match guard.get(&job_id) {
        Some(job) => (StatusCode::OK, Json(serde_json::to_value(job).unwrap())).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "job not found" })),
        )
            .into_response(),
    }
}

/// GET /api/experts — every distinct advisor whose chunks live in Qdrant.
/// This is the union of: (a) the static KB-curated baseline (Hale, AiScout,
/// Marsh, Whitfield, Wandering Engineer, Kessler) and (b) every YouTube-ingested
/// expert added through `/api/experts/confirm`. Used by the Chat tab's
/// scope dropdown so the user can narrow chat to ANY queryable expert,
/// not just the YouTube-ingested ones (which is what `/api/creators` returns).
pub async fn list_all_experts(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut all: HashSet<String> = state.experts.clone();
    if let Ok(dynamic) = state.vector_store.list_creators().await {
        all.extend(dynamic);
    }
    let mut sorted: Vec<String> = all.into_iter().collect();
    sorted.sort();
    (
        StatusCode::OK,
        Json(serde_json::json!({ "experts": sorted })),
    )
        .into_response()
}

/// GET /api/limits — current YouTube-fetch rate budget and corpus-size stats.
/// The frontend uses this to show a banner above the search field so the user
/// knows whether they have headroom before kicking off another ingest.
pub async fn get_limits(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let (fetches_last_hour, minutes_until_safe) = state.fetch_rate();
    let experts_total = state
        .vector_store
        .list_creators()
        .await
        .map(|v| v.len())
        .unwrap_or(0);
    let chunks_total = state.vector_store.count_points().await.unwrap_or(0);

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "fetches_last_hour": fetches_last_hour,
            "safe_budget_per_hour": crate::SAFE_FETCH_BUDGET_PER_HOUR,
            "minutes_until_safe": minutes_until_safe,
            "experts_total": experts_total,
            "chunks_total": chunks_total,
        })),
    )
        .into_response()
}

/// DELETE /api/experts/:name — drop every Qdrant point where
/// `advisor == name AND source_type == "creator_transcript"`. The source-type
/// guard protects the original 6 KB-curated advisors (their chunks are
/// `kb_entry`) — so calling this for "Dana Whitfield" removes the YouTube
/// transcripts ingested via the Experts tab but leaves his curated baseline.
pub async fn delete_expert(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let name = name.trim().to_string();
    if name.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "name is required" })),
        )
            .into_response();
    }
    match state.vector_store.delete_creator(&name).await {
        Ok(0) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": format!("no removable chunks found for '{name}' (transcripts, documents or Wikipedia). Curated knowledge-base entries are never deleted.")
            })),
        )
            .into_response(),
        Ok(deleted) => {
            info!("deleted {deleted} chunks for expert '{name}'");
            (
                StatusCode::OK,
                Json(serde_json::json!({ "deleted": deleted, "name": name })),
            )
                .into_response()
        }
        Err(e) => {
            warn!("delete_expert('{name}') failed: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ExportParams {
    /// When set, export only this one video instead of the whole expert.
    pub video: Option<String>,
}

/// GET /api/experts/{name}/export — download an expert's ingested transcript
/// text as a .txt file, sectioned by video (with the YouTube URL) so it's
/// readable offline. Wikipedia content gets its own section. Chunks are
/// stitched back together in chunk_idx order.
///
/// With `?video=<video_id>`, exports just that one video instead.
pub async fn export_expert(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(params): Query<ExportParams>,
) -> impl IntoResponse {
    let name = name.trim().to_string();
    if name.is_empty() {
        return (StatusCode::BAD_REQUEST, "expert name is required").into_response();
    }

    let mut chunks = match state.vector_store.fetch_expert_chunks(&name).await {
        Ok(c) => c,
        Err(e) => {
            warn!("export_expert('{name}') failed: {e}");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("export failed: {e}"),
            )
                .into_response();
        }
    };
    if chunks.is_empty() {
        return (
            StatusCode::NOT_FOUND,
            format!("no ingested content found for '{name}'"),
        )
            .into_response();
    }

    // Optional single-video filter. Remember its title so the download is named
    // after the video rather than the expert.
    let single_video_title = match params.video.as_deref().map(str::trim).filter(|v| !v.is_empty())
    {
        Some(vid) => {
            let title = chunks
                .iter()
                .find(|c| c.video_id == vid)
                .map(|c| c.title.clone());
            chunks.retain(|c| c.video_id == vid);
            if chunks.is_empty() {
                return (
                    StatusCode::NOT_FOUND,
                    format!("no content for video '{vid}' under '{name}'"),
                )
                    .into_response();
            }
            Some(title.unwrap_or_default())
        }
        None => None,
    };

    // Group chunks by video (or a per-title key for Wikipedia), keep the title
    // + source type, then stitch each group's text in chunk_idx order.
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<String, (String, String, Vec<(i64, String)>)> = BTreeMap::new();
    for c in chunks {
        let key = if c.video_id.is_empty() {
            format!("wiki::{}", c.title)
        } else {
            c.video_id.clone()
        };
        let entry = groups
            .entry(key)
            .or_insert_with(|| (c.title.clone(), c.source_type.clone(), Vec::new()));
        entry.2.push((c.chunk_idx, c.content));
    }

    let bar = "=".repeat(64);
    let mut doc = String::new();
    doc.push_str(&name);
    doc.push('\n');
    doc.push_str(&format!(
        "Exported from AI Advisor — {}\n",
        chrono::Utc::now().to_rfc3339()
    ));
    doc.push_str(&format!("{} source(s)\n", groups.len()));
    for (key, (title, source_type, mut parts)) in groups {
        parts.sort_by_key(|(idx, _)| *idx);
        let body = parts
            .into_iter()
            .map(|(_, t)| t)
            .collect::<Vec<_>>()
            .join(" ");
        doc.push_str(&format!("\n\n{bar}\n"));
        if source_type == "wikipedia_article" {
            let t = if title.is_empty() { name.clone() } else { title };
            doc.push_str(&format!("WIKIPEDIA: {t}\n"));
        } else {
            let t = if title.is_empty() {
                "(untitled)".to_string()
            } else {
                title
            };
            doc.push_str(&format!("VIDEO: {t}\n"));
            doc.push_str(&format!("https://www.youtube.com/watch?v={key}\n"));
        }
        doc.push_str(&format!("{bar}\n\n"));
        doc.push_str(body.trim());
        doc.push('\n');
    }

    // Filename: the video title for a single-video export, else the expert.
    let file_base = match &single_video_title {
        Some(t) if !t.is_empty() => t.clone(),
        _ => name.clone(),
    };
    let safe_name: String = file_base
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == ' ' || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let headers = [
        (
            axum::http::header::CONTENT_TYPE,
            "text/plain; charset=utf-8".to_string(),
        ),
        (
            axum::http::header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{safe_name}.txt\""),
        ),
    ];
    (StatusCode::OK, headers, doc).into_response()
}

/// GET /api/experts/{name}/videos — list the distinct videos ingested for an
/// expert as `{ video_id, title }`, sorted by title. Drives the per-video
/// download links in the UI.
pub async fn list_expert_videos(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let name = name.trim().to_string();
    let chunks = match state.vector_store.fetch_expert_chunks(&name).await {
        Ok(c) => c,
        Err(e) => {
            warn!("list_expert_videos('{name}') failed: {e}");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response();
        }
    };
    // distinct video_id -> title (first occurrence wins; a video's chunks all
    // carry the same title, so first is fine).
    let mut by_id: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
    for c in chunks {
        if c.video_id.is_empty() {
            continue;
        }
        by_id.entry(c.video_id).or_insert(c.title);
    }
    let mut videos: Vec<_> = by_id
        .into_iter()
        .map(|(video_id, title)| serde_json::json!({ "video_id": video_id, "title": title }))
        .collect();
    videos.sort_by(|a, b| {
        a["title"]
            .as_str()
            .unwrap_or("")
            .cmp(b["title"].as_str().unwrap_or(""))
    });
    (
        StatusCode::OK,
        Json(serde_json::json!({ "videos": videos })),
    )
        .into_response()
}

#[derive(Debug, Deserialize)]
pub struct IngestUrlRequest {
    pub name: String,
    pub url: String,
    #[serde(default)]
    pub max_videos: Option<usize>,
}

/// POST /api/experts/ingest-url — ingest directly from a pasted YouTube URL.
/// A single-video URL ingests just that video's transcript; a channel/playlist
/// URL ingests up to max_videos from it (same as confirm). Stored under `name`.
/// Returns a job_id to poll via GET /api/creators/:job_id.
pub async fn ingest_url(
    State(state): State<Arc<AppState>>,
    Json(req): Json<IngestUrlRequest>,
) -> impl IntoResponse {
    let name = req.name.trim().to_string();
    let url = req.url.trim().to_string();
    if name.is_empty() || url.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "name and url are required" })),
        )
            .into_response();
    }

    // Rumble (and other non-YouTube sites) have no transcript API — we pull
    // subtitles via yt-dlp instead. Detected by host and routed separately.
    let is_rumble = url.to_lowercase().contains("rumble.com");
    let single_video = if is_rumble {
        None
    } else {
        ingestion::extract_video_id(&url)
    };
    let max_videos = req.max_videos.unwrap_or(10).clamp(1, 50);
    // Reserve budget: 1 for a single video (YouTube or Rumble), else max_videos.
    let want = if is_rumble || single_video.is_some() {
        1
    } else {
        max_videos
    };
    let granted = match state.try_reserve_fetches(want) {
        Ok(g) => g,
        Err(mins) => return budget_exceeded(mins),
    };

    let job_id = uuid::Uuid::new_v4().to_string();
    let initial = JobState::new(job_id.clone(), name.clone());
    {
        let mut jobs = state.jobs.lock().expect("jobs mutex poisoned");
        jobs.insert(job_id.clone(), initial);
    }

    let state_cloned = state.clone();
    let job_id_cloned = job_id.clone();
    let name_cloned = name.clone();
    let url_cloned = url.clone();
    tokio::spawn(async move {
        set_status(&state_cloned, &job_id_cloned, JobStatus::Running);
        let jobs = state_cloned.jobs.clone();
        let jid = job_id_cloned.clone();
        let update = move |p: &JobProgress| {
            let mut guard = jobs.lock().expect("jobs mutex poisoned");
            if let Some(job) = guard.get_mut(&jid) {
                match p {
                    JobProgress::Listed { total } => job.videos_total = *total,
                    JobProgress::VideoDone { done } => job.videos_done = *done,
                    JobProgress::Indexed { chunks } => job.chunks_indexed = *chunks,
                }
            }
        };

        let result = if is_rumble {
            info!("[job {job_id_cloned}] rumble URL '{url_cloned}' as '{name_cloned}' (yt-dlp subtitles)");
            ingestion::index_url_via_subtitles(
                &name_cloned,
                &url_cloned,
                state_cloned.embedder.clone(),
                state_cloned.vector_store.clone(),
                update,
            )
            .await
        } else if let Some(video_id) = single_video {
            info!("[job {job_id_cloned}] single video {video_id} as '{name_cloned}'");
            let (title, _channel) =
                ingestion::fetch_video_meta(&video_id).await;
            let topic = title.unwrap_or_else(|| format!("Video {video_id}"));
            ingestion::index_single_video(
                &name_cloned,
                &video_id,
                &topic,
                state_cloned.embedder.clone(),
                state_cloned.vector_store.clone(),
                update,
            )
            .await
        } else {
            info!("[job {job_id_cloned}] channel/playlist URL '{url_cloned}' as '{name_cloned}'");
            ingestion::index_creator(
                &name_cloned,
                &url_cloned,
                granted,
                state_cloned.embedder.clone(),
                state_cloned.vector_store.clone(),
                update,
            )
            .await
        };
        finalize_job(&state_cloned, &job_id_cloned, result);
    });

    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({ "job_id": job_id, "status": "started" })),
    )
        .into_response()
}

fn default_translate() -> bool {
    true
}

#[derive(Debug, Deserialize)]
pub struct IngestDocumentRequest {
    /// Expert this document is filed under. Reusing an existing name adds to
    /// that expert; a new name creates one.
    pub name: String,
    /// Original filename — drives format handling and is stored as the chunk
    /// topic so answers can cite which document they came from.
    pub filename: String,
    /// The file's text, read by the browser.
    pub content: String,
    /// Translate to English before indexing. On by default: retrieval is
    /// multilingual either way, but the chat model answers in the language of
    /// the context it is given, so an untranslated corpus yields untranslated
    /// answers.
    #[serde(default = "default_translate")]
    pub translate: bool,
}

/// Set the human-readable progress note on a job.
fn set_note(state: &Arc<AppState>, job_id: &str, note: impl Into<String>) {
    let mut guard = state.jobs.lock().expect("jobs mutex poisoned");
    if let Some(job) = guard.get_mut(job_id) {
        job.note = Some(note.into());
    }
}

/// POST /api/experts/ingest-document — index an uploaded text file.
///
/// Parsing happens synchronously so a bad upload (binary file, unsupported
/// extension, empty after stripping) fails fast with a specific message
/// instead of becoming a job the user has to poll to discover was doomed.
/// Everything after that — language detection, translation, embedding — runs
/// in the background under the same job machinery as video ingestion.
///
/// No YouTube fetch budget is consumed: nothing here touches YouTube.
pub async fn ingest_document(
    State(state): State<Arc<AppState>>,
    Json(req): Json<IngestDocumentRequest>,
) -> impl IntoResponse {
    let name = req.name.trim().to_string();
    let filename = req.filename.trim().to_string();
    if name.is_empty() || filename.is_empty() || req.content.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "name, filename and file content are all required"
            })),
        )
            .into_response();
    }

    let extracted = match crate::documents::extract(&filename, &req.content) {
        Ok(text) => text,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response();
        }
    };

    let job_id = uuid::Uuid::new_v4().to_string();
    {
        let mut jobs = state.jobs.lock().expect("jobs mutex poisoned");
        jobs.insert(job_id.clone(), JobState::new(job_id.clone(), name.clone()));
    }

    let state_cloned = state.clone();
    let job_id_cloned = job_id.clone();
    let want_translation = req.translate;

    tokio::spawn(async move {
        set_status(&state_cloned, &job_id_cloned, JobStatus::Running);
        {
            let mut guard = state_cloned.jobs.lock().expect("jobs mutex poisoned");
            if let Some(job) = guard.get_mut(&job_id_cloned) {
                job.videos_total = 1;
            }
        }
        info!(
            "[job {job_id_cloned}] document '{filename}' as '{name}' ({} chars, translate={want_translation})",
            extracted.chars().count()
        );

        let model = std::env::var("OPENROUTER_MODEL")
            .unwrap_or_else(|_| "z-ai/glm-4.6".to_string());

        // --- language handling ------------------------------------------
        let mut text = extracted;
        let language_note: String;

        if want_translation {
            set_note(&state_cloned, &job_id_cloned, "Detecting language...");
            let detected = crate::translate::detect_language(
                &state_cloned.http_client,
                &state_cloned.openrouter_api_key,
                &model,
                &text,
            )
            .await;

            match detected {
                Some(lang) if crate::translate::is_english(&lang) => {
                    language_note = "Already English - indexed as written.".to_string();
                    info!("[job {job_id_cloned}] detected English; no translation needed");
                }
                Some(lang) => {
                    info!("[job {job_id_cloned}] detected {lang}; translating to English");
                    let state_for_progress = state_cloned.clone();
                    let jid = job_id_cloned.clone();
                    let lang_for_progress = lang.clone();
                    let translated = crate::translate::translate_to_english(
                        &state_cloned.http_client,
                        &state_cloned.openrouter_api_key,
                        &model,
                        &text,
                        &lang,
                        move |done, total| {
                            set_note(
                                &state_for_progress,
                                &jid,
                                format!(
                                    "Translating {lang_for_progress} to English - {done}/{total} segments"
                                ),
                            );
                        },
                    )
                    .await;

                    match translated {
                        Ok(english) => {
                            language_note = format!("Translated from {lang} to English.");
                            text = english;
                        }
                        Err(e) => {
                            // Indexing the original still works — the embedder
                            // is multilingual — so degrade rather than fail,
                            // but say so plainly.
                            warn!("[job {job_id_cloned}] translation failed: {e}");
                            language_note = format!(
                                "Translation failed ({e}). Indexed in {lang}; \
                                 English questions still retrieve it, but answers \
                                 may come back in {lang}."
                            );
                        }
                    }
                }
                None => {
                    language_note =
                        "Could not detect the language - indexed as written.".to_string();
                }
            }
        } else {
            language_note = "Translation off - indexed as written.".to_string();
        }

        set_note(
            &state_cloned,
            &job_id_cloned,
            format!("{language_note} Embedding..."),
        );

        // --- index ------------------------------------------------------
        let result = ingestion::index_document(
            &name,
            &filename,
            &text,
            state_cloned.embedder.clone(),
            state_cloned.vector_store.clone(),
        )
        .await
        .map(|chunks| ingestion::IndexSummary {
            chunks,
            videos_total: 1,
            videos_listed: 1,
            ..Default::default()
        });

        {
            let mut guard = state_cloned.jobs.lock().expect("jobs mutex poisoned");
            if let Some(job) = guard.get_mut(&job_id_cloned) {
                job.videos_done = 1;
                job.note = Some(language_note.clone());
            }
        }
        finalize_job(&state_cloned, &job_id_cloned, result);
    });

    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({ "job_id": job_id, "status": "started" })),
    )
        .into_response()
}

/// GET /api/creators — list every creator handle currently indexed in Qdrant.
pub async fn list_creators(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.vector_store.list_creators_with_counts().await {
        Ok(creators) => {
            let arr: Vec<_> = creators
                .into_iter()
                .map(|(name, videos, ingested_at)| {
                    serde_json::json!({
                        "name": name,
                        "videos": videos,
                        "ingested_at": ingested_at,
                    })
                })
                .collect();
            (
                StatusCode::OK,
                Json(serde_json::json!({ "creators": arr })),
            )
                .into_response()
        }
        Err(e) => {
            warn!("list_creators failed: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    }
}

/// Request body for `POST /api/experts/search`.
#[derive(Debug, Deserialize)]
pub struct ExpertSearchRequest {
    pub name: String,
}

/// POST /api/experts/search — given a person's name, return a Wikipedia bio
/// (so the user can verify they got the right person) plus YouTube channel
/// candidates ranked by how often they appear in `ytsearch10:<name>` results.
/// Backend-only; the frontend wires this into the search field.
pub async fn search_expert(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ExpertSearchRequest>,
) -> impl IntoResponse {
    let name = req.name.trim().to_string();
    if name.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "name is required" })),
        )
            .into_response();
    }
    let result = crate::expert_search::search_expert(
        &state.http_client,
        &name,
    )
    .await;
    (StatusCode::OK, Json(serde_json::to_value(result).unwrap())).into_response()
}

/// Request body for `POST /api/experts/confirm`.
#[derive(Debug, Deserialize)]
pub struct ExpertConfirmRequest {
    /// Display name (e.g. "Dana Whitfield") — what gets stored as the advisor in Qdrant.
    pub name: String,
    /// YouTube identifier — bare handle, @handle, or full channel URL.
    pub handle: String,
    #[serde(default)]
    pub max_videos: Option<usize>,
}

/// POST /api/experts/confirm — kicks off the existing creator-ingestion pipeline
/// but stores results under `name` instead of the raw handle. Returns a job_id
/// the caller polls via `GET /api/creators/:job_id` (same job tracker as 3b).
pub async fn confirm_expert(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ExpertConfirmRequest>,
) -> impl IntoResponse {
    let name = req.name.trim().to_string();
    let handle = req.handle.trim().to_string();
    if name.is_empty() || handle.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "name and handle are required" })),
        )
            .into_response();
    }
    let max_videos = req.max_videos.unwrap_or(5).clamp(1, 50);

    // Enforce the rolling fetch budget (same as /api/creators).
    let max_videos = match state.try_reserve_fetches(max_videos) {
        Ok(granted) => granted,
        Err(mins) => return budget_exceeded(mins),
    };

    let job_id = uuid::Uuid::new_v4().to_string();
    let initial = JobState::new(job_id.clone(), name.clone());
    {
        let mut jobs = state.jobs.lock().expect("jobs mutex poisoned");
        jobs.insert(job_id.clone(), initial);
    }

    let state_cloned = state.clone();
    let job_id_cloned = job_id.clone();
    let name_cloned = name.clone();
    let handle_cloned = handle.clone();
    tokio::spawn(async move {
        info!(
            "[job {job_id_cloned}] starting expert ingestion for '{name_cloned}' (handle '{handle_cloned}')"
        );
        set_status(&state_cloned, &job_id_cloned, JobStatus::Running);

        let jobs = state_cloned.jobs.clone();
        let jid = job_id_cloned.clone();
        let update = move |p: &JobProgress| {
            let mut guard = jobs.lock().expect("jobs mutex poisoned");
            if let Some(job) = guard.get_mut(&jid) {
                match p {
                    JobProgress::Listed { total } => job.videos_total = *total,
                    JobProgress::VideoDone { done } => job.videos_done = *done,
                    JobProgress::Indexed { chunks } => job.chunks_indexed = *chunks,
                }
            }
        };

        // Run YouTube ingest and Wikipedia article ingest in parallel.
        // Per Beads ai_advisor-4y9, Wikipedia is the on-topic baseline so
        // experts without a personal YouTube channel still get usable
        // chunks even when the picked archive channel is mostly off-topic.
        let yt_future = ingestion::index_creator(
            &name_cloned,
            &handle_cloned,
            max_videos,
            state_cloned.embedder.clone(),
            state_cloned.vector_store.clone(),
            update,
        );
        let wiki_future = ingest_wikipedia_for(
            state_cloned.clone(),
            name_cloned.clone(),
        );
        let (yt_result, wiki_count) = tokio::join!(yt_future, wiki_future);

        // Fold Wikipedia chunks into the summary so the job tile shows the
        // combined chunk count and surfaces "wikipedia_ingested" status.
        let combined = match yt_result {
            Ok(mut summary) => {
                summary.wikipedia_ingested = Some(wiki_count > 0);
                summary.chunks += wiki_count;
                Ok(summary)
            }
            Err(e) if wiki_count > 0 => {
                // YouTube blew up entirely (e.g. yt-dlp missing) but Wikipedia
                // saved the run. Don't fail the job — surface partial success.
                warn!(
                    "[job {job_id_cloned}] YouTube ingest errored ({e}); Wikipedia rescued {wiki_count} chunks"
                );
                Ok(ingestion::IndexSummary {
                    chunks: wiki_count,
                    videos_total: 0,
                    wikipedia_ingested: Some(true),
                    ..Default::default()
                })
            }
            Err(e) => Err(e),
        };

        finalize_job(&state_cloned, &job_id_cloned, combined);
    });

    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({ "job_id": job_id, "status": "started" })),
    )
        .into_response()
}

/// Best-effort Wikipedia ingest invoked alongside the YouTube job during
/// `confirm_expert`. Returns the chunk count (0 on any failure — Wikipedia
/// is additive, never blocks the job).
async fn ingest_wikipedia_for(state: Arc<AppState>, name: String) -> usize {
    let article = match crate::expert_search::wikipedia_article_text(
        &state.http_client,
        &name,
    )
    .await
    {
        Ok(Some(a)) => a,
        Ok(None) => {
            info!("[wiki] no article body for '{name}'; skipping baseline ingest");
            return 0;
        }
        Err(e) => {
            warn!("[wiki] article fetch for '{name}' failed: {e}");
            return 0;
        }
    };

    match ingestion::index_wikipedia(
        &name,
        &name,
        &article,
        state.embedder.clone(),
        state.vector_store.clone(),
    )
    .await
    {
        Ok(n) => {
            info!("[wiki] '{name}' ingested {n} chunks from Wikipedia");
            n
        }
        Err(e) => {
            warn!("[wiki] index_wikipedia failed for '{name}': {e}");
            0
        }
    }
}
