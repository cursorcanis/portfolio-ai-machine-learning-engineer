use std::path::PathBuf;

use anyhow::{Context, Result};
use tracing::info;

use ai_advisor_rs::embeddings::Embedder;
use ai_advisor_rs::knowledge::load_knowledge_base;
use ai_advisor_rs::vector_store::{UpsertItem, VectorStore};

const CHUNK_CHARS: usize = 500;
const CHUNK_OVERLAP: usize = 50;
const EMBED_BATCH: usize = 32;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ai_advisor_rs=info,migrate_kb=info".into()),
        )
        .init();

    let data_dir = PathBuf::from("..")
        .join("ai-advisor")
        .join("ai-advisor")
        .join("data");
    info!("Data dir: {}", data_dir.display());

    let qdrant_url =
        std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://127.0.0.1:6334".to_string());
    info!("Connecting to Qdrant at {qdrant_url}");
    let store = VectorStore::connect(&qdrant_url)?;
    store.ensure_collection().await?;

    info!("Loading embedder (this may download ~470 MB on first run)...");
    let embedder = Embedder::new()?;

    // --- Stage 1: knowledge base entries ---
    let kb = load_knowledge_base(&data_dir);
    info!("Loaded {} KB entries", kb.len());

    let mut items: Vec<UpsertItem> = Vec::new();
    let mut texts_to_embed: Vec<String> = Vec::new();

    for entry in &kb {
        // Embed the concatenation of topic + content so the vector carries
        // both the headline and the body.
        let text_for_embed = format!("{}\n{}", entry.topic, entry.content);
        texts_to_embed.push(text_for_embed);
        items.push(UpsertItem {
            key: format!("kb::{}", entry.id),
            vector: Vec::new(), // filled in after batch embed
            source_type: "kb_entry".into(),
            advisor: entry.advisor.clone(),
            topic: entry.topic.clone(),
            content: entry.content.clone(),
            chunk_idx: None,
            video_id: None,
            keywords: entry.keywords.clone(),
            speaker_mode: None,
        });
    }

    // --- Stage 2: transcripts ---
    let transcripts_dir = data_dir.join("transcripts");
    if transcripts_dir.exists() {
        let mut entries: Vec<_> = std::fs::read_dir(&transcripts_dir)
            .with_context(|| format!("reading {}", transcripts_dir.display()))?
            .filter_map(|e| e.ok())
            .collect();
        entries.sort_by_key(|e| e.path());

        for entry in entries {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("txt") {
                continue;
            }
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            let content = std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            let chunks = chunk_text(&content, CHUNK_CHARS, CHUNK_OVERLAP);
            info!(
                "Transcript {} -> {} chunks ({} chars)",
                stem,
                chunks.len(),
                content.len()
            );

            let advisor = guess_advisor_from_stem(&stem);
            // One classification per file — chunks are too short to judge.
            let speaker_mode =
                ai_advisor_rs::ingestion::classify_speaker_mode(&content).to_string();
            for (idx, chunk) in chunks.iter().enumerate() {
                texts_to_embed.push(chunk.clone());
                items.push(UpsertItem {
                    key: format!("transcript::{stem}::{idx}"),
                    vector: Vec::new(),
                    source_type: "transcript_chunk".into(),
                    advisor: advisor.clone(),
                    topic: format!("Transcript: {stem}"),
                    content: chunk.clone(),
                    chunk_idx: Some(idx as u32),
                    video_id: None,
                    keywords: Vec::new(),
                    speaker_mode: Some(speaker_mode.clone()),
                });
            }
        }
    } else {
        info!("No transcripts/ directory at {}", transcripts_dir.display());
    }

    // --- Stage 3: batch embed ---
    info!("Embedding {} texts...", texts_to_embed.len());
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(texts_to_embed.len());
    for batch in texts_to_embed.chunks(EMBED_BATCH) {
        let batch_vec: Vec<String> = batch.to_vec();
        let out = embedder.embed_batch(batch_vec)?;
        vectors.extend(out);
    }
    assert_eq!(vectors.len(), items.len(), "embedding count mismatch");

    for (item, vector) in items.iter_mut().zip(vectors.into_iter()) {
        item.vector = vector;
    }

    // --- Stage 4: upsert ---
    let kb_count = items
        .iter()
        .filter(|i| i.source_type == "kb_entry")
        .count();
    let transcript_count = items.len() - kb_count;
    info!(
        "Upserting {kb_count} KB entries + {transcript_count} transcript chunks to Qdrant..."
    );
    store.upsert_batch(items).await?;

    info!("Migration complete.");
    Ok(())
}

/// Split text into overlapping char windows. Simple but effective for our scale.
fn chunk_text(text: &str, size: usize, overlap: usize) -> Vec<String> {
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

/// Best-effort mapping from a transcript filename stem to an advisor name.
/// Kept generous for seed data; Phase 3b will attach the real creator tag.
fn guess_advisor_from_stem(stem: &str) -> String {
    let low = stem.to_lowercase();
    if low.contains("aiscout") || low.contains("ai_scout") {
        "AiScout".to_string()
    } else if low.contains("devcast") || low.contains("dev_cast") {
        "DevCast".to_string()
    } else {
        stem.to_string()
    }
}

