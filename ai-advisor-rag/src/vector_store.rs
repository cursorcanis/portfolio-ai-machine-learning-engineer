use std::collections::HashSet;
use std::sync::Mutex;

use anyhow::{Context, Result};
use qdrant_client::qdrant::value::Kind;
use qdrant_client::qdrant::points_selector::PointsSelectorOneOf;
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, DeletePointsBuilder, Distance, Filter, PointStruct,
    ScrollPointsBuilder, SearchPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::{Payload, Qdrant};
use serde_json::json;
use uuid::Uuid;

use crate::embeddings::EMBED_DIM;
use crate::models::RetrievedChunk;

/// One stored chunk with the payload fields needed to rebuild readable
/// per-video transcript text for the export feature.
pub struct ExportChunk {
    pub video_id: String,
    pub title: String,
    pub chunk_idx: i64,
    pub content: String,
    pub source_type: String,
}

pub const COLLECTION: &str = "ai_advisor_kb";

/// Fixed UUID namespace for deriving stable point IDs from string keys.
const NAMESPACE: Uuid = Uuid::from_bytes([
    0xa1, 0xad, 0x15, 0x07, 0xa1, 0xad, 0x15, 0x07, 0xa1, 0xad, 0x15, 0x07, 0xa1, 0xad, 0x15, 0x07,
]);

/// Derive a stable UUID for a point key (so re-runs overwrite rather than duplicate).
pub fn point_id(key: &str) -> String {
    Uuid::new_v5(&NAMESPACE, key.as_bytes()).to_string()
}

pub struct VectorStore {
    client: Qdrant,
    /// Memoized `list_creators` result. That call scrolls every user-added
    /// chunk in the collection, and every unscoped chat request needs it, so
    /// recomputing it per message made chat latency grow with the corpus.
    creators_cache: Mutex<CreatorsCache>,
}

/// `names` is the cached list (None = must rescan). `generation` is bumped on
/// every write that can change the list, so a scan that started before a
/// write cannot store its now-stale result after the write invalidated it.
#[derive(Default)]
struct CreatorsCache {
    generation: u64,
    names: Option<Vec<String>>,
}

impl VectorStore {
    /// Connect to a running Qdrant instance. Default URL matches the local
    /// binary/Docker defaults.
    pub fn connect(url: &str) -> Result<Self> {
        let client = Qdrant::from_url(url)
            .build()
            .with_context(|| format!("failed to build Qdrant client for {url}"))?;
        Ok(Self {
            client,
            creators_cache: Mutex::new(CreatorsCache::default()),
        })
    }

    /// Drop the cached creator list. Called after every write that can add or
    /// remove a user-added expert: `upsert_batch` and `delete_creator`.
    /// `migrate_kb` (a separate process) only writes `kb_entry` and
    /// `transcript_chunk` points, which `list_creators` never returns, so it
    /// cannot leave this cache stale.
    fn invalidate_creators(&self) {
        let mut cache = self.creators_cache.lock().expect("creators_cache poisoned");
        cache.generation += 1;
        cache.names = None;
    }

    /// Create the collection if it doesn't exist yet. Idempotent.
    pub async fn ensure_collection(&self) -> Result<()> {
        if self.client.collection_exists(COLLECTION).await? {
            return Ok(());
        }
        self.client
            .create_collection(
                CreateCollectionBuilder::new(COLLECTION)
                    .vectors_config(VectorParamsBuilder::new(EMBED_DIM, Distance::Cosine)),
            )
            .await?;
        Ok(())
    }

    /// Upsert a batch of (key, vector, payload) tuples. Keys are hashed to stable UUIDs.
    pub async fn upsert_batch(&self, items: Vec<UpsertItem>) -> Result<()> {
        // One ingest timestamp for the whole batch (Unix seconds). Re-ingesting
        // an expert overwrites its chunks and thus refreshes this — so it reads
        // as "last ingested". Surfaced per-expert in the Experts list.
        let ingested_at = chrono::Utc::now().timestamp();
        let points: Vec<PointStruct> = items
            .into_iter()
            .map(|item| {
                let payload: Payload = Payload::try_from(json!({
                    "source_type": item.source_type,
                    "advisor": item.advisor,
                    "topic": item.topic,
                    "content": item.content,
                    "chunk_idx": item.chunk_idx,
                    "video_id": item.video_id,
                    "keywords": item.keywords,
                    "ingested_at": ingested_at,
                    "speaker_mode": item.speaker_mode,
                }))
                .expect("payload serialization");
                PointStruct::new(point_id(&item.key), item.vector, payload)
            })
            .collect();

        let result = self
            .client
            .upsert_points(UpsertPointsBuilder::new(COLLECTION, points).wait(true))
            .await;
        // Invalidate even on error: a failed batch may still have landed in part.
        self.invalidate_creators();
        result?;
        Ok(())
    }

    /// List every distinct `advisor` payload value among points tagged with
    /// any user-added source type — currently `creator_transcript` (YouTube
    /// transcripts ingested via the Experts tab) and `wikipedia_article`
    /// (the on-topic Wikipedia baseline added per `ai_advisor-4y9`).
    ///
    /// Curated KB chunks (`source_type == "kb_entry"`) are intentionally
    /// excluded so the Indexed-Experts pill list at the bottom of the UI
    /// only shows experts the user added (and can therefore delete).
    ///
    /// Served from `creators_cache` when warm; see [`VectorStore::invalidate_creators`].
    pub async fn list_creators(&self) -> Result<Vec<String>> {
        let generation = {
            let cache = self.creators_cache.lock().expect("creators_cache poisoned");
            if let Some(names) = &cache.names {
                return Ok(names.clone());
            }
            cache.generation
        };

        let names = self.scan_creators().await?;

        let mut cache = self.creators_cache.lock().expect("creators_cache poisoned");
        if cache.generation == generation {
            cache.names = Some(names.clone());
        }
        Ok(names)
    }

    /// Uncached `list_creators`: scroll every user-added chunk and collect
    /// the distinct advisor names.
    async fn scan_creators(&self) -> Result<Vec<String>> {
        let filter = Filter {
            should: vec![
                Condition::matches("source_type", "creator_transcript".to_string()),
                Condition::matches("source_type", "wikipedia_article".to_string()),
                Condition::matches("source_type", "document".to_string()),
            ],
            ..Default::default()
        };
        let mut seen: HashSet<String> = HashSet::new();
        let mut offset: Option<qdrant_client::qdrant::PointId> = None;
        loop {
            let mut builder = ScrollPointsBuilder::new(COLLECTION)
                .filter(filter.clone())
                .limit(256)
                .with_payload(true)
                .with_vectors(false);
            if let Some(o) = offset.clone() {
                builder = builder.offset(o);
            }
            let resp = self.client.scroll(builder).await?;
            for pt in &resp.result {
                if let Some(v) = pt.payload.get("advisor") {
                    if let Some(Kind::StringValue(s)) = v.kind.as_ref() {
                        if !s.is_empty() {
                            seen.insert(s.clone());
                        }
                    }
                }
            }
            match resp.next_page_offset {
                Some(next) => offset = Some(next),
                None => break,
            }
        }
        let mut out: Vec<String> = seen.into_iter().collect();
        out.sort();
        Ok(out)
    }

    /// Like `list_creators`, but also returns the number of distinct videos
    /// ingested per expert (distinct `video_id` among `creator_transcript`
    /// points). Wikipedia-only experts come back with 0 videos. Sorted by name.
    /// Drives the per-expert "(N videos)" labels in the UI.
    /// Returns (name, distinct_video_count, most_recent_ingested_at_unix_secs).
    /// `ingested_at` is None for experts ingested before timestamps were added.
    pub async fn list_creators_with_counts(&self) -> Result<Vec<(String, usize, Option<i64>)>> {
        let filter = Filter {
            should: vec![
                Condition::matches("source_type", "creator_transcript".to_string()),
                Condition::matches("source_type", "wikipedia_article".to_string()),
                Condition::matches("source_type", "document".to_string()),
            ],
            ..Default::default()
        };
        // advisor -> set of distinct video_ids (empty set = wiki-only / 0 videos)
        let mut videos_by_advisor: std::collections::BTreeMap<String, HashSet<String>> =
            std::collections::BTreeMap::new();
        // advisor -> most recent ingested_at seen
        let mut ingested_by_advisor: std::collections::BTreeMap<String, i64> =
            std::collections::BTreeMap::new();
        let mut offset: Option<qdrant_client::qdrant::PointId> = None;
        loop {
            let mut builder = ScrollPointsBuilder::new(COLLECTION)
                .filter(filter.clone())
                .limit(256)
                .with_payload(true)
                .with_vectors(false);
            if let Some(o) = offset.clone() {
                builder = builder.offset(o);
            }
            let resp = self.client.scroll(builder).await?;
            for pt in &resp.result {
                let advisor = match pt.payload.get("advisor").and_then(|v| v.kind.as_ref()) {
                    Some(Kind::StringValue(s)) if !s.is_empty() => s.clone(),
                    _ => continue,
                };
                let entry = videos_by_advisor.entry(advisor.clone()).or_default();
                if let Some(Kind::StringValue(vid)) =
                    pt.payload.get("video_id").and_then(|v| v.kind.as_ref())
                {
                    if !vid.is_empty() {
                        entry.insert(vid.clone());
                    }
                }
                if let Some(Kind::IntegerValue(ts)) =
                    pt.payload.get("ingested_at").and_then(|v| v.kind.as_ref())
                {
                    let cur = ingested_by_advisor.entry(advisor).or_insert(i64::MIN);
                    if *ts > *cur {
                        *cur = *ts;
                    }
                }
            }
            match resp.next_page_offset {
                Some(next) => offset = Some(next),
                None => break,
            }
        }
        Ok(videos_by_advisor
            .into_iter()
            .map(|(name, vids)| {
                let ts = ingested_by_advisor
                    .get(&name)
                    .copied()
                    .filter(|t| *t != i64::MIN);
                (name, vids.len(), ts)
            })
            .collect())
    }

    /// Fetch every user-added chunk for one expert (advisor == name AND source
    /// type creator_transcript or wikipedia_article), with the payload fields
    /// needed to rebuild readable per-video text. Used by the export endpoint.
    pub async fn fetch_expert_chunks(&self, name: &str) -> Result<Vec<ExportChunk>> {
        let filter = Filter {
            must: vec![Condition::matches("advisor", name.to_string())],
            should: vec![
                Condition::matches("source_type", "creator_transcript".to_string()),
                Condition::matches("source_type", "wikipedia_article".to_string()),
                Condition::matches("source_type", "document".to_string()),
            ],
            ..Default::default()
        };
        let mut out: Vec<ExportChunk> = Vec::new();
        let mut offset: Option<qdrant_client::qdrant::PointId> = None;
        loop {
            let mut builder = ScrollPointsBuilder::new(COLLECTION)
                .filter(filter.clone())
                .limit(256)
                .with_payload(true)
                .with_vectors(false);
            if let Some(o) = offset.clone() {
                builder = builder.offset(o);
            }
            let resp = self.client.scroll(builder).await?;
            for pt in &resp.result {
                let get_str = |k: &str| -> String {
                    match pt.payload.get(k).and_then(|v| v.kind.as_ref()) {
                        Some(Kind::StringValue(s)) => s.clone(),
                        _ => String::new(),
                    }
                };
                let chunk_idx = match pt.payload.get("chunk_idx").and_then(|v| v.kind.as_ref()) {
                    Some(Kind::IntegerValue(i)) => *i,
                    Some(Kind::DoubleValue(d)) => *d as i64,
                    _ => 0,
                };
                out.push(ExportChunk {
                    video_id: get_str("video_id"),
                    title: get_str("topic"),
                    chunk_idx,
                    content: get_str("content"),
                    source_type: get_str("source_type"),
                });
            }
            match resp.next_page_offset {
                Some(next) => offset = Some(next),
                None => break,
            }
        }
        Ok(out)
    }

    /// Total number of points in the collection. Cheap — uses Qdrant's
    /// `collection_info` endpoint, no scrolling.
    pub async fn count_points(&self) -> Result<u64> {
        let info = self.client.collection_info(COLLECTION).await?;
        Ok(info
            .result
            .and_then(|r| r.points_count)
            .unwrap_or(0))
    }

    /// Delete every point tagged `advisor == name` AND user-added source
    /// type (`creator_transcript` OR `wikipedia_article`). Curated KB
    /// chunks (`source_type == "kb_entry"`) are protected so even if a
    /// user-added expert shares a name with a baseline advisor, the
    /// curated content survives. Returns the number of points deleted.
    pub async fn delete_creator(&self, name: &str) -> Result<usize> {
        // Count first via scroll so we can report a deleted-count.
        // Qdrant's delete API doesn't return a count by itself.
        // Filter semantics: must (advisor == name) AND should-match-any
        // (source_type ∈ {creator_transcript, wikipedia_article}).
        let filter = Filter {
            must: vec![Condition::matches("advisor", name.to_string())],
            should: vec![
                Condition::matches("source_type", "creator_transcript".to_string()),
                Condition::matches("source_type", "wikipedia_article".to_string()),
                Condition::matches("source_type", "document".to_string()),
            ],
            ..Default::default()
        };

        let mut count = 0usize;
        let mut offset: Option<qdrant_client::qdrant::PointId> = None;
        loop {
            let mut builder = ScrollPointsBuilder::new(COLLECTION)
                .filter(filter.clone())
                .limit(1024)
                .with_payload(false)
                .with_vectors(false);
            if let Some(o) = offset.clone() {
                builder = builder.offset(o);
            }
            let resp = self.client.scroll(builder).await?;
            count += resp.result.len();
            match resp.next_page_offset {
                Some(next) => offset = Some(next),
                None => break,
            }
        }

        if count == 0 {
            return Ok(0);
        }

        let result = self
            .client
            .delete_points(
                DeletePointsBuilder::new(COLLECTION)
                    .points(PointsSelectorOneOf::Filter(filter))
                    .wait(true),
            )
            .await;
        self.invalidate_creators();
        result?;

        Ok(count)
    }

    /// Search top-k nearest chunks, optionally filtered to a single advisor.
    /// Returns them in descending score order.
    pub async fn search(
        &self,
        query: Vec<f32>,
        top_k: u64,
        advisor_filter: Option<&str>,
    ) -> Result<Vec<RetrievedChunk>> {
        let mut builder = SearchPointsBuilder::new(COLLECTION, query, top_k).with_payload(true);
        if let Some(advisor) = advisor_filter {
            builder = builder.filter(Filter::must([Condition::matches(
                "advisor",
                advisor.to_string(),
            )]));
        }
        let resp = self.client.search_points(builder).await?;

        let chunks = resp
            .result
            .into_iter()
            .map(|pt| {
                let score = pt.score;
                let payload = pt.payload;
                let get_str = |k: &str| -> String {
                    payload
                        .get(k)
                        .and_then(|v| v.kind.as_ref())
                        .and_then(|kind| match kind {
                            Kind::StringValue(s) => Some(s.clone()),
                            _ => None,
                        })
                        .unwrap_or_default()
                };
                RetrievedChunk {
                    source_type: get_str("source_type"),
                    advisor: get_str("advisor"),
                    topic: get_str("topic"),
                    content: get_str("content"),
                    speaker_mode: Some(get_str("speaker_mode")).filter(|s| !s.is_empty()),
                    score,
                }
            })
            .collect();

        Ok(chunks)
    }
}

/// Input to `upsert_batch` — one row per vector.
pub struct UpsertItem {
    pub key: String,
    pub vector: Vec<f32>,
    pub source_type: String,
    pub advisor: String,
    pub topic: String,
    pub content: String,
    pub chunk_idx: Option<u32>,
    pub video_id: Option<String>,
    pub keywords: Vec<String>,
    /// "single" (monologue) or "multiple" (conversation/panel), classified
    /// from the transcript at ingest time. None for non-video sources
    /// (KB entries, Wikipedia) and for chunks ingested before this feature.
    pub speaker_mode: Option<String>,
}
