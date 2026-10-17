//! Expert lookup: name → Wikipedia bio + YouTube channel candidates.
//!
//! The user types "Dana Whitfield" in the search field; we return:
//!   - a Wikipedia summary so the user can verify they got the right person
//!   - a list of YouTube channels mentioning that name, ranked by how often
//!     they appear in `ytsearch` results (most-mentioned ≈ their main channel)
//!
//! From there the frontend asks the user to confirm a candidate; that
//! confirmation hits `POST /api/experts/confirm` which kicks off the existing
//! `index_creator` ingestion pipeline.

use std::collections::HashMap;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use tokio::process::Command;
use tracing::{info, warn};


const YT_SEARCH_LIMIT: usize = 10;

/// Wikipedia REST summary — minimal subset of the `page/summary` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WikipediaSummary {
    pub title: String,
    pub description: Option<String>,
    pub extract: Option<String>,
    pub url: Option<String>,
    pub thumbnail: Option<String>,
}

/// One YouTube channel that surfaced in a search for the expert's name.
#[derive(Debug, Clone, Serialize)]
pub struct ChannelCandidate {
    pub channel: String,
    pub channel_id: String,
    pub channel_url: String,
    /// How many videos in the search results came from this channel.
    /// Higher = more likely their main channel.
    pub appearance_count: u32,
}

/// Combined search response.
#[derive(Debug, Clone, Serialize)]
pub struct ExpertSearchResult {
    pub name: String,
    pub wikipedia: Option<WikipediaSummary>,
    pub youtube_candidates: Vec<ChannelCandidate>,
}

/// Top-level entry point: hit Wikipedia + YouTube in parallel and return both.
pub async fn search_expert(
    client: &reqwest::Client,
    name: &str,
) -> ExpertSearchResult {
    let name = name.trim();
    let (wiki, channels) = tokio::join!(
        wikipedia_summary(client, name),
        youtube_channel_candidates(name)
    );

    ExpertSearchResult {
        name: name.to_string(),
        wikipedia: match wiki {
            Ok(s) => Some(s),
            Err(e) => {
                warn!("wikipedia lookup for '{name}' failed: {e}");
                None
            }
        },
        youtube_candidates: match channels {
            Ok(c) => c,
            Err(e) => {
                warn!("youtube channel search for '{name}' failed: {e}");
                Vec::new()
            }
        },
    }
}

/// Hit `https://en.wikipedia.org/api/rest_v1/page/summary/<title>` and shape
/// the response into our summary struct. The endpoint redirects through
/// disambiguation for ambiguous names; for clearly missing names it 404s.
pub async fn wikipedia_summary(client: &reqwest::Client, name: &str) -> Result<WikipediaSummary> {
    // Wikipedia paths use underscores for spaces; everything else gets
    // percent-encoded by reqwest's URL parser.
    let title = name.replace(' ', "_");
    let url = format!("https://en.wikipedia.org/api/rest_v1/page/summary/{title}");
    info!("wikipedia GET {url}");

    let resp = client
        .get(&url)
        .header("User-Agent", "ai-advisor-rs/0.1 (expert-search)")
        .timeout(std::time::Duration::from_secs(8))
        .send()
        .await
        .context("wikipedia request failed")?;

    if !resp.status().is_success() {
        return Err(anyhow!("wikipedia returned {}", resp.status()));
    }

    let body: serde_json::Value = resp.json().await.context("invalid wikipedia JSON")?;

    // `type: "disambiguation"` means "many people share this name" — we still
    // return the extract so the user can pick a more specific search term.
    let title = body
        .get("title")
        .and_then(|x| x.as_str())
        .unwrap_or(name)
        .to_string();
    let description = body
        .get("description")
        .and_then(|x| x.as_str())
        .map(str::to_string);
    let extract = body
        .get("extract")
        .and_then(|x| x.as_str())
        .map(str::to_string);
    let url = body
        .get("content_urls")
        .and_then(|x| x.get("desktop"))
        .and_then(|x| x.get("page"))
        .and_then(|x| x.as_str())
        .map(str::to_string);
    let thumbnail = body
        .get("thumbnail")
        .and_then(|x| x.get("source"))
        .and_then(|x| x.as_str())
        .map(str::to_string);

    Ok(WikipediaSummary {
        title,
        description,
        extract,
        url,
        thumbnail,
    })
}

/// Fetch the full plaintext body of a Wikipedia article via the MediaWiki
/// Action API (`action=query&prop=extracts&explaintext=true`).
///
/// Used for the on-topic baseline source per Beads `ai_advisor-4y9` — for
/// figures with no personal YouTube channel (e.g. Lenny Bruce), Wikipedia
/// is a guaranteed-on-topic backstop alongside whatever YouTube returns.
///
/// Returns the extracted plaintext (markdown-ish — section headings preserved
/// as `== Heading ==`). Returns `Ok(None)` when the page doesn't exist or has
/// an empty extract. Disambiguation pages also yield an extract; the caller
/// decides whether to ingest them.
pub async fn wikipedia_article_text(
    client: &reqwest::Client,
    title: &str,
) -> Result<Option<String>> {
    // Wikipedia paths use underscores for spaces; the API also accepts spaces.
    let title = title.replace(' ', "_");
    let url = format!(
        "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=true&format=json&redirects=1&titles={title}"
    );
    info!("wikipedia article GET {url}");

    let resp = client
        .get(&url)
        .header("User-Agent", "ai-advisor-rs/0.1 (article-ingest)")
        .send()
        .await
        .context("wikipedia article request failed")?;

    if !resp.status().is_success() {
        return Err(anyhow!("wikipedia article API returned {}", resp.status()));
    }

    let body: serde_json::Value = resp
        .json()
        .await
        .context("invalid wikipedia article JSON")?;

    // Response shape: { query: { pages: { "<pageid>": { extract: "..." } } } }
    // The pages object has exactly one entry (or one with `missing: ""` for 404s).
    let pages = match body.pointer("/query/pages").and_then(|p| p.as_object()) {
        Some(p) => p,
        None => return Ok(None),
    };

    let extract = pages
        .values()
        .next()
        .and_then(|page| page.get("extract"))
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    if extract.is_empty() {
        Ok(None)
    } else {
        Ok(Some(extract))
    }
}

/// Run `yt-dlp "ytsearch10:<name>" --flat-playlist --dump-json`, then count
/// how many results came from each channel. Returns the top channels sorted
/// by appearance count.
pub async fn youtube_channel_candidates(
    name: &str,
) -> Result<Vec<ChannelCandidate>> {
    let query = format!("ytsearch{YT_SEARCH_LIMIT}:{name}");

    let mut args: Vec<String> = Vec::new();
    args.extend([
        // Fail fast when YouTube is slow / rate-limiting. The default is a 20s
        // read timeout retried 3x (~60s), which would hang the whole /search
        // request until the frontend proxy gives up with a 500.
        "--socket-timeout".into(),
        "5".into(),
        "--retries".into(),
        "1".into(),
        "--extractor-retries".into(),
        "0".into(),
        "--no-warnings".into(),
        "--flat-playlist".into(),
        "--dump-json".into(),
        query,
    ]);

    // Hard ceiling so a stuck yt-dlp can't stall the request past the frontend
    // proxy's own timeout (which would surface as a 500). kill_on_drop ensures
    // the timeout actually terminates yt-dlp instead of leaving a zombie that
    // keeps hammering YouTube and worsens the throttling. On timeout we return
    // an error, which the caller turns into "no channels surfaced" — the
    // Wikipedia half of the search still comes back.
    let fut = Command::new("yt-dlp").args(&args).kill_on_drop(true).output();
    let output = match tokio::time::timeout(std::time::Duration::from_secs(12), fut).await {
        Ok(res) => res.context("failed to spawn yt-dlp for channel search")?,
        Err(_) => {
            return Err(anyhow!(
                "yt-dlp channel search timed out (YouTube slow or rate-limiting)"
            ));
        }
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!(
            "yt-dlp search exited with status {}: {}",
            output.status,
            stderr.lines().last().unwrap_or("").trim()
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut counts: HashMap<String, ChannelCandidate> = HashMap::new();

    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let channel = v
            .get("channel")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let channel_id = v
            .get("channel_id")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let channel_url = v
            .get("channel_url")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        if channel_id.is_empty() {
            continue;
        }
        counts
            .entry(channel_id.clone())
            .and_modify(|c| c.appearance_count += 1)
            .or_insert(ChannelCandidate {
                channel,
                channel_id,
                channel_url,
                appearance_count: 1,
            });
    }

    let mut candidates: Vec<ChannelCandidate> = counts.into_values().collect();
    candidates.sort_by(|a, b| b.appearance_count.cmp(&a.appearance_count));
    Ok(candidates)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_candidates_sort_by_appearance() {
        let mut cands = vec![
            ChannelCandidate {
                channel: "A".into(),
                channel_id: "a".into(),
                channel_url: "".into(),
                appearance_count: 1,
            },
            ChannelCandidate {
                channel: "B".into(),
                channel_id: "b".into(),
                channel_url: "".into(),
                appearance_count: 5,
            },
            ChannelCandidate {
                channel: "C".into(),
                channel_id: "c".into(),
                channel_url: "".into(),
                appearance_count: 3,
            },
        ];
        cands.sort_by(|a, b| b.appearance_count.cmp(&a.appearance_count));
        assert_eq!(
            cands.iter().map(|c| c.channel.as_str()).collect::<Vec<_>>(),
            vec!["B", "C", "A"]
        );
    }
}
