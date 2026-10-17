//! Detect a document's language and translate it to English before indexing.
//!
//! Retrieval itself does not need this — the embedder is multilingual
//! (Entry 018), so an English question already matches a foreign passage. What
//! translation buys is the *answer*: the chat model mirrors the language of the
//! context it is given, so an untranslated corpus produces untranslated
//! replies, and quoted passages come back in the source language.
//!
//! Uses the OpenRouter model the app is already configured with, so there is
//! no second provider, key, or quota to manage.

use anyhow::{anyhow, Result};
use reqwest::Client;
use serde::Deserialize;
use tracing::{info, warn};

/// Translate in pieces: the model's *output* budget is the binding limit, not
/// its context window. ~3500 chars in yields comfortably under 2k tokens out.
const SEGMENT_CHARS: usize = 3000;

/// Refuse rather than quietly spend an enormous amount on one upload. At this
/// size the user is better served splitting the file, or indexing it
/// untranslated (retrieval still works).
pub const MAX_TRANSLATE_CHARS: usize = 250_000;

/// How much of the document to show the detector.
const DETECT_SAMPLE_CHARS: usize = 700;

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    content: Option<String>,
}

/// One non-streaming completion. `system` sets the task, `user` carries the payload.
async fn complete(
    client: &Client,
    api_key: &str,
    model: &str,
    system: &str,
    user: &str,
    max_tokens: u32,
) -> Result<String> {
    let body = serde_json::json!({
        "model": model,
        "max_tokens": max_tokens,
        // Deterministic: translation is not a place for creative variance.
        "temperature": 0,
        // The configured model may be a reasoning model (GLM-4.6 is). Reasoning
        // tokens are drawn from the SAME max_tokens budget as the reply, so a
        // tight budget gets spent thinking and returns empty content — which is
        // exactly how the first version of this failed. Translation is
        // mechanical, so switch reasoning off where the provider supports it
        // and leave generous headroom where it does not.
        "reasoning": { "enabled": false },
        "messages": [
            { "role": "system", "content": system },
            { "role": "user", "content": user }
        ]
    });

    let resp = client
        .post("https://openrouter.ai/api/v1/chat/completions")
        .bearer_auth(api_key)
        .header("HTTP-Referer", "http://localhost:3000")
        .header("X-Title", "AI Advisor")
        .json(&body)
        .send()
        .await
        .map_err(|e| anyhow!("translation request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(anyhow!("translation model returned {status}: {}", text.chars().take(200).collect::<String>()));
    }

    let parsed: ChatResponse = resp
        .json()
        .await
        .map_err(|e| anyhow!("could not parse translation response: {e}"))?;

    parsed
        .choices
        .first()
        .and_then(|c| c.message.content.clone())
        .map(|s| s.trim().to_string())
        .ok_or_else(|| anyhow!("translation model returned no content"))
}

/// Best-effort language identification. Returns a human-readable name such as
/// "Italian", or None when the model gives something unusable.
pub async fn detect_language(
    client: &Client,
    api_key: &str,
    model: &str,
    text: &str,
) -> Option<String> {
    let sample: String = text.chars().take(DETECT_SAMPLE_CHARS).collect();
    let out = complete(
        client,
        api_key,
        model,
        "You identify what language a text is written in. Reply with ONLY the \
         English name of the language, one word where possible, no punctuation \
         or explanation. Example replies: English. Italian. Brazilian Portuguese.",
        &sample,
        // Not 16: a reasoning model needs room to think before it answers, and
        // thinking is billed against this same budget.
        512,
    )
    .await;

    match out {
        Ok(s) => {
            // Guard against a chatty model: a language name is short.
            let name = s.trim().trim_end_matches('.').trim().to_string();
            if name.is_empty() || name.chars().count() > 30 {
                warn!("language detection returned something unusable: {s:?}");
                None
            } else {
                Some(name)
            }
        }
        Err(e) => {
            warn!("language detection failed: {e}");
            None
        }
    }
}

/// True when a detected language name means "already English".
pub fn is_english(language: &str) -> bool {
    let l = language.trim().to_ascii_lowercase();
    l == "english" || l.starts_with("english ") || l.ends_with(" english")
}

/// Split on paragraph boundaries, packing up to `SEGMENT_CHARS` per piece so a
/// sentence is never cut in half mid-translation.
fn segment(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();

    for para in text.split("\n\n") {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }
        // A single paragraph longer than the budget gets split on sentence ends.
        if para.chars().count() > SEGMENT_CHARS {
            if !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
            let mut piece = String::new();
            for sentence in split_sentences(para) {
                if piece.chars().count() + sentence.chars().count() > SEGMENT_CHARS
                    && !piece.is_empty()
                {
                    out.push(std::mem::take(&mut piece));
                }
                piece.push_str(&sentence);
            }
            if !piece.is_empty() {
                out.push(piece);
            }
            continue;
        }
        if current.chars().count() + para.chars().count() + 2 > SEGMENT_CHARS
            && !current.is_empty()
        {
            out.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(para);
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// Crude sentence splitter, keeping the terminator attached.
fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        cur.push(c);
        if matches!(c, '.' | '!' | '?' | '\u{3002}') {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out
}

/// Translate `text` into English, segment by segment.
///
/// `on_progress(done, total)` is called after each segment so a long document
/// can report movement instead of appearing hung.
pub async fn translate_to_english<F>(
    client: &Client,
    api_key: &str,
    model: &str,
    text: &str,
    source_language: &str,
    mut on_progress: F,
) -> Result<String>
where
    F: FnMut(usize, usize),
{
    if text.chars().count() > MAX_TRANSLATE_CHARS {
        return Err(anyhow!(
            "document is {} characters, over the {MAX_TRANSLATE_CHARS} translation limit. \
             Split the file, or index it untranslated - retrieval is multilingual, so an \
             English question still finds it.",
            text.chars().count()
        ));
    }

    let segments = segment(text);
    let total = segments.len();
    info!("translating {total} segment(s) from {source_language} to English");

    let system = format!(
        "You are a translator. Translate the user's {source_language} text into English. \
         Rules: output ONLY the translation, with no preamble, notes, or quotation marks \
         around it. Preserve paragraph breaks. Keep proper nouns, names, and technical \
         terms intact. Do not summarize, shorten, or explain - translate everything you \
         are given."
    );

    let mut out = String::with_capacity(text.len());
    for (i, seg) in segments.iter().enumerate() {
        // Headroom for three things at once: English can be longer than the
        // source, tokens are not characters, and any reasoning the model still
        // does comes out of this same budget.
        let max_tokens = ((seg.chars().count() as f32 * 1.5) as u32 + 1500).clamp(512, 8192);
        let translated = complete(client, api_key, model, &system, seg, max_tokens).await?;
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str(translated.trim());
        on_progress(i + 1, total);
    }

    if out.trim().is_empty() {
        return Err(anyhow!("translation produced no text"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_detection_accepts_the_shapes_a_model_returns() {
        assert!(is_english("English"));
        assert!(is_english("english"));
        assert!(is_english(" English "));
        assert!(is_english("American English"));
        assert!(!is_english("Italian"));
        assert!(!is_english("Englishman's Portuguese"));
    }

    #[test]
    fn segments_stay_under_budget_and_keep_every_paragraph() {
        let para = "Sentence one is here. ".repeat(40); // ~880 chars
        let doc = vec![para.trim(); 12].join("\n\n"); // ~10.5k chars
        let segs = segment(&doc);
        assert!(segs.len() > 1, "should split a 10k document");
        for s in &segs {
            assert!(
                s.chars().count() <= SEGMENT_CHARS + 200,
                "segment too long: {}",
                s.chars().count()
            );
        }
        // Nothing dropped: every paragraph survives somewhere.
        let joined = segs.join(" ");
        assert_eq!(
            joined.matches("Sentence one is here.").count(),
            doc.matches("Sentence one is here.").count()
        );
    }

    #[test]
    fn a_single_huge_paragraph_is_split_on_sentences() {
        let doc = "This is a sentence. ".repeat(500); // ~10k chars, no blank lines
        let segs = segment(&doc);
        assert!(segs.len() > 1, "a wall of text must still be split");
        for s in &segs {
            assert!(s.chars().count() <= SEGMENT_CHARS + 200, "segment too long");
        }
    }

    #[test]
    fn short_text_stays_one_segment() {
        let segs = segment("Just a line.\n\nAnd another.");
        assert_eq!(segs.len(), 1);
    }
}
