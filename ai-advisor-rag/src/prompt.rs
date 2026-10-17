use std::collections::{BTreeMap, HashSet};

use crate::models::RetrievedChunk;

/// Appended AFTER the retrieved context in every prompt variant.
///
/// Sources are indexed in their original language (the multilingual embedder
/// retrieves across languages, so no translation happens at ingest). Several
/// thousand tokens of, say, Italian transcript sitting directly above the
/// answer will pull the model into replying in Italian — a rule placed
/// mid-prompt loses to that, which is exactly what happened in testing. A
/// trailing directive wins on recency, same lesson as Entry 016.
const LANGUAGE_DIRECTIVE: &str = "\n\nCRITICAL — RESPONSE LANGUAGE: The retrieved context above may be in a different language than the user's question. Write your ENTIRE response in the same language the user wrote their most recent message in. Translate the substance of the source material into that language. Do NOT reply in the language of the source material just because the context is in it.";

/// Tone annotation for context that comes from single-narrator monologues.
pub const SINGLE_SPEAKER_NOTE: &str = "You are chatting with a single person. \
The derived tone of the response is of a single person";

/// Tone annotation for context that includes multi-speaker conversations.
pub const MULTI_SPEAKER_NOTE: &str = "Chatting with this expert will be the \
combination of opinions of all people in this video. Therefore, the tone of \
the response will be an aggregate tone of all persons included in the video.";

/// Pick the tone annotation for one advisor's retrieved chunks, from the
/// `speaker_mode` each video was labeled with at ingest time. Any
/// multi-speaker source in the mix wins (the context already blends voices);
/// otherwise single-narrator; None when no chunk carries a label (KB entries,
/// Wikipedia, pre-feature ingests).
fn speaker_note(chunks: &[&RetrievedChunk]) -> Option<&'static str> {
    let mut any_single = false;
    for c in chunks {
        match c.speaker_mode.as_deref() {
            Some("multiple") => return Some(MULTI_SPEAKER_NOTE),
            Some("single") => any_single = true,
            _ => {}
        }
    }
    if any_single {
        Some(SINGLE_SPEAKER_NOTE)
    } else {
        None
    }
}

/// Voice guide descriptions for each advisor persona.
const VOICE_GUIDES: &str = "\
- Dana Whitfield: Intellectual, urgent, humanistic. Speaks like a professor who genuinely cares. \
Uses metaphors from biology and media theory. Challenges the listener to think deeper.
- Morgan Hale: Sharp, analytical, measured. Speaks like a journalist building an argument brick by brick. \
References systems and structures. Asks rhetorical questions.
- Wandering Engineer: Direct, no-nonsense, everyman. Speaks like a guy at a bar who's seen the world and \
is tired of the BS. Blunt, practical, occasionally cynical.
- AiScout: Conversational, practical, honest. Speaks like a YouTuber breaking down AI news for real \
developers. Evaluates models based on actual usability, not hype. Cares about accessibility and free \
access. Keeps it real about open vs closed source.
- Elliot Marsh: Prophetic, literary, uncompromising. Speaks like a war correspondent turned moral \
philosopher. Vivid imagery, historical references, righteous anger.
- Sam Kessler: Technical, skeptical, urgent. Speaks like a hacker who's seen behind the curtain and \
wants to warn you. Breaks down complex tech into plain language. Treats privacy as non-negotiable.";

/// Build the full system prompt from the top-k retrieved chunks and an optional
/// keyword-matched transcript. Chunks are grouped by advisor so the model sees
/// each voice as a coherent context block.
///
/// `scoped_to`: when `Some(name)`, the prompt narrows to that single expert —
/// no list of other advisors, no static voice-guide block, the retrieved
/// chunks themselves become the model's voice reference. Use this for
/// "focused chat with one expert" mode.
pub fn build_system_prompt(
    experts: &HashSet<String>,
    retrieved: &[RetrievedChunk],
    transcript: &str,
    scoped_to: Option<&str>,
) -> String {
    // Group retrieved chunks by advisor, preserving the global order (score desc)
    // within each group.
    let mut grouped: BTreeMap<String, Vec<&RetrievedChunk>> = BTreeMap::new();
    for chunk in retrieved {
        let key = if chunk.advisor.is_empty() {
            "Unattributed".to_string()
        } else {
            chunk.advisor.clone()
        };
        grouped.entry(key).or_default().push(chunk);
    }

    let context_block = if grouped.is_empty() {
        "(no specific knowledge-base context matched this query)".to_string()
    } else {
        let mut out = String::new();
        for (advisor, chunks) in &grouped {
            out.push_str(&format!("\n### {advisor}\n"));
            if let Some(note) = speaker_note(chunks) {
                out.push_str(&format!("[Tone note: {note}]\n"));
            }
            for c in chunks {
                if !c.topic.is_empty() {
                    out.push_str(&format!("**{}** (score {:.2})\n", c.topic, c.score));
                } else {
                    out.push_str(&format!("(score {:.2})\n", c.score));
                }
                out.push_str(&c.content);
                out.push_str("\n\n");
            }
        }
        out
    };

    let base = if let Some(name) = scoped_to {
        // Scoped mode: single-expert focus. The retrieved chunks ARE the voice
        // reference — we don't pull from the static voice-guide table because
        // the expert may not be one of the original 6, and even if they are,
        // their actual transcripts are a fresher voice signal than the guide.
        //
        // The framing itself is speaker-mode-aware: when the sources are
        // multi-speaker conversations, "channel {name} in first person" is the
        // wrong instruction — the persona is the aggregate of everyone in the
        // videos, and the model must say so instead of impersonating one voice.
        let all_chunks: Vec<&RetrievedChunk> = retrieved.iter().collect();
        if speaker_note(&all_chunks) == Some(MULTI_SPEAKER_NOTE) {
            format!(
                r#"
You are channeling "{name}". The source material below comes from conversations involving MULTIPLE people (interviews, panels, or dialogues) — it is NOT a single narrator.

IMPORTANT RULES:
- {MULTI_SPEAKER_NOTE}
- Blend the viewpoints of EVERYONE speaking in the source material into one combined voice. Do NOT impersonate only one participant.
- Where the participants disagree or emphasize different things, synthesize the range of opinions and, when relevant, say so (e.g. "one voice in this conversation argues X, another stresses Y").
- The retrieved context below is transcripts of those conversations. Treat the full set of voices in it as your reference.
- Do NOT reference, mention, or compare to other experts or advisors outside this material.
- If the user asks about a topic the participants haven't directly addressed, extrapolate from their combined worldview rather than refusing.
- Answer in the SAME language the user wrote their question in. Retrieved context may be in another language (sources are indexed in their original language); translate its substance into the user's language rather than replying in the source language.
- TypeScript only for any code examples.

Retrieved context from "{name}" (multi-speaker sources, grouped chronologically):
{context_block}
"#
            )
        } else {
            format!(
                r#"
You are channeling {name}. Every response in this conversation should be in {name}'s authentic voice — their cadence, vocabulary, attitude, and characteristic concerns.

IMPORTANT RULES:
- Speak AS {name}, not ABOUT them. Use first person.
- The retrieved context below is from {name}'s actual writing or transcripts. Treat it as your voice reference — match its tone and style.
- Do NOT reference, mention, or compare to other experts or advisors. The user has scoped this conversation to {name} specifically.
- If the user asks about a topic {name} hasn't directly addressed, extrapolate from their broader worldview rather than refusing.
- If the retrieved context carries a [Tone note: ...], follow it when deriving the voice: single-person sources mean one individual's tone; multi-speaker sources mean an aggregate tone of everyone in those videos.
- Answer in the SAME language the user wrote their question in. Retrieved context may be in another language (sources are indexed in their original language); translate its substance into the user's language rather than replying in the source language.
- TypeScript only for any code examples.

Retrieved context from {name} (grouped chronologically by source):
{context_block}
"#
            )
        }
    } else {
        let experts_joined = experts
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            r#"
You have access to a knowledge base of experts and thought leaders.
The experts are: {experts_joined}.

IMPORTANT RULES FOR RESPONDING:
- When the user asks about a specific advisor or a topic tied to one, adopt that advisor's voice, tone, and personality. Speak AS them, not ABOUT them. Use their style, their cadence, their attitude. Channel them like you're doing an impression.
- When multiple advisors are relevant, give each one their own paragraph and speak in each of their distinct voices.
- When no specific advisor is relevant, respond in a neutral helpful tone.
- When an advisor's context block carries a [Tone note: ...], follow it: single-person sources keep that one person's individual tone; multi-speaker sources blend an aggregate tone of all people in those videos.
- Answer in the SAME language the user wrote their question in. Retrieved context may be in another language (sources are indexed in their original language); translate its substance into the user's language rather than replying in the source language.
- You only use TypeScript in your responses for code examples.
- If the user asks about an advisor not in the knowledge base, say you don't have info on them.
- Don't just parrot the knowledge base entries — riff on them, expand on them, and respond to the user's specific question in the advisor's authentic voice.

Here are the advisor voice guides:
{VOICE_GUIDES}

Here is the most relevant context retrieved for this query (grouped by advisor):
{context_block}
"#
        )
    };

    let mut out = if transcript.is_empty() {
        base
    } else {
        format!(
            "{base}\n\nHere is a full transcript from the relevant expert:\n{transcript}\n\n\
             Use this transcript to provide deeper, more specific answers in the advisor's voice."
        )
    };
    // Must be appended here, not inside the templates above: the optional
    // full transcript is added AFTER the context block, and the directive
    // only works when nothing follows it. Same ordering rule as
    // build_split_prompt.
    out.push_str(LANGUAGE_DIRECTIVE);
    out
}

/// Build the system prompt as a (static, dynamic) tuple for providers that
/// support prompt caching (Anthropic). The static prefix contains the truly
/// invariant content — voice guides + the rules block — and is the cache
/// target. The dynamic suffix contains the per-request bits — the current
/// experts set, the retrieved chunks, and any transcript.
///
/// Caching note: Sonnet 4.6 has a **2048-token minimum cacheable prefix**
/// (Opus 4.7/4.6/Haiku 4.5: 4096). The current static prefix is ~430 tokens
/// (just the voice guides + rules), so the cache marker we set on the static
/// block will silently no-op until the prefix grows past the threshold. The
/// marker is wired in correctly so savings kick in automatically once it does.
///
/// In scoped mode the static prefix is empty — the prompt is single-expert
/// and inherently dynamic, so caching doesn't apply.
pub fn build_split_prompt(
    experts: &HashSet<String>,
    retrieved: &[RetrievedChunk],
    transcript: &str,
    scoped_to: Option<&str>,
) -> (String, String) {
    if scoped_to.is_some() {
        // Scoped mode: no cacheable prefix, return everything as the dynamic
        // half. Caller passes an empty static block to the LLM.
        return (
            String::new(),
            build_system_prompt(experts, retrieved, transcript, scoped_to),
        );
    }

    let static_part = format!(
        r#"
You have access to a knowledge base of experts and thought leaders.

IMPORTANT RULES FOR RESPONDING:
- When the user asks about a specific advisor or a topic tied to one, adopt that advisor's voice, tone, and personality. Speak AS them, not ABOUT them. Use their style, their cadence, their attitude. Channel them like you're doing an impression.
- When multiple advisors are relevant, give each one their own paragraph and speak in each of their distinct voices.
- When no specific advisor is relevant, respond in a neutral helpful tone.
- When an advisor's context block carries a [Tone note: ...], follow it: single-person sources keep that one person's individual tone; multi-speaker sources blend an aggregate tone of all people in those videos.
- Answer in the SAME language the user wrote their question in. Retrieved context may be in another language (sources are indexed in their original language); translate its substance into the user's language rather than replying in the source language.
- You only use TypeScript in your responses for code examples.
- If the user asks about an advisor not in the knowledge base, say you don't have info on them.
- Don't just parrot the knowledge base entries — riff on them, expand on them, and respond to the user's specific question in the advisor's authentic voice.

Here are the advisor voice guides:
{VOICE_GUIDES}
"#
    );

    let mut grouped: BTreeMap<String, Vec<&RetrievedChunk>> = BTreeMap::new();
    for chunk in retrieved {
        let key = if chunk.advisor.is_empty() {
            "Unattributed".to_string()
        } else {
            chunk.advisor.clone()
        };
        grouped.entry(key).or_default().push(chunk);
    }
    let context_block = if grouped.is_empty() {
        "(no specific knowledge-base context matched this query)".to_string()
    } else {
        let mut out = String::new();
        for (advisor, chunks) in &grouped {
            out.push_str(&format!("\n### {advisor}\n"));
            if let Some(note) = speaker_note(chunks) {
                out.push_str(&format!("[Tone note: {note}]\n"));
            }
            for c in chunks {
                if !c.topic.is_empty() {
                    out.push_str(&format!("**{}** (score {:.2})\n", c.topic, c.score));
                } else {
                    out.push_str(&format!("(score {:.2})\n", c.score));
                }
                out.push_str(&c.content);
                out.push_str("\n\n");
            }
        }
        out
    };

    let experts_joined = experts
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join(", ");

    let mut dynamic_part = format!(
        "Currently available experts in the knowledge base: {experts_joined}.\n\
         \n\
         Here is the most relevant context retrieved for this query (grouped by advisor):\n\
         {context_block}"
    );
    if !transcript.is_empty() {
        dynamic_part.push_str(&format!(
            "\n\nHere is a full transcript from the relevant expert:\n{transcript}\n\n\
             Use this transcript to provide deeper, more specific answers in the advisor's voice."
        ));
    }
    // Must be last: it has to sit after ALL retrieved material, including the
    // optional full transcript appended above.
    dynamic_part.push_str(LANGUAGE_DIRECTIVE);

    (static_part, dynamic_part)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(advisor: &str, content: &str) -> RetrievedChunk {
        RetrievedChunk {
            source_type: "creator_transcript".into(),
            advisor: advisor.into(),
            topic: "t".into(),
            content: content.into(),
            score: 0.9,
            speaker_mode: Some("single".into()),
        }
    }

    /// The directive only works if nothing follows it. The optional keyword
    /// transcript is appended after the context block, so a directive embedded
    /// in the template would end up buried mid-prompt — which is the exact
    /// failure it exists to prevent.
    #[test]
    fn language_directive_is_last_even_with_a_full_transcript() {
        let experts: HashSet<String> = ["Someone".to_string()].into_iter().collect();
        let retrieved = vec![chunk("Someone", "una frase in italiano")];

        for scope in [Some("Someone"), None] {
            for transcript in ["", "a very long full transcript in the source language"] {
                let p = build_system_prompt(&experts, &retrieved, transcript, scope);
                assert!(
                    p.trim_end().ends_with(LANGUAGE_DIRECTIVE.trim_end()),
                    "directive must be last (scope={scope:?}, transcript_empty={})",
                    transcript.is_empty()
                );
                assert_eq!(
                    p.matches("CRITICAL — RESPONSE LANGUAGE").count(),
                    1,
                    "directive must appear exactly once"
                );
            }
        }

        // The split builder feeds the LLM static + dynamic concatenated, so the
        // dynamic half is what has to end with it.
        let (_static_part, dynamic) =
            build_split_prompt(&experts, &retrieved, "a full transcript", None);
        assert!(dynamic.trim_end().ends_with(LANGUAGE_DIRECTIVE.trim_end()));
    }
}
