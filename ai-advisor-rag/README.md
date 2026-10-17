# AI Advisor: Rust RAG backend

A retrieval-augmented chat backend that answers questions from video creators' transcripts and uploaded documents, grounded in what the sources actually say. This is the public showcase copy of a personal prototype; see [What was removed](#what-was-removed-from-this-public-copy).

## Stack
- **Rust:** Axum HTTP server, Tokio, token streaming over server-sent events.
- **Retrieval:** Qdrant vector store. Embeddings via `fastembed` using `multilingual-e5-small` (384 dims, ~100 languages in one vector space).
- **Generation:** an LLM via OpenRouter.
- **Ingestion:** async jobs that list a channel's videos (yt-dlp), fetch captions (`scripts/fetch_transcript.py`), chunk them (500 characters, 50 overlap) and batch-embed. Uploaded text documents are parsed in `documents.rs` and optionally translated (`translate.rs`).
- **History:** the backend was migrated incrementally from a Next.js/Node API to Rust using a strangler-fig approach.

## The multilingual retrieval decision
The first version embedded with the English-only `bge-small-en-v1.5`, so Italian sources retrieved poorly and ingestion depended on a rate-limited translation endpoint. Switching to `multilingual-e5-small`, with E5's `query:`/`passage:` prefixes enforced inside `embeddings.rs`, was benchmarked separately: 70 questions over 689 Wikipedia chunks using this app's chunking and top-8 retrieval.

| English question about an Italian source | before | after |
|---|---|---|
| correct passage in top 8 | 52% | 85% |
| correct passage ranked 1st | 28% | 50% |

No loss was detected on a small English set. The full benchmark method, limitations and charts are in a separate portfolio write-up, available on request.

Answers can still drift into the source's language. That's why the language directive is placed after all retrieved context (`prompt.rs`, tested), and why uploaded documents can optionally be translated.

## Run
```bash
# Qdrant on :6333/:6334 (see scripts/start-qdrant.ps1), then:
export OPENROUTER_API_KEY=...        # required for chat and translation
pip install youtube-transcript-api   # caption fetcher used by ingestion
cargo run --release                  # serves on :3002
cargo test --lib                     # 31 unit tests
```

## What was removed from this public copy
- The original personas imitated real public figures. They are replaced with fictional examples (e.g. "Dana Whitfield").
- Authenticated YouTube access (per-user cookie upload and cookie-based yt-dlp/transcript requests), plus browser-header rotation and proxy fallback in the caption fetcher. The public copy makes only plain, rate-limited requests.
- Git history, the knowledge-base content, local launch scripts, and machine-specific build config.

The original repository has 34 unit tests. The 3 removed here covered the cookie module.
