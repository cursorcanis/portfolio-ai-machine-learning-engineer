#!/usr/bin/env python3
"""Per-video transcript fetcher - invoked by the Rust backend.

Reads one video ID on argv, prints a single JSON object on stdout, exits 0.
Status codes inside the JSON ("ok" / "no_transcript" / "error") let the
caller distinguish "skip this video" from "fix your Python install".

RESILIENCE: retry with exponential backoff + jitter on transient errors / 429s,
and keep volume modest (the backend enforces a per-hour fetch budget).

Env knobs:
    YT_FETCH_RETRIES            Max attempts on transient block (default 4).
    YT_FETCH_BASE_DELAY         Base backoff seconds, doubled each retry (default 1.5).
    YT_LANGS                    Comma-separated preferred languages (default "en").
    YT_TRANSLATE_TO             Translate a foreign track to this language
                                (default "en"); set empty to keep the original.

LANGUAGE FALLBACK: when none of the preferred languages exist, we take the
best track the video actually has (manually-created preferred over
auto-generated) rather than reporting "no transcript".

Translation to YT_TRANSLATE_TO is best-effort, NOT required for correctness:
retrieval embeds with multilingual-e5-small, which puts ~100 languages in one
vector space, so an English question already matches an Italian passage. We
still translate when YouTube offers it (English text reads better in quoted
answers), but a failure here is cosmetic - the chunk is indexed either way.
Note YouTube only offers translation for auto-generated tracks; manually
created captions report is_translatable=False and are indexed as-is.

Install: pip install --user --break-system-packages youtube-transcript-api
"""
import json
import os
import random
import re
import sys
import time


def _pick_fallback(transcript_list):
    """Best track when none of the preferred languages exist.

    Manually-created captions beat auto-generated ones (cleaner text, real
    punctuation). Returns None when the video exposes no tracks at all.
    """
    manual, generated = [], []
    for t in transcript_list:
        (generated if t.is_generated else manual).append(t)
    for pool in (manual, generated):
        if pool:
            return pool[0]
    return None


def _fetch_any(api, video_id, languages, translate_to, base_delay=1.5):
    """Fetch the preferred language, else fall back to any available track.

    Returns (fetched, original_language_code, translated, translation_error).
    Raises NoTranscriptFound only when the video truly exposes no tracks.
    """
    from youtube_transcript_api import NoTranscriptFound

    try:
        fetched = api.fetch(video_id, languages=languages)
        return fetched, fetched.language_code, False, None
    except NoTranscriptFound:
        pass  # fall through to the any-language path below

    chosen = _pick_fallback(api.list(video_id))
    if chosen is None:
        raise NoTranscriptFound(video_id, languages, [])

    original = chosen.language_code
    # Only translate when the track is in a different language than the
    # target. Compare base codes so "en-US" is not translated into "en".
    wants_translation = (
        translate_to
        and original.split("-")[0].lower() != translate_to.split("-")[0].lower()
        and getattr(chosen, "is_translatable", False)
    )
    if not wants_translation:
        return chosen.fetch(), original, False, None

    # The translation endpoint is rate-limited independently of the plain
    # fetch, so give it its own short backoff rather than degrading to
    # untranslated text on the first refusal.
    translation_error = None
    for attempt in range(3):
        try:
            return chosen.translate(translate_to).fetch(), original, True, None
        except Exception as e:  # noqa: BLE001 - classified by name
            translation_error = f"{type(e).__name__}"
            if attempt < 2 and _transient_block(type(e).__name__):
                time.sleep(base_delay * (2 ** attempt) + random.uniform(0, base_delay))
                continue
            break

    # Keep the original-language text - still worth indexing - but report the
    # degradation instead of hiding it, so the caller can log it.
    return chosen.fetch(), original, False, translation_error


def _transient_block(exc_name: str) -> bool:
    """Heuristic: is this exception a transient block worth retrying?"""
    transient = {
        "IpBlocked", "RequestBlocked", "YouTubeRequestFailed",
        "TooManyRequests", "ConnectionError", "Timeout", "ReadTimeout",
    }
    return exc_name in transient


def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "reason": "missing video_id argv"}))
        return 0

    raw = sys.argv[1]
    video_id = re.sub(r".*[?&]v=", "", raw).split("&")[0]
    langs_env = os.environ.get("YT_LANGS")
    languages = (
        sys.argv[2].split(",") if len(sys.argv) > 2
        else (langs_env.split(",") if langs_env else ["en"])
    )

    try:
        from youtube_transcript_api import (
            YouTubeTranscriptApi,
            NoTranscriptFound,
            TranscriptsDisabled,
        )
    except ImportError as e:
        print(json.dumps({
            "status": "error",
            "video_id": video_id,
            "reason": f"youtube-transcript-api not installed: {e}",
        }))
        return 0

    retries = int(os.environ.get("YT_FETCH_RETRIES", "4"))
    base_delay = float(os.environ.get("YT_FETCH_BASE_DELAY", "1.5"))

    translate_to = os.environ.get("YT_TRANSLATE_TO", "en").strip()

    last_err = None
    for attempt in range(retries):
        try:
            api = YouTubeTranscriptApi()
            transcript, original_lang, translated, translation_error = _fetch_any(
                api, video_id, languages, translate_to, base_delay
            )
            text = " ".join(s.text for s in transcript.snippets).strip()
            payload = {
                "status": "ok",
                "video_id": video_id,
                "language": transcript.language_code,
                "original_language": original_lang,
                "translated": translated,
                "is_generated": transcript.is_generated,
                "text": text,
            }
            if translation_error:
                payload["translation_error"] = translation_error
            print(json.dumps(payload, ensure_ascii=False))
            return 0
        except TranscriptsDisabled:
            # The uploader turned captions off entirely - no language helps.
            print(json.dumps({
                "status": "no_transcript",
                "video_id": video_id,
                "reason": "captions_disabled",
            }))
            return 0
        except NoTranscriptFound:
            # Reached only when the video exposes no tracks in ANY language.
            print(json.dumps({
                "status": "no_transcript",
                "video_id": video_id,
                "reason": "language_unavailable",
            }))
            return 0
        except Exception as e:  # noqa: BLE001 - we classify by name below
            last_err = e
            name = type(e).__name__
            if attempt < retries - 1 and _transient_block(name):
                # Exponential backoff with jitter, so repeated blocks back off
                # politely instead of hammering the endpoint.
                delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
                time.sleep(delay)
                continue
            break

    print(json.dumps({
        "status": "error",
        "video_id": video_id,
        "reason": f"{type(last_err).__name__}: {last_err}" if last_err else "unknown",
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
