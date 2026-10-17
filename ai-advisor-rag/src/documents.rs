//! Turn an uploaded text file into clean prose ready for chunking.
//!
//! The browser reads the file and posts its text, so everything here is
//! string work — no new dependencies, no temp files. Each supported format
//! gets just enough handling to strip the scaffolding that would otherwise be
//! embedded and retrieved as if it were content: subtitle timecodes, HTML
//! tags, RTF control words.
//!
//! Binary formats (PDF, DOCX, ...) are rejected with a specific message
//! rather than silently indexing mojibake — a PDF read as text is a wall of
//! `%PDF` and stream operators, which embeds as convincing-looking noise.

use anyhow::{anyhow, Result};

/// Upper bound on a single upload, in characters. Roughly 100k tokens, which
/// is a large book chapter; past this the job is almost certainly a mistake.
pub const MAX_DOC_CHARS: usize = 400_000;

/// Extensions we can read as text. Anything else is refused up front.
const TEXT_EXTENSIONS: &[&str] = &[
    "txt", "text", "log", "md", "markdown", "mdown", "rst", "csv", "tsv",
    "json", "jsonl", "ndjson", "yaml", "yml", "toml", "ini", "conf",
    "html", "htm", "xml", "srt", "vtt", "sbv", "rtf", "tex", "org",
];

/// Lowercase extension of a filename, or "" when there is none.
pub fn extension_of(filename: &str) -> String {
    filename
        .rsplit_once('.')
        .map(|(_, ext)| ext.trim().to_ascii_lowercase())
        .unwrap_or_default()
}

/// True when the payload looks like binary rather than text: NUL bytes or a
/// recognizable magic prefix. Catches the common "user dragged a PDF" case.
pub fn looks_binary(raw: &str) -> bool {
    if raw.contains('\u{0}') {
        return true;
    }
    let head = raw.trim_start();
    head.starts_with("%PDF")
        || head.starts_with("PK\u{3}\u{4}") // zip container: docx/odt/xlsx
        || head.starts_with("{\\rtf1\\ansi\\ansicpg")
            && raw.matches("\\pict").count() > 0
}

/// Human-readable list for error messages and the file picker.
pub fn supported_list() -> String {
    TEXT_EXTENSIONS
        .iter()
        .map(|e| format!(".{e}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Extract indexable prose from an uploaded file.
///
/// Returns an error (rather than empty text) whenever the caller should be
/// told something concrete: wrong format, binary payload, nothing left after
/// stripping.
pub fn extract(filename: &str, raw: &str) -> Result<String> {
    let ext = extension_of(filename);

    if looks_binary(raw) {
        return Err(anyhow!(
            "'{filename}' looks like a binary file, not text. PDF and Word documents \
             are not supported yet - export or save it as .txt or .md first. \
             Supported: {}",
            supported_list()
        ));
    }

    // An unknown extension is allowed through when the content is plainly
    // text; refusing on extension alone would reject perfectly good files
    // that merely lack one.
    if !ext.is_empty() && !TEXT_EXTENSIONS.contains(&ext.as_str()) {
        return Err(anyhow!(
            "'.{ext}' files are not supported. Supported: {}",
            supported_list()
        ));
    }

    let text = match ext.as_str() {
        "srt" | "vtt" | "sbv" => strip_subtitles(raw),
        "html" | "htm" | "xml" => strip_markup(raw),
        "rtf" => strip_rtf(raw),
        "md" | "markdown" | "mdown" => strip_markdown(raw),
        _ => raw.to_string(),
    };

    let cleaned = normalize_whitespace(&text);
    if cleaned.is_empty() {
        return Err(anyhow!(
            "'{filename}' contained no readable text after parsing."
        ));
    }
    if cleaned.chars().count() > MAX_DOC_CHARS {
        return Err(anyhow!(
            "'{filename}' is {} characters, over the {MAX_DOC_CHARS} limit. \
             Split it into smaller files and upload them one at a time.",
            cleaned.chars().count()
        ));
    }
    Ok(cleaned)
}

/// Drop cue numbers, timecodes and positioning from SRT / VTT / SBV, keeping
/// only spoken text. Consecutive duplicate lines are collapsed because
/// rolling-caption formats repeat each line as it scrolls.
fn strip_subtitles(raw: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    for line in raw.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        // "WEBVTT", "NOTE ...", "STYLE", cue settings
        if t.starts_with("WEBVTT") || t.starts_with("NOTE") || t.starts_with("STYLE")
            || t.starts_with("REGION")
        {
            continue;
        }
        // Timecode lines: contain "-->" (SRT/VTT) or are SBV "0:00:01.000,0:00:03.000"
        if t.contains("-->") {
            continue;
        }
        if t.matches(':').count() >= 2 && t.contains(',') && !t.contains(' ') {
            continue;
        }
        // Bare cue index
        if t.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let cleaned = strip_markup(t);
        let cleaned = cleaned.trim();
        if cleaned.is_empty() {
            continue;
        }
        if out.last().map(|p| p == cleaned).unwrap_or(false) {
            continue;
        }
        out.push(cleaned.to_string());
    }
    out.join(" ")
}

/// Remove tags and the contents of script/style, then decode the handful of
/// entities that actually show up in prose.
fn strip_markup(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut in_tag = false;
    let mut skip_depth = 0usize;
    let lower = raw.to_ascii_lowercase();
    let bytes: Vec<char> = raw.chars().collect();
    let lower_chars: Vec<char> = lower.chars().collect();
    let mut i = 0usize;

    while i < bytes.len() {
        let c = bytes[i];
        if c == '<' {
            // Opening of a block whose text is not content.
            let rest: String = lower_chars[i..(i + 8).min(lower_chars.len())].iter().collect();
            if rest.starts_with("<script") || rest.starts_with("<style") {
                skip_depth += 1;
            } else if rest.starts_with("</script") || rest.starts_with("</style") {
                skip_depth = skip_depth.saturating_sub(1);
            }
            in_tag = true;
            i += 1;
            continue;
        }
        if c == '>' {
            in_tag = false;
            // Tags act as word boundaries; without this "<p>a</p><p>b</p>"
            // would come out as "ab".
            if skip_depth == 0 {
                out.push(' ');
            }
            i += 1;
            continue;
        }
        if !in_tag && skip_depth == 0 {
            out.push(c);
        }
        i += 1;
    }

    out.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
}

/// Strip RTF control words and grouping braces, keeping literal text.
fn strip_rtf(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let chars: Vec<char> = raw.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        match chars[i] {
            '\\' => {
                i += 1;
                // Escaped literal
                if i < chars.len() && !chars[i].is_ascii_alphanumeric() {
                    out.push(chars[i]);
                    i += 1;
                    continue;
                }
                // Control word: letters, optional numeric arg, optional space
                while i < chars.len() && chars[i].is_ascii_alphabetic() {
                    i += 1;
                }
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '-') {
                    i += 1;
                }
                if i < chars.len() && chars[i] == ' ' {
                    i += 1;
                }
                out.push(' ');
            }
            '{' | '}' => {
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    out
}

/// Light Markdown flattening: drop fenced code blocks and the punctuation that
/// carries formatting rather than meaning. Link text is kept, the URL dropped.
fn strip_markdown(raw: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut in_fence = false;
    for line in raw.lines() {
        let t = line.trim_end();
        let trimmed = t.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        let mut s = trimmed.to_string();
        // Heading and quote markers
        s = s.trim_start_matches('#').trim_start_matches('>').to_string();
        // Table pipes and horizontal rules
        if s.chars().all(|c| c == '-' || c == '=' || c == '|' || c == ' ' || c == ':')
            && !s.trim().is_empty()
        {
            continue;
        }
        let s = s
            .replace("**", "")
            .replace("__", "")
            .replace('`', "");
        let s = strip_md_links(&s);
        let s = s.trim();
        if !s.is_empty() {
            out.push(s.to_string());
        }
    }
    out.join("\n")
}

/// `[label](url)` -> `label`; `![alt](url)` -> `alt`.
fn strip_md_links(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        if chars[i] == '[' {
            if let Some(close) = (i + 1..chars.len()).find(|&j| chars[j] == ']') {
                // Only treat it as a link when "(" follows the "]".
                if close + 1 < chars.len() && chars[close + 1] == '(' {
                    if let Some(paren) = (close + 2..chars.len()).find(|&j| chars[j] == ')') {
                        let label: String = chars[i + 1..close].iter().collect();
                        out.push_str(&label);
                        i = paren + 1;
                        continue;
                    }
                }
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// Collapse runs of spaces/tabs, cap blank lines at one, trim each line.
fn normalize_whitespace(raw: &str) -> String {
    let mut lines: Vec<String> = Vec::new();
    let mut blank_run = 0usize;
    for line in raw.replace('\r', "").lines() {
        let mut collapsed = String::with_capacity(line.len());
        let mut last_space = false;
        for c in line.chars() {
            let is_space = c == ' ' || c == '\t';
            if is_space {
                if !last_space {
                    collapsed.push(' ');
                }
            } else {
                collapsed.push(c);
            }
            last_space = is_space;
        }
        let t = collapsed.trim().to_string();
        if t.is_empty() {
            blank_run += 1;
            if blank_run > 1 {
                continue;
            }
            lines.push(String::new());
        } else {
            blank_run = 0;
            lines.push(t);
        }
    }
    lines.join("\n").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extension_is_lowercased_and_optional() {
        assert_eq!(extension_of("notes.TXT"), "txt");
        assert_eq!(extension_of("archive.tar.gz"), "gz");
        assert_eq!(extension_of("README"), "");
    }

    #[test]
    fn binary_uploads_are_refused_with_a_useful_message() {
        let err = extract("paper.pdf", "%PDF-1.7\n1 0 obj").unwrap_err().to_string();
        assert!(err.contains("binary"), "got: {err}");
        assert!(err.contains(".txt"), "should suggest a way out: {err}");

        // A .docx is a zip; the extension check catches it first.
        let err = extract("report.docx", "PK\u{3}\u{4}junk").unwrap_err().to_string();
        assert!(err.contains("not supported"), "got: {err}");
    }

    #[test]
    fn nul_bytes_count_as_binary() {
        assert!(looks_binary("abc\u{0}def"));
        assert!(!looks_binary("plain text"));
    }

    #[test]
    fn subtitles_lose_timecodes_and_repeats() {
        let srt = "1\n00:00:01,000 --> 00:00:03,000\nHello there\n\n\
                   2\n00:00:03,000 --> 00:00:05,000\nHello there\n\n\
                   3\n00:00:05,000 --> 00:00:07,000\nSecond line\n";
        let out = extract("clip.srt", srt).unwrap();
        assert_eq!(out, "Hello there Second line");
    }

    #[test]
    fn vtt_header_and_tags_are_dropped() {
        let vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<c.colorE5E5E5>Spoken words</c>\n";
        assert_eq!(extract("clip.vtt", vtt).unwrap(), "Spoken words");
    }

    #[test]
    fn html_keeps_prose_and_drops_script() {
        let html = "<html><head><style>p{color:red}</style></head>\
                    <body><p>First para</p><p>Second&nbsp;para</p>\
                    <script>alert('x')</script></body></html>";
        let out = extract("page.html", html).unwrap();
        assert!(out.contains("First para"), "got: {out}");
        assert!(out.contains("Second para"), "got: {out}");
        assert!(!out.contains("alert"), "script body leaked: {out}");
        assert!(!out.contains("color:red"), "style body leaked: {out}");
    }

    #[test]
    fn adjacent_tags_do_not_glue_words_together() {
        let out = extract("x.html", "<p>alpha</p><p>beta</p>").unwrap();
        assert!(out.contains("alpha beta"), "got: {out}");
    }

    #[test]
    fn markdown_drops_code_fences_and_link_urls() {
        let md = "# Title\n\nSome **bold** text and a [link](https://example.com).\n\n\
                  ```rust\nlet x = 1;\n```\n\nAfter the block.\n";
        let out = extract("notes.md", md).unwrap();
        assert!(out.contains("Title"), "got: {out}");
        assert!(out.contains("Some bold text and a link."), "got: {out}");
        assert!(!out.contains("let x = 1"), "code fence leaked: {out}");
        assert!(!out.contains("example.com"), "url leaked: {out}");
    }

    #[test]
    fn rtf_control_words_are_stripped() {
        let rtf = r"{\rtf1\ansi\deff0 {\fonttbl{\f0 Times;}}\f0\fs24 Hello from RTF\par}";
        let out = extract("note.rtf", rtf).unwrap();
        assert!(out.contains("Hello from RTF"), "got: {out}");
        assert!(!out.contains("fonttbl"), "got: {out}");
    }

    #[test]
    fn whitespace_is_normalized_but_paragraphs_survive() {
        let out = extract("a.txt", "one   two\t\tthree\n\n\n\nnext para\n").unwrap();
        assert_eq!(out, "one two three\n\nnext para");
    }

    #[test]
    fn extensionless_plain_text_is_accepted() {
        assert_eq!(extract("LICENSE", "Some licence text").unwrap(), "Some licence text");
    }

    #[test]
    fn empty_after_parsing_is_an_error() {
        assert!(extract("empty.vtt", "WEBVTT\n\n").is_err());
    }

    #[test]
    fn oversized_uploads_are_refused() {
        let huge = "a".repeat(MAX_DOC_CHARS + 1);
        let err = extract("big.txt", &huge).unwrap_err().to_string();
        assert!(err.contains("limit"), "got: {err}");
    }
}
