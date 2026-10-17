use std::collections::HashSet;
use std::path::Path;

use crate::models::KnowledgeEntry;

/// Load the knowledge base from the JSON file on disk.
pub fn load_knowledge_base(data_dir: &Path) -> Vec<KnowledgeEntry> {
    let kb_path = data_dir.join("knowledge-base.json");
    let raw = std::fs::read_to_string(&kb_path)
        .unwrap_or_else(|e| panic!("Failed to read knowledge base at {}: {}", kb_path.display(), e));
    serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("Failed to parse knowledge base JSON: {}", e))
}

/// Extract the unique set of advisor names from the knowledge base.
pub fn advisor_names(kb: &[KnowledgeEntry]) -> HashSet<String> {
    kb.iter().map(|e| e.advisor.clone()).collect()
}

/// Check if the user message mentions a specific advisor and return the
/// corresponding transcript content if it exists on disk.
pub fn get_transcript_for_message(message: &str, data_dir: &Path) -> String {
    let lower = message.to_lowercase();

    let checks: &[(&[&str], &str)] = &[
        (&["aiscout", "ai scout"], "transcripts/AiScout.txt"),
        (&["devcast", "dev cast"], "transcripts/devcast.txt"),
    ];

    for (keywords, rel_path) in checks {
        if keywords.iter().any(|kw| lower.contains(kw)) {
            let full_path = data_dir.join(rel_path);
            if full_path.exists() {
                return std::fs::read_to_string(&full_path).unwrap_or_default();
            }
        }
    }

    String::new()
}
