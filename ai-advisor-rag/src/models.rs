use serde::{Deserialize, Serialize};

/// A single entry in the knowledge base JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEntry {
    pub id: String,
    pub topic: String,
    pub advisor: String,
    pub keywords: Vec<String>,
    pub content: String,
}

/// A chunk returned from the vector store, ready to splice into a prompt.
#[derive(Debug, Clone)]
pub struct RetrievedChunk {
    pub source_type: String,
    pub advisor: String,
    pub topic: String,
    pub content: String,
    /// "single" / "multiple" speaker classification from ingest time;
    /// None for pre-feature chunks and non-video sources.
    pub speaker_mode: Option<String>,
    pub score: f32,
}

/// A single part of a chat message (mirrors Gemini's format).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagePart {
    pub text: String,
}

/// A single turn in the conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub parts: Vec<MessagePart>,
}

/// The incoming request body from the frontend.
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub message: String,
    #[serde(default)]
    pub history: Vec<ChatMessage>,
    /// Optional advisor to scope retrieval to. When set:
    ///   - Qdrant search filters to chunks tagged with this advisor name.
    ///   - The system prompt narrows to this single expert (no other voice
    ///     guides, no "experts available" list).
    /// Unset = legacy behavior (retrieve across the full collection).
    #[serde(default)]
    pub expert: Option<String>,
}

/// Error response body.
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

// --- OpenRouter API types (OpenAI-compatible /v1/chat/completions) ---

#[derive(Debug, Serialize)]
pub struct OpenRouterRequest {
    pub model: String,
    pub messages: Vec<OpenRouterMessage>,
    pub max_tokens: u32,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    /// Reasoning-model controls. For GLM-4.6 and other reasoning models we set
    /// `exclude: true` so the model still thinks under the hood but the
    /// chain-of-thought stays out of the streamed output — otherwise reasoning
    /// burns the entire token budget in delta.reasoning while delta.content
    /// stays empty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
}

#[derive(Debug, Serialize)]
pub struct StreamOptions {
    pub include_usage: bool,
}

#[derive(Debug, Serialize)]
pub struct ReasoningConfig {
    /// Turn reasoning OFF entirely. Required for GLM-4.6 — `exclude` alone
    /// hides reasoning from the stream but the model still burns the whole
    /// token budget thinking, so content never emerges.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// Keep reasoning ON but suppress it from the streamed output. Kept here
    /// for flexibility — not currently used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclude: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct OpenRouterMessage {
    pub role: String,
    pub content: String,
}

/// One streaming chunk in OpenAI-compatible SSE.
/// Final chunk may carry `usage` when `stream_options.include_usage` is true.
#[derive(Debug, Deserialize)]
pub struct OpenRouterStreamChunk {
    #[serde(default)]
    pub choices: Vec<OpenRouterChoice>,
    #[serde(default)]
    pub usage: Option<OpenRouterUsage>,
}

#[derive(Debug, Deserialize)]
pub struct OpenRouterChoice {
    #[serde(default)]
    pub delta: OpenRouterDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct OpenRouterDelta {
    #[serde(default)]
    pub content: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OpenRouterUsage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
}
