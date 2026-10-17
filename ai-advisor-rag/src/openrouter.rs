use bytes::Bytes;
use futures::Stream;
use reqwest::Client;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::mpsc;
use tracing::{debug, error, info};

use crate::models::*;

/// Streams text chunks from OpenRouter (OpenAI-compatible /v1/chat/completions).
pub struct OpenRouterStream {
    rx: mpsc::Receiver<Result<String, String>>,
}

impl OpenRouterStream {
    pub fn start(
        client: &Client,
        api_key: &str,
        request: OpenRouterRequest,
        request_id: String,
    ) -> Self {
        let url = "https://openrouter.ai/api/v1/chat/completions";

        let client = client.clone();
        let api_key = api_key.to_string();
        let (tx, rx) = mpsc::channel::<Result<String, String>>(64);

        tokio::spawn(async move {
            let resp = match client
                .post(url)
                .bearer_auth(&api_key)
                // OpenRouter "site identity" headers — optional but documented.
                .header("HTTP-Referer", "http://localhost:3000")
                .header("X-Title", "AI Advisor")
                .json(&request)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    error!("[{request_id}] OpenRouter request failed: {e}");
                    let _ = tx.send(Err(format!("Request failed: {e}"))).await;
                    return;
                }
            };

            if !resp.status().is_success() {
                let status = resp.status();
                let body_text = resp.text().await.unwrap_or_default();
                error!("[{request_id}] OpenRouter returned {status}: {body_text}");
                let _ = tx.send(Err(format!("OpenRouter error {status}"))).await;
                return;
            }

            let mut stream = resp.bytes_stream();
            use futures::StreamExt;
            let mut buffer = String::new();

            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        error!("[{request_id}] Stream read error: {e}");
                        break;
                    }
                };

                let text = String::from_utf8_lossy(&chunk);
                buffer.push_str(&text);

                while let Some(line_end) = buffer.find('\n') {
                    let line = buffer[..line_end].trim().to_string();
                    buffer = buffer[line_end + 1..].to_string();

                    // OpenAI-compat SSE intermixes ": comment" keepalive lines
                    // with "data: {...}" payloads. Skip everything that isn't
                    // a data line, and bail on the terminator.
                    let Some(data) = line.strip_prefix("data: ") else { continue };
                    if data.trim() == "[DONE]" {
                        continue;
                    }
                    match serde_json::from_str::<OpenRouterStreamChunk>(data) {
                        Ok(chunk) => {
                            for choice in &chunk.choices {
                                if let Some(text) = &choice.delta.content {
                                    if !text.is_empty() {
                                        if tx.send(Ok(text.clone())).await.is_err() {
                                            return;
                                        }
                                    }
                                }
                            }
                            if let Some(u) = chunk.usage {
                                info!(
                                    "[{request_id}] Usage - prompt: {}, completion: {}, total: {}",
                                    u.prompt_tokens, u.completion_tokens, u.total_tokens
                                );
                            }
                        }
                        Err(e) => {
                            if !data.trim().is_empty() {
                                debug!("[{request_id}] Skipping non-JSON/unmapped SSE data: {e}");
                            }
                        }
                    }
                }
            }
        });

        Self { rx }
    }
}

impl Stream for OpenRouterStream {
    type Item = Result<Bytes, axum::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.rx.poll_recv(cx) {
            Poll::Ready(Some(Ok(text))) => Poll::Ready(Some(Ok(Bytes::from(text)))),
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(axum::Error::new(
                std::io::Error::new(std::io::ErrorKind::Other, e),
            )))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}
