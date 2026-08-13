//! Embedding client — talks to the fleet-gateway (OpenAI-compatible API).

use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EmbedError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("api error ({status}): {body}")]
    Api { status: u16, body: String },
    #[error("empty embeddings response")]
    Empty,
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
}

/// OpenAI-compatible embedding request.
#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

/// OpenAI-compatible embedding response.
#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

/// Client for the fleet-gateway embedding endpoint.
pub struct EmbeddingClient {
    client: Client,
    endpoint: String,
    model: String,
    expected_dims: usize,
}

impl EmbeddingClient {
    pub fn new(gateway_url: &str, model: &str, expected_dims: usize) -> Result<Self, EmbedError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()?;

        Ok(Self {
            client,
            endpoint: format!("{}/embeddings", gateway_url.trim_end_matches('/')),
            model: model.to_string(),
            expected_dims,
        })
    }

    /// Embed a batch of texts. Returns vectors in the same order.
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let req = EmbeddingRequest {
            model: self.model.clone(),
            input: texts.to_vec(),
        };

        let resp = self.client.post(&self.endpoint).json(&req).send().await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(EmbedError::Api {
                status: status.as_u16(),
                body,
            });
        }

        let data: EmbeddingResponse = resp.json().await?;

        if data.data.is_empty() {
            return Err(EmbedError::Empty);
        }

        // Validate dimensions
        if data.data[0].embedding.len() != self.expected_dims {
            return Err(EmbedError::DimMismatch {
                expected: self.expected_dims,
                got: data.data[0].embedding.len(),
            });
        }

        Ok(data.data.into_iter().map(|d| d.embedding).collect())
    }

    /// Embed a single text.
    pub async fn embed_one(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let mut results = self.embed_batch(&[text.to_string()]).await?;
        results.pop().ok_or(EmbedError::Empty)
    }
}
