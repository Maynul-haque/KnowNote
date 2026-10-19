//! Embedding backends.
//!
//! Two of them, behind one trait. The local backend runs the pinned model
//! through ONNX Runtime in this process; the remote backend forwards to an
//! endpoint the user configured. The chat path cannot tell the difference,
//! which is the point: a notebook's vectors carry the space they were made
//! in, so switching a notebook from one backend to the other is a re-index,
//! not a silent corruption.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use knownote_embedding::{EmbeddingBackend, EmbeddingRole, EmbeddingSpace};
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::local_model::{local_embedding_space, LOCAL_EMBEDDING_FILES, LOCAL_EMBEDDING_MODEL};
use crate::{EmbedError, Result};

/// The local backend: the pinned model, run in this process.
#[derive(Clone)]
pub struct LocalEmbeddingBackend {
    inner: Arc<LocalEmbeddingInner>,
}

struct LocalEmbeddingInner {
    /// The session is behind a lock because ONNX Runtime's `run` takes
    /// `&mut self`. The lock is held for the length of one inference, so
    /// concurrent callers queue rather than run at once — which is the right
    /// behaviour for a model that is already using every core.
    session: Mutex<Session>,
    /// The environment has to outlive the session.
    _environment: Environment,
    tokenizer: Tokenizer,
    space: EmbeddingSpace,
    max_length: usize,
}

impl LocalEmbeddingBackend {
    /// Load the pinned model from a cache directory.
    ///
    /// The files are the ones the manifest names, at the revision the manifest
    /// names. Nothing is fetched here: the downloader owns that, so a missing
    /// model is a missing model rather than a surprise network call.
    pub fn load(cache_dir: impl Into<PathBuf>) -> Result<Self> {
        let cache_dir = cache_dir.into();
        let root = cache_dir
            .join(LOCAL_EMBEDDING_MODEL.id)
            .join(LOCAL_EMBEDDING_MODEL.revision);

        let model_path = root.join("onnx/model_quantized.onnx");
        let tokenizer_path = root.join("tokenizer.json");

        for file in LOCAL_EMBEDDING_FILES {
            let path = root.join(file.path);
            if !path.exists() {
                return Err(EmbedError::ModelMissing(format!(
                    "{} is not in the cache",
                    file.path
                )));
            }
        }

        let environment = ort::init()
            .build()
            .map_err(|error| EmbedError::Inference(error.to_string()))?;

        let mut builder = Session::builder(&environment)
            .map_err(|error| EmbedError::Inference(error.to_string()))?;
        let session = builder
            .commit_from_file(&model_path)
            .map_err(|error| {
                EmbedError::Inference(format!("cannot load {}: {error}", model_path.display()))
            })?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| {
            EmbedError::Inference(format!("cannot load {}: {error}", tokenizer_path.display()))
        })?;

        Ok(Self {
            inner: Arc::new(LocalEmbeddingInner {
                session: Mutex::new(session),
                _environment: environment,
                tokenizer,
                space: local_embedding_space(),
                max_length: LOCAL_EMBEDDING_MODEL.max_length,
            }),
        })
    }

    /// Embed a batch of texts. Blocking: this is the work the async wrapper
    /// hands to a blocking thread.
    fn embed_blocking(&self, texts: &[String], role: EmbeddingRole) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let prefixed: Vec<String> = texts
            .iter()
            .map(|text| format!("{}{}", role.prefix(), text))
            .collect();

        let encodings = self
            .inner
            .tokenizer
            .encode_batch(prefixed, true)
            .map_err(|error| EmbedError::Inference(error.to_string()))?;

        let mut session = self
            .inner
            .session
            .lock()
            .map_err(|_| EmbedError::Inference("the inference session is poisoned".to_string()))?;

        let mut vectors = Vec::with_capacity(encodings.len());

        for encoding in encodings {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();

            // The tokenizer can return a pair for a single piece; that never
            // happens for the encodings this model produces, but a mismatch
            // here would silently shift the pooling, so it is checked.
            if ids.len() != mask.len() {
                return Err(EmbedError::Inference(
                    "the tokenizer produced a mask of a different length than the ids".to_string(),
                ));
            }

            let length = ids.len().min(self.inner.max_length);
            let input_ids: Vec<i64> = ids[..length].iter().map(|id| *id as i64).collect();
            let attention_mask: Vec<i64> = mask[..length].iter().map(|value| *value as i64).collect();
            let token_type_ids: Vec<i64> = vec![0; length];

            let shape = [1usize, length];

            let outputs = session
                .run(ort::inputs! {
                    "input_ids" => Tensor::from_array((shape, input_ids))
                        .map_err(|error| EmbedError::Inference(error.to_string()))?,
                    "attention_mask" => Tensor::from_array((shape, attention_mask.clone()))
                        .map_err(|error| EmbedError::Inference(error.to_string()))?,
                    "token_type_ids" => Tensor::from_array((shape, token_type_ids))
                        .map_err(|error| EmbedError::Inference(error.to_string()))?,
                })
                .map_err(|error| EmbedError::Inference(error.to_string()))?;

            let (_, hidden) = outputs["last_hidden_state"]
                .try_extract_tensor::<f32>()
                .map_err(|error| EmbedError::Inference(error.to_string()))?;

            vectors.push(mean_pool(
                hidden,
                length,
                LOCAL_EMBEDDING_MODEL.dimensions,
                &attention_mask,
            ));
        }

        Ok(vectors)
    }
}

/// Mean-pool the token vectors and normalise the result.
///
/// The attention mask is what keeps the padding out of the average, so it is
/// applied to the numerator and the denominator alike.
fn mean_pool(hidden: &[f32], length: usize, dimensions: usize, mask: &[i64]) -> Vec<f32> {
    let mut pooled = vec![0f32; dimensions];
    let mut counted = 0f32;

    for token in 0..length {
        if mask.get(token).copied().unwrap_or(0) == 0 {
            continue;
        }
        let base = token * dimensions;
        for dimension in 0..dimensions {
            pooled[dimension] += hidden[base + dimension];
        }
        counted += 1.0;
    }

    if counted > 0.0 {
        for value in &mut pooled {
            *value /= counted;
        }
    }

    if LOCAL_EMBEDDING_MODEL.normalize {
        let norm = pooled.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut pooled {
                *value /= norm;
            }
        }
    }

    pooled
}

#[async_trait]
impl EmbeddingBackend for LocalEmbeddingBackend {
    fn dimensions(&self) -> usize {
        LOCAL_EMBEDDING_MODEL.dimensions
    }

    fn space(&self) -> EmbeddingSpace {
        self.inner.space.clone()
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // The inference is CPU-bound and can take a while for a large batch, so
        // it runs on a blocking thread rather than on the runtime's workers.
        let backend = self.clone();
        let texts = texts.to_vec();
        tokio::task::spawn_blocking(move || backend.embed_blocking(&texts, EmbeddingRole::Document))
            .await
            .map_err(|error| EmbedError::Inference(error.to_string()))?
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let backend = self.clone();
        let text = text.to_string();
        let vectors = tokio::task::spawn_blocking(move || {
            backend.embed_blocking(&[text], EmbeddingRole::Query)
        })
        .await
        .map_err(|error| EmbedError::Inference(error.to_string()))??;

        Ok(vectors.into_iter().next().unwrap_or_default())
    }
}

/// The remote backend: an embedding endpoint the user configured.
pub struct RemoteEmbeddingBackend {
    client: reqwest::Client,
    endpoint: String,
    api_key: Option<String>,
    model: String,
    dimensions: usize,
    space: EmbeddingSpace,
}

impl RemoteEmbeddingBackend {
    /// A backend that forwards to an OpenAI-compatible embeddings endpoint.
    pub fn new(
        endpoint: impl Into<String>,
        api_key: Option<String>,
        model: impl Into<String>,
        dimensions: usize,
    ) -> Self {
        let model = model.into();
        Self {
            client: reqwest::Client::new(),
            endpoint: endpoint.into(),
            api_key,
            space: EmbeddingSpace::remote(model.clone(), dimensions),
            model,
            dimensions,
        }
    }
}

#[async_trait]
impl EmbeddingBackend for RemoteEmbeddingBackend {
    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn space(&self) -> EmbeddingSpace {
        self.space.clone()
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let body = serde_json::json!({
            "model": self.model,
            "input": texts,
        });

        let mut request = self
            .client
            .post(&self.endpoint)
            .header("Content-Type", "application/json")
            .body(serde_json::to_vec(&body).map_err(|error| {
                EmbedError::Inference(format!("cannot encode the request: {error}"))
            })?);

        if let Some(key) = &self.api_key {
            request = request.header("Authorization", format!("Bearer {key}"));
        }

        let response = request
            .send()
            .await
            .map_err(|error| EmbedError::Network(error.to_string()))?;

        if !response.status().is_success() {
            return Err(EmbedError::Network(format!(
                "the embedding endpoint answered {}",
                response.status()
            )));
        }

        let payload: serde_json::Value = response
            .json()
            .await
            .map_err(|error| EmbedError::Network(error.to_string()))?;

        let entries = payload
            .get("data")
            .and_then(|data| data.as_array())
            .ok_or_else(|| EmbedError::Network("the embedding response has no data".to_string()))?;

        let mut vectors = Vec::with_capacity(entries.len());
        for entry in entries {
            let vector = entry
                .get("embedding")
                .and_then(|value| value.as_array())
                .ok_or_else(|| {
                    EmbedError::Network("an embedding entry has no embedding".to_string())
                })?;
            vectors.push(
                vector
                    .iter()
                    .map(|value| value.as_f64().unwrap_or(0.0) as f32)
                    .collect(),
            );
        }

        Ok(vectors)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.embed_documents(&[text.to_string()]).await?.remove(0))
    }
}
