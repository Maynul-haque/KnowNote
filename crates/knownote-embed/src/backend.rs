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
}

impl LocalEmbeddingBackend {
    /// Load the pinned model from a cache directory, using the ONNX Runtime
    /// that is already loaded into the process.
    ///
    /// The files are the ones the manifest names, at the revision the manifest
    /// names. Nothing is fetched here: the downloader owns that, so a missing
    /// model is a missing model rather than a surprise network call.
    pub fn load(cache_dir: impl Into<PathBuf>) -> Result<Self> {
        Self::load_with_runtime(cache_dir, None)
    }

    /// Load the pinned model, loading a specific ONNX Runtime.
    ///
    /// The runtime is loaded dynamically rather than linked, so the
    /// application ships the runtime it chose rather than the one the build
    /// machine happened to have. The CPU-only build of ONNX Runtime is a
    /// fraction of the size of the DirectML one, and a 384-dimensional
    /// embedding model does not need a GPU.
    ///
    /// This has to run before any other `ort` call in the process: the
    /// dynamic loader registers the library once, and every later call uses
    /// it.
    pub fn load_with_runtime(
        cache_dir: impl Into<PathBuf>,
        runtime: Option<&std::path::Path>,
    ) -> Result<Self> {
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

        let environment = match runtime {
            Some(path) => ort::init_from(path)
                .map_err(|error| {
                    EmbedError::Inference(format!(
                        "cannot load the ONNX Runtime from {}: {error}",
                        path.display()
                    ))
                })?
                .build()
                .map_err(|error| EmbedError::Inference(error.to_string()))?,
            None => ort::init()
                .build()
                .map_err(|error| EmbedError::Inference(error.to_string()))?,
        };

        let mut builder = Session::builder(&environment)
            .map_err(|error| EmbedError::Inference(error.to_string()))?;
        let session = builder
            .commit_from_file(&model_path)
            .map_err(|error| {
                EmbedError::Inference(format!("cannot load {}: {error}", model_path.display()))
            })?;

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| {
            EmbedError::Inference(format!("cannot load {}: {error}", tokenizer_path.display()))
        })?;

        // The batch is padded to its longest member and truncated to the
        // model's limit. Both settings change the vectors, so they are set
        // explicitly rather than left to the tokenizer's defaults.
        //
        // The pad token is the one the tokenizer's own vocabulary declares,
        // not a guess: the wrong id pads with a real word, and the attention
        // then reads it.
        let pad_id = tokenizer
            .get_vocab(true)
            .get("<pad>")
            .copied()
            .unwrap_or(0);
        let pad_token = "<pad>".to_string();

        let _ = tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            direction: tokenizers::PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id,
            pad_type_id: 0,
            pad_token,
        }));
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: LOCAL_EMBEDDING_MODEL.max_length,
                strategy: tokenizers::TruncationStrategy::LongestFirst,
                direction: tokenizers::TruncationDirection::Right,
                stride: 0,
            }))
            .map_err(|error| EmbedError::Inference(error.to_string()))?;

        Ok(Self {
            inner: Arc::new(LocalEmbeddingInner {
                session: Mutex::new(session),
                _environment: environment,
                tokenizer,
                space: local_embedding_space(),
            }),
        })
    }

    /// Embed a batch of texts. Blocking: this is the work the async wrapper
    /// hands to a blocking thread.
    ///
    /// The whole batch goes through the graph in one call, the way the
    /// application embeds a document's chunks. That is not an optimisation:
    /// a padded batch and an unpadded one give slightly different vectors,
    /// because the attention is computed over the padded length. Embedding one
    /// text at a time would therefore produce a *different index* from the
    /// same document, and the retrieval would not match.
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

        // Every row is the same length: the tokenizer padded the batch to its
        // longest member.
        let length = encodings.first().map(|encoding| encoding.len()).unwrap_or(0);
        if length == 0 {
            return Ok(vec![Vec::new(); encodings.len()]);
        }

        let batch = encodings.len();
        let mut input_ids: Vec<i64> = Vec::with_capacity(batch * length);
        let mut attention_mask: Vec<i64> = Vec::with_capacity(batch * length);
        let mut token_type_ids: Vec<i64> = Vec::with_capacity(batch * length);

        for encoding in &encodings {
            if encoding.len() != length {
                return Err(EmbedError::Inference(
                    "the tokenizer produced rows of different lengths".to_string(),
                ));
            }
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            for index in 0..length {
                input_ids.push(ids[index] as i64);
                attention_mask.push(mask[index] as i64);
                token_type_ids.push(0);
            }
        }

        let shape = [batch, length];

        let mut session = self
            .inner
            .session
            .lock()
            .map_err(|_| EmbedError::Inference("the inference session is poisoned".to_string()))?;

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

        let mut vectors = Vec::with_capacity(batch);
        for row in 0..batch {
            let mask = &attention_mask[row * length..(row + 1) * length];
            let hidden_row = &hidden[row * length * LOCAL_EMBEDDING_MODEL.dimensions
                ..(row + 1) * length * LOCAL_EMBEDDING_MODEL.dimensions];
            vectors.push(mean_pool(
                hidden_row,
                length,
                LOCAL_EMBEDDING_MODEL.dimensions,
                mask,
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
