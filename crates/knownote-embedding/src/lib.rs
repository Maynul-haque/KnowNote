//! The embedding contract.
//!
//! This crate holds the *contract* and nothing else: what a backend has to
//! provide, and the identity of the space a vector belongs to.
//!
//! It is separate from the backends on purpose. The store needs "something
//! that can turn text into vectors" — it does not need ONNX Runtime, a
//! tokenizer or an HTTP client, and making it depend on them would mean the
//! store's tests could not run without a 150 MB model and a native inference
//! runtime on the machine.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Which side of the embedding a text is on.
///
/// The E5 models were trained with an asymmetric prefix, and dropping it
/// measurably changes the vectors, so it is not optional.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingRole {
    /// A passage that will be retrieved.
    Document,
    /// A question that is being matched against passages.
    Query,
}

impl EmbeddingRole {
    /// The prefix this role prepends.
    pub fn prefix(self) -> &'static str {
        match self {
            EmbeddingRole::Document => "passage: ",
            EmbeddingRole::Query => "query: ",
        }
    }
}

/// Which kind of backend produced the vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingBackendKind {
    Local,
    Remote,
}

impl EmbeddingBackendKind {
    /// The name stored in the database.
    pub fn as_str(self) -> &'static str {
        match self {
            EmbeddingBackendKind::Local => "local",
            EmbeddingBackendKind::Remote => "remote",
        }
    }
}

/// The identity of an embedding space.
///
/// Two vectors are only comparable when they were produced in the same space.
/// Comparing the width is not enough: a different model can have the same
/// width, and then the old vectors are incomparable without anything looking
/// wrong. The revision is part of the identity for the same reason.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingSpace {
    /// A stable identifier for the space.
    pub space_id: String,
    /// Whether the vectors came from the local model or a remote endpoint.
    pub backend: EmbeddingBackendKind,
    /// The model name.
    pub model: String,
    /// The model revision, when there is one.
    pub revision: String,
    /// The width of a vector.
    pub dimensions: usize,
}

impl EmbeddingSpace {
    /// The space a local model produces.
    pub fn local(model: impl Into<String>, revision: impl Into<String>, dimensions: usize) -> Self {
        let model = model.into();
        let revision = revision.into();
        Self {
            space_id: format!("{model}@{revision}"),
            backend: EmbeddingBackendKind::Local,
            model,
            revision,
            dimensions,
        }
    }

    /// The space a remote endpoint produces.
    pub fn remote(model: impl Into<String>, dimensions: usize) -> Self {
        let model = model.into();
        Self {
            space_id: format!("remote:{model}#{}", dimensions),
            backend: EmbeddingBackendKind::Remote,
            model,
            revision: String::new(),
            dimensions,
        }
    }
}

/// Something that can turn text into vectors.
#[async_trait]
pub trait EmbeddingBackend: Send + Sync {
    /// The width of a vector.
    fn dimensions(&self) -> usize;

    /// The space the vectors belong to.
    fn space(&self) -> EmbeddingSpace;

    /// Embed several passages.
    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError>;

    /// Embed a question.
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError>;
}

/// Everything that can go wrong while embedding.
#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("the embedding model is not in the cache: {0}")]
    ModelMissing(String),

    #[error("the download was cancelled")]
    Cancelled,

    #[error("the downloaded file did not match the manifest: {0}")]
    Verification(String),

    #[error("network error: {0}")]
    Network(String),

    #[error("inference error: {0}")]
    Inference(String),

    #[error("{0}")]
    Io(String),
}

impl From<std::io::Error> for EmbedError {
    fn from(error: std::io::Error) -> Self {
        EmbedError::Io(error.to_string())
    }
}
