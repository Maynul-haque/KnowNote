//! Embedding for KnowNote.
//!
//! The local backend runs a pinned model through ONNX Runtime in this
//! process, so retrieval keeps working with the network off. The remote
//! backend forwards to an endpoint the user configured. Both produce vectors
//! in a named *space*, and a notebook's vectors are only comparable with a
//! query vector from the same space — which is why the space identity
//! includes the model revision and not just the width.

pub mod backend;
pub mod download;
pub mod local_model;

pub use backend::{LocalEmbeddingBackend, RemoteEmbeddingBackend};
pub use download::{CancelHandle, DownloadProgress, ModelDownloader};
pub use knownote_embedding::{
    EmbeddingBackend, EmbeddingBackendKind, EmbeddingRole, EmbeddingSpace,
};
pub use local_model::{
    build_resolve_url, local_embedding_space, local_embedding_total_bytes, EmbeddingModelFile,
    LocalEmbeddingModel, LOCAL_EMBEDDING_FILES, LOCAL_EMBEDDING_MODEL,
};

/// Everything that can go wrong while embedding.
pub use knownote_embedding::EmbedError;

/// The result type the crate uses.
pub type Result<T> = std::result::Result<T, EmbedError>;
