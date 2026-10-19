//! KnowNote's storage: the workspace on a fast drive, the bulk on a drive you choose.
//!
//! The store is split by *how often something is read*, not by what it is
//! about. A dense query never reads chunk text — it embeds the question,
//! scans the vectors, and gets back five chunk ids; the text of those five is
//! fetched afterwards. So the hot path is vectors and indexes, and the cold
//! path is text, and they want to live on different media.
//!
//! See [`layout::StorageLayout`] for the tiers and [`schema`] for the SQL.

pub mod layout;
pub mod retrieval;
pub mod schema;
pub mod store;

pub use layout::{ContentMigration, StorageLayout};
pub use store::Store;

/// Everything that can go wrong in the store.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("{0}")]
    Io(String),

    #[error("the content store is not available: {0}")]
    ContentUnavailable(String),

    #[error("{0}")]
    Invalid(String),
}

impl From<knownote_embedding::EmbedError> for StoreError {
    fn from(error: knownote_embedding::EmbedError) -> Self {
        StoreError::Invalid(error.to_string())
    }
}

/// The result type the store uses.
pub type Result<T> = std::result::Result<T, StoreError>;
