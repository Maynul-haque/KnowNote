//! The loader contract.
//!
//! A loader turns a file into the canonical text plus whatever structure it
//! found — pages, sections, or neither. The structure is what lets a citation
//! resolve to a page and a heading rather than to a paragraph of flat text, so
//! a loader that throws its structure away is not just less useful, it makes
//! every citation in that document point at the wrong place.

use std::path::Path;

use knownote_core::structure::DocumentStructure;

/// What a loader produced.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentLoadResult {
    /// The canonical text. Every offset in the store refers to this string.
    pub content: String,
    /// The document's title, when the format has one.
    pub title: Option<String>,
    pub mime_type: String,
    /// The structure the loader found.
    pub structure: Option<DocumentStructure>,
    /// Anything else the loader wants to record.
    pub metadata: Option<serde_json::Value>,
}

/// A format KnowNote can read.
pub trait DocumentLoader: Send + Sync {
    /// The MIME types this loader claims.
    fn supported_mime_types(&self) -> &[&'static str];

    /// The file extensions this loader claims, without the dot.
    fn supported_extensions(&self) -> &[&'static str];

    /// Read a file.
    fn load_from_path(&self, path: &Path) -> Result<DocumentLoadResult, IngestError>;

    /// Read bytes.
    fn load_from_buffer(&self, buffer: &[u8]) -> Result<DocumentLoadResult, IngestError>;

    /// Whether this loader handles the path or MIME type.
    fn can_load(&self, path_or_mime_type: &str) -> bool {
        let lower = path_or_mime_type.to_lowercase();
        if self.supported_mime_types().contains(&lower.as_str()) {
            return true;
        }
        match extension_of(&lower) {
            Some(extension) => self.supported_extensions().contains(&extension.as_str()),
            None => false,
        }
    }
}

/// The extension of a path or a MIME type, lowercased and without the dot.
pub fn extension_of(path_or_mime_type: &str) -> Option<String> {
    let trimmed = path_or_mime_type.trim();
    if trimmed.is_empty() {
        return None;
    }

    // A MIME type has a slash and no extension.
    if trimmed.contains('/') {
        return None;
    }

    let extension = trimmed.rsplit('.').next()?;
    if extension == trimmed {
        // No dot at all.
        return None;
    }
    Some(extension.to_lowercase())
}

/// Everything that can go wrong while reading a document.
#[derive(Debug, thiserror::Error)]
pub enum IngestError {
    #[error("cannot read {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("the file is not valid UTF-8: {0}")]
    Encoding(String),

    #[error("{0}")]
    Parse(String),

    #[error("no loader handles {0}")]
    Unsupported(String),
}
