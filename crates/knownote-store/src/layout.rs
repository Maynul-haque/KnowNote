//! Where things live on disk.
//!
//! # The tiers
//!
//! The store is split by *how often something is read*, not by what it is
//! about. That distinction is what makes the split work for a spinning disk:
//!
//! | Tier | Where | What | How often it is read |
//! |---|---|---|---|
//! | **Hot** | SSD | the vector index, the full-text index | on every query |
//! | **Bulk** | a drive you choose | chunk text, document blocks, extracted text, source files | written once at import, read a handful of times per query |
//! | **Meta** | SSD | notebooks, sessions, messages, notes, items, settings | small and transactional |
//!
//! The reason this works is that a dense query never reads chunk text. It
//! embeds the question, scans the vectors, and gets back five chunk ids. The
//! text of those five chunks is fetched afterwards, once. So the hot path is
//! vectors and the cold path is text, and they want to live on different
//! media.
//!
//! The bulk tier is configurable because the question it answers is "where do
//! I have room": a 200 GB corpus of PDFs does not belong on the boot drive.
//!
//! # What is *not* configurable
//!
//! The index tier stays on the fast drive even when the bulk tier is not. A
//! brute-force vector scan over 150 MB is random I/O; on a spinning disk that
//! is seconds per query. The vectors are also bigger than the text they
//! describe, so moving them buys little and costs a lot.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// A file name for the fast tier: the workspace, the chunk index, the
/// full-text index and the vector index.
const APP_DB: &str = "knownote.db";
/// A file name for the bulk tier.
const CONTENT_DB: &str = "content.db";

/// Where each tier of the store lives.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageLayout {
    /// The application data directory. Holds the fast tier.
    pub data_dir: PathBuf,
    /// Where the bulk content lives. Configurable, and the reason this type
    /// exists.
    pub content_dir: PathBuf,
    /// Where downloaded models are cached.
    pub models_dir: PathBuf,
}

impl StorageLayout {
    /// A layout with every tier inside `data_dir`.
    pub fn new(data_dir: impl Into<PathBuf>) -> Self {
        let data_dir = data_dir.into();
        Self {
            models_dir: data_dir.join("models"),
            content_dir: data_dir.clone(),
            data_dir,
        }
    }

    /// A layout with the bulk content on another drive.
    pub fn with_content_root(
        data_dir: impl Into<PathBuf>,
        content_root: impl Into<PathBuf>,
    ) -> Self {
        let mut layout = Self::new(data_dir);
        layout.content_dir = content_root.into();
        layout
    }

    /// The fast tier: the workspace, the chunk index, the full-text index and
    /// the vector index.
    pub fn app_db_path(&self) -> PathBuf {
        self.data_dir.join(APP_DB)
    }

    /// The bulk content database: chunk text, document blocks, extracted text.
    pub fn content_db_path(&self) -> PathBuf {
        self.content_dir.join(CONTENT_DB)
    }

    /// The library's copies of the original files.
    pub fn sources_dir(&self) -> PathBuf {
        self.content_dir.join("sources")
    }

    /// The embedding model cache.
    pub fn model_cache_dir(&self) -> PathBuf {
        self.models_dir.clone()
    }

    /// Create every directory the layout needs.
    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        for dir in [
            &self.data_dir,
            &self.models_dir,
            &self.content_dir,
            &self.sources_dir(),
        ] {
            fs::create_dir_all(dir)?;
        }
        Ok(())
    }

    /// Whether the bulk tier is on a different drive from the metadata tier.
    pub fn content_is_separate(&self) -> bool {
        self.content_dir != self.data_dir
    }

    /// Move the bulk tier to `new_root`.
    ///
    /// The content database and the source library are copied, checked, and
    /// only then is the layout switched over. The old copy is left in place
    /// rather than deleted: a move that deletes its source before the new one
    /// is proven good is a move that can lose data.
    ///
    /// The returned report says what was moved and what was left behind.
    pub fn migrate_content_root(
        &self,
        new_root: &Path,
    ) -> Result<ContentMigration, crate::StoreError> {
        if new_root == self.content_dir {
            return Ok(ContentMigration {
                moved: false,
                bytes_moved: 0,
                source: self.content_dir.clone(),
                target: new_root.to_path_buf(),
            });
        }

        fs::create_dir_all(new_root)
            .map_err(|error| crate::StoreError::Io(format!("cannot create {new_root:?}: {error}")))?;

        let mut bytes_moved = 0u64;

        // The content database.
        let source_db = self.content_db_path();
        let target_db = new_root.join(CONTENT_DB);
        if source_db.exists() {
            bytes_moved += copy_verified(&source_db, &target_db)?;
        }

        // The source library.
        let source_sources = self.sources_dir();
        let target_sources = new_root.join("sources");
        if source_sources.exists() {
            bytes_moved += copy_tree_verified(&source_sources, &target_sources)?;
        }

        Ok(ContentMigration {
            moved: true,
            bytes_moved,
            source: self.content_dir.clone(),
            target: new_root.to_path_buf(),
        })
    }
}

/// What a content migration did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentMigration {
    /// Whether anything was moved. False when the target was already the
    /// content root.
    pub moved: bool,
    /// How many bytes were copied.
    pub bytes_moved: u64,
    /// Where the content came from.
    pub source: PathBuf,
    /// Where the content went.
    pub target: PathBuf,
}

/// Copy a file and prove the copy is the same length before returning.
fn copy_verified(source: &Path, target: &Path) -> Result<u64, crate::StoreError> {
    let bytes = fs::copy(source, target).map_err(|error| {
        crate::StoreError::Io(format!("cannot copy {source:?} to {target:?}: {error}"))
    })?;

    let copied = fs::metadata(target)
        .map_err(|error| crate::StoreError::Io(format!("cannot stat {target:?}: {error}")))?
        .len();

    if copied != bytes {
        return Err(crate::StoreError::Io(format!(
            "the copy of {source:?} is {copied} bytes, the original is {bytes}"
        )));
    }

    Ok(bytes)
}

/// Copy a directory tree and prove every file made it.
fn copy_tree_verified(source: &Path, target: &Path) -> Result<u64, crate::StoreError> {
    let mut total = 0u64;

    for entry in walk(source)? {
        let relative = entry
            .strip_prefix(source)
            .map_err(|error| crate::StoreError::Io(format!("{error}")))?;
        let destination = target.join(relative);

        if entry.is_dir() {
            fs::create_dir_all(&destination).map_err(|error| {
                crate::StoreError::Io(format!("cannot create {destination:?}: {error}"))
            })?;
            continue;
        }

        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).map_err(|error| {
                crate::StoreError::Io(format!("cannot create {parent:?}: {error}"))
            })?;
        }

        total += copy_verified(&entry, &destination)?;
    }

    Ok(total)
}

/// Every file and directory under `root`, depth first.
fn walk(root: &Path) -> Result<Vec<PathBuf>, crate::StoreError> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];

    while let Some(directory) = stack.pop() {
        let entries = fs::read_dir(&directory).map_err(|error| {
            crate::StoreError::Io(format!("cannot read {directory:?}: {error}"))
        })?;

        for entry in entries {
            let entry = entry.map_err(|error| {
                crate::StoreError::Io(format!("cannot read {directory:?}: {error}"))
            })?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path.clone());
            }
            found.push(path);
        }
    }

    Ok(found)
}
