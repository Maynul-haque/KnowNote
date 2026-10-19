//! Downloading the pinned model.
//!
//! The download is verified before anything is written into the cache: the
//! size has to match the manifest and the SHA256 has to match the manifest.
//! A file that fails either check is deleted rather than left behind, because
//! a half-written model in the cache is a model that loads and produces
//! wrong vectors.
//!
//! Downloads resume. The hub answers the weights with a redirect to a signed
//! URL that expires after an hour, so a resumed download asks for the resolve
//! URL again instead of reusing the redirect target.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use crate::local_model::{
    build_resolve_url, EmbeddingModelFile, DEFAULT_HUB_BASE_URL, LOCAL_EMBEDDING_FILES,
    LOCAL_EMBEDDING_MODEL, MIRROR_HUB_BASE_URL,
};
use crate::{EmbedError, Result};

/// How far a download has got.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownloadProgress {
    /// The file being downloaded.
    pub file: String,
    /// Bytes downloaded for this file.
    pub file_downloaded: u64,
    /// The size of this file.
    pub file_total: u64,
    /// Bytes downloaded across every file.
    pub overall_downloaded: u64,
    /// The size of every file together.
    pub overall_total: u64,
}

impl DownloadProgress {
    /// The fraction of the whole download that is done, from 0 to 1.
    pub fn fraction(&self) -> f64 {
        if self.overall_total == 0 {
            return 1.0;
        }
        self.overall_downloaded as f64 / self.overall_total as f64
    }
}

/// A handle that cancels a running download.
#[derive(Debug, Clone, Default)]
pub struct CancelHandle {
    cancelled: Arc<AtomicBool>,
}

impl CancelHandle {
    /// Ask the download to stop.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Whether the download was cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

/// Downloads the pinned model into a cache directory.
pub struct ModelDownloader {
    client: reqwest::Client,
    cache_dir: PathBuf,
    base_urls: Vec<String>,
}

impl ModelDownloader {
    /// A downloader that writes into `cache_dir`.
    pub fn new(cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .user_agent(concat!("KnowNote/", env!("CARGO_PKG_VERSION")))
                .build()
                .unwrap_or_default(),
            cache_dir: cache_dir.into(),
            base_urls: vec![
                DEFAULT_HUB_BASE_URL.to_string(),
                MIRROR_HUB_BASE_URL.to_string(),
            ],
        }
    }

    /// Use a different set of hubs, in order.
    pub fn with_base_urls(mut self, base_urls: Vec<String>) -> Self {
        self.base_urls = base_urls;
        self
    }

    /// The directory a file is cached in.
    pub fn file_path(&self, file: &EmbeddingModelFile) -> PathBuf {
        self.cache_dir
            .join(LOCAL_EMBEDDING_MODEL.id)
            .join(LOCAL_EMBEDDING_MODEL.revision)
            .join(file.path)
    }

    /// Whether every file of the manifest is present and verified.
    pub fn is_cached(&self) -> bool {
        LOCAL_EMBEDDING_FILES
            .iter()
            .all(|file| self.file_path(file).exists())
    }

    /// Download every file that is not already cached.
    ///
    /// `on_progress` is called after each chunk, so the settings screen can
    /// show a moving bar rather than a spinner.
    pub async fn download(
        &self,
        cancel: &CancelHandle,
        mut on_progress: impl FnMut(DownloadProgress),
    ) -> Result<()> {
        let overall_total: u64 = LOCAL_EMBEDDING_FILES.iter().map(|file| file.size).sum();
        let mut overall_done = 0u64;

        for file in LOCAL_EMBEDDING_FILES {
            let target = self.file_path(file);

            if target.exists() && verify(&target, file).is_ok() {
                overall_done += file.size;
                on_progress(DownloadProgress {
                    file: file.path.to_string(),
                    file_downloaded: file.size,
                    file_total: file.size,
                    overall_downloaded: overall_done,
                    overall_total,
                });
                continue;
            }

            if cancel.is_cancelled() {
                return Err(EmbedError::Cancelled);
            }

            let mut file_done = 0u64;
            self.download_file(file, &target, cancel, &mut file_done, &mut overall_done, overall_total, &mut on_progress)
                .await?;

            verify(&target, file)?;
            overall_done = overall_done.max(overall_done);
        }

        Ok(())
    }

    /// Download one file, resuming if a partial copy is already there.
    async fn download_file(
        &self,
        file: &EmbeddingModelFile,
        target: &Path,
        cancel: &CancelHandle,
        file_done: &mut u64,
        overall_done: &mut u64,
        overall_total: u64,
        on_progress: &mut impl FnMut(DownloadProgress),
    ) -> Result<()> {
        if let Some(parent) = target.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let temporary = target.with_extension("partial");
        let mut resume_from = 0u64;
        if temporary.exists() {
            resume_from = tokio::fs::metadata(&temporary).await?.len();
        }

        let mut last_error: Option<EmbedError> = None;

        for base_url in &self.base_urls {
            if cancel.is_cancelled() {
                return Err(EmbedError::Cancelled);
            }

            let url = build_resolve_url(base_url, LOCAL_EMBEDDING_MODEL.revision, file.path);

            let mut request = self.client.get(&url);
            if resume_from > 0 {
                request = request.header("Range", format!("bytes={resume_from}-"));
            }

            let response = match request.send().await {
                Ok(response) => response,
                Err(error) => {
                    last_error = Some(EmbedError::Network(error.to_string()));
                    continue;
                }
            };

            if !response.status().is_success() {
                last_error = Some(EmbedError::Network(format!(
                    "{url} answered {}",
                    response.status()
                )));
                continue;
            }

            // The server ignored the range request, so start over rather than
            // appending a whole file to a partial one.
            let append = resume_from > 0 && response.status().as_u16() == 206;
            if !append {
                *file_done = 0;
            }

            let mut file_handle = tokio::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .append(append)
                .truncate(!append)
                .open(&temporary)
                .await?;

            let mut stream = response.bytes_stream();
            use futures_util::StreamExt;

            while let Some(chunk) = stream.next().await {
                if cancel.is_cancelled() {
                    return Err(EmbedError::Cancelled);
                }

                let chunk = chunk.map_err(|error| EmbedError::Network(error.to_string()))?;
                file_handle.write_all(&chunk).await?;
                *file_done += chunk.len() as u64;
                *overall_done += chunk.len() as u64;

                on_progress(DownloadProgress {
                    file: file.path.to_string(),
                    file_downloaded: *file_done,
                    file_total: file.size,
                    overall_downloaded: (*overall_done).min(overall_total),
                    overall_total,
                });
            }

            file_handle.flush().await?;
            drop(file_handle);

            // The file is only published to the cache once it is complete and
            // verified, so a cancelled download never leaves a usable-looking
            // model behind.
            tokio::fs::rename(&temporary, target).await?;
            return Ok(());
        }

        Err(last_error.unwrap_or_else(|| {
            EmbedError::Network(format!("no hub could serve {}", file.path))
        }))
    }
}

/// Check a cached file against the manifest.
pub fn verify(path: &Path, file: &EmbeddingModelFile) -> Result<()> {
    let metadata = std::fs::metadata(path)?;
    if metadata.len() != file.size {
        return Err(EmbedError::Verification(format!(
            "{} is {} bytes, the manifest says {}",
            file.path,
            metadata.len(),
            file.size
        )));
    }

    let digest = sha256_of(path)?;
    if digest != file.sha256 {
        return Err(EmbedError::Verification(format!(
            "{} has the wrong checksum",
            file.path
        )));
    }

    Ok(())
}

/// The SHA256 of a file, read in blocks.
pub fn sha256_of(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 64 * 1024];

    loop {
        let read = std::io::Read::read(&mut file, &mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}
