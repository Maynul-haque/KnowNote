//! Download the pinned embedding model into a cache directory.
//!
//! ```text
//! cargo run -p knownote-embed --example download_model -- <cache-dir>
//! ```

use knownote_embed::download::{CancelHandle, ModelDownloader};
use knownote_embed::local_model::local_embedding_total_bytes;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| ".knownote-models".to_string());

    println!(
        "downloading the pinned model into {cache_dir} ({} MB)",
        local_embedding_total_bytes() / (1024 * 1024)
    );

    let downloader = ModelDownloader::new(&cache_dir);
    if downloader.is_cached() {
        println!("the model is already in the cache");
        return Ok(());
    }

    let cancel = CancelHandle::default();
    let mut last_percent = -1i64;

    downloader
        .download(&cancel, |progress| {
            let percent = (progress.fraction() * 100.0) as i64;
            if percent != last_percent {
                last_percent = percent;
                println!(
                    "  {percent:3}%  {} ({}/{} bytes)",
                    progress.file,
                    progress.file_downloaded,
                    progress.file_total
                );
            }
        })
        .await?;

    println!("the model is in the cache and its checksums match");
    Ok(())
}
