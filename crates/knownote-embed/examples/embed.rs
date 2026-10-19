//! Embed texts with the pinned model and print the vectors.
//!
//! This exists so the Rust inference can be compared against the TypeScript
//! one: the same texts, the same model, the same vectors.
//!
//! ```text
//! cargo run -p knownote-embed --example embed -- <cache-dir> <runtime-dll> "text one" "text two"
//! ```

use knownote_embed::{EmbeddingBackend, LocalEmbeddingBackend};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let cache_dir = arguments
        .next()
        .ok_or("usage: embed <cache-dir> <runtime-dll> [text ...]")?;
    let runtime = arguments.next().ok_or(
        "usage: embed <cache-dir> <runtime-dll> [text ...]  (pass an empty string for the built-in runtime)",
    )?;

    let texts: Vec<String> = arguments.collect();
    if texts.is_empty() {
        return Err("no texts to embed".into());
    }

    let backend = if runtime.is_empty() {
        LocalEmbeddingBackend::load(&cache_dir)?
    } else {
        LocalEmbeddingBackend::load_with_runtime(&cache_dir, Some(std::path::Path::new(&runtime)))?
    };

    // The documents go through the graph in one call, the way the application
    // embeds a document's chunks. A question is embedded on its own, the way
    // a query is embedded. Batching changes the vectors, so comparing the two
    // has to compare the same batching.
    let documents = backend.embed_documents(&texts).await?;
    for (text, vector) in texts.iter().zip(documents.iter()) {
        println!("document\t{}\t{}", text.replace('\n', " "), format_vector(vector));
    }

    for text in &texts {
        let vector = backend.embed_query(text).await?;
        println!("query\t{}\t{}", text.replace('\n', " "), format_vector(&vector));
    }

    Ok(())
}

/// The vector, printed with full precision: the comparison is exact, not
/// approximate.
fn format_vector(vector: &[f32]) -> String {
    vector
        .iter()
        .map(|value| format!("{value:.9}"))
        .collect::<Vec<_>>()
        .join(",")
}
