//! The command-line entry point for the evaluation harness.

use std::path::PathBuf;

use knownote_core::retrieval::RetrievalStrategy;
use knownote_embed::LocalEmbeddingBackend;
use knownote_eval::harness::{run_eval_harness, stabilize, to_deterministic_report, EvalOptions};
use knownote_eval::types::EvalSplit;
use knownote_store::{StorageLayout, Store};

/// A very small argument reader: `--name value` and `--name=value`.
struct Arguments {
    values: std::collections::HashMap<String, String>,
}

impl Arguments {
    fn parse() -> Self {
        let mut values = std::collections::HashMap::new();
        let mut arguments = std::env::args().skip(1);

        while let Some(argument) = arguments.next() {
            let Some(name) = argument.strip_prefix("--") else {
                continue;
            };
            if let Some((name, value)) = name.split_once('=') {
                values.insert(name.to_string(), value.to_string());
            } else if let Some(value) = arguments.next() {
                values.insert(name.to_string(), value);
            }
        }

        Self { values }
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.values.get(name).map(String::as_str)
    }

    fn get_or(&self, name: &str, fallback: &str) -> String {
        self.get(name).unwrap_or(fallback).to_string()
    }

    fn number(&self, name: &str, fallback: usize) -> usize {
        self.get(name)
            .and_then(|value| value.parse().ok())
            .unwrap_or(fallback)
    }

    fn decimal(&self, name: &str, fallback: f64) -> f64 {
        self.get(name)
            .and_then(|value| value.parse().ok())
            .unwrap_or(fallback)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = Arguments::parse();

    let root = PathBuf::from(std::env::var("KNOWNOTE_REPO_ROOT").unwrap_or_else(|_| ".".to_string()));
    let corpus_dir = PathBuf::from(arguments.get_or("corpus", &root.join("eval/corpus").display().to_string()));
    let questions_path = PathBuf::from(arguments.get_or(
        "questions",
        &root.join("eval/questions.jsonl").display().to_string(),
    ));
    let splits_path = PathBuf::from(arguments.get_or(
        "splits",
        &root.join("eval/splits.json").display().to_string(),
    ));

    let models_dir = PathBuf::from(arguments.get_or("models", "./.knownote-models"));
    let runtime = arguments.get("runtime").map(PathBuf::from);

    let split = match arguments.get_or("split", "all").as_str() {
        "validation" => EvalSplit::Validation,
        "test" => EvalSplit::Test,
        _ => EvalSplit::All,
    };

    let strategy = match arguments.get_or("strategy", "dense").as_str() {
        "sparse" => RetrievalStrategy::Sparse,
        "hybrid" => RetrievalStrategy::Hybrid,
        _ => RetrievalStrategy::Dense,
    };

    // The store is a throwaway: the harness never touches the user's own
    // database, which is the whole point of measuring on a clean index.
    let working = std::env::temp_dir().join(format!(
        "knownote-eval-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&working);
    let layout = StorageLayout::new(&working);
    let store = Store::open(layout)?;

    let embedder = match runtime {
        Some(path) => LocalEmbeddingBackend::load_with_runtime(&models_dir, Some(&path))?,
        None => LocalEmbeddingBackend::load(&models_dir)?,
    };

    let options = EvalOptions {
        corpus_label: "eval/corpus".to_string(),
        corpus_dir,
        questions_path,
        splits_path,
        baseline: arguments.get_or("baseline", "v1.6-rust"),
        split,
        candidate_k: arguments.number("candidate-k", 20),
        context_k: arguments.number("context-k", 3),
        threshold: arguments.decimal("threshold", 0.5),
        chunk_options: knownote_core::ChunkOptions {
            chunk_size: Some(arguments.number("chunk-size", 1000)),
            chunk_overlap: Some(arguments.number("chunk-overlap", 100)),
            min_chunk_size: Some(arguments.number("min-chunk-size", 100)),
            allow_span_pages: Some(arguments.get("allow-span-pages").is_some()),
            respect_headings: Some(arguments.get("respect-headings").is_some()),
            separators: None,
        },
        strategy,
    };

    let report = run_eval_harness(&store, &embedder, &options).await?;
    let report = stabilize(&report);

    let output = arguments.get_or("out", "");
    if output.is_empty() {
        println!(
            "{}",
            serde_json::to_string_pretty(&to_deterministic_report(&report))?
        );
    } else {
        std::fs::write(&output, serde_json::to_string_pretty(&to_deterministic_report(&report))?)?;
        println!("wrote {output}");
    }

    // The metrics line, so two runs can be compared by eye as well as by file.
    println!(
        "[eval] metrics recall@1={:.6} recall@5={:.6} recall@10={:.6} mrr={:.6} \
         ndcg@10={:.6} hit@5={:.6} map@10={:.6} contextP={:.6} contextR={:.6}",
        report.metrics.recall_at1,
        report.metrics.recall_at5,
        report.metrics.recall_at10,
        report.metrics.mrr,
        report.metrics.ndcg_at10,
        report.metrics.hit_rate_at5,
        report.metrics.map_at10,
        report.metrics.context_precision,
        report.metrics.context_recall
    );

    let _ = std::fs::remove_dir_all(&working);
    Ok(())
}
