//! The retrieval reproduces the committed baseline.
//!
//! This is the gate that says the rewrite is the same product: not "close
//! enough", but the same numbers, question by question, against
//! `docs/eval/baseline-v1.6.json`.
//!
//! It needs the pinned model and an ONNX Runtime, so it is skipped rather
//! than failed when they are absent. The TypeScript harness behaves the same
//! way: `npm run eval` refuses to run without the model and says so.
//!
//! ```text
//! KNOWNOTE_EVAL_MODEL_CACHE=<dir> KNOWNOTE_ONNXRUNTIME=<onnxruntime.dll> \
//!   cargo test -p knownote-eval --test baseline -- --ignored
//! ```

use std::path::PathBuf;

use knownote_eval::harness::{run_eval_harness, EvalOptions};
use knownote_eval::types::EvalSplit;
use knownote_store::StorageLayout;

fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("the repository root exists")
}

/// The model cache, or `None` when it is not there.
fn model_cache() -> Option<PathBuf> {
    let candidate = std::env::var("KNOWNOTE_EVAL_MODEL_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repository_root().join(".knownote-models"));

    let model = candidate
        .join("Xenova")
        .join("multilingual-e5-small")
        .join("761b726dd34fb83930e26aab4e9ac3899aa1fa78")
        .join("onnx")
        .join("model_quantized.onnx");

    model.exists().then_some(candidate)
}

/// The ONNX Runtime, or `None` when it is not there.
fn onnx_runtime() -> Option<PathBuf> {
    let candidate = PathBuf::from(std::env::var("KNOWNOTE_ONNXRUNTIME").ok()?);
    candidate.exists().then_some(candidate)
}

#[tokio::test]
#[ignore = "needs the pinned model and an ONNX Runtime"]
async fn the_retrieval_reproduces_the_committed_baseline() {
    let Some(cache) = model_cache() else {
        eprintln!("skipping: the pinned model is not in the cache");
        return;
    };
    let Some(runtime) = onnx_runtime() else {
        eprintln!("skipping: no ONNX Runtime was given");
        return;
    };

    let root = repository_root();
    let baseline_path = root.join("docs").join("eval").join("baseline-v1.6.json");
    let baseline: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&baseline_path).expect("the baseline is readable"),
    )
    .expect("the baseline is valid JSON");

    let working = std::env::temp_dir().join(format!(
        "knownote-eval-baseline-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&working);

    let store = knownote_store::Store::open(StorageLayout::new(&working)).expect("the store opens");
    let embedder = knownote_embed::LocalEmbeddingBackend::load_with_runtime(&cache, Some(&runtime))
        .expect("the model loads");

    let options = EvalOptions {
        corpus_dir: root.join("eval").join("corpus"),
        corpus_label: "eval/corpus".to_string(),
        questions_path: root.join("eval").join("questions.jsonl"),
        splits_path: root.join("eval").join("splits.json"),
        baseline: "v1.6".to_string(),
        split: EvalSplit::All,
        candidate_k: 20,
        context_k: 3,
        threshold: 0.5,
        chunk_options: knownote_core::ChunkOptions::default(),
        strategy: knownote_core::retrieval::RetrievalStrategy::Dense,
    };

    let report = run_eval_harness(&store, &embedder, &options)
        .await
        .expect("the harness runs");
    let report = knownote_eval::stabilize(&report);

    let _ = std::fs::remove_dir_all(&working);

    // The configuration has to match too, or the metrics would be describing
    // a different run.
    let expected_config = &baseline["config"];
    assert_eq!(
        report.config.documents, expected_config["documents"],
        "a different number of documents was indexed"
    );
    assert_eq!(
        report.config.questions, expected_config["questions"],
        "a different number of questions was run"
    );
    assert_eq!(
        report.config.chunk_count, expected_config["chunkCount"],
        "a different number of chunks was indexed"
    );
    assert_eq!(
        report.config.candidate_k, expected_config["candidateK"],
        "the candidate width differs"
    );
    assert_eq!(
        report.config.context_k, expected_config["contextK"],
        "the context width differs"
    );

    // Every metric.
    let expected_metrics = &baseline["metrics"];
    let pairs: [(&str, &str); 9] = [
        ("recallAt1", "recall_at1"),
        ("recallAt5", "recall_at5"),
        ("recallAt10", "recall_at10"),
        ("mrr", "mrr"),
        ("ndcgAt10", "ndcg_at10"),
        ("hitRateAt5", "hit_rate_at5"),
        ("mapAt10", "map_at10"),
        ("contextPrecision", "context_precision"),
        ("contextRecall", "context_recall"),
    ];

    for (baseline_name, ours) in pairs {
        let expected = expected_metrics[baseline_name]
            .as_f64()
            .unwrap_or_else(|| panic!("the baseline has {baseline_name}"));
        let actual = match ours {
            "recall_at1" => report.metrics.recall_at1,
            "recall_at5" => report.metrics.recall_at5,
            "recall_at10" => report.metrics.recall_at10,
            "mrr" => report.metrics.mrr,
            "ndcg_at10" => report.metrics.ndcg_at10,
            "hit_rate_at5" => report.metrics.hit_rate_at5,
            "map_at10" => report.metrics.map_at10,
            "context_precision" => report.metrics.context_precision,
            "context_recall" => report.metrics.context_recall,
            _ => unreachable!(),
        };
        assert!(
            (actual - expected).abs() < 1e-6,
            "{baseline_name} is {actual}, the baseline says {expected}"
        );
    }

    // And every question, because an average can hide a moved question.
    let expected_questions = baseline["perQuestion"]
        .as_array()
        .expect("the baseline has its questions");
    assert_eq!(
        report.per_question.len(),
        expected_questions.len(),
        "a different number of questions was reported"
    );

    for (index, expected) in expected_questions.iter().enumerate() {
        let actual = &report.per_question[index];
        assert_eq!(actual.id, expected["id"], "the questions are in a different order");
        assert_eq!(
            actual.retrieved_count, expected["retrievedCount"],
            "question {} retrieved a different number of passages",
            actual.id
        );
        assert_eq!(
            actual.first_relevant_rank, expected["firstRelevantRank"],
            "question {} found its first relevant passage at a different rank",
            actual.id
        );
        assert_eq!(
            actual.matches_by_rank.len(),
            expected["matchesByRank"].as_array().expect("the baseline has the matches").len(),
            "question {} matched a different number of blocks",
            actual.id
        );
        for (rank, (actual_matches, expected_matches)) in actual
            .matches_by_rank
            .iter()
            .zip(expected["matchesByRank"].as_array().expect("the baseline has the matches"))
            .enumerate()
        {
            let expected_matches: Vec<usize> = expected_matches
                .as_array()
                .expect("the matches are a list")
                .iter()
                .map(|value| value.as_u64().expect("a match is a number") as usize)
                .collect();
            assert_eq!(
                actual_matches, &expected_matches,
                "question {} matched different blocks at rank {}",
                actual.id,
                rank + 1
            );
        }
    }
}

/// The metric functions are ported, so they are checked against the
/// definitions rather than against the harness.
#[test]
fn the_metrics_match_their_definitions() {
    use knownote_eval::metrics::*;

    // Two ground-truth locations; the first passage covers the first, the
    // third covers the second.
    let matches = vec![vec![0], vec![], vec![1]];

    assert_eq!(first_relevant_rank(&matches), 1);
    assert_eq!(recall_at_k(&matches, 2, 1), 0.5);
    assert_eq!(recall_at_k(&matches, 2, 3), 1.0);
    assert_eq!(reciprocal_rank(&matches), 1.0);
    assert_eq!(hit_rate_at_k(&matches, 1), 1.0);
    assert_eq!(hit_rate_at_k(&matches, 1), 1.0);
    assert_eq!(evidence_precision_at_k(&matches, 3), 2.0 / 3.0);

    // Average precision: the first passage is a hit, the third is a hit.
    // The precision at a hit is the number of hits so far over the rank, so
    // this is (1/1 + 2/3) over the two ground-truth locations.
    let expected_ap = (1.0 + 2.0 / 3.0) / 2.0;
    assert!((average_precision_at_k(&matches, 2, 10) - expected_ap).abs() < 1e-9);

    // nDCG with binary gains: two hits, at ranks 1 and 3. The ideal ranking
    // would put both first, so the discount is log base 2 over the rank.
    let expected_ndcg = (1.0 / 2.0f64.log2() + 1.0 / 4.0f64.log2())
        / (1.0 / 2.0f64.log2() + 1.0 / 3.0f64.log2());
    assert!((ndcg_at_k(&matches, 2, 10) - expected_ndcg).abs() < 1e-9);

    // A question with no ground truth scores zero, not a division by zero.
    let empty: Vec<Vec<usize>> = vec![vec![]];
    assert_eq!(recall_at_k(&empty, 0, 10), 0.0);
    assert_eq!(ndcg_at_k(&empty, 0, 10), 0.0);
    assert_eq!(average_precision_at_k(&empty, 0, 10), 0.0);
    assert_eq!(reciprocal_rank(&empty), 0.0);
}
