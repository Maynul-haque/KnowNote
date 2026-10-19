//! The report the harness produces.
//!
//! A port of the report shapes in `src/main/eval/types.ts`.
//!
//! The committed baseline is compared byte for byte by CI, so the shapes and
//! the rounding are part of the contract: a report that says the same thing
//! in a different order or with a different number of decimals would fail the
//! diff for a reason that has nothing to do with retrieval.

use serde::{Deserialize, Serialize};

/// The metrics a run produced.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvalMetrics {
    pub recall_at1: f64,
    pub recall_at5: f64,
    pub recall_at10: f64,
    pub mrr: f64,
    pub ndcg_at10: f64,
    pub hit_rate_at5: f64,
    pub map_at10: f64,
    pub context_precision: f64,
    pub context_recall: f64,
}

/// One question's result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuestionReport {
    pub id: String,
    pub question: String,
    #[serde(rename = "type")]
    pub question_type: String,
    pub answerable: bool,
    pub first_relevant_rank: usize,
    pub relevant_count: usize,
    pub retrieved_count: usize,
    pub context_chars: usize,
    pub matches_by_rank: Vec<Vec<usize>>,
    /// Only when the run asked for the scores.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retrieved_scores: Option<Vec<f64>>,
}

/// One query class's results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvalTypeBreakdown {
    #[serde(rename = "type")]
    pub question_type: String,
    pub questions: usize,
    pub metrics: EvalMetrics,
}

/// The unanswerable questions' results.
///
/// They are a group of their own and never averaged into the ranking
/// metrics: they have no ground truth, so recall for them is 0/0 rather than
/// 0, and counting "should have refused" as "missed" would distort the table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UnanswerableReport {
    pub questions: usize,
    pub abstention_count: usize,
    /// The direction is the opposite of the other metrics: when there is
    /// nothing relevant, returning nothing is the right answer.
    pub retrieval_abstention_rate: f64,
    /// The number of candidates that passed the threshold, not the number
    /// handed to the prompt: the harness fetches `candidateK` passages to be
    /// able to score recall@10, so reading this as the prompt width would
    /// overstate the problem.
    pub mean_candidates_retrieved: f64,
    pub mean_context_passages: f64,
}

/// What the run was configured with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvalConfig {
    pub embedding: String,
    pub chunking: ChunkingConfig,
    pub retrieval: String,
    pub split: String,
    pub candidate_k: usize,
    pub context_k: usize,
    pub threshold: f64,
    pub corpus: String,
    pub documents: usize,
    pub questions: usize,
    pub chunk_count: usize,
}

/// The chunking the run used.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChunkingConfig {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub min_chunk_size: usize,
    pub allow_span_pages: bool,
    pub respect_headings: bool,
}

/// The timings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvalTiming {
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub indexing_ms: f64,
}

/// The whole report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvalReport {
    pub baseline: String,
    pub generated_by: String,
    pub config: EvalConfig,
    pub metrics: EvalMetrics,
    pub by_type: Vec<EvalTypeBreakdown>,
    pub unanswerable: UnanswerableReport,
    pub timing: EvalTiming,
    pub per_question: Vec<QuestionReport>,
}

/// The report without the timings, so two runs diff cleanly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvalDeterministicReport {
    pub baseline: String,
    pub generated_by: String,
    pub config: EvalConfig,
    pub metrics: EvalMetrics,
    pub by_type: Vec<EvalTypeBreakdown>,
    pub unanswerable: UnanswerableReport,
    pub per_question: Vec<QuestionReport>,
}

/// Round a metric to a stable number of decimals, so the JSON diffs cleanly.
pub fn round_metric(value: f64) -> f64 {
    // `toFixed` in JavaScript rounds half away from zero; Rust's `round`
    // rounds half away from zero too, but on a tie the two can differ
    // because the intermediate product is computed differently. Six decimals
    // is far below any difference that matters.
    (value * 1_000_000.0).round() / 1_000_000.0
}

/// Round every metric in a set.
pub fn round_metrics(metrics: &EvalMetrics) -> EvalMetrics {
    EvalMetrics {
        recall_at1: round_metric(metrics.recall_at1),
        recall_at5: round_metric(metrics.recall_at5),
        recall_at10: round_metric(metrics.recall_at10),
        mrr: round_metric(metrics.mrr),
        ndcg_at10: round_metric(metrics.ndcg_at10),
        hit_rate_at5: round_metric(metrics.hit_rate_at5),
        map_at10: round_metric(metrics.map_at10),
        context_precision: round_metric(metrics.context_precision),
        context_recall: round_metric(metrics.context_recall),
    }
}
