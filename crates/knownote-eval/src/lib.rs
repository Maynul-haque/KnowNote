//! The retrieval evaluation harness.
//!
//! Run it over the eval corpus and compare the numbers with the committed
//! baseline:
//!
//! ```text
//! cargo run -p knownote-eval -- --corpus eval/corpus --models <model-cache>
//! ```
//!
//! The baseline every change is measured against is
//! `docs/eval/baseline-v1.6.json`.

pub mod harness;
pub mod metrics;
pub mod report;
pub mod types;

pub use harness::{
    run_eval_harness, stabilize, to_deterministic_report, EvalError, EvalOptions,
};
pub use report::{
    ChunkingConfig, EvalConfig, EvalDeterministicReport, EvalMetrics, EvalReport, EvalTiming,
    EvalTypeBreakdown, QuestionReport, UnanswerableReport,
};
pub use types::{EvalQuestion, EvalRelevantLocation, EvalSplit};
