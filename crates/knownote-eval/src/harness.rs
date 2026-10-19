//! The harness.
//!
//! A port of `src/main/eval/harness.ts`.
//!
//! It indexes a corpus through the normal ingestion path, runs the real
//! retriever, and reports the retrieval metrics. It is the reference point
//! every retrieval change has to be measured against, so the numbers are
//! produced here and never written by hand.
//!
//! The ground truth is resolved from corpus identity to runtime identity
//! *after* ingestion, because the document id is generated and the block id
//! embeds it. Nothing in the committed dataset names a runtime id.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use knownote_core::blocks::{assign_block_ids, build_document_blocks};
use knownote_core::chunking::ChunkingService;
use knownote_core::ChunkBlock;
use knownote_core::retrieval::{RetrievalRequest, RetrievalStrategy};
use knownote_embedding::EmbeddingBackend;
use knownote_ingest::{DocumentLoader, MarkdownLoader};
use knownote_store::ingest::{ensure_vector_table, persist_document, DocumentInput};
use knownote_store::Store;

use crate::metrics::{
    average_precision_at_k, evidence_precision_at_k, first_relevant_rank, hit_rate_at_k, mean,
    ndcg_at_k, percentile, recall_at_k, reciprocal_rank,
};
use crate::report::{
    round_metrics, ChunkingConfig, EvalConfig, EvalMetrics, EvalReport, EvalTiming,
    EvalTypeBreakdown, QuestionReport, UnanswerableReport,
};
use crate::types::{
    assert_question_shape, parse_split_assignment, question_type_of, select_split, EvalQuestion,
    EvalSplit,
};

/// The notebook the corpus is indexed into.
///
/// The id has to be usable as part of a SQL identifier, because the vector
/// table is named after it.
const NOTEBOOK_ID: &str = "evalnotebook";

/// How the harness is configured.
pub struct EvalOptions {
    /// Where the corpus is read from.
    pub corpus_dir: PathBuf,
    /// The label recorded in the report, so the JSON does not depend on where
    /// the run happened.
    pub corpus_label: String,
    pub questions_path: PathBuf,
    pub splits_path: PathBuf,
    pub baseline: String,
    /// Which split to run.
    pub split: EvalSplit,
    /// The first stage's width, and the depth the ranking metrics are scored
    /// at.
    pub candidate_k: usize,
    /// How many passages the production prompt would take.
    pub context_k: usize,
    /// The similarity floor. Zero keeps the ranking intact.
    pub threshold: f64,
    /// The chunking configuration.
    pub chunk_options: knownote_core::ChunkOptions,
    /// The retrieval strategy.
    pub strategy: RetrievalStrategy,
}

/// Everything that can go wrong in the harness.
#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("{0}")]
    Invalid(String),

    #[error("the corpus could not be read: {0}")]
    Corpus(String),

    #[error("{0}")]
    Store(#[from] knownote_store::StoreError),

    #[error("{0}")]
    Embed(#[from] knownote_embedding::EmbedError),

    #[error("{0}")]
    Ingest(#[from] knownote_ingest::IngestError),
}

/// The ground truth, resolved to runtime identities.
struct ResolvedGroundTruth {
    block_id: String,
}

/// Run the harness.
pub async fn run_eval_harness(
    store: &Store,
    embedder: &dyn EmbeddingBackend,
    options: &EvalOptions,
) -> Result<EvalReport, EvalError> {
    // A context window wider than the retrieval depth can never be filled: the
    // harness fetches `candidateK` passages and the context metrics look at
    // `contextK` of them.
    //
    // Without this check the sweep silently produced rows where two different
    // context widths were *identical*, not because they assessed the same
    // thing but because the passages did not exist. A wrong number that looks
    // like a measurement is worse than a failure.
    if options.context_k > options.candidate_k {
        return Err(EvalError::Invalid(format!(
            "contextK ({}) cannot exceed candidateK ({}): the harness retrieves candidateK \
             passages, so a wider context window can never be filled.",
            options.context_k, options.candidate_k
        )));
    }

    let indexing_started = Instant::now();
    let (document_ids, chunk_count) = index_corpus(store, embedder, options).await?;
    let indexing_ms = indexing_started.elapsed().as_secs_f64() * 1000.0;

    let questions_raw = std::fs::read_to_string(&options.questions_path).map_err(|error| {
        EvalError::Corpus(format!(
            "cannot read {}: {error}",
            options.questions_path.display()
        ))
    })?;
    let all_questions = parse_questions(&questions_raw)?;

    // The dataset's own contract first: an answerable question must have
    // ground truth and an unanswerable one must not. Getting them the wrong
    // way round does not make the harness fail — it makes the numbers quietly
    // meaningless.
    for question in &all_questions {
        assert_question_shape(question)
            .map_err(|error| EvalError::Invalid(error))?;
    }

    let assignment = load_split_assignment(options)?;
    let questions = select_split(&all_questions, options.split, &assignment)
        .map_err(EvalError::Invalid)?;
    if questions.is_empty() {
        return Err(EvalError::Invalid(format!(
            "eval split {:?} selected no questions from {}",
            options.split,
            options.questions_path.display()
        )));
    }

    let mut per_question: Vec<QuestionReport> = Vec::new();
    let mut latencies: Vec<f64> = Vec::new();

    for question in &questions {
        let ground_truth = resolve_ground_truth(store, question, &document_ids)?;
        let ground_truth_ids: Vec<String> = ground_truth
            .iter()
            .map(|entry| entry.block_id.clone())
            .collect();

        let started = Instant::now();
        // The ranking metrics are scored on the full candidate depth, not
        // truncated to `contextK`: recall@10 needs at least ten results, and
        // truncating only takes a prefix of the candidate list, so the prefix
        // is ordered the same way the whole list was.
        let result = knownote_store::retrieval::search(
            store,
            embedder,
            &RetrievalRequest {
                notebook_id: NOTEBOOK_ID.to_string(),
                query: question.question.clone(),
                candidate_k: Some(options.candidate_k),
                top_k: Some(options.candidate_k),
                threshold: Some(options.threshold),
                filter: None,
                strategy: Some(options.strategy),
            },
        )
        .await?;
        latencies.push(started.elapsed().as_secs_f64() * 1000.0);

        let matches_by_rank: Vec<Vec<usize>> = result
            .evidence
            .iter()
            .map(|evidence| {
                let block_ids: Vec<&str> = evidence
                    .locator
                    .blocks
                    .iter()
                    .map(|block| block.block_id.as_str())
                    .collect();
                ground_truth_ids
                    .iter()
                    .enumerate()
                    .filter(|(_, block_id)| block_ids.contains(&block_id.as_str()))
                    .map(|(index, _)| index)
                    .collect()
            })
            .collect();

        let context_chars: usize = result
            .evidence
            .iter()
            .take(options.context_k)
            .map(|evidence| evidence.content.chars().count())
            .sum();

        per_question.push(QuestionReport {
            id: question.id.clone(),
            question: question.question.clone(),
            question_type: question_type_of(question),
            answerable: question.is_answerable(),
            first_relevant_rank: first_relevant_rank(&matches_by_rank),
            relevant_count: ground_truth.len(),
            retrieved_count: result.evidence.len(),
            context_chars,
            matches_by_rank,
            retrieved_scores: None,
        });
    }

    // The unanswerable questions do not go into the ranking metrics.
    let answerable: Vec<&QuestionReport> = per_question
        .iter()
        .filter(|question| question.answerable)
        .collect();
    let unanswerable_questions: Vec<&QuestionReport> = per_question
        .iter()
        .filter(|question| !question.answerable)
        .collect();

    let metrics = summarize(&answerable, options.context_k);

    // One row per query class, sorted by name, so the same JSON diffs cleanly
    // between two runs.
    let mut classes: Vec<&str> = answerable
        .iter()
        .map(|question| question.question_type.as_str())
        .collect();
    classes.sort_unstable();
    classes.dedup();

    let by_type: Vec<EvalTypeBreakdown> = classes
        .into_iter()
        .map(|class| {
            let group: Vec<&QuestionReport> = answerable
                .iter()
                .copied()
                .filter(|question| question.question_type == class)
                .collect();
            EvalTypeBreakdown {
                question_type: class.to_string(),
                questions: group.len(),
                metrics: summarize(&group, options.context_k),
            }
        })
        .collect();

    let abstention_count = unanswerable_questions
        .iter()
        .filter(|question| question.retrieved_count == 0)
        .count();
    let unanswerable = UnanswerableReport {
        questions: unanswerable_questions.len(),
        abstention_count,
        retrieval_abstention_rate: if unanswerable_questions.is_empty() {
            0.0
        } else {
            abstention_count as f64 / unanswerable_questions.len() as f64
        },
        mean_candidates_retrieved: mean(
            &unanswerable_questions
                .iter()
                .map(|question| question.retrieved_count as f64)
                .collect::<Vec<_>>(),
        ),
        mean_context_passages: mean(
            &unanswerable_questions
                .iter()
                .map(|question| {
                    question.retrieved_count.min(options.context_k) as f64
                })
                .collect::<Vec<_>>(),
        ),
    };

    let chunking = options.chunk_options.resolve_over(&knownote_core::default_chunk_options());

    Ok(EvalReport {
        baseline: options.baseline.clone(),
        generated_by: "cargo run -p knownote-eval".to_string(),
        config: EvalConfig {
            embedding: format!(
                "{}@{} {} ({}d, local)",
                knownote_embed::LOCAL_EMBEDDING_MODEL.id,
                knownote_embed::LOCAL_EMBEDDING_MODEL.revision,
                knownote_embed::LOCAL_EMBEDDING_MODEL.dtype,
                knownote_embed::LOCAL_EMBEDDING_MODEL.dimensions
            ),
            chunking: ChunkingConfig {
                chunk_size: chunking.chunk_size,
                chunk_overlap: chunking.chunk_overlap,
                min_chunk_size: chunking.min_chunk_size,
                allow_span_pages: chunking.allow_span_pages,
                respect_headings: chunking.respect_headings,
            },
            retrieval: options.strategy.as_str().to_string(),
            split: options.split.as_str().to_string(),
            candidate_k: options.candidate_k,
            context_k: options.context_k,
            threshold: options.threshold,
            corpus: options.corpus_label.clone(),
            documents: document_ids.len(),
            questions: questions.len(),
            chunk_count,
        },
        metrics,
        by_type,
        unanswerable,
        timing: EvalTiming {
            latency_p50_ms: percentile(&latencies, 50.0),
            latency_p95_ms: percentile(&latencies, 95.0),
            indexing_ms,
        },
        per_question,
    })
}

/// The metrics of a set of questions.
///
/// One function for the overall average and for each class, because a grouped
/// average that used a different definition would not be comparable with the
/// overall one.
fn summarize(questions: &[&QuestionReport], context_k: usize) -> EvalMetrics {
    let recall = |k: usize| {
        mean(
            &questions
                .iter()
                .map(|question| recall_at_k(&question.matches_by_rank, question.relevant_count, k))
                .collect::<Vec<_>>(),
        )
    };

    EvalMetrics {
        recall_at1: recall(1),
        recall_at5: recall(5),
        recall_at10: recall(10),
        mrr: mean(
            &questions
                .iter()
                .map(|question| reciprocal_rank(&question.matches_by_rank))
                .collect::<Vec<_>>(),
        ),
        ndcg_at10: mean(
            &questions
                .iter()
                .map(|question| ndcg_at_k(&question.matches_by_rank, question.relevant_count, 10))
                .collect::<Vec<_>>(),
        ),
        hit_rate_at5: mean(
            &questions
                .iter()
                .map(|question| hit_rate_at_k(&question.matches_by_rank, 5))
                .collect::<Vec<_>>(),
        ),
        map_at10: mean(
            &questions
                .iter()
                .map(|question| {
                    average_precision_at_k(&question.matches_by_rank, question.relevant_count, 10)
                })
                .collect::<Vec<_>>(),
        ),
        // The two context metrics share one window, because they are two
        // sides of the same question: how much of what went into the prompt
        // was relevant, and how much of what was needed actually got in.
        context_precision: mean(
            &questions
                .iter()
                .map(|question| evidence_precision_at_k(&question.matches_by_rank, context_k))
                .collect::<Vec<_>>(),
        ),
        context_recall: recall(context_k),
    }
}

/// Index every corpus document.
///
/// Returns the corpus-relative path of each document and its runtime id, and
/// the number of chunks written.
async fn index_corpus(
    store: &Store,
    embedder: &dyn EmbeddingBackend,
    options: &EvalOptions,
) -> Result<(BTreeMap<String, String>, usize), EvalError> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_secs() as i64)
        .unwrap_or(0);

    store
        .app()
        .execute(
            "INSERT OR REPLACE INTO notebooks (id, title, created_at, updated_at)
             VALUES (?1, 'Eval corpus', ?2, ?2)",
            rusqlite::params![NOTEBOOK_ID, now],
        )
        .map_err(|error| EvalError::Store(error.into()))?;

    let mut files: Vec<String> = std::fs::read_dir(&options.corpus_dir)
        .map_err(|error| {
            EvalError::Corpus(format!(
                "cannot read {}: {error}",
                options.corpus_dir.display()
            ))
        })?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name().to_string_lossy().to_string())
        .filter(|name| !name.starts_with('.'))
        .collect();
    files.sort();

    let loader = MarkdownLoader::new();
    let chunker = ChunkingService::new();
    let mut document_ids = BTreeMap::new();
    let mut chunk_count = 0usize;

    for file in &files {
        let path = options.corpus_dir.join(file);
        let result = loader
            .load_from_path(&path)
            .map_err(|error| EvalError::Corpus(format!("{}: {error}", path.display())))?;

        let document_id = format!("doc_{}", sanitize(file));
        let blocks = assign_block_ids(
            &document_id,
            build_document_blocks(&result.content, result.structure.as_ref()),
        );

        let chunk_blocks: Vec<ChunkBlock> = blocks
            .iter()
            .map(|block| ChunkBlock {
                id: block.id.clone(),
                kind: block.draft.kind,
                page: block.draft.page,
                level: block.draft.level,
                text: block.draft.text.clone(),
                start_offset: block.draft.start_offset,
                end_offset: block.draft.end_offset,
            })
            .collect();

        let chunks = chunker.chunk_blocks(
            &result.content,
            &chunk_blocks,
            Some(&options.chunk_options),
        );

        let texts: Vec<String> = chunks
            .iter()
            .map(|chunk| chunk.content.clone())
            .collect();
        let vectors = embedder.embed_documents(&texts).await?;

        ensure_vector_table(store, NOTEBOOK_ID, embedder.dimensions())?;

        let written = persist_document(
            store,
            &DocumentInput {
                document_id: &document_id,
                notebook_id: NOTEBOOK_ID,
                title: file,
                content: &result.content,
                structure: result.structure.as_ref(),
                blocks: &blocks,
                chunks: &chunks,
                vectors: &vectors,
            },
        )?;

        chunk_count += written;
        document_ids.insert(file.clone(), document_id);
    }

    Ok((document_ids, chunk_count))
}

/// A file name that is safe to use in an id.
fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Resolve the ground truth to runtime identities.
fn resolve_ground_truth(
    store: &Store,
    question: &EvalQuestion,
    document_ids: &BTreeMap<String, String>,
) -> Result<Vec<ResolvedGroundTruth>, EvalError> {
    let mut resolved = Vec::new();

    for location in &question.relevant {
        let document_id = document_ids.get(&location.document).ok_or_else(|| {
            EvalError::Invalid(format!(
                "question {} references {:?}, which is not in the corpus",
                question.id, location.document
            ))
        })?;

        let block: Option<(String, Option<i64>, String)> = store
            .content()
            .query_row(
                "SELECT id, page, text FROM document_blocks
                 WHERE document_id = ?1 AND \"order\" = ?2",
                rusqlite::params![document_id, location.block as i64],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .ok();

        let Some((block_id, page, text)) = block else {
            return Err(EvalError::Invalid(format!(
                "question {}: {:?} has no block with ordinal {}",
                question.id, location.document, location.block
            )));
        };

        if let Some(quote) = &location.quote {
            if !normalize(&text).contains(&normalize(quote)) {
                return Err(EvalError::Invalid(format!(
                    "question {}: quote drifted — block {} of {:?} does not contain {quote:?}. \
                     Update the ground truth deliberately.",
                    question.id, location.block, location.document
                )));
            }
        }

        if let Some(expected_page) = location.page {
            if page != Some(expected_page) {
                return Err(EvalError::Invalid(format!(
                    "question {}: {:?} block {} is on page {:?}, ground truth says {expected_page}",
                    question.id, location.document, location.block, page
                )));
            }
        }

        resolved.push(ResolvedGroundTruth { block_id });
    }

    Ok(resolved)
}

/// Normalise text for the quote check.
fn normalize(text: &str) -> String {
    text.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Parse the questions file.
fn parse_questions(raw: &str) -> Result<Vec<EvalQuestion>, EvalError> {
    raw.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            serde_json::from_str(line).map_err(|error| {
                EvalError::Invalid(format!(
                    "questions.jsonl line {} is not valid JSON: {error}",
                    index + 1
                ))
            })
        })
        .collect()
}

/// Read the split manifest.
///
/// The `all` split does not need the manifest, so a run of `all` does not
/// fail because the manifest is missing.
fn load_split_assignment(options: &EvalOptions) -> Result<crate::types::SplitAssignment, EvalError> {
    if options.split == EvalSplit::All {
        return Ok(Default::default());
    }

    let raw = std::fs::read_to_string(&options.splits_path).map_err(|error| {
        EvalError::Corpus(format!(
            "cannot read {}: {error}",
            options.splits_path.display()
        ))
    })?;

    parse_split_assignment(&raw, &options.splits_path.display().to_string())
        .map_err(EvalError::Invalid)
}

/// Round the metrics so the JSON diffs cleanly.
pub fn stabilize(report: &EvalReport) -> EvalReport {
    let mut stabilized = report.clone();
    stabilized.metrics = round_metrics(&report.metrics);
    for entry in &mut stabilized.by_type {
        entry.metrics = round_metrics(&entry.metrics);
    }
    stabilized
}

/// Strip the timings, so the committed JSON is diff-stable.
pub fn to_deterministic_report(
    report: &EvalReport,
) -> crate::report::EvalDeterministicReport {
    crate::report::EvalDeterministicReport {
        baseline: report.baseline.clone(),
        generated_by: report.generated_by.clone(),
        config: report.config.clone(),
        metrics: report.metrics,
        by_type: report.by_type.clone(),
        unanswerable: report.unanswerable.clone(),
        per_question: report.per_question.clone(),
    }
}

/// The default corpus path, relative to the repository root.
pub fn default_corpus_dir(root: &Path) -> PathBuf {
    root.join("eval").join("corpus")
}
