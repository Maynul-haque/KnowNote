//! The retrieval contract.
//!
//! Retrieval is a first-class capability, not something the chat happens to
//! do. The chat, the MCP server and the search panel all go through the same
//! retriever and get the same evidence, so none of them has to grow its own
//! RAG path — and the eval harness has one stable entry point to measure.
//!
//! The difference between `RetrievedEvidence` and a plain search result is
//! that provenance is a field, not something looked up later: a citation has
//! to land on *which source, which page, which passage*, and the evidence has
//! to carry that from the moment it is retrieved.

use serde::{Deserialize, Serialize};

/// A block an evidence passage covers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceBlock {
    pub block_id: String,
    pub kind: String,
    pub level: Option<i64>,
    pub page: Option<i64>,
    pub bbox: Option<serde_json::Value>,
    pub text: String,
    pub start_offset: usize,
    pub end_offset: usize,
    /// The block's position in the document.
    pub order: usize,
    /// Where the covered range starts inside the block.
    pub start_in_block: usize,
    /// Where the covered range ends inside the block.
    pub end_in_block: usize,
}

/// Where the evidence sits in its source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceLocator {
    pub page_start: Option<i64>,
    pub page_end: Option<i64>,
    /// The covered blocks, in document order.
    pub blocks: Vec<EvidenceBlock>,
}

/// Which source the evidence came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceSource {
    pub title: String,
    pub kind: String,
}

/// One piece of retrieved evidence, ready to be cited.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievedEvidence {
    pub chunk_id: String,
    pub document_id: String,
    pub content: String,
    pub score: f64,
    pub chunk_index: usize,
    pub source: EvidenceSource,
    pub locator: EvidenceLocator,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// A filter that narrows a search.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RetrievalFilter {
    /// The documents to search within. Empty or absent means the whole
    /// notebook.
    #[serde(default)]
    pub document_ids: Vec<String>,
}

/// Which retrieval path to take.
///
/// `Dense` finds passages that mean the same thing, `Sparse` finds passages
/// that use the same words, `Hybrid` wants both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RetrievalStrategy {
    Dense,
    Sparse,
    Hybrid,
}

impl Default for RetrievalStrategy {
    fn default() -> Self {
        RetrievalStrategy::Dense
    }
}

impl RetrievalStrategy {
    /// The name stored in a trace.
    pub fn as_str(self) -> &'static str {
        match self {
            RetrievalStrategy::Dense => "dense",
            RetrievalStrategy::Sparse => "sparse",
            RetrievalStrategy::Hybrid => "hybrid",
        }
    }
}

/// One retrieval request.
///
/// The two K's are separate because they answer different questions:
///
/// * `candidate_k` — how wide each channel is in the first stage. Favours recall.
/// * `top_k` — how much evidence is finally delivered. Favours precision.
///
/// Collapsing them would let "the vector store returned N rows" impersonate
/// two-stage retrieval: with hybrid, each channel taking `top_k` rows would
/// leave a fusion pool of at most `2 * top_k`, which is then cut back to
/// `top_k` — so the fusion would barely have room to happen.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievalRequest {
    pub notebook_id: String,
    pub query: String,
    /// The first-stage width. Never smaller than `top_k`.
    pub candidate_k: Option<usize>,
    /// How much evidence to deliver.
    pub top_k: Option<usize>,
    /// The similarity floor for the dense channel.
    pub threshold: Option<f64>,
    /// A filter narrowing the search.
    pub filter: Option<RetrievalFilter>,
    /// Which path to take. `Dense` by default.
    pub strategy: Option<RetrievalStrategy>,
}

/// The first-stage width when nothing says otherwise.
pub const DEFAULT_CANDIDATE_K: usize = 20;

/// How much evidence is delivered when nothing says otherwise.
pub const DEFAULT_TOP_K: usize = 5;

/// The dense channel's similarity floor when nothing says otherwise.
pub const DEFAULT_DENSE_THRESHOLD: f64 = 0.5;

/// The first stage's real width: `candidate_k`, but never less than `top_k`.
///
/// Without this, "take a wider pool than you deliver" only happens most of the
/// time: any call with `top_k > candidate_k` — the search panel's limit, the
/// MCP server's `top_k` — would silently return fewer results.
pub fn effective_candidate_k(candidate_k: Option<usize>, top_k: Option<usize>) -> usize {
    let top_k = top_k.unwrap_or(DEFAULT_TOP_K);
    candidate_k.unwrap_or(DEFAULT_CANDIDATE_K).max(top_k)
}

/// The similarity floor that actually applies to the **dense channel**.
///
/// `Hybrid` runs dense too, so it has a threshold as well. Only `Sparse` has
/// none, because BM25 has no notion of a similarity floor.
///
/// One function produces this value so that the number passed to the dense
/// channel and the number written into the trace cannot come apart. They once
/// could: the hybrid's dense leg used 0.5 while the trace recorded
/// `undefined`, which made the snapshot unable to reproduce the search.
pub fn dense_channel_threshold(
    strategy: RetrievalStrategy,
    requested: Option<f64>,
) -> Option<f64> {
    if strategy == RetrievalStrategy::Sparse {
        return None;
    }
    Some(requested.unwrap_or(DEFAULT_DENSE_THRESHOLD))
}

/// The parameters a retrieval actually ran with.
///
/// This is snapshotted with every answer, so it has to be produced by the
/// retrieval layer rather than guessed by the caller.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievalTrace {
    pub strategy: String,
    /// The documents the search was limited to.
    pub scope: Vec<String>,
    /// The first-stage width, after `effective_candidate_k`.
    pub candidate_k: usize,
    /// How much evidence was delivered.
    pub top_k: usize,
    /// The floor that applied to the dense channel.
    ///
    /// Absent means the dense channel did not run, not that the floor was
    /// zero.
    pub dense_threshold: Option<f64>,
    pub duration_ms: f64,
}

/// What a retrieval produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievalResult {
    pub evidence: Vec<RetrievedEvidence>,
    pub trace: RetrievalTrace,
}

/// One channel's hit, before fusion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateHit {
    pub chunk_id: String,
    pub score: f64,
}

/// A fused hit with its final rank.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredChunkHit {
    pub chunk_id: String,
    pub score: f64,
    pub rank: usize,
}

/// The RRF constant from the original paper. It damps the top ranks' advantage.
pub const RRF_K: f64 = 60.0;

/// Reciprocal Rank Fusion across channels.
///
/// A chunk's fused score is `sum(1 / (k + rank))` over the channels that
/// returned it, so a chunk ranked well by one channel and missing from
/// another still scores, and agreement between the channels is what moves a
/// chunk up.
///
/// Scores are not comparable across channels — cosine and BM25 are different
/// scales — which is exactly why only the ranks are used.
pub fn rrf_fuse(channels: &[&[CandidateHit]], k: f64) -> Vec<ScoredChunkHit> {
    let mut fused: Vec<(String, f64)> = Vec::new();

    for channel in channels {
        for (index, hit) in channel.iter().enumerate() {
            let contribution = 1.0 / (k + index as f64 + 1.0);
            match fused.iter_mut().find(|(id, _)| id == &hit.chunk_id) {
                Some(entry) => entry.1 += contribution,
                None => fused.push((hit.chunk_id.clone(), contribution)),
            }
        }
    }

    let mut hits: Vec<ScoredChunkHit> = fused
        .into_iter()
        .map(|(chunk_id, score)| ScoredChunkHit {
            chunk_id,
            score,
            rank: 0,
        })
        .collect();

    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| collate_root_ascii(&a.chunk_id, &b.chunk_id))
    });

    for (index, hit) in hits.iter_mut().enumerate() {
        hit.rank = index + 1;
    }

    hits
}

/// Turn free text into an FTS5 `MATCH` expression.
///
/// Every whitespace-separated term is quoted, so the user's text can never be
/// parsed as FTS operators — `-`, `*`, `:`, `NEAR`, unbalanced quotes. A
/// query that is a syntax error is a query that returns nothing, which reads
/// as "no results" rather than "your search is malformed".
///
/// The terms are **OR**ed. AND was the first version and it made BM25 useless
/// for chat retrieval: the terms of a natural-language question almost never
/// all appear in one chunk, so the sparse channel returned nothing and hybrid
/// silently degenerated to dense. BM25 already ranks a chunk that matches
/// more terms higher, so OR keeps the best passages first while still finding
/// them.
///
/// Returns `None` when there is nothing to search for.
pub fn build_fts_match_query(raw: &str) -> Option<String> {
    let terms: Vec<&str> = raw.split_whitespace().filter(|term| !term.is_empty()).collect();
    if terms.is_empty() {
        return None;
    }
    Some(
        terms
            .iter()
            .map(|term| format!("\"{}\"", term.replace('"', "\"\"")))
            .collect::<Vec<_>>()
            .join(" OR "),
    )
}

/// Compare two strings the way the original implementation's tie-break did.
///
/// The original sorted equal-scored chunks with `localeCompare`, which under
/// ICU's root collation orders whitespace, then punctuation, then symbols,
/// then digits, then letters — and compares letters without regard to case
/// first, breaking a remaining tie in favour of lowercase.
///
/// A byte comparison would order those groups differently, so the tie-break
/// would not match. The difference only shows between chunks with exactly the
/// same score, but those are common enough in the fusion pool to be worth
/// getting right.
fn collate_root_ascii(left: &str, right: &str) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    let left: Vec<char> = left.chars().collect();
    let right: Vec<char> = right.chars().collect();

    for (a, b) in left.iter().zip(right.iter()) {
        let (a_primary, a_tertiary) = collation_weights(*a);
        let (b_primary, b_tertiary) = collation_weights(*b);

        match a_primary.cmp(&b_primary) {
            Ordering::Equal => {}
            other => return other,
        }

        // The primary weights are equal: the letters are the same, only the
        // case differs. Lowercase sorts first.
        if a_tertiary != b_tertiary {
            return a_tertiary.cmp(&b_tertiary);
        }
    }

    left.len().cmp(&right.len())
}

/// The collation weight of one character: a primary class and a tertiary
/// tie-break.
fn collation_weights(c: char) -> (u8, u8) {
    match c {
        // Whitespace.
        '\t'..='\r' | ' ' => (0, 0),
        // Punctuation.
        '!' | '"' | '#' | '%' | '&' | '\'' | '(' | ')' | '*' | ',' | '-' | '.' | '/' | ':'
        | ';' | '?' | '@' | '[' | '\\' | ']' | '_' | '{' | '}' => (1, c as u8),
        // Symbols.
        '$' | '+' | '<' | '=' | '>' | '^' | '`' | '|' | '~' => (2, c as u8),
        // Digits.
        '0'..='9' => (3, c as u8),
        // Letters: case-insensitive at the primary level.
        'A'..='Z' => (4, c as u8 + 32),
        'a'..='z' => (4, c as u8),
        // Anything else falls back to its code point.
        _ => (5, c as u8),
    }
}
