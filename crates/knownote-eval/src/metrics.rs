//! Retrieval metrics.
//!
//! A port of `src/main/eval/metrics.ts`. These are pure functions over the
//! per-question match matrix, so the arithmetic is unit-testable on its own
//! and the harness only has to supply the retrieval output.
//!
//! Relevance is binary: a passage either covers a ground-truth block or it
//! does not. The dataset records locations, not degrees of relevance.

/// The ground-truth indices each retrieved rank matched, in rank order.
pub type MatchMatrix = Vec<Vec<usize>>;

/// The 1-based rank of the first relevant item, or 0 when none is relevant.
pub fn first_relevant_rank(matches_by_rank: &MatchMatrix) -> usize {
    match matches_by_rank.iter().position(|matches| !matches.is_empty()) {
        Some(index) => index + 1,
        None => 0,
    }
}

/// The fraction of the ground truth covered by the first `k` ranks.
pub fn recall_at_k(matches_by_rank: &MatchMatrix, ground_truth_count: usize, k: usize) -> f64 {
    if ground_truth_count == 0 {
        return 0.0;
    }

    let mut covered: Vec<usize> = Vec::new();
    for matches in matches_by_rank.iter().take(k) {
        for match_index in matches {
            if !covered.contains(match_index) {
                covered.push(*match_index);
            }
        }
    }

    covered.len() as f64 / ground_truth_count as f64
}

/// The reciprocal rank of the first relevant item.
pub fn reciprocal_rank(matches_by_rank: &MatchMatrix) -> f64 {
    let rank = first_relevant_rank(matches_by_rank);
    if rank == 0 {
        0.0
    } else {
        1.0 / rank as f64
    }
}

/// Hit rate@k: 1 when any of the first `k` ranks covers the ground truth.
///
/// Deliberately the bluntest metric here. Recall@k says *how much* of the
/// ground truth was found; the hit rate says only whether the answer was
/// findable at all. When a question needs two passages, a run that finds one
/// scores 0.5 recall and 1.0 hit rate — and the second is the one that says
/// the model had a chance.
pub fn hit_rate_at_k(matches_by_rank: &MatchMatrix, k: usize) -> f64 {
    if matches_by_rank.iter().take(k).any(|matches| !matches.is_empty()) {
        1.0
    } else {
        0.0
    }
}

/// Average precision@k, averaged over the ground-truth locations.
///
/// Precision is measured at each rank that recovers a *fresh* ground-truth
/// location — the same rule nDCG uses — and then normalised by the number of
/// ground-truth locations. That makes AP the one metric here that combines
/// ranking position with coverage: pulling a second relevant passage from rank
/// 9 to rank 2 moves it, while recall@5 sits still.
pub fn average_precision_at_k(
    matches_by_rank: &MatchMatrix,
    ground_truth_count: usize,
    k: usize,
) -> f64 {
    if ground_truth_count == 0 {
        return 0.0;
    }

    let mut covered: Vec<usize> = Vec::new();
    let mut found = 0usize;
    let mut sum = 0.0f64;
    let limit = matches_by_rank.len().min(k);

    for (index, matches) in matches_by_rank.iter().take(limit).enumerate() {
        let mut fresh = false;
        for match_index in matches {
            if *match_index < ground_truth_count && !covered.contains(match_index) {
                covered.push(*match_index);
                fresh = true;
            }
        }
        if fresh {
            found += 1;
            sum += found as f64 / (index + 1) as f64;
        }
    }

    sum / ground_truth_count as f64
}

/// nDCG@k with binary gains.
///
/// A rank only gains when it covers a ground-truth location that no earlier
/// rank already covered. Without that, several passages recovering the *same*
/// block each scored a gain while the ideal ranking only counted the block
/// once, and nDCG went above 1 — which is not a valid ranking metric.
/// Smaller chunks overlap more and recover the same block from several ranks,
/// so the bug only became visible under the chunking experiment.
pub fn ndcg_at_k(matches_by_rank: &MatchMatrix, ground_truth_count: usize, k: usize) -> f64 {
    if ground_truth_count == 0 {
        return 0.0;
    }

    let mut dcg = 0.0f64;
    let mut covered: Vec<usize> = Vec::new();
    let limit = matches_by_rank.len().min(k);

    for (index, matches) in matches_by_rank.iter().take(limit).enumerate() {
        let mut fresh = false;
        for match_index in matches {
            // An index at or beyond the ground-truth count is not part of the
            // declared ground truth and cannot be a gain; ignoring it keeps
            // the DCG at or below the ideal by construction.
            if *match_index < ground_truth_count && !covered.contains(match_index) {
                covered.push(*match_index);
                fresh = true;
            }
        }
        if fresh {
            dcg += 1.0 / ((index + 2) as f64).log2();
        }
    }

    let mut idcg = 0.0f64;
    let ideal = ground_truth_count.min(k);
    for index in 0..ideal {
        idcg += 1.0 / ((index + 2) as f64).log2();
    }

    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

/// Evidence precision@k: of the first `k` retrieved passages, the share that
/// cover a ground-truth block.
///
/// This is retrieval precision, **not** answer citation recall. No model runs
/// in this harness and no answer is produced, so a metric that claimed to be
/// about an answer's citations would be a false claim.
///
/// Each retrieved passage is counted once, however many ground-truth blocks it
/// covers. Overlapping chunks can cover the same block, and each still counts
/// as a separate passage — a reader would see two citations there, and this
/// metric describes what they would be shown.
pub fn evidence_precision_at_k(matches_by_rank: &MatchMatrix, k: usize) -> f64 {
    let retrieved = &matches_by_rank[..matches_by_rank.len().min(k)];
    if retrieved.is_empty() {
        return 0.0;
    }
    let grounded = retrieved.iter().filter(|matches| !matches.is_empty()).count();
    grounded as f64 / retrieved.len() as f64
}

/// The mean of a list.
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// The nearest-rank percentile. `p` is from 0 to 100.
pub fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let rank = (p / 100.0 * sorted.len() as f64).ceil() as usize;
    let index = rank.saturating_sub(1).min(sorted.len() - 1);
    sorted[index]
}
