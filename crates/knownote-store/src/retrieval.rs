//! The retrievers, over the tiered store.
//!
//! Three strategies, one path. The difference is only how the candidates are
//! produced:
//!
//! * `dense` — passages that mean the same thing.
//! * `sparse` — passages that use the same words.
//! * `hybrid` — both, fused.
//!
//! # The tier split, in practice
//!
//! Every step of a dense query touches only the fast tier. The query is
//! embedded, the vector is matched against the notebook's `vec0` table, and
//! the source filter is a subquery over the chunk index — all in the fast
//! database. The bulk tier is read exactly once, at the end, and only for the
//! handful of chunks that survived: their text, their blocks, their spans.
//!
//! A sparse query is the same story: the inverted index is on the fast drive
//! and the text is fetched afterwards.

use knownote_core::retrieval::{
    build_fts_match_query, dense_channel_threshold, effective_candidate_k, rrf_fuse,
    CandidateHit, EvidenceBlock, EvidenceLocator, EvidenceSource, RetrievedEvidence,
    RetrievalRequest, RetrievalResult, RetrievalStrategy, RetrievalTrace, DEFAULT_TOP_K,
};
use knownote_embedding::EmbeddingBackend;

use crate::schema;
use crate::store::Store;
use crate::{Result, StoreError};

/// Runs a retrieval and returns the evidence and the trace.
pub async fn search(
    store: &Store,
    embedder: &dyn EmbeddingBackend,
    request: &RetrievalRequest,
) -> Result<RetrievalResult> {
    let strategy = request.strategy.unwrap_or_default();
    let candidate_k = effective_candidate_k(request.candidate_k, request.top_k);
    let top_k = request.top_k.unwrap_or(DEFAULT_TOP_K);
    let started = std::time::Instant::now();

    // The value passed to the dense channel and the value written into the
    // trace come from the same variable, so they cannot come apart.
    let dense_threshold = dense_channel_threshold(strategy, request.threshold);

    let hits: Vec<CandidateHit> = match strategy {
        RetrievalStrategy::Sparse => sparse_candidates(store, request, candidate_k)
            .into_iter()
            .take(top_k)
            .collect(),
        RetrievalStrategy::Hybrid => {
            let dense =
                dense_candidates(store, embedder, request, candidate_k, dense_threshold).await?;
            let sparse = sparse_candidates(store, request, candidate_k);
            let channels = [&dense[..], &sparse[..]];
            rrf_fuse(&channels, knownote_core::retrieval::RRF_K)
                .into_iter()
                .take(top_k)
                .map(|hit| CandidateHit {
                    chunk_id: hit.chunk_id,
                    score: hit.score,
                })
                .collect()
        }
        RetrievalStrategy::Dense => {
            dense_candidates(store, embedder, request, candidate_k, dense_threshold)
                .await?
                .into_iter()
                .take(top_k)
                .collect()
        }
    };

    let evidence = if hits.is_empty() {
        Vec::new()
    } else {
        hydrate_evidence(store, &hits)?
    };

    Ok(RetrievalResult {
        evidence,
        trace: RetrievalTrace {
            strategy: strategy.as_str().to_string(),
            scope: request
                .filter
                .as_ref()
                .map(|filter| filter.document_ids.clone())
                .unwrap_or_default(),
            candidate_k,
            top_k,
            dense_threshold,
            duration_ms: started.elapsed().as_secs_f64() * 1000.0,
        },
    })
}

/// The dense channel: embed the question, then take the nearest vectors.
///
/// The source filter is applied *before* the nearest-neighbour search, not
/// after taking the top rows. A search limited to one document that took the
/// global top five first would come back with nothing, because the best
/// chunk of the wanted document might have been sixth. SQLite treats the
/// subquery as a constraint on the `vec0` table, so the candidate set is
/// narrowed before the neighbours are computed.
async fn dense_candidates(
    store: &Store,
    embedder: &dyn EmbeddingBackend,
    request: &RetrievalRequest,
    candidate_k: usize,
    threshold: Option<f64>,
) -> Result<Vec<CandidateHit>> {
    let query_vector = embedder.embed_query(&request.query).await?;

    let table = schema::vector_table_name(&request.notebook_id);
    if !schema::is_safe_identifier(&table) {
        return Err(StoreError::Invalid(format!(
            "refusing to use {table:?} as a table name"
        )));
    }

    let document_ids = request
        .filter
        .as_ref()
        .map(|filter| filter.document_ids.clone())
        .unwrap_or_default();

    let where_clause = if document_ids.is_empty() {
        " WHERE embedding MATCH ? AND k = ?".to_string()
    } else {
        let placeholders = vec!["?"; document_ids.len()].join(", ");
        format!(
            " WHERE chunk_id IN (SELECT chunk_id FROM chunk_ref WHERE document_id IN ({placeholders})) \
             AND embedding MATCH ? AND k = ?"
        )
    };

    let sql = format!(
        "SELECT embedding_id, chunk_id, distance FROM {table}{where_clause} ORDER BY distance ASC"
    );

    let vector_bytes: Vec<u8> = query_vector
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();

    let mut statement = store.app().prepare(&sql)?;
    let mut rows = if document_ids.is_empty() {
        statement.query(rusqlite::params![vector_bytes, candidate_k as i64])?
    } else {
        // The document ids are bound first: they belong to the subquery, which
        // appears before the match in the statement.
        let mut parameters: Vec<Box<dyn rusqlite::ToSql>> = document_ids
            .iter()
            .map(|id| Box::new(id.clone()) as Box<dyn rusqlite::ToSql>)
            .collect();
        parameters.push(Box::new(vector_bytes));
        parameters.push(Box::new(candidate_k as i64));
        statement.query(rusqlite::params_from_iter(parameters))?
    };

    let mut hits = Vec::new();
    while let Some(row) = rows.next()? {
        let chunk_id: String = row.get(1)?;
        let distance: f64 = row.get(2)?;

        // `score = 1 - distance / 2 = (1 + cosine) / 2`.
        //
        // This is *not* the cosine. It is an affine map: a score of 0.5 is a
        // cosine of 0, and only a score of 1 is a cosine of 1. The threshold in
        // the settings is this score, so reading 0.5 as "cosine at least 0.5"
        // overstates the floor considerably.
        let score = 1.0 - distance / 2.0;

        if let Some(threshold) = threshold {
            if score < threshold {
                continue;
            }
        }

        hits.push(CandidateHit { chunk_id, score });
    }

    Ok(hits)
}

/// The sparse channel: BM25 over the full-text index.
///
/// The index lives on the fast drive and holds no text of its own, so this
/// query never reads the bulk tier.
fn sparse_candidates(
    store: &Store,
    request: &RetrievalRequest,
    candidate_k: usize,
) -> Vec<CandidateHit> {
    let Some(match_expression) = build_fts_match_query(&request.query) else {
        return Vec::new();
    };

    let document_ids = request
        .filter
        .as_ref()
        .map(|filter| filter.document_ids.clone())
        .unwrap_or_default();

    let document_clause = if document_ids.is_empty() {
        String::new()
    } else {
        let placeholders = vec!["?"; document_ids.len()].join(", ");
        format!(" AND ref.document_id IN ({placeholders})")
    };

    let sql = format!(
        "SELECT ref.chunk_id, bm25(chunks_fts) AS score \
         FROM chunks_fts \
         JOIN chunk_ref AS ref ON ref.seq = chunks_fts.rowid \
         WHERE chunks_fts MATCH ? AND ref.notebook_id = ?{document_clause} \
         ORDER BY score ASC LIMIT ?"
    );

    let mut statement = match store.app().prepare(&sql) {
        Ok(statement) => statement,
        Err(error) => {
            tracing::warn!("the full-text search could not be prepared: {error}");
            return Vec::new();
        }
    };

    let mut parameters: Vec<Box<dyn rusqlite::ToSql>> = vec![
        Box::new(match_expression),
        Box::new(request.notebook_id.clone()),
    ];
    parameters.extend(
        document_ids
            .iter()
            .map(|id| Box::new(id.clone()) as Box<dyn rusqlite::ToSql>),
    );
    parameters.push(Box::new(candidate_k as i64));

    let mut rows = match statement.query(rusqlite::params_from_iter(parameters)) {
        Ok(rows) => rows,
        Err(error) => {
            tracing::warn!("the full-text search failed: {error}");
            return Vec::new();
        }
    };

    let mut hits = Vec::new();
    loop {
        match rows.next() {
            Ok(Some(row)) => {
                let chunk_id: String = row.get(0).unwrap_or_default();
                // BM25 returns a negative number where a more negative value is
                // a better match, so the sign is flipped to make bigger mean
                // more relevant.
                let score: f64 = row.get(1).unwrap_or(0.0);
                hits.push(CandidateHit {
                    chunk_id,
                    score: -score,
                });
            }
            Ok(None) => break,
            Err(error) => {
                tracing::warn!("the full-text search failed part way through: {error}");
                break;
            }
        }
    }

    hits
}

/// Fill in the evidence for a list of candidates.
///
/// This is the only step of a retrieval that reads the bulk tier, and it reads
/// it for the handful of chunks that survived rather than for the whole
/// candidate pool. The lookups are batched: a constant number of queries
/// regardless of how many candidates there are, because a per-chunk query on
/// the hot path is what makes a retrieval slow.
fn hydrate_evidence(store: &Store, hits: &[CandidateHit]) -> Result<Vec<RetrievedEvidence>> {
    if hits.is_empty() {
        return Ok(Vec::new());
    }

    let chunk_ids: Vec<String> = hits.iter().map(|hit| hit.chunk_id.clone()).collect();

    // The chunk's own metadata, from the fast tier.
    let placeholders = vec!["?"; chunk_ids.len()].join(", ");
    let sql = format!(
        "SELECT chunk_id, document_id, chunk_index, start_offset, end_offset, page_start, page_end \
         FROM chunk_ref WHERE chunk_id IN ({placeholders})"
    );

    let mut metadata = std::collections::HashMap::new();
    {
        let mut statement = store.app().prepare(&sql)?;
        let mut rows = statement.query(rusqlite::params_from_iter(
            chunk_ids.iter().map(|id| id.clone()),
        ))?;
        while let Some(row) = rows.next()? {
            let chunk_id: String = row.get(0)?;
            metadata.insert(
                chunk_id,
                ChunkMetadata {
                    document_id: row.get(1)?,
                    chunk_index: row.get::<_, i64>(2)? as usize,
                    page_start: row.get(5)?,
                    page_end: row.get(6)?,
                },
            );
        }
    }

    // The chunk's text, from the bulk tier.
    let mut texts = std::collections::HashMap::new();
    {
        let sql = format!("SELECT chunk_id, content FROM chunk_text WHERE chunk_id IN ({placeholders})");
        let mut statement = store.content().prepare(&sql)?;
        let mut rows = statement.query(rusqlite::params_from_iter(
            chunk_ids.iter().map(|id| id.clone()),
        ))?;
        while let Some(row) = rows.next()? {
            texts.insert(row.get::<_, String>(0)?, row.get::<_, String>(1)?);
        }
    }

    // The blocks the chunks cover, and the blocks themselves.
    let (spans, blocks) = load_provenance(store, &chunk_ids)?;

    // The source titles.
    let document_ids: Vec<String> = metadata
        .values()
        .map(|entry| entry.document_id.clone())
        .collect();
    let mut documents = std::collections::HashMap::new();
    if !document_ids.is_empty() {
        let placeholders = vec!["?"; document_ids.len()].join(", ");
        let sql = format!("SELECT id, title, type FROM documents WHERE id IN ({placeholders})");
        let mut statement = store.app().prepare(&sql)?;
        let mut rows = statement.query(rusqlite::params_from_iter(
            document_ids.iter().map(|id| id.clone()),
        ))?;
        while let Some(row) = rows.next()? {
            documents.insert(
                row.get::<_, String>(0)?,
                EvidenceSource {
                    title: row.get(1)?,
                    kind: row.get(2)?,
                },
            );
        }
    }

    let mut evidence = Vec::with_capacity(hits.len());
    for hit in hits {
        let Some(entry) = metadata.get(&hit.chunk_id) else {
            // A chunk that was deleted between the search and the hydration is
            // dropped rather than returned without provenance.
            continue;
        };

        let content = texts.get(&hit.chunk_id).cloned().unwrap_or_default();
        let source = documents.get(&entry.document_id).cloned().unwrap_or(EvidenceSource {
            title: String::new(),
            kind: String::new(),
        });

        let mut locator_blocks: Vec<EvidenceBlock> = Vec::new();
        for span in spans.get(&hit.chunk_id).into_iter().flatten() {
            if let Some(block) = blocks.get(&span.block_id) {
                locator_blocks.push(EvidenceBlock {
                    block_id: block.id.clone(),
                    kind: block.kind.clone(),
                    level: block.level,
                    page: block.page,
                    bbox: block.bbox.clone(),
                    text: block.text.clone(),
                    start_offset: block.start_offset,
                    end_offset: block.end_offset,
                    order: block.order,
                    start_in_block: span.start_in_block,
                    end_in_block: span.end_in_block,
                });
            }
        }

        evidence.push(RetrievedEvidence {
            chunk_id: hit.chunk_id.clone(),
            document_id: entry.document_id.clone(),
            content,
            score: hit.score,
            chunk_index: entry.chunk_index,
            source,
            locator: EvidenceLocator {
                page_start: entry.page_start,
                page_end: entry.page_end,
                blocks: locator_blocks,
            },
            metadata: None,
        });
    }

    Ok(evidence)
}

/// The chunk's metadata, from the fast tier.
struct ChunkMetadata {
    document_id: String,
    chunk_index: usize,
    page_start: Option<i64>,
    page_end: Option<i64>,
}

/// A block, from the bulk tier.
struct BlockRow {
    id: String,
    kind: String,
    level: Option<i64>,
    page: Option<i64>,
    bbox: Option<serde_json::Value>,
    text: String,
    start_offset: usize,
    end_offset: usize,
    order: usize,
}

/// A chunk's range inside a block.
struct SpanRow {
    block_id: String,
    start_in_block: usize,
    end_in_block: usize,
}

/// Load the block spans and the blocks the given chunks cover.
fn load_provenance(
    store: &Store,
    chunk_ids: &[String],
) -> Result<(
    std::collections::HashMap<String, Vec<SpanRow>>,
    std::collections::HashMap<String, BlockRow>,
)> {
    let placeholders = vec!["?"; chunk_ids.len()].join(", ");
    let sql = format!(
        "SELECT chunk_id, block_id, start_in_block, end_in_block \
         FROM chunk_blocks WHERE chunk_id IN ({placeholders})"
    );

    let mut spans: std::collections::HashMap<String, Vec<SpanRow>> = Default::default();
    let mut block_ids: Vec<String> = Vec::new();

    {
        let mut statement = store.content().prepare(&sql)?;
        let mut rows = statement.query(rusqlite::params_from_iter(
            chunk_ids.iter().map(|id| id.clone()),
        ))?;
        while let Some(row) = rows.next()? {
            let chunk_id: String = row.get(0)?;
            let block_id: String = row.get(1)?;
            block_ids.push(block_id.clone());
            spans.entry(chunk_id).or_default().push(SpanRow {
                block_id,
                start_in_block: row.get::<_, i64>(2)? as usize,
                end_in_block: row.get::<_, i64>(3)? as usize,
            });
        }
    }

    let mut blocks = std::collections::HashMap::new();
    if !block_ids.is_empty() {
        block_ids.sort();
        block_ids.dedup();

        let placeholders = vec!["?"; block_ids.len()].join(", ");
        let sql = format!(
            "SELECT id, kind, level, page, bbox, text, start_offset, end_offset, \"order\" \
             FROM document_blocks WHERE id IN ({placeholders})"
        );
        let mut statement = store.content().prepare(&sql)?;
        let mut rows = statement.query(rusqlite::params_from_iter(
            block_ids.iter().map(|id| id.clone()),
        ))?;
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            blocks.insert(
                id.clone(),
                BlockRow {
                    id,
                    kind: row.get(1)?,
                    level: row.get(2)?,
                    page: row.get(3)?,
                    bbox: row
                        .get::<_, Option<String>>(4)?
                        .and_then(|raw| serde_json::from_str(&raw).ok()),
                    text: row.get(5)?,
                    start_offset: row.get::<_, i64>(6)? as usize,
                    end_offset: row.get::<_, i64>(7)? as usize,
                    order: row.get::<_, i64>(8)? as usize,
                },
            );
        }
    }

    Ok((spans, blocks))
}
