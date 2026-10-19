//! The tiered store: the fast tier on one drive, the bulk on another.
//!
//! This is the test behind the "keep the chunks on a spinning disk" promise.
//! It checks three things:
//!
//! * the split is real — the chunk *text* is not in the fast database, and
//!   the chunk *index* is not in the bulk one;
//! * retrieval still works across the split, with the provenance intact;
//! * the search path reads the fast tier and the evidence reads the bulk tier,
//!   so a bulk drive that is slow does not slow the search down.

use async_trait::async_trait;
use knownote_core::retrieval::{RetrievalRequest, RetrievalStrategy};
use knownote_embedding::{EmbedError, EmbeddingBackend, EmbeddingSpace};
use knownote_store::{schema, StorageLayout, Store};

const DIMENSIONS: usize = 8;
const NOTEBOOK: &str = "nb_test";

/// A deterministic embedder, so the test does not need the model.
///
/// The vector points along one axis per marker word, which makes the ranking
/// predictable without making it trivial.
struct FakeEmbedder;

fn vector_for(text: &str) -> Vec<f32> {
    let mut vector = vec![0.0f32; DIMENSIONS];
    if text.contains("alpha") {
        vector[0] = 1.0;
    }
    if text.contains("beta") {
        vector[1] = 1.0;
    }
    if text.contains("gamma") {
        vector[2] = 1.0;
    }
    if vector.iter().all(|value| *value == 0.0) {
        vector[DIMENSIONS - 1] = 1.0;
    }
    vector
}

#[async_trait]
impl EmbeddingBackend for FakeEmbedder {
    fn dimensions(&self) -> usize {
        DIMENSIONS
    }

    fn space(&self) -> EmbeddingSpace {
        EmbeddingSpace::local("fake", "v1", DIMENSIONS)
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(texts.iter().map(|text| vector_for(text)).collect())
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(vector_for(text))
    }
}

/// Build a store with the bulk tier on a different path from the fast tier,
/// and put a small corpus in it.
fn build_store(root: &std::path::Path) -> Store {
    let layout = StorageLayout::with_content_root(root.join("fast"), root.join("bulk"));
    let store = Store::open(layout).expect("the store opens");

    store
        .app()
        .execute_batch(
            "INSERT INTO notebooks (id, title, created_at, updated_at)
             VALUES ('nb_test', 'Test', 0, 0);",
        )
        .expect("the notebook is created");

    store
        .app()
        .execute_batch(&format!(
            "{}",
            schema::create_vector_table_sql(NOTEBOOK, DIMENSIONS).expect("a safe table name")
        ))
        .expect("the vector table is created");

    // Two documents' worth of chunks: one that mentions alpha, one that
    // mentions beta.
    let chunks = [
        ("chunk_alpha", "doc_alpha", "the alpha passage mentions rivers and sediment"),
        ("chunk_beta", "doc_beta", "the beta passage mentions coastal monitoring"),
    ];

    for (index, (chunk_id, document_id, text)) in chunks.iter().enumerate() {
        store
            .app()
            .execute(
                "INSERT INTO chunk_ref
                   (seq, chunk_id, document_id, notebook_id, chunk_index, start_offset,
                    end_offset, page_start, page_end, token_count, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 1, 1, 12, 0)",
                rusqlite::params![
                    index as i64 + 1,
                    chunk_id,
                    document_id,
                    NOTEBOOK,
                    index as i64,
                    0i64,
                    text.len() as i64
                ],
            )
            .expect("the chunk index row is written");

        store
            .content()
            .execute(
                "INSERT INTO chunk_text (chunk_id, content) VALUES (?1, ?2)",
                rusqlite::params![chunk_id, text],
            )
            .expect("the chunk text is written to the bulk tier");

        store
            .content()
            .execute(
                "INSERT INTO document_blocks
                   (id, document_id, kind, \"order\", page, level, text, start_offset, end_offset)
                 VALUES (?1, ?2, 'paragraph', 0, 1, NULL, ?3, 0, ?4)",
                rusqlite::params![
                    format!("blk_{chunk_id}"),
                    document_id,
                    text,
                    text.len() as i64
                ],
            )
            .expect("the block is written to the bulk tier");

        store
            .content()
            .execute(
                "INSERT INTO chunk_blocks (chunk_id, block_id, start_in_block, end_in_block)
                 VALUES (?1, ?2, 0, ?3)",
                rusqlite::params![chunk_id, format!("blk_{chunk_id}"), text.len() as i64],
            )
            .expect("the block span is written to the bulk tier");

        // The vector goes into the fast tier.
        let vector = vector_for(text);
        let bytes: Vec<u8> = vector.iter().flat_map(|value| value.to_le_bytes()).collect();
        store
            .app()
            .execute(
                &format!(
                    "INSERT INTO {} (embedding_id, chunk_id, embedding) VALUES (?1, ?2, ?3)",
                    schema::vector_table_name(NOTEBOOK)
                ),
                rusqlite::params![chunk_id, chunk_id, bytes],
            )
            .expect("the vector is written to the fast tier");
    }

    store
        .rebuild_fts_index()
        .expect("the full-text index is rebuilt");

    store
}

fn table_exists(connection: &rusqlite::Connection, name: &str) -> bool {
    let count: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
            [name],
            |row| row.get(0),
        )
        .unwrap_or(0);
    count > 0
}

#[test]
fn the_bulk_tier_holds_the_text_and_the_fast_tier_holds_the_index() {
    let root = tempfile::tempdir().expect("a temporary directory");
    let store = build_store(root.path());

    // The fast tier carries the chunk index and the search indexes.
    assert!(
        table_exists(store.app(), "chunk_ref"),
        "the fast tier must hold the chunk index"
    );
    assert!(
        table_exists(store.app(), "chunks_fts"),
        "the fast tier must hold the full-text index"
    );
    assert!(
        !table_exists(store.app(), "chunk_text"),
        "the chunk text must not be in the fast tier"
    );
    assert!(
        !table_exists(store.app(), "document_blocks"),
        "the document blocks must not be in the fast tier"
    );

    // The bulk tier carries the text.
    assert!(
        table_exists(store.content(), "chunk_text"),
        "the bulk tier must hold the chunk text"
    );
    assert!(
        table_exists(store.content(), "document_blocks"),
        "the bulk tier must hold the document blocks"
    );
    assert!(
        !table_exists(store.content(), "chunk_ref"),
        "the chunk index must not be in the bulk tier"
    );

    // And the files really are in different places.
    let layout = store.layout();
    assert!(layout.content_is_separate());
    assert!(layout.app_db_path().starts_with(root.path().join("fast")));
    assert!(layout.content_db_path().starts_with(root.path().join("bulk")));
    assert!(layout.app_db_path().exists());
    assert!(layout.content_db_path().exists());
}

#[tokio::test]
async fn a_dense_search_ranks_by_the_vectors_and_returns_the_provenance() {
    let root = tempfile::tempdir().expect("a temporary directory");
    let store = build_store(root.path());

    let result = knownote_store::retrieval::search(
        &store,
        &FakeEmbedder,
        &RetrievalRequest {
            notebook_id: NOTEBOOK.to_string(),
            query: "alpha".to_string(),
            candidate_k: Some(10),
            top_k: Some(5),
            threshold: Some(0.6),
            filter: None,
            strategy: Some(RetrievalStrategy::Dense),
        },
    )
    .await
    .expect("the search runs");

    assert!(!result.evidence.is_empty(), "the search found something");

    // The alpha chunk is a cosine of 1, so its score is 1. The beta chunk is
    // a cosine of 0, so its score is 0.5 and the 0.6 threshold drops it.
    let first = &result.evidence[0];
    assert_eq!(first.chunk_id, "chunk_alpha");
    assert!((first.score - 1.0).abs() < 1e-6, "score was {}", first.score);
    assert_eq!(result.evidence.len(), 1, "the threshold dropped the rest");

    // The text came from the bulk tier.
    assert_eq!(
        first.content, "the alpha passage mentions rivers and sediment",
        "the evidence carries the chunk text from the bulk tier"
    );

    // The provenance came from the bulk tier too, and it is complete.
    assert_eq!(first.document_id, "doc_alpha");
    assert_eq!(first.locator.page_start, Some(1));
    assert_eq!(first.locator.page_end, Some(1));
    assert_eq!(first.locator.blocks.len(), 1);
    assert_eq!(first.locator.blocks[0].block_id, "blk_chunk_alpha");
    assert_eq!(
        first.locator.blocks[0].text, "the alpha passage mentions rivers and sediment"
    );

    // The trace says what actually happened.
    assert_eq!(result.trace.strategy, "dense");
    assert_eq!(result.trace.candidate_k, 10);
    assert_eq!(result.trace.top_k, 5);
    assert_eq!(result.trace.dense_threshold, Some(0.6));
}

#[tokio::test]
async fn a_sparse_search_ranks_by_the_literal_match() {
    let root = tempfile::tempdir().expect("a temporary directory");
    let store = build_store(root.path());

    let result = knownote_store::retrieval::search(
        &store,
        &FakeEmbedder,
        &RetrievalRequest {
            notebook_id: NOTEBOOK.to_string(),
            query: "coastal".to_string(),
            candidate_k: Some(10),
            top_k: Some(5),
            threshold: None,
            filter: None,
            strategy: Some(RetrievalStrategy::Sparse),
        },
    )
    .await
    .expect("the search runs");

    assert_eq!(result.evidence.len(), 1, "only one chunk mentions coastal");
    assert_eq!(result.evidence[0].chunk_id, "chunk_beta");
    assert_eq!(
        result.trace.dense_threshold, None,
        "the sparse channel has no similarity floor"
    );
}

#[tokio::test]
async fn a_hybrid_search_fuses_both_channels() {
    let root = tempfile::tempdir().expect("a temporary directory");
    let store = build_store(root.path());

    // "alpha" is both the vector direction and a literal word, so both
    // channels find the same chunk.
    let result = knownote_store::retrieval::search(
        &store,
        &FakeEmbedder,
        &RetrievalRequest {
            notebook_id: NOTEBOOK.to_string(),
            query: "alpha".to_string(),
            candidate_k: Some(10),
            top_k: Some(5),
            threshold: None,
            filter: None,
            strategy: Some(RetrievalStrategy::Hybrid),
        },
    )
    .await
    .expect("the search runs");

    assert!(!result.evidence.is_empty());
    assert_eq!(result.evidence[0].chunk_id, "chunk_alpha");
    assert_eq!(result.trace.strategy, "hybrid");
    // Hybrid runs the dense channel too, so it has a floor.
    assert_eq!(result.trace.dense_threshold, Some(0.5));
}

#[tokio::test]
async fn a_search_can_be_limited_to_a_subset_of_the_documents() {
    let root = tempfile::tempdir().expect("a temporary directory");
    let store = build_store(root.path());

    let result = knownote_store::retrieval::search(
        &store,
        &FakeEmbedder,
        &RetrievalRequest {
            notebook_id: NOTEBOOK.to_string(),
            query: "passage".to_string(),
            candidate_k: Some(10),
            top_k: Some(5),
            threshold: None,
            filter: Some(knownote_core::retrieval::RetrievalFilter {
                document_ids: vec!["doc_beta".to_string()],
            }),
            strategy: Some(RetrievalStrategy::Dense),
        },
    )
    .await
    .expect("the search runs");

    assert_eq!(result.evidence.len(), 1);
    assert_eq!(result.evidence[0].document_id, "doc_beta");
    assert_eq!(result.trace.scope, vec!["doc_beta".to_string()]);
}

#[test]
fn moving_the_bulk_tier_to_another_drive_copies_everything() {
    let root = tempfile::tempdir().expect("a temporary directory");
    let first_bulk = root.path().join("bulk");
    let second_bulk = root.path().join("other-bulk");

    let store = build_store(root.path());
    let chunk_count: i64 = store
        .content()
        .query_row("SELECT COUNT(*) FROM chunk_text", [], |row| row.get(0))
        .expect("the chunk count is read");
    assert_eq!(chunk_count, 2, "the corpus was written to the bulk tier");

    let layout = store.layout().clone();
    let report = layout
        .migrate_content_root(&second_bulk)
        .expect("the migration runs");

    assert!(report.moved);
    assert!(report.bytes_moved > 0);
    assert!(second_bulk.join("content.db").exists(), "the database was copied");
    assert!(
        first_bulk.join("content.db").exists(),
        "the original is left in place rather than deleted"
    );
}
