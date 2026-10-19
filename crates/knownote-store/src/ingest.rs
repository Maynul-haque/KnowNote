//! The write path: getting a document into the store.
//!
//! A document arrives as its canonical text, its blocks and its chunks, and
//! leaves as rows in both tiers plus a vector per chunk. The order matters:
//! the chunk index goes in the fast tier and the text in the bulk tier, and
//! the two are written separately because they are separate databases.
//!
//! A crash between the two writes leaves a chunk that is indexed but has no
//! text, and the full-text index can be rebuilt from the chunk text — that is
//! the repair path, and it is why the text is the copy that is kept.

use knownote_core::blocks::IdentifiedBlockDraft;
use knownote_core::structure::DocumentStructure;
use knownote_core::ChunkResult;

use crate::schema;
use crate::store::Store;
use crate::{Result, StoreError};

/// Everything needed to write one document.
pub struct DocumentInput<'a> {
    pub document_id: &'a str,
    pub notebook_id: &'a str,
    pub title: &'a str,
    /// The canonical text. Every offset refers to this string.
    pub content: &'a str,
    pub structure: Option<&'a DocumentStructure>,
    /// The document's blocks, with their ids.
    pub blocks: &'a [IdentifiedBlockDraft],
    /// The chunks, with the block spans they cover.
    pub chunks: &'a [ChunkResult],
    /// One vector per chunk, in the same order.
    pub vectors: &'a [Vec<f32>],
}

/// Write a document, its blocks, its chunks and its vectors.
///
/// Returns the number of chunks written.
pub fn persist_document(store: &Store, input: &DocumentInput) -> Result<usize> {
    if input.chunks.len() != input.vectors.len() {
        return Err(StoreError::Invalid(format!(
            "document {} has {} chunks but {} vectors",
            input.document_id,
            input.chunks.len(),
            input.vectors.len()
        )));
    }

    let now = now_seconds();

    // The document's own row, in the fast tier.
    store
        .app()
        .execute(
            "INSERT OR REPLACE INTO documents
               (id, notebook_id, title, type, content_hash, status, source_state,
                chunk_count, created_at, updated_at)
             VALUES (?1, ?2, ?3, 'file', NULL, 'indexed', 'available', ?4, ?5, ?5)",
            rusqlite::params![
                input.document_id,
                input.notebook_id,
                input.title,
                input.chunks.len() as i64,
                now
            ],
        )
        .map_err(|error| StoreError::Invalid(format!("cannot write the document: {error}")))?;

    // The extracted text, in the bulk tier.
    let structure = input
        .structure
        .map(|structure| serde_json::to_string(structure).unwrap_or_default());
    store
        .content()
        .execute(
            "INSERT OR REPLACE INTO document_content (document_id, content, structure)
             VALUES (?1, ?2, ?3)",
            rusqlite::params![input.document_id, input.content, structure],
        )
        .map_err(|error| StoreError::Invalid(format!("cannot write the document text: {error}")))?;

    // The blocks, in the bulk tier.
    for block in input.blocks {
        let bbox = block
            .draft
            .bbox
            .map(|bbox| serde_json::to_string(&bbox).unwrap_or_default());
        let metadata = block
            .draft
            .metadata
            .as_ref()
            .map(|metadata| serde_json::to_string(metadata).unwrap_or_default());

        store
            .content()
            .execute(
                "INSERT OR REPLACE INTO document_blocks
                   (id, document_id, kind, \"order\", page, level, text, start_offset,
                    end_offset, bbox, metadata)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                rusqlite::params![
                    block.id,
                    input.document_id,
                    block.draft.kind.as_str(),
                    block.draft.order as i64,
                    block.draft.page,
                    block.draft.level,
                    block.draft.text,
                    block.draft.start_offset as i64,
                    block.draft.end_offset as i64,
                    bbox,
                    metadata
                ],
            )
            .map_err(|error| StoreError::Invalid(format!("cannot write a block: {error}")))?;
    }

    // The chunks: the index in the fast tier, the text in the bulk one.
    for (index, chunk) in input.chunks.iter().enumerate() {
        let chunk_id = format!("{}_{}", input.document_id, index);

        let sequence: i64 = store
            .app()
            .query_row(
                "INSERT INTO chunk_ref
                   (chunk_id, document_id, notebook_id, chunk_index, start_offset,
                    end_offset, page_start, page_end, token_count, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
                 RETURNING seq",
                rusqlite::params![
                    chunk_id,
                    input.document_id,
                    input.notebook_id,
                    chunk.index as i64,
                    chunk.start_offset as i64,
                    chunk.end_offset as i64,
                    chunk.page_start,
                    chunk.page_end,
                    chunk.token_count as i64,
                    now
                ],
                |row| row.get::<_, i64>(0),
            )
            .map_err(|error| StoreError::Invalid(format!("cannot write the chunk index: {error}")))?;

        store
            .content()
            .execute(
                "INSERT OR REPLACE INTO chunk_text (chunk_id, content) VALUES (?1, ?2)",
                rusqlite::params![chunk_id, chunk.content],
            )
            .map_err(|error| StoreError::Invalid(format!("cannot write the chunk text: {error}")))?;

        for span in &chunk.block_spans {
            store
                .content()
                .execute(
                    "INSERT OR REPLACE INTO chunk_blocks
                       (chunk_id, block_id, start_in_block, end_in_block)
                     VALUES (?1, ?2, ?3, ?4)",
                    rusqlite::params![
                        chunk_id,
                        span.block_id,
                        span.start_in_block as i64,
                        span.end_in_block as i64
                    ],
                )
                .map_err(|error| {
                    StoreError::Invalid(format!("cannot write a chunk block span: {error}"))
                })?;
        }

        // The vector, in the fast tier.
        let vector = &input.vectors[index];
        let bytes: Vec<u8> = vector.iter().flat_map(|value| value.to_le_bytes()).collect();
        let table = schema::vector_table_name(input.notebook_id);
        if !schema::is_safe_identifier(&table) {
            return Err(StoreError::Invalid(format!(
                "refusing to use {table:?} as a table name"
            )));
        }
        store
            .app()
            .execute(
                &format!(
                    "INSERT OR REPLACE INTO {table} (embedding_id, chunk_id, embedding)
                     VALUES (?1, ?2, ?3)"
                ),
                rusqlite::params![chunk_id, chunk_id, bytes],
            )
            .map_err(|error| StoreError::Invalid(format!("cannot write a vector: {error}")))?;

        // The full-text row, in the fast tier. The index is contentless, so
        // the text is the only thing it holds.
        store
            .app()
            .execute(
                "INSERT INTO chunks_fts (rowid, content) VALUES (?1, ?2)",
                rusqlite::params![sequence, chunk.content],
            )
            .map_err(|error| {
                StoreError::Invalid(format!("cannot write the full-text row: {error}"))
            })?;
    }

    Ok(input.chunks.len())
}

/// Create a notebook's vector table if it does not exist.
///
/// The width is fixed when the table is created, so a notebook whose vectors
/// would change width needs an explicit rebuild rather than a silent
/// replacement.
pub fn ensure_vector_table(
    store: &Store,
    notebook_id: &str,
    dimensions: usize,
) -> Result<()> {
    let sql = schema::create_vector_table_sql(notebook_id, dimensions)
        .map_err(StoreError::Invalid)?;

    store.app().execute_batch(&sql).map_err(|error| {
        StoreError::Invalid(format!("cannot create the vector table: {error}"))
    })?;

    let now = now_seconds();
    store
        .app()
        .execute(
            "INSERT OR REPLACE INTO vec_metadata
               (notebook_id, table_name, dimensions, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?4)",
            rusqlite::params![
                notebook_id,
                schema::vector_table_name(notebook_id),
                dimensions as i64,
                now
            ],
        )
        .map_err(|error| {
            StoreError::Invalid(format!("cannot record the vector table: {error}"))
        })?;

    Ok(())
}

/// The current time in seconds since the epoch.
fn now_seconds() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_secs() as i64)
        .unwrap_or(0)
}
