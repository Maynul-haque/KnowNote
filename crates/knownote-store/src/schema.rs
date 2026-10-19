//! The SQL that defines each tier of the store.
//!
//! The schema is split by access pattern rather than by subject, which is what
//! makes it possible to put the bulk of the data on a spinning disk without
//! putting the hot path there.
//!
//! | Database | Where | Holds |
//! |---|---|---|
//! | `knownote.db` | fast drive | the workspace, the chunk index, the full-text index and the vector index |
//! | `content.db` | configurable drive | the bulk: chunk text, document blocks, extracted text |
//!
//! The chunk *index* is the piece that makes the split work. It carries
//! everything the retrieval path needs to rank and filter — the document a
//! chunk belongs to, its offsets, its page range, its token count — but not
//! its text. So a dense query can narrow the candidate set and score it
//! entirely on the fast drive, and the text of the handful of chunks that
//! survive is the only thing read from the slow one.
//!
//! The indexes live in the same database as the chunk index on purpose. The
//! vector search filters by source *before* the nearest-neighbour search, and
//! that filter is a subquery over the chunk index; the full-text search
//! filters by notebook and document the same way. A subquery cannot cross a
//! database boundary, so splitting the indexes into their own files would
//! force the hot path to read from the slow one.

/// The workspace schema: the fast tier.
pub const APP_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS notebooks (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  description TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notebooks_updated ON notebooks(updated_at);

CREATE TABLE IF NOT EXISTS chat_sessions (
  id TEXT PRIMARY KEY,
  notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  title_is_auto INTEGER NOT NULL DEFAULT 0,
  summary TEXT,
  total_tokens INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'active',
  parent_session_id TEXT REFERENCES chat_sessions(id) ON DELETE SET NULL,
  retrieval_scope TEXT,
  last_opened_at INTEGER,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_notebook ON chat_sessions(notebook_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_sessions_notebook_opened ON chat_sessions(notebook_id, last_opened_at);

CREATE TABLE IF NOT EXISTS chat_messages (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  reasoning_content TEXT,
  metadata TEXT,
  status TEXT,
  finish_reason TEXT,
  error TEXT,
  usage TEXT,
  finished_at INTEGER,
  attempt_of TEXT,
  created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id, created_at);

CREATE TABLE IF NOT EXISTS notes (
  id TEXT PRIMARY KEY,
  notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_notebook ON notes(notebook_id, updated_at);

-- The library snapshot: a source exists once in the library, and a notebook
-- mounts it. The text lives in the bulk tier; this row is the identity.
CREATE TABLE IF NOT EXISTS library_sources (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  type TEXT NOT NULL,
  source_uri TEXT,
  local_file_path TEXT,
  content_hash TEXT,
  mime_type TEXT,
  file_size INTEGER,
  metadata TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY,
  notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  type TEXT NOT NULL,
  source_uri TEXT,
  local_file_path TEXT,
  source_note_id TEXT REFERENCES notes(id) ON DELETE SET NULL,
  source_id TEXT REFERENCES library_sources(id) ON DELETE SET NULL,
  content_hash TEXT,
  mime_type TEXT,
  file_size INTEGER,
  metadata TEXT,
  status TEXT NOT NULL DEFAULT 'pending',
  source_state TEXT NOT NULL DEFAULT 'available',
  source_mtime_ms INTEGER,
  error_message TEXT,
  chunk_count INTEGER DEFAULT 0,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_notebook ON documents(notebook_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_notebook_source ON documents(notebook_id, source_id);

CREATE TABLE IF NOT EXISTS folder_watches (
  id TEXT PRIMARY KEY,
  notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
  path TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_folder_watches_notebook_path ON folder_watches(notebook_id, path);

CREATE TABLE IF NOT EXISTS ingestion_runs (
  id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  stage TEXT NOT NULL,
  progress INTEGER NOT NULL DEFAULT 0,
  error_message TEXT,
  started_at INTEGER NOT NULL,
  finished_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_ingestion_runs_document ON ingestion_runs(document_id, started_at);

-- The chunk index: everything about a chunk except its text.
--
-- `seq` is the integer the full-text index uses as its rowid, so the two can
-- be joined without carrying the chunk id through the FTS table.
CREATE TABLE IF NOT EXISTS chunk_ref (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  chunk_id TEXT NOT NULL UNIQUE,
  document_id TEXT NOT NULL,
  notebook_id TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  start_offset INTEGER,
  end_offset INTEGER,
  page_start INTEGER,
  page_end INTEGER,
  token_count INTEGER,
  created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunk_ref_document ON chunk_ref(document_id);
CREATE INDEX IF NOT EXISTS idx_chunk_ref_notebook ON chunk_ref(notebook_id);

CREATE TABLE IF NOT EXISTS embeddings (
  id TEXT PRIMARY KEY,
  chunk_id TEXT NOT NULL,
  notebook_id TEXT NOT NULL,
  model TEXT NOT NULL,
  dimensions INTEGER NOT NULL,
  created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk ON embeddings(chunk_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_notebook ON embeddings(notebook_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model);

-- The identity of a notebook's embedding space. Comparing dimensions is not
-- enough: a different model can have the same width, and the old vectors are
-- then incomparable without anything looking wrong.
CREATE TABLE IF NOT EXISTS notebook_embedding_spaces (
  notebook_id TEXT PRIMARY KEY REFERENCES notebooks(id) ON DELETE CASCADE,
  space_id TEXT NOT NULL,
  backend TEXT NOT NULL,
  model TEXT NOT NULL,
  revision TEXT NOT NULL DEFAULT '',
  dimensions INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS mind_maps (
  id TEXT PRIMARY KEY,
  notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 1,
  tree_data TEXT NOT NULL,
  chunk_mapping TEXT NOT NULL,
  metadata TEXT,
  status TEXT NOT NULL DEFAULT 'generating',
  error_message TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mindmaps_notebook ON mind_maps(notebook_id, updated_at);

CREATE TABLE IF NOT EXISTS quizzes (
  id TEXT PRIMARY KEY,
  notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 1,
  questions_data TEXT NOT NULL,
  chunk_mapping TEXT NOT NULL,
  metadata TEXT,
  status TEXT NOT NULL DEFAULT 'generating',
  error_message TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_quizzes_notebook ON quizzes(notebook_id, updated_at);

CREATE TABLE IF NOT EXISTS quiz_sessions (
  id TEXT PRIMARY KEY,
  quiz_id TEXT NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
  notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
  answers TEXT,
  score INTEGER,
  total_questions INTEGER NOT NULL,
  correct_count INTEGER,
  completed_at INTEGER,
  created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_quiz_sessions_quiz ON quiz_sessions(quiz_id, created_at);

CREATE TABLE IF NOT EXISTS anki_cards (
  id TEXT PRIMARY KEY,
  notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 1,
  cards_data TEXT NOT NULL,
  chunk_mapping TEXT NOT NULL,
  metadata TEXT,
  status TEXT NOT NULL DEFAULT 'generating',
  error_message TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ankicards_notebook ON anki_cards(notebook_id, updated_at);

CREATE TABLE IF NOT EXISTS items (
  id TEXT PRIMARY KEY,
  notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
  type TEXT NOT NULL,
  resource_id TEXT NOT NULL,
  "order" INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_items_notebook_order ON items(notebook_id, "order");
CREATE INDEX IF NOT EXISTS idx_items_type ON items(type);
CREATE INDEX IF NOT EXISTS idx_items_resource ON items(type, resource_id);

-- The full-text index.
--
-- The table is *contentless*: it stores the inverted index but not the text.
-- The text lives in the bulk tier and is fetched for the handful of rows that
-- survive a search, so the index on the fast drive stays a fraction of the
-- corpus size.
--
-- `columnsize=1` is not optional. A contentless table defaults to storing no
-- column sizes, and BM25 needs the document length to normalise by, so
-- without this the ranking silently degrades.
--
-- The rows are addressed by the chunk index's `seq`, which is why the join
-- below is local rather than a cross-database one.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  content,
  content='',
  columnsize=1
);

-- The vector index.
--
-- One `vec0` table per notebook, each carrying its own width, so vectors
-- from two embedding spaces are never compared. The table name is derived
-- from the notebook id and is validated against a strict identifier
-- whitelist before it reaches SQL.
CREATE TABLE IF NOT EXISTS vec_metadata (
  notebook_id TEXT PRIMARY KEY,
  table_name TEXT NOT NULL,
  dimensions INTEGER NOT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
"#;

/// The bulk schema: the tier that can live on a slow drive.
pub const CONTENT_SCHEMA: &str = r#"
-- The extracted text of a document. This is the canonical string every
-- offset in the store refers to, so it is the one thing that must never be
-- rewritten.
CREATE TABLE IF NOT EXISTS document_content (
  document_id TEXT PRIMARY KEY,
  content TEXT NOT NULL,
  structure TEXT
);

-- The canonical text of a library snapshot, shared by every notebook that
-- mounts it.
CREATE TABLE IF NOT EXISTS library_source_content (
  source_id TEXT PRIMARY KEY,
  content TEXT NOT NULL,
  structure TEXT
);

-- The text of a chunk. The chunk's metadata lives in the fast tier.
CREATE TABLE IF NOT EXISTS chunk_text (
  chunk_id TEXT PRIMARY KEY,
  content TEXT NOT NULL
);

-- A block is where a citation lands: page, heading level, character span and
-- the text itself.
CREATE TABLE IF NOT EXISTS document_blocks (
  id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  "order" INTEGER NOT NULL,
  page INTEGER,
  level INTEGER,
  text TEXT NOT NULL,
  start_offset INTEGER NOT NULL,
  end_offset INTEGER NOT NULL,
  bbox TEXT,
  metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_blocks_document_order ON document_blocks(document_id, "order");
CREATE INDEX IF NOT EXISTS idx_blocks_document_page ON document_blocks(document_id, page);

-- The chunk to block mapping.
--
-- A retrieval result is a chunk id, and a citation has to be able to walk
-- back to a page and a character span, so every chunk records the range of
-- each block it covers. A chunk can cover part of a block, which is why the
-- range is here rather than being implied by the block's own span.
CREATE TABLE IF NOT EXISTS chunk_blocks (
  chunk_id TEXT NOT NULL,
  block_id TEXT NOT NULL,
  start_in_block INTEGER NOT NULL,
  end_in_block INTEGER NOT NULL,
  PRIMARY KEY (chunk_id, block_id)
);
CREATE INDEX IF NOT EXISTS idx_chunk_blocks_block ON chunk_blocks(block_id);
"#;

/// The name of the `vec0` table that holds a notebook's vectors.
pub fn vector_table_name(notebook_id: &str) -> String {
    format!("vec_{notebook_id}")
}

/// Whether a string is safe to use as a SQL identifier.
///
/// A table name reaches SQL as a string, so it is checked rather than
/// escaped. The rule is deliberately narrow — a letter or underscore
/// followed by letters, digits and underscores — because anything wider would
/// let a name that SQLite has to quote reach the statement unquoted.
pub fn is_safe_identifier(identifier: &str) -> bool {
    let mut characters = identifier.chars();
    match characters.next() {
        Some(first) if first.is_ascii_alphabetic() || first == '_' => {}
        _ => return false,
    }
    characters.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The `CREATE VIRTUAL TABLE` statement for a notebook's vector table.
pub fn create_vector_table_sql(notebook_id: &str, dimensions: usize) -> Result<String, String> {
    let table = vector_table_name(notebook_id);
    if !is_safe_identifier(&table) {
        return Err(format!("refusing to use {table:?} as a table name"));
    }
    Ok(format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0(\
         embedding_id TEXT PRIMARY KEY, \
         chunk_id TEXT, \
         embedding FLOAT[{dimensions}] distance_metric=cosine)"
    ))
}

/// The statement that drops a notebook's vector table.
pub fn drop_vector_table_sql(notebook_id: &str) -> Result<String, String> {
    let table = vector_table_name(notebook_id);
    if !is_safe_identifier(&table) {
        return Err(format!("refusing to use {table:?} as a table name"));
    }
    Ok(format!("DROP TABLE IF EXISTS {table}"))
}
