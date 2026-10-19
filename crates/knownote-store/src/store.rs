//! Opening the two databases that make up the store.

use std::path::Path;

use rusqlite::Connection;

use crate::layout::StorageLayout;
use crate::schema;
use crate::{Result, StoreError};

/// The two connections the store holds.
pub struct Store {
    layout: StorageLayout,
    /// The fast tier: the workspace, the chunk index, the full-text index and
    /// the vector index.
    app: Connection,
    /// The bulk tier: chunk text, document blocks, extracted text.
    content: Connection,
}

impl Store {
    /// Open the store, creating whatever does not exist yet.
    pub fn open(layout: StorageLayout) -> Result<Self> {
        layout.ensure_dirs().map_err(|error| {
            StoreError::Io(format!("cannot create the store directories: {error}"))
        })?;

        // The vector extension has to be registered *before* any connection is
        // opened. SQLite's auto-extension runs for every connection that is
        // opened afterwards, so registering it later would leave the already
        // open connection without the `vec0` module.
        register_vector_extension()?;

        let app = open_app(&layout.app_db_path())?;
        let content = open_content(&layout.content_db_path())?;

        Ok(Self {
            layout,
            app,
            content,
        })
    }

    /// The layout this store was opened with.
    pub fn layout(&self) -> &StorageLayout {
        &self.layout
    }

    /// The fast tier.
    pub fn app(&self) -> &Connection {
        &self.app
    }

    /// The bulk tier.
    pub fn content(&self) -> &Connection {
        &self.content
    }

    /// Whether the bulk tier is reachable.
    ///
    /// The content drive can be unplugged. This checks the directory rather
    /// than the database, because a missing directory is the failure a user
    /// can act on, and a locked or corrupt file is not.
    pub fn content_available(&self) -> bool {
        self.layout.content_dir.is_dir()
    }

    /// Rebuild the full-text index from the chunk text.
    ///
    /// The full-text index lives on the fast drive while the text lives on the
    /// bulk drive, so the two are separate databases and cannot be updated in
    /// one transaction. This is the repair path: it attaches the bulk database
    /// for the length of one statement, so the rebuild is a single join
    /// rather than a loop of round trips, and a crash between the two writes
    /// is recoverable.
    pub fn rebuild_fts_index(&self) -> Result<usize> {
        let content_path = self.layout.content_db_path();

        self.app.execute_batch(&format!(
            "ATTACH DATABASE '{}' AS bulk;",
            escape_sql_string(&content_path.to_string_lossy()),
        ))?;

        let outcome = (|| -> Result<usize> {
            let transaction = self.app.unchecked_transaction()?;
            transaction.execute_batch("DELETE FROM chunks_fts")?;
            let inserted = transaction.execute(
                "INSERT INTO chunks_fts(rowid, content)
                 SELECT ref.seq, text.content
                 FROM chunk_ref AS ref
                 JOIN bulk.chunk_text AS text ON text.chunk_id = ref.chunk_id",
                [],
            )?;
            transaction.commit()?;
            Ok(inserted)
        })();

        // Detach even when the rebuild failed: a connection left holding an
        // attached database cannot be reopened cleanly.
        let _ = self.app.execute_batch("DETACH DATABASE bulk;");

        outcome
    }
}

/// Open the fast tier.
fn open_app(path: &Path) -> Result<Connection> {
    let connection = Connection::open(path)?;
    configure(&connection, JournalMode::Wal)?;
    connection.execute_batch(schema::APP_SCHEMA)?;
    Ok(connection)
}

/// Open the bulk tier.
///
/// The journal mode is the important difference: WAL needs a shared-memory
/// file, which a removable or exFAT-formatted drive cannot provide, and the
/// bulk tier is exactly the kind of drive a user points this at.
fn open_content(path: &Path) -> Result<Connection> {
    let connection = Connection::open(path)?;
    configure(&connection, JournalMode::Delete)?;
    connection.execute_batch(schema::CONTENT_SCHEMA)?;
    Ok(connection)
}

/// The journal mode a database uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JournalMode {
    /// Write-ahead logging. Fast, but needs a shared-memory file.
    Wal,
    /// The rollback journal. Slower under concurrency, but works on any
    /// filesystem, including removable and network drives.
    Delete,
}

/// Apply the pragmas every connection wants.
fn configure(connection: &Connection, journal: JournalMode) -> Result<()> {
    let journal = match journal {
        JournalMode::Wal => "WAL",
        JournalMode::Delete => "DELETE",
    };

    connection.execute_batch(&format!(
        "PRAGMA journal_mode = {journal};
         PRAGMA synchronous = NORMAL;
         PRAGMA busy_timeout = 5000;
         PRAGMA foreign_keys = ON;"
    ))?;

    Ok(())
}

/// Register the `vec0` extension with every connection this process opens.
///
/// The extension is compiled into the binary by the `sqlite-vec` crate, so
/// there is no shared library to find and no platform-specific path to get
/// wrong. That is the whole reason the Rust port does not need the
/// `load_extension` dance the TypeScript version had to do for three
/// platforms.
#[allow(unsafe_code)]
fn register_vector_extension() -> Result<()> {
    use std::sync::Once;

    static ONCE: Once = Once::new();
    let mut failure: Option<String> = None;

    ONCE.call_once(|| {
        // SAFETY: `sqlite3_vec_init` is the entry point the sqlite-vec crate
        // exports. It has the signature SQLite's auto-extension callback
        // expects apart from the parameters, which SQLite supplies; the
        // transmute below is the pattern the sqlite-vec crate itself
        // documents. The callback runs inside SQLite's own initialisation,
        // touches no state of ours, and is registered once per process.
        unsafe {
            let raw: rusqlite::auto_extension::RawAutoExtension =
                std::mem::transmute(sqlite_vec::sqlite3_vec_init as *const ());
            if let Err(error) = rusqlite::auto_extension::register_auto_extension(raw) {
                failure = Some(error.to_string());
            }
        }
    });

    match failure {
        Some(error) => Err(StoreError::Invalid(format!(
            "cannot register the sqlite-vec extension: {error}"
        ))),
        None => Ok(()),
    }
}

/// Escape a string for embedding in a statement.
///
/// Only used for the database paths the store itself opened, so this is a
/// belt-and-braces measure rather than the primary defence.
fn escape_sql_string(value: &str) -> String {
    value.replace('\'', "''")
}
