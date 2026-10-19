//! KnowNote core: document chunking, provenance and citation assembly.
//!
//! This crate is the Rust port of the parts of the Electron app that decide
//! *what text goes into which chunk* and *where in the source that text came
//! from*. It is deliberately free of I/O, of Electron and of SQLite, so the
//! rules can be tested against the committed eval baseline on their own.
//!
//! # Offsets
//!
//! Every offset in this crate is a **byte** offset into the UTF-8 canonical
//! document text (`documents.content`). The TypeScript original used UTF-16
//! code-unit indices, because that is what a JavaScript string is indexed by.
//! The two differ for any character outside the Basic Multilingual Plane, so
//! the port is explicit about which unit it uses at every boundary:
//!
//! * **Positions** (chunk and block offsets) are byte offsets. They are what
//!   gets persisted, and `&content[start..end]` is the invariant.
//! * **Size comparisons** (`chunk_size`, `chunk_overlap`, `min_chunk_size`)
//!   are counted in UTF-16 code units, so the chunk boundaries this crate
//!   produces are the same boundaries the original produced. Only the numbers
//!   stored in the database change unit.
//!
//! That combination is what makes the port verifiable: the chunk *contents*
//! and the chunk *boundaries* are identical to the TypeScript implementation,
//! so a differential test against the original can assert equality, and the
//! only thing a database migration has to do is re-express the same positions
//! in bytes.

pub mod blocks;
pub mod chunking;
pub mod retrieval;
pub mod structure;
pub mod text;
pub mod tokens;
pub mod types;

pub use blocks::{assign_block_ids, build_document_blocks, DocumentBlockDraft, IdentifiedBlockDraft};
pub use chunking::ChunkingService;
pub use retrieval::{
    dense_channel_threshold, effective_candidate_k, rrf_fuse, CandidateHit, RetrievedEvidence,
    RetrievalFilter, RetrievalRequest, RetrievalResult, RetrievalStrategy, RetrievalTrace,
    ScoredChunkHit, DEFAULT_CANDIDATE_K, DEFAULT_DENSE_THRESHOLD, DEFAULT_TOP_K, RRF_K,
};
pub use structure::{DocumentStructure, NormalizedBox, PageBlockInfo, PageInfo, SectionInfo};
pub use types::{
    default_chunk_options, BlockKind, ChunkBlock, ChunkBlockSpan, ChunkOptions, ChunkResult,
    ResolvedChunkOptions, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_MIN_CHUNK_SIZE,
    DEFAULT_SEPARATORS,
};
