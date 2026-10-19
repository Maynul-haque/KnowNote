//! Shared types for the chunking and provenance layer.

use serde::{Deserialize, Serialize};

/// The kind of a document block.
///
/// Mirrors `BlockKind` in `src/main/services/blocks/documentBlocks.ts`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockKind {
    Heading,
    Paragraph,
    ListItem,
    Table,
    Figure,
    Caption,
    Code,
}

impl BlockKind {
    /// Headings are atomic: a chunk never starts or ends inside one.
    pub fn is_heading(self) -> bool {
        matches!(self, BlockKind::Heading)
    }

    /// Code and table blocks keep their leading and trailing whitespace,
    /// because that whitespace is part of what the block shows.
    pub fn preserves_whitespace(self) -> bool {
        matches!(self, BlockKind::Code | BlockKind::Table)
    }
}

/// A block of a document, as the chunker sees it.
///
/// This is a subset of a `document_blocks` row: the chunker needs the text,
/// its position in the canonical document, its page and its kind.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkBlock {
    pub id: String,
    pub kind: BlockKind,
    /// 1-based page number, or `None` for a document without pages.
    pub page: Option<i64>,
    /// Heading level, or `None` for a block that is not a heading.
    pub level: Option<i64>,
    pub text: String,
    /// Byte offset of `text` inside the canonical document text.
    pub start_offset: usize,
    /// Byte offset one past the end of `text` inside the canonical document text.
    pub end_offset: usize,
}

/// The character range of one block that a chunk covers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkBlockSpan {
    pub block_id: String,
    /// Byte offset of the covered range inside the block's own text.
    pub start_in_block: usize,
    /// Byte offset one past the end of the covered range inside the block's text.
    pub end_in_block: usize,
}

/// A chunk: a contiguous slice of the canonical document text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkResult {
    /// Exactly `content[start_offset..end_offset]` of the canonical text.
    pub content: String,
    /// Dense, ascending index within the document.
    pub index: usize,
    pub start_offset: usize,
    pub end_offset: usize,
    /// Estimated token count, for budgeting a prompt.
    pub token_count: usize,
    /// The blocks this chunk covers, in document order.
    pub block_spans: Vec<ChunkBlockSpan>,
    /// The lowest page number among the covered blocks, if any.
    pub page_start: Option<i64>,
    /// The highest page number among the covered blocks, if any.
    pub page_end: Option<i64>,
}

/// Overrides for a chunking run. Every field left as `None` keeps the default.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChunkOptions {
    /// Target chunk length, counted in UTF-16 code units.
    pub chunk_size: Option<usize>,
    /// Overlap between consecutive chunks, in UTF-16 code units.
    pub chunk_overlap: Option<usize>,
    /// Where the character-window fallback is allowed to break a chunk.
    pub separators: Option<Vec<String>>,
    /// A document no longer than this is emitted as a single chunk.
    pub min_chunk_size: Option<usize>,
    /// Whether a chunk may span a page boundary. Off by default.
    pub allow_span_pages: Option<bool>,
    /// Whether a chunk may span a heading. Off by default.
    pub respect_headings: Option<bool>,
}

/// A [`ChunkOptions`] with every field resolved to a concrete value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedChunkOptions {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub separators: Vec<String>,
    pub min_chunk_size: usize,
    pub allow_span_pages: bool,
    pub respect_headings: bool,
}

/// The default chunk size, in UTF-16 code units.
///
/// `1000` with an overlap of `100` is the measured result of the v1.5 chunking
/// experiment: against the v1.4 corpus they moved Recall@1 from 0.7667 to
/// 0.8333, MRR from 0.8492 to 0.9278 and nDCG@10 from 0.8899 to 0.9437,
/// while the index shrank from 36 chunks to 19. The comparison is in
/// `docs/eval/chunking-v1.5.md`; the frozen baseline is
/// `docs/eval/baseline-v1.5.json`.
pub const DEFAULT_CHUNK_SIZE: usize = 1000;

/// The default overlap between chunks, in UTF-16 code units.
pub const DEFAULT_CHUNK_OVERLAP: usize = 100;

/// The default size below which a whole document is one chunk.
pub const DEFAULT_MIN_CHUNK_SIZE: usize = 100;

/// The default separators, in the order the fallback breaks a window.
///
/// The order is the priority order: a section break beats a paragraph break,
/// a paragraph break beats a line break, and a space is the last resort.
pub const DEFAULT_SEPARATORS: &[&str] = &[
    "\n\n\n",     // several blank lines (section break)
    "\n\n",       // paragraph break
    "\n",         // line break
    "\u{3002}", // 。 Chinese full stop
    ".",          // English full stop
    "\u{ff01}", // ！
    "!",
    "\u{ff1f}", // ？
    "?",
    "\u{ff1b}", // ；
    ";",
    "\u{ff0c}", // ，
    ",",
    " ", // last resort
];

/// The default chunking parameters.
///
/// The values live in code rather than in a report so that the report cannot
/// drift from the code: changing a default without re-running the baseline
/// shows up as a delta in the baseline itself.
pub fn default_chunk_options() -> ResolvedChunkOptions {
    ResolvedChunkOptions {
        chunk_size: DEFAULT_CHUNK_SIZE,
        chunk_overlap: DEFAULT_CHUNK_OVERLAP,
        min_chunk_size: DEFAULT_MIN_CHUNK_SIZE,
        allow_span_pages: false,
        respect_headings: false,
        separators: DEFAULT_SEPARATORS
            .iter()
            .map(|separator| separator.to_string())
            .collect(),
    }
}

impl Default for ResolvedChunkOptions {
    fn default() -> Self {
        default_chunk_options()
    }
}

impl ChunkOptions {
    /// Merge these overrides over `base`.
    ///
    /// A field left as `None` keeps the base value, so an override touches
    /// only what it names.
    pub fn resolve_over(&self, base: &ResolvedChunkOptions) -> ResolvedChunkOptions {
        ResolvedChunkOptions {
            chunk_size: self.chunk_size.unwrap_or(base.chunk_size),
            chunk_overlap: self.chunk_overlap.unwrap_or(base.chunk_overlap),
            separators: self
                .separators
                .clone()
                .unwrap_or_else(|| base.separators.clone()),
            min_chunk_size: self.min_chunk_size.unwrap_or(base.min_chunk_size),
            allow_span_pages: self.allow_span_pages.unwrap_or(base.allow_span_pages),
            respect_headings: self.respect_headings.unwrap_or(base.respect_headings),
        }
    }
}
