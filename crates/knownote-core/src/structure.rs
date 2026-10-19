//! The structure a loader reports about a document.
//!
//! A port of `src/main/services/loaders/types.ts`. The loaders describe what
//! they found — pages, sections, or neither — and the blocks layer turns that
//! into the units a citation can point at.
//!
//! Every offset here is a byte offset into the canonical document text, the
//! same unit the chunker uses.

use serde::{Deserialize, Serialize};

/// A rectangle normalised to 0..1, with the origin at the top left.
///
/// The page coordinate system puts the origin top-left so the renderer can
/// use it without a flip.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NormalizedBox {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

/// A paragraph inside a page.
///
/// The text and its offsets are anchored to the whole canonical content, not
/// to the page, and the box is normalised to the page.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PageBlockInfo {
    pub text: String,
    pub start_offset: usize,
    pub end_offset: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bbox: Option<NormalizedBox>,
}

/// A page of a paginated document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PageInfo {
    /// 1-based page number.
    pub page_number: i64,
    pub content: String,
    pub start_offset: usize,
    pub end_offset: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bbox: Option<NormalizedBox>,
    /// Paragraph-level blocks inside the page, when the loader produced them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blocks: Vec<PageBlockInfo>,
}

/// A section of a document with a heading hierarchy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SectionInfo {
    /// Heading level, 1 to 6.
    pub level: i64,
    pub title: String,
    /// The section's own content, without its children.
    pub content: String,
    pub start_offset: usize,
    pub end_offset: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<SectionInfo>,
}

/// What a loader found in a document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum DocumentStructure {
    /// No usable structure.
    Flat,
    /// A paginated document: PDF, PowerPoint.
    Pages {
        #[serde(default)]
        pages: Vec<PageInfo>,
    },
    /// A document with a heading hierarchy: Markdown, Word.
    Sections {
        #[serde(default)]
        sections: Vec<SectionInfo>,
    },
}
