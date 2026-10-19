//! Turning a loaded document into blocks.
//!
//! A port of `src/main/services/blocks/documentBlocks.ts`.
//!
//! A block is where a citation lands. Every block carries the offsets that
//! index the canonical document text, so the block's text is *always*
//! `content[start_offset..end_offset]` — never a re-serialised copy. That is
//! the mechanism that lets a citation resolve to a location in the source
//! rather than to a chunk that might be re-split tomorrow.
//!
//! The rules, in the order they are applied:
//!
//! * A paginated document gets one block per page, or per paragraph within
//!   the page when the loader produced paragraph-level geometry.
//! * A document with usable heading offsets gets a heading block per heading
//!   and paragraph blocks for the text between them.
//! * Anything else gets flat paragraph blocks, split on blank lines.

use crate::structure::{DocumentStructure, NormalizedBox, PageInfo, SectionInfo};
use crate::text::is_blank_js;
use serde::{Deserialize, Serialize};

/// A block before it has been given an id and a document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocumentBlockDraft {
    pub kind: crate::types::BlockKind,
    /// Dense, ascending order within the document.
    pub order: usize,
    pub page: Option<i64>,
    pub level: Option<i64>,
    pub text: String,
    pub start_offset: usize,
    pub end_offset: usize,
    pub bbox: Option<NormalizedBox>,
    pub metadata: Option<serde_json::Value>,
}

/// A block that has been given an id and its document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IdentifiedBlockDraft {
    pub id: String,
    pub document_id: String,
    #[serde(flatten)]
    pub draft: DocumentBlockDraft,
}

/// Give every block an id.
///
/// The id is derived from the document id and the order, so it is unique and
/// readable. Re-indexing reuses the same document id but deletes the old
/// blocks first, so an order-based id never collides with a stale one.
pub fn assign_block_ids(
    document_id: &str,
    drafts: Vec<DocumentBlockDraft>,
) -> Vec<IdentifiedBlockDraft> {
    drafts
        .into_iter()
        .map(|draft| IdentifiedBlockDraft {
            id: format!("blk_{document_id}_{}", draft.order),
            document_id: document_id.to_string(),
            draft,
        })
        .collect()
}

/// Build the block sequence of a document.
pub fn build_document_blocks(
    content: &str,
    structure: Option<&DocumentStructure>,
) -> Vec<DocumentBlockDraft> {
    let mut blocks = match structure {
        Some(DocumentStructure::Pages { pages }) if !pages.is_empty() => {
            build_page_blocks(content, pages)
        }
        _ => {
            let headings = match structure {
                Some(DocumentStructure::Sections { sections }) if !sections.is_empty() => {
                    usable_headings(content, sections)
                }
                _ => Vec::new(),
            };
            if headings.is_empty() {
                build_flat_blocks(content)
            } else {
                build_section_blocks(content, &headings)
            }
        }
    };

    for (order, block) in blocks.iter_mut().enumerate() {
        block.order = order;
    }
    blocks
}

/// A paginated document.
///
/// The loader's paragraph-level blocks are preferred when it produced them;
/// otherwise the page is one block.
fn build_page_blocks(content: &str, pages: &[PageInfo]) -> Vec<DocumentBlockDraft> {
    let mut blocks = Vec::new();

    for page in pages {
        if !page.blocks.is_empty() {
            for page_block in &page.blocks {
                let text = slice(content, page_block.start_offset, page_block.end_offset);
                if is_blank_js(&text) {
                    continue;
                }

                blocks.push(DocumentBlockDraft {
                    kind: crate::types::BlockKind::Paragraph,
                    order: 0,
                    page: Some(page.page_number),
                    level: None,
                    // The end offset is written back from the text that was
                    // actually cut, so trailing whitespace in the loader's
                    // range cannot push the block past the content.
                    end_offset: page_block.start_offset + text.len(),
                    text,
                    start_offset: page_block.start_offset,
                    bbox: page_block.bbox,
                    metadata: None,
                });
            }
            continue;
        }

        let text = slice(content, page.start_offset, page.end_offset);
        if is_blank_js(&text) {
            continue;
        }

        blocks.push(DocumentBlockDraft {
            kind: crate::types::BlockKind::Paragraph,
            order: 0,
            page: Some(page.page_number),
            level: None,
            end_offset: page.start_offset + text.len(),
            text,
            start_offset: page.start_offset,
            bbox: page.bbox,
            metadata: None,
        });
    }

    blocks
}

/// A heading reference that survived the checks in [`usable_headings`].
struct HeadingRef {
    level: i64,
    start_offset: usize,
}

/// A document with headings.
///
/// Only the heading's start offset is trusted; the body runs from the end of
/// the heading line to the next heading. Parent and child headings therefore
/// never contain each other.
fn build_section_blocks(content: &str, headings: &[HeadingRef]) -> Vec<DocumentBlockDraft> {
    let mut blocks = Vec::new();
    let mut cursor = 0usize;

    for heading in headings {
        if heading.start_offset > cursor {
            blocks.extend(paragraph_blocks(content, cursor, heading.start_offset));
        }

        let line_end = end_of_line(content, heading.start_offset);
        blocks.push(DocumentBlockDraft {
            kind: crate::types::BlockKind::Heading,
            order: 0,
            page: None,
            level: Some(heading.level),
            text: slice(content, heading.start_offset, line_end),
            start_offset: heading.start_offset,
            end_offset: line_end,
            bbox: None,
            metadata: None,
        });

        cursor = line_end;
    }

    if cursor < content.len() {
        blocks.extend(paragraph_blocks(content, cursor, content.len()));
    }

    blocks
}

/// A document with no usable structure: flat paragraph blocks.
fn build_flat_blocks(content: &str) -> Vec<DocumentBlockDraft> {
    paragraph_blocks(content, 0, content.len())
}

/// Split a range into paragraph blocks.
fn paragraph_blocks(
    content: &str,
    from: usize,
    to: usize,
) -> Vec<DocumentBlockDraft> {
    split_paragraphs(content, from, to)
        .into_iter()
        .map(|(start, end)| DocumentBlockDraft {
            kind: crate::types::BlockKind::Paragraph,
            order: 0,
            page: None,
            level: None,
            text: slice(content, start, end),
            start_offset: start,
            end_offset: end,
            bbox: None,
            metadata: None,
        })
        .collect()
}

/// The headings of a document that can be used, in document order.
///
/// Two things are refused. Duplicate offsets — the web loader reports every
/// section at offset zero — and a heading whose title is not actually at the
/// offset, which would let a wrong offset masquerade as structure.
fn usable_headings(content: &str, sections: &[SectionInfo]) -> Vec<HeadingRef> {
    let flat = flatten_sections(sections);
    if flat.is_empty() {
        return Vec::new();
    }

    let mut starts: Vec<usize> = flat.iter().map(|section| section.start_offset).collect();
    let unique = {
        let mut sorted = starts.clone();
        sorted.sort_unstable();
        sorted.dedup();
        sorted.len()
    };
    if unique != starts.len() {
        // Two sections at the same offset: the offsets carry no information,
        // so the structure is refused rather than half-trusted.
        return Vec::new();
    }
    starts.clear();

    let mut usable = Vec::new();
    for section in &flat {
        if section.start_offset >= content.len() {
            return Vec::new();
        }
        if !heading_matches_content(content, section) {
            return Vec::new();
        }
        usable.push(HeadingRef {
            level: section.level,
            start_offset: section.start_offset,
        });
    }

    usable.sort_by_key(|heading| heading.start_offset);
    usable
}

/// Flatten a section tree into document order.
fn flatten_sections(sections: &[SectionInfo]) -> Vec<SectionInfo> {
    let mut flat = Vec::new();
    fn walk(nodes: &[SectionInfo], flat: &mut Vec<SectionInfo>) {
        for node in nodes {
            flat.push(node.clone());
            if !node.children.is_empty() {
                walk(&node.children, flat);
            }
        }
    }
    walk(sections, &mut flat);
    flat
}

/// Whether the heading's title really is at its offset.
///
/// The offset points at the `#` markers in Markdown, so the title appears a
/// little way in rather than at the offset itself.
fn heading_matches_content(content: &str, section: &SectionInfo) -> bool {
    let window_end = std::cmp::min(
        content.len(),
        section.start_offset + section.level as usize + section.title.len() + 4,
    );
    slice(content, section.start_offset, window_end).contains(&section.title)
}

/// The offset of the end of the line that starts at `from`.
fn end_of_line(content: &str, from: usize) -> usize {
    match content[from..].find('\n') {
        Some(offset) => from + offset,
        None => content.len(),
    }
}

/// Split a range on blank lines.
///
/// The ranges are exact and do not overlap, and the caller slices the text out
/// of them, so nothing is rewritten here.
fn split_paragraphs(content: &str, from: usize, to: usize) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut i = from;

    while i < to {
        while i < to && is_whitespace_at(content, i) {
            i += char_len_at(content, i);
        }
        if i >= to {
            break;
        }

        let start = i;
        let mut end = i;

        while i < to {
            if content.as_bytes()[i] == b'\n' && is_blank_line_break(content, i, to) {
                break;
            }
            if !is_whitespace_at(content, i) {
                end = i + char_len_at(content, i);
            }
            i += char_len_at(content, i);
        }

        if end > start {
            ranges.push((start, end));
        }

        while i < to && is_whitespace_at(content, i) {
            i += char_len_at(content, i);
        }
    }

    ranges
}

/// Whether the newline at `index` opens a blank line.
fn is_blank_line_break(content: &str, index: usize, to: usize) -> bool {
    let mut i = index + 1;
    while i < to {
        if content.as_bytes()[i] == b'\n' {
            return true;
        }
        if !is_whitespace_at(content, i) {
            return false;
        }
        i += char_len_at(content, i);
    }
    true
}

/// Whether the character at `index` is one of the chunker's whitespace characters.
fn is_whitespace_at(content: &str, index: usize) -> bool {
    crate::text::is_whitespace(content[index..].chars().next().unwrap_or('\0'))
}

/// The byte length of the character at `index`.
fn char_len_at(content: &str, index: usize) -> usize {
    content[index..].chars().next().map(char::len_utf8).unwrap_or(1)
}

/// Cut a range out of the content, clamped to the content's bounds.
fn slice(content: &str, start: usize, end: usize) -> String {
    let start = std::cmp::min(start, content.len());
    let end = std::cmp::min(std::cmp::max(end, start), content.len());
    content[start..end].to_string()
}
