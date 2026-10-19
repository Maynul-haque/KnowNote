//! The chunker.
//!
//! A port of `src/main/services/ChunkingService.ts`. The port keeps the
//! original's rules exactly, because those rules are what the committed eval
//! baseline measures:
//!
//! * A heading is an atomic unit. A chunk never starts or ends inside one.
//! * A block is a semantic boundary. Nothing is ever split across two blocks.
//! * Inside a block the text is refined line by line, then sentence by
//!   sentence, and only a fragment that is still too long is hard-split.
//! * A chunk does not span a page boundary unless `allow_span_pages` says so.
//! * The overlap between two chunks is aligned to unit boundaries, so the two
//!   chunks share byte-identical text.
//!
//! # The one invariant that matters
//!
//! `chunk.content` is always exactly `content[start_offset..end_offset]` of
//! the canonical document text. The chunker never rewrites text, never
//! normalises whitespace and never computes offsets against a different
//! string. That is what lets a citation point at a location in the source
//! instead of at a chunk, and it is asserted by the tests.

use crate::text::{
    char_at, is_blank_js, is_sentence_end, is_whitespace, trim_leading_whitespace,
    trim_trailing_whitespace, utf16_len, utf16_len_range, Utf16Index,
};
use crate::tokens::estimate_tokens;
use crate::types::{
    ChunkBlock, ChunkBlockSpan, ChunkOptions, ChunkResult, ResolvedChunkOptions,
};

/// A semantic unit: a slice of one block, in canonical-document byte offsets.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Unit {
    block_index: usize,
    start: usize,
    end: usize,
    page: Option<i64>,
}

/// A chunk before its index and token estimate are attached.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ChunkDraft {
    content: String,
    start_offset: usize,
    end_offset: usize,
    block_spans: Vec<ChunkBlockSpan>,
    page_start: Option<i64>,
    page_end: Option<i64>,
}

/// Splits a document into chunks.
#[derive(Debug, Clone)]
pub struct ChunkingService {
    default_options: ResolvedChunkOptions,
}

impl Default for ChunkingService {
    fn default() -> Self {
        Self::new()
    }
}

impl ChunkingService {
    pub fn new() -> Self {
        Self {
            default_options: ResolvedChunkOptions::default(),
        }
    }

    /// Build a service with different defaults.
    pub fn with_default_options(default_options: ResolvedChunkOptions) -> Self {
        Self { default_options }
    }

    /// The options this service uses when a call does not override them.
    pub fn default_options(&self) -> &ResolvedChunkOptions {
        &self.default_options
    }

    /// Replace the defaults.
    pub fn set_default_options(&mut self, options: ResolvedChunkOptions) {
        self.default_options = options;
    }

    /// Split a document that has blocks.
    ///
    /// Offsets and content are anchored on the canonical text. Boundaries fall
    /// only between semantic units, a heading is never split, and a chunk does
    /// not span a page unless `allow_span_pages` says it may. `block_spans`
    /// records the range of each block the chunk covers, for the caller to
    /// persist.
    pub fn chunk_blocks(
        &self,
        content: &str,
        blocks: &[ChunkBlock],
        options: Option<&ChunkOptions>,
    ) -> Vec<ChunkResult> {
        let opts = self.resolve(options);

        let mut ordered: Vec<ChunkBlock> = blocks
            .iter()
            .filter(|block| block.end_offset > block.start_offset)
            .cloned()
            .collect();
        // A stable sort, like the original: two blocks that start at the same
        // offset keep the order the caller gave them.
        ordered.sort_by(|a, b| a.start_offset.cmp(&b.start_offset));

        if ordered.is_empty() {
            return self.chunk(content, options);
        }

        let units = build_units(&ordered, opts.chunk_size);
        let packed = pack_units(content, &ordered, units, &opts);

        packed
            .into_iter()
            .enumerate()
            .map(|(index, draft)| ChunkResult {
                index,
                token_count: estimate_tokens(&draft.content),
                content: draft.content,
                start_offset: draft.start_offset,
                end_offset: draft.end_offset,
                block_spans: draft.block_spans,
                page_start: draft.page_start,
                page_end: draft.page_end,
            })
            .collect()
    }

    /// Split text that has no blocks.
    ///
    /// A character window over the canonical text. Nothing is pre-processed
    /// and no length is rewritten, so every offset slices back exactly.
    ///
    /// This path is a fallback for content that arrived without structure. It
    /// works in UTF-16 code units, like the original, so that the window
    /// boundaries it produces are the ones the original produced.
    pub fn chunk(&self, text: &str, options: Option<&ChunkOptions>) -> Vec<ChunkResult> {
        let opts = self.resolve(options);

        if text.is_empty() || is_blank_js(text) {
            return Vec::new();
        }

        let index = Utf16Index::new(text);
        let text_len = index.len();
        let mut chunks: Vec<ChunkResult> = Vec::new();

        // Trim the whitespace at both ends of the window, then emit it.
        let push = |start: usize, end: usize, chunks: &mut Vec<ChunkResult>| {
            let start = trim_leading_utf16(text, &index, start, end);
            let end = trim_trailing_utf16(text, &index, start, end);
            if end <= start {
                return;
            }

            let byte_start = index.byte_of(start);
            let byte_end = index.byte_of(end);
            let content = text[byte_start..byte_end].to_string();

            chunks.push(ChunkResult {
                index: chunks.len(),
                token_count: estimate_tokens(&content),
                content,
                start_offset: byte_start,
                end_offset: byte_end,
                block_spans: Vec::new(),
                page_start: None,
                page_end: None,
            });
        };

        if text_len <= opts.min_chunk_size {
            push(0, text_len, &mut chunks);
            return chunks;
        }

        let mut start = 0usize;
        while start < text_len {
            let mut end = std::cmp::min(start + opts.chunk_size, text_len);
            if end < text_len {
                let split = find_separator_split(text, &index, start, end, &opts.separators);
                if split > start {
                    end = split;
                }
            }

            push(start, end, &mut chunks);
            if end >= text_len {
                break;
            }

            start = std::cmp::max(end.saturating_sub(opts.chunk_overlap), start + 1);
        }

        chunks
    }

    fn resolve(&self, options: Option<&ChunkOptions>) -> ResolvedChunkOptions {
        match options {
            None => self.default_options.clone(),
            Some(options) => options.resolve_over(&self.default_options),
        }
    }
}

/// Split a block sequence into semantic units.
///
/// A heading is one atomic unit. A paragraph is refined line by line, then
/// sentence by sentence; only a fragment that is still too long is hard-split.
/// Nothing crosses a block boundary, because a block boundary is already a
/// paragraph-level semantic boundary.
fn build_units(blocks: &[ChunkBlock], chunk_size: usize) -> Vec<Unit> {
    let mut units: Vec<Unit> = Vec::new();

    for (block_index, block) in blocks.iter().enumerate() {
        let text = block.text.as_str();

        if block.kind.is_heading() {
            push_unit(&mut units, block_index, block, 0, text.len(), true);
            continue;
        }

        // Code and table blocks keep their leading and trailing whitespace;
        // for every other kind that whitespace is not part of the content.
        let preserve_whitespace = block.kind.preserves_whitespace();

        for line in line_ranges(text) {
            if utf16_len_range(text, line.start, line.end) <= chunk_size {
                push_unit(
                    &mut units,
                    block_index,
                    block,
                    line.start,
                    line.end,
                    !preserve_whitespace,
                );
                continue;
            }

            for sentence in sentence_ranges(text, line.start, line.end) {
                if utf16_len_range(text, sentence.start, sentence.end) <= chunk_size {
                    push_unit(
                        &mut units,
                        block_index,
                        block,
                        sentence.start,
                        sentence.end,
                        !preserve_whitespace,
                    );
                    continue;
                }

                for piece in hard_split(text, sentence.start, sentence.end, chunk_size) {
                    push_unit(
                        &mut units,
                        block_index,
                        block,
                        piece.start,
                        piece.end,
                        !preserve_whitespace,
                    );
                }
            }
        }
    }

    units
}

/// Emit one unit, trimming the whitespace at the edges of the block-relative range.
fn push_unit(
    units: &mut Vec<Unit>,
    block_index: usize,
    block: &ChunkBlock,
    start: usize,
    end: usize,
    trim: bool,
) {
    let (start, end) = if trim {
        let start = trim_leading_whitespace(&block.text, start, end);
        let end = trim_trailing_whitespace(&block.text, start, end);
        (start, end)
    } else {
        (start, end)
    };

    if end > start {
        units.push(Unit {
            block_index,
            start: block.start_offset + start,
            end: block.start_offset + end,
            page: block.page,
        });
    }
}

/// Greedy packing: merge units until the chunk is full.
///
/// A chunk breaks at a page boundary unless `allow_span_pages` says otherwise.
/// The overlap is aligned to unit starts, so the overlapping region is
/// byte-identical in both chunks.
fn pack_units(
    content: &str,
    blocks: &[ChunkBlock],
    units: Vec<Unit>,
    opts: &ResolvedChunkOptions,
) -> Vec<ChunkDraft> {
    let mut chunks: Vec<ChunkDraft> = Vec::new();
    let mut current: Vec<Unit> = Vec::new();

    for unit in units {
        // A chunk never spans a heading when `respect_headings` is set, and
        // there is no overlap across that break: carrying the tail of the
        // previous section into the next one is exactly what the rule avoids.
        if opts.respect_headings
            && !current.is_empty()
            && blocks[unit.block_index].kind.is_heading()
        {
            chunks.push(materialize(content, blocks, &current));
            current.clear();
        }

        if current.is_empty() {
            current.push(unit);
            continue;
        }

        let last_page = current[current.len() - 1].page;
        let page_break = !opts.allow_span_pages
            && unit.page.is_some()
            && last_page.is_some()
            && unit.page != last_page;
        let would_overflow =
            utf16_len_range(content, current[0].start, unit.end) > opts.chunk_size;

        if page_break || would_overflow {
            let overlap = if page_break {
                Vec::new()
            } else {
                overlap_units(content, &current, opts.chunk_overlap)
            };
            chunks.push(materialize(content, blocks, &current));
            current = overlap;
        }

        current.push(unit);
    }

    if !current.is_empty() {
        chunks.push(materialize(content, blocks, &current));
    }

    chunks
}

/// Take units from the tail of a full chunk to open the next one.
///
/// The overlap never exceeds `chunk_overlap` characters and never covers the
/// whole previous chunk, because then the next chunk would only keep growing.
fn overlap_units(content: &str, current: &[Unit], chunk_overlap: usize) -> Vec<Unit> {
    if chunk_overlap == 0 {
        return Vec::new();
    }

    let end = current[current.len() - 1].end;
    let mut picked: Vec<Unit> = Vec::new();

    for index in (0..current.len()).rev() {
        if utf16_len_range(content, current[index].start, end) > chunk_overlap {
            break;
        }
        picked.insert(0, current[index].clone());
    }

    if picked.len() == current.len() && picked.len() > 1 {
        picked.remove(0);
    }

    picked
}

/// Cut a chunk's content and block ranges out of the unit range.
///
/// The content is always an exact slice of the canonical text.
fn materialize(content: &str, blocks: &[ChunkBlock], units: &[Unit]) -> ChunkDraft {
    let start_offset = units[0].start;
    let end_offset = units[units.len() - 1].end;
    let pieces = &content[start_offset..end_offset];

    let mut block_spans: Vec<ChunkBlockSpan> = Vec::new();
    let mut pages: Vec<i64> = Vec::new();

    for unit in units {
        let block = &blocks[unit.block_index];
        if let Some(page) = block.page {
            pages.push(page);
        }

        let start_in_block = unit.start - block.start_offset;
        let end_in_block = unit.end - block.start_offset;

        match block_spans.last_mut() {
            Some(last) if last.block_id == block.id => {
                last.start_in_block = last.start_in_block.min(start_in_block);
                last.end_in_block = last.end_in_block.max(end_in_block);
            }
            _ => block_spans.push(ChunkBlockSpan {
                block_id: block.id.clone(),
                start_in_block,
                end_in_block,
            }),
        }
    }

    ChunkDraft {
        content: pieces.to_string(),
        start_offset,
        end_offset,
        block_spans,
        page_start: pages.iter().copied().min(),
        page_end: pages.iter().copied().max(),
    }
}

/// Find where to break a window, scanning right to left for the highest-priority separator.
///
/// Positions are UTF-16 code-unit indices, like the original.
fn find_separator_split(
    text: &str,
    index: &Utf16Index,
    start: usize,
    end: usize,
    separators: &[String],
) -> usize {
    let search_start = std::cmp::max(start + (end - start) / 2, start);
    let mut best_pos: i64 = -1;
    let mut best_priority = separators.len();

    let mut position = end;
    loop {
        let byte = index.byte_of(position);

        for (priority, separator) in separators.iter().enumerate() {
            if text[byte..].starts_with(separator.as_str()) {
                // Only a strictly better priority replaces the current best,
                // so for equal priorities the rightmost match wins.
                if priority < best_priority {
                    best_pos = (position + utf16_len(separator)) as i64;
                    best_priority = priority;
                }
                break;
            }
        }

        if best_priority <= 2 || position == search_start {
            break;
        }
        position -= 1;
    }

    if best_pos > start as i64 {
        best_pos as usize
    } else {
        end
    }
}

/// The byte ranges of every line, without the newline.
fn line_ranges(text: &str) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0;

    for (offset, byte) in text.bytes().enumerate() {
        if byte == b'\n' {
            ranges.push(start..offset);
            start = offset + 1;
        }
    }
    ranges.push(start..text.len());

    ranges
}

/// Split a range on sentence-ending punctuation.
fn sentence_ranges(
    text: &str,
    from: usize,
    to: usize,
) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = from;

    let mut chars = text[from..to].char_indices().peekable();
    while let Some((offset, c)) = chars.next() {
        if !is_sentence_end(c) {
            continue;
        }

        let mut end = from + offset + c.len_utf8();
        while let Some(&(next_offset, next)) = chars.peek() {
            if !is_sentence_end(next) {
                break;
            }
            end = from + next_offset + next.len_utf8();
            chars.next();
        }

        ranges.push(start..end);
        start = end;
    }

    if start < to {
        ranges.push(start..to);
    }

    ranges
}

/// The last resort for a fragment that is still too long: pack it by whitespace.
///
/// A word that is itself longer than the limit is cut hard.
fn hard_split(
    text: &str,
    from: usize,
    to: usize,
    max_length: usize,
) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = from;
    let mut i = from;
    let mut last_whitespace: Option<usize> = None;
    // The length of `text[start..i]` in UTF-16 code units, tracked as the
    // cursor moves so the comparison matches the original.
    let mut length = 0usize;

    while i < to {
        if length >= max_length {
            let cut = match last_whitespace {
                Some(whitespace) if whitespace > start => whitespace,
                _ => i,
            };
            ranges.push(start..cut);
            start = cut;

            while start < to && is_whitespace(char_at(text, start)) {
                start += char_at(text, start).len_utf8();
            }

            i = start;
            last_whitespace = None;
            length = 0;
            continue;
        }

        let c = char_at(text, i);
        if is_whitespace(c) {
            last_whitespace = Some(i);
        }
        i += c.len_utf8();
        length += c.len_utf16();
    }

    if start < to {
        ranges.push(start..to);
    }

    ranges
}

/// Trim leading whitespace in UTF-16 index space.
fn trim_leading_utf16(text: &str, index: &Utf16Index, start: usize, end: usize) -> usize {
    let byte_start = index.byte_of(start);
    let byte_end = index.byte_of(end);
    let trimmed = trim_leading_whitespace(text, byte_start, byte_end);

    if trimmed >= byte_end {
        // Everything in the range was whitespace.
        return end;
    }
    index.utf16_of_byte(trimmed)
}

/// Trim trailing whitespace in UTF-16 index space.
fn trim_trailing_utf16(text: &str, index: &Utf16Index, start: usize, end: usize) -> usize {
    let byte_start = index.byte_of(start);
    let byte_end = index.byte_of(end);
    let trimmed = trim_trailing_whitespace(text, byte_start, byte_end);

    if trimmed <= byte_start {
        // Everything in the range was whitespace.
        return start;
    }
    index.utf16_of_byte(trimmed)
}
