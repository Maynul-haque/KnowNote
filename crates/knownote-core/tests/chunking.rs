//! The chunker contract.
//!
//! A port of `test/chunkBlocks.test.ts`. Every test here asserts the same
//! contract the original does:
//!
//! ```text
//! content[start_offset..end_offset] == chunk.content
//! ```
//!
//! plus the properties the provenance layer depends on: dense chunk indices,
//! block spans that point back into real blocks, page ranges, overlap that
//! shares byte-identical text, and headings that are never split.

use knownote_core::{BlockKind, ChunkBlock, ChunkOptions, ChunkingService};

/// A small heading + paragraph document with hand-computed canonical offsets.
struct Synthetic {
    content: String,
    blocks: Vec<ChunkBlock>,
}

fn synthetic() -> Synthetic {
    let heading1 = "Overview";
    let paragraph1 = "First paragraph sentence one. Sentence two.";
    let heading2 = "Details";
    let paragraph2 = "Second paragraph with more text.";
    let content = [heading1, paragraph1, heading2, paragraph2].join("\n\n");

    let offset_of = |text: &str| content.find(text).expect("block text is in the document");

    let block = |id: &str, kind: BlockKind, level: Option<i64>, text: &str| ChunkBlock {
        id: id.to_string(),
        kind,
        page: None,
        level,
        text: text.to_string(),
        start_offset: offset_of(text),
        end_offset: offset_of(text) + text.len(),
    };

    let blocks = vec![
        block("b1", BlockKind::Heading, Some(1), heading1),
        block("b2", BlockKind::Paragraph, None, paragraph1),
        block("b3", BlockKind::Heading, Some(2), heading2),
        block("b4", BlockKind::Paragraph, None, paragraph2),
    ];

    Synthetic { content, blocks }
}

fn assert_canonical_offsets(content: &str, chunks: &[knownote_core::ChunkResult]) {
    let indices: Vec<usize> = chunks.iter().map(|chunk| chunk.index).collect();
    let expected: Vec<usize> = (0..chunks.len()).collect();
    assert_eq!(indices, expected, "chunk index must be dense and ascending");

    for chunk in chunks {
        assert_eq!(
            &content[chunk.start_offset..chunk.end_offset],
            chunk.content,
            "chunk {} does not slice back to the canonical document",
            chunk.index
        );
        assert!(
            chunk.token_count > 0,
            "chunk {} has no token estimate",
            chunk.index
        );
    }
}

fn assert_spans_resolve(
    content: &str,
    blocks: &[ChunkBlock],
    chunks: &[knownote_core::ChunkResult],
) {
    for chunk in chunks {
        let mut previous_order: i64 = -1;

        for span in &chunk.block_spans {
            let order = blocks
                .iter()
                .position(|block| block.id == span.block_id)
                .expect("chunk references an unknown block");
            let block = &blocks[order];

            assert!(
                span.start_in_block <= span.end_in_block,
                "chunk {} span for {} is inverted",
                chunk.index,
                span.block_id
            );
            assert!(
                span.end_in_block <= block.text.len(),
                "chunk {} span for {} leaves the block",
                chunk.index,
                span.block_id
            );
            assert!(
                span.end_in_block > span.start_in_block,
                "chunk {} span for {} is empty",
                chunk.index,
                span.block_id
            );

            let span_text = &content
                [block.start_offset + span.start_in_block..block.start_offset + span.end_in_block];
            assert!(
                chunk.content.contains(span_text),
                "chunk {} content does not contain the text of {}",
                chunk.index,
                span.block_id
            );

            assert!(
                order as i64 > previous_order,
                "block spans must follow document order"
            );
            previous_order = order as i64;
        }
    }
}

#[test]
fn chunk_offsets_and_content_address_the_canonical_document_exactly() {
    let Synthetic { content, blocks } = synthetic();
    let chunks = ChunkingService::new().chunk_blocks(
        &content,
        &blocks,
        Some(&ChunkOptions {
            chunk_size: Some(40),
            ..Default::default()
        }),
    );

    assert!(
        chunks.len() > 1,
        "a small chunk size must produce several chunks"
    );
    assert_canonical_offsets(&content, &chunks);
    assert_spans_resolve(&content, &blocks, &chunks);
}

#[test]
fn a_heading_block_is_never_split_even_when_it_forces_a_chunk_boundary() {
    let Synthetic { content, blocks } = synthetic();
    let chunks = ChunkingService::new().chunk_blocks(
        &content,
        &blocks,
        Some(&ChunkOptions {
            chunk_size: Some(20),
            chunk_overlap: Some(5),
            ..Default::default()
        }),
    );

    assert_canonical_offsets(&content, &chunks);

    let mut seen = 0;
    for chunk in &chunks {
        for span in &chunk.block_spans {
            let block = blocks
                .iter()
                .find(|candidate| candidate.id == span.block_id)
                .expect("span points at a real block");
            if !block.kind.is_heading() {
                continue;
            }
            seen += 1;
            assert_eq!(
                span.start_in_block, 0,
                "chunk {} starts inside a heading",
                chunk.index
            );
            assert_eq!(
                span.end_in_block,
                block.text.len(),
                "chunk {} cut heading {} short",
                chunk.index,
                block.id
            );
        }
    }

    assert!(seen >= 2, "every heading should be covered at least once");
}

#[test]
fn overlapping_chunks_share_identical_bytes() {
    let Synthetic { content, blocks } = synthetic();
    let chunks = ChunkingService::new().chunk_blocks(
        &content,
        &blocks,
        Some(&ChunkOptions {
            chunk_size: Some(30),
            chunk_overlap: Some(15),
            ..Default::default()
        }),
    );

    let mut overlaps = 0;
    for index in 0..chunks.len().saturating_sub(1) {
        let current = &chunks[index];
        let next = &chunks[index + 1];
        if next.start_offset >= current.end_offset {
            continue;
        }

        overlaps += 1;
        let from_current = &current.content[next.start_offset - current.start_offset..];
        let from_next = &next.content[..current.end_offset - next.start_offset];
        assert_eq!(
            from_current, from_next,
            "chunk {} and {} do not share the same overlap text",
            index,
            index + 1
        );
    }

    assert!(overlaps > 0, "expected at least one overlapping chunk pair");
}

#[test]
fn the_char_window_fallback_also_produces_canonical_offsets_and_overlap() {
    let content = (0..40)
        .map(|index| format!("Sentence number {index}."))
        .collect::<Vec<_>>()
        .join(" ");

    let chunks = ChunkingService::new().chunk(
        &content,
        Some(&ChunkOptions {
            chunk_size: Some(120),
            chunk_overlap: Some(40),
            ..Default::default()
        }),
    );

    assert!(chunks.len() > 1);
    assert_canonical_offsets(&content, &chunks);
    assert!(chunks.iter().all(|chunk| chunk.block_spans.is_empty()));

    for index in 0..chunks.len().saturating_sub(1) {
        let current = &chunks[index];
        let next = &chunks[index + 1];
        assert!(
            next.start_offset < current.end_offset,
            "windows should overlap"
        );
        assert_eq!(
            &current.content[next.start_offset - current.start_offset..],
            &next.content[..current.end_offset - next.start_offset]
        );
    }
}

#[test]
fn no_blocks_and_no_text_produces_no_chunks() {
    let service = ChunkingService::new();
    assert!(service.chunk_blocks("", &[], None).is_empty());
    assert!(service.chunk("   \n\n  ", None).is_empty());
}

#[test]
fn cross_page_chunks_only_appear_when_explicitly_allowed() {
    // Three pages, one block each, the shape the PDF loader produces.
    let pages = ["Page one text.", "Page two text.", "Page three text."];
    let content = pages.join("\n\n");
    let blocks: Vec<ChunkBlock> = pages
        .iter()
        .enumerate()
        .map(|(index, text)| {
            let start = content.find(text).expect("page text is in the document");
            ChunkBlock {
                id: format!("p{}", index + 1),
                kind: BlockKind::Paragraph,
                page: Some(index as i64 + 1),
                level: None,
                text: text.to_string(),
                start_offset: start,
                end_offset: start + text.len(),
            }
        })
        .collect();

    let service = ChunkingService::new();

    let separated = service.chunk_blocks(&content, &blocks, None);
    assert_canonical_offsets(&content, &separated);
    let ranges: Vec<(Option<i64>, Option<i64>)> = separated
        .iter()
        .map(|chunk| (chunk.page_start, chunk.page_end))
        .collect();
    assert_eq!(
        ranges,
        vec![(Some(1), Some(1)), (Some(2), Some(2)), (Some(3), Some(3))]
    );

    let merged = service.chunk_blocks(
        &content,
        &blocks,
        Some(&ChunkOptions {
            allow_span_pages: Some(true),
            ..Default::default()
        }),
    );
    assert_eq!(merged.len(), 1, "a large chunk with allow_span_pages merges");
    assert_eq!(merged[0].page_start, Some(1));
    assert_eq!(merged[0].page_end, Some(3));
    assert_canonical_offsets(&content, &merged);
}

#[test]
fn page_range_matches_the_min_and_max_page_of_the_mapped_blocks() {
    let pages = ["Page one text.", "Page two text.", "Page three text."];
    let content = pages.join("\n\n");
    let blocks: Vec<ChunkBlock> = pages
        .iter()
        .enumerate()
        .map(|(index, text)| {
            let start = content.find(text).expect("page text is in the document");
            ChunkBlock {
                id: format!("p{}", index + 1),
                kind: BlockKind::Paragraph,
                page: Some(index as i64 + 1),
                level: None,
                text: text.to_string(),
                start_offset: start,
                end_offset: start + text.len(),
            }
        })
        .collect();

    let chunks = ChunkingService::new().chunk_blocks(&content, &blocks, None);

    for chunk in &chunks {
        let mut pages: Vec<i64> = Vec::new();
        for span in &chunk.block_spans {
            let block = blocks
                .iter()
                .find(|candidate| candidate.id == span.block_id)
                .expect("span points at a real block");
            if let Some(page) = block.page {
                pages.push(page);
            }
        }
        assert_eq!(
            (chunk.page_start, chunk.page_end),
            (pages.iter().min().copied(), pages.iter().max().copied()),
            "chunk {} page range does not match its blocks",
            chunk.index
        );
    }
}

#[test]
fn a_chunk_resolves_to_its_ordered_block_spans() {
    let Synthetic { content, blocks } = synthetic();
    let chunks = ChunkingService::new().chunk_blocks(&content, &blocks, None);
    let chunk = &chunks[0];

    assert_eq!(
        chunk.block_spans.len(),
        blocks.len(),
        "the whole synthetic document is one chunk"
    );

    let span_order: Vec<usize> = chunk
        .block_spans
        .iter()
        .map(|span| {
            blocks
                .iter()
                .position(|block| block.id == span.block_id)
                .expect("span points at a real block")
        })
        .collect();
    let expected: Vec<usize> = (0..blocks.len()).collect();
    assert_eq!(span_order, expected, "block spans follow document order");
}

#[test]
fn respect_headings_keeps_a_chunk_from_spanning_a_section() {
    let Synthetic { content, blocks } = synthetic();
    let chunks = ChunkingService::new().chunk_blocks(
        &content,
        &blocks,
        Some(&ChunkOptions {
            respect_headings: Some(true),
            ..Default::default()
        }),
    );

    assert_canonical_offsets(&content, &chunks);
    assert_spans_resolve(&content, &blocks, &chunks);

    // No chunk may cover a heading and a following paragraph together.
    for chunk in &chunks {
        let kinds: Vec<BlockKind> = chunk
            .block_spans
            .iter()
            .filter_map(|span| {
                blocks
                    .iter()
                    .find(|block| block.id == span.block_id)
                    .map(|block| block.kind)
            })
            .collect();
        let heading_positions: Vec<usize> = kinds
            .iter()
            .enumerate()
            .filter(|(_, kind)| kind.is_heading())
            .map(|(index, _)| index)
            .collect();
        for position in heading_positions {
            assert!(
                position + 1 >= kinds.len() || !kinds[position + 1].is_heading(),
                "a chunk must not span two sections"
            );
        }
    }
}
