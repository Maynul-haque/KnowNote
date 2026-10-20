//! The PDF text layout algorithm, pinned with synthetic items.
//!
//! These are the item-for-item ports of `test/pdfTextLayout.test.ts`. The
//! original pins its geometry the same way — no PDF fixture, no engine, just
//! the rules — and the port has to satisfy exactly those cases: items on one
//! baseline form a line, gaps become spaces, a split word does not, a vertical
//! gap or a first-line indent starts a paragraph, boxes normalise to the page,
//! and the paragraphs still slice back out of the canonical text they produced.

use std::string::ToString;

use knownote_core::blocks::build_document_blocks;
use knownote_core::structure::{DocumentStructure, NormalizedBox, PageBlockInfo, PageInfo};
use knownote_ingest::{layout_page_text, PositionedTextItem};

/// A viewport-space text item; font height defaults to 10.
fn item(text: &str, left: f64, baseline: f64, width: f64) -> PositionedTextItem {
    PositionedTextItem {
        text: ToString::to_string(text),
        left,
        baseline,
        width,
        height: 10.0,
    }
}

/// Two lines in one paragraph, then a second paragraph after a blank line.
fn two_paragraph_page() -> Vec<PositionedTextItem> {
    vec![
        item("First", 50.0, 100.0, 40.0),
        item("line", 96.0, 100.0, 20.0),
        item("continues", 50.0, 114.0, 60.0),
        item("Second paragraph.", 50.0, 140.0, 110.0),
    ]
}

#[test]
fn items_on_the_same_baseline_become_one_line_and_words_are_spaced_apart() {
    let paragraphs = layout_page_text(&two_paragraph_page(), 800.0, 1000.0);
    assert_eq!(paragraphs.len(), 2, "the page has two paragraphs");

    // "First" ends at x=90, "line" starts at x=96: a gap, so a space is
    // inserted between the items even though neither contains one.
    assert_eq!(paragraphs[0].text, "First line continues");
}

#[test]
fn a_word_split_across_items_is_not_given_an_inner_space() {
    let words = [item("Know", 50.0, 100.0, 32.0), item("Note", 82.0, 100.0, 34.0)];

    let paragraphs = layout_page_text(&words, 800.0, 1000.0);
    assert_eq!(paragraphs.len(), 1);
    assert_eq!(paragraphs[0].text, "KnowNote");
}

#[test]
fn a_vertical_gap_starts_a_new_paragraph_preserving_reading_order() {
    let paragraphs = layout_page_text(&two_paragraph_page(), 800.0, 1000.0);

    let texts: Vec<&str> = paragraphs.iter().map(|p| p.text.as_str()).collect();
    assert_eq!(texts, ["First line continues", "Second paragraph."]);
}

#[test]
fn a_first_line_indent_starts_a_new_paragraph_even_without_a_vertical_gap() {
    let page = [
        item("one line", 50.0, 100.0, 60.0),
        // Indented by 20 (> one font height), with a vertical gap of only 2.
        item("and the next", 70.0, 112.0, 70.0),
    ];

    let paragraphs = layout_page_text(&page, 800.0, 1000.0);

    let texts: Vec<&str> = paragraphs.iter().map(|p| p.text.as_str()).collect();
    assert_eq!(texts, ["one line", "and the next"]);
}

#[test]
fn paragraph_bboxes_are_normalized_to_the_page_and_cover_only_their_own_items() {
    let paragraphs = layout_page_text(&two_paragraph_page(), 800.0, 1000.0);
    assert_eq!(paragraphs.len(), 2);

    // The first paragraph occupies x 50..116 and y 90..114.
    let first = &paragraphs[0].bbox;
    assert_eq!(first.x, 50.0 / 800.0);
    assert_eq!(first.y, 90.0 / 1000.0);
    assert_eq!(first.w, 66.0 / 800.0);
    assert_eq!(first.h, 24.0 / 1000.0);

    let second = &paragraphs[1].bbox;
    assert_eq!(second.x, 50.0 / 800.0);
    assert_eq!(second.y, 130.0 / 1000.0);
    assert_eq!(second.w, 110.0 / 800.0);
    assert_eq!(second.h, 10.0 / 1000.0);
}

#[test]
fn items_arrive_in_any_order_but_read_top_to_bottom_left_to_right() {
    let shuffled = [
        item("second", 50.0, 114.0, 40.0),
        item("world", 100.0, 100.0, 40.0),
        item("hello", 50.0, 100.0, 40.0),
    ];

    let paragraphs = layout_page_text(&shuffled, 800.0, 1000.0);

    // Both lines are only 4 apart, so they belong to one paragraph, and the
    // order inside the line still follows x — and across lines, reading order.
    assert_eq!(paragraphs.len(), 1, "a 4pt gap is under the paragraph threshold");
    assert_eq!(paragraphs[0].text, "hello world second");
}

#[test]
fn blank_and_non_finite_items_are_ignored() {
    let page = [
        item("   ", 50.0, 100.0, 40.0),
        item("real", 50.0, 100.0, 40.0),
        PositionedTextItem {
            text: "bad".to_string(),
            left: f64::NAN,
            baseline: 100.0,
            width: 10.0,
            height: 10.0,
        },
    ];

    let paragraphs = layout_page_text(&page, 800.0, 1000.0);

    let texts: Vec<&str> = paragraphs.iter().map(|p| p.text.as_str()).collect();
    assert_eq!(texts, ["real"]);
}

#[test]
fn an_empty_page_produces_no_paragraphs() {
    assert!(layout_page_text(&[], 800.0, 1000.0).is_empty());
}

// --- block consumption -------------------------------------------------------

/// The canonical content of one page: paragraphs joined by a blank line, with
/// the paragraph offsets anchored to it — the same thing the PDF loader does
/// per page.
fn page_structure(paragraphs: &[(String, Option<NormalizedBox>)]) -> (String, Vec<PageInfo>) {
    let content = paragraphs
        .iter()
        .map(|(text, _)| text)
        .cloned()
        .collect::<Vec<_>>()
        .join("\n\n");

    let mut cursor = 0usize;
    let blocks: Vec<PageBlockInfo> = paragraphs
        .iter()
        .map(|(text, bbox)| {
            let start_offset = cursor;
            cursor += text.len() + 2;
            PageBlockInfo {
                text: text.clone(),
                start_offset,
                end_offset: start_offset + text.len(),
                bbox: *bbox,
            }
        })
        .collect();

    // Offsets in `blocks` must address the trimmed canonical content.
    let trimmed = content.trim().to_string();
    let shift = content.len() - content.trim_start().len();
    let blocks: Vec<PageBlockInfo> = blocks
        .into_iter()
        .map(|mut block| {
            block.start_offset -= shift;
            block.end_offset -= shift;
            block
        })
        .collect();

    let page = PageInfo {
        page_number: 1,
        content: trimmed.clone(),
        start_offset: 0,
        end_offset: trimmed.len(),
        bbox: None,
        blocks,
    };

    (trimmed, vec![page])
}

#[test]
fn a_multi_paragraph_page_becomes_multiple_blocks_in_reading_order() {
    let paragraphs = layout_page_text(&two_paragraph_page(), 800.0, 1000.0);
    let boxed: Vec<_> = paragraphs
        .iter()
        .map(|paragraph| (paragraph.text.clone(), Some(paragraph.bbox)))
        .collect();

    let (content, pages) = page_structure(&boxed);
    let structure = DocumentStructure::Pages { pages };
    let blocks = build_document_blocks(&content, Some(&structure));

    let texts: Vec<&str> = blocks.iter().map(|block| block.text.as_str()).collect();
    assert_eq!(texts, ["First line continues", "Second paragraph."]);
    assert!(blocks.iter().all(|block| block.page == Some(1)));

    // The whole provenance chain rests on this.
    for block in &blocks {
        assert_eq!(
            &content[block.start_offset..block.end_offset],
            block.text,
            "block {} does not slice back to its text",
            block.order
        );
    }
}

#[test]
fn a_page_without_paragraph_blocks_still_falls_back_to_one_page_level_block() {
    let content = "legacy page text";
    let structure = DocumentStructure::Pages {
        pages: vec![PageInfo {
            page_number: 1,
            content: content.to_string(),
            start_offset: 0,
            end_offset: content.len(),
            bbox: None,
            blocks: Vec::new(),
        }],
    };

    let blocks = build_document_blocks(&content, Some(&structure));

    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].text, content);
    assert_eq!(
        &content[blocks[0].start_offset..blocks[0].end_offset],
        blocks[0].text
    );
}
