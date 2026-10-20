//! The PDF loader against its committed pin.
//!
//! `pdf.js` and PDFium are different extraction engines, so the Markdown
//! loader's mode of proof — byte-compare against the TypeScript original — is
//! not available here. What is available is two always-on gates and one engine
//! gate:
//!
//! * the pin fixture itself is checked without a PDFium anywhere: if it is
//!   stale or corrupt, every case fails loudly here;
//! * a malformed input produces a clean error, never a panic, in every
//!   environment;
//! * and when `KNOWNOTE_PDFIUM` points at a real PDFium, the loader must
//!   reproduce the committed pin byte for byte, offset for offset, box for
//!   box. That is the gate an extraction change, a pdfium update or a layout
//!   regression has to pass, and it runs in `cargo test --ignored`.

use std::path::PathBuf;

use knownote_core::structure::{DocumentStructure, NormalizedBox};
use knownote_ingest::{DocumentLoader, PdfLoader};
use serde::Deserialize;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn pin_path() -> PathBuf {
    repo_root()
        .join("crates")
        .join("knownote-core")
        .join("tests")
        .join("fixtures")
        .join("pdf-loader.json")
}

fn fixture_pdf(name: &str) -> PathBuf {
    repo_root().join("test").join("fixtures").join(name)
}

/// Whether two normalised boxes describe the same geometry.
///
/// The pin stores boxes as decimal text, and a decimal round-trip can move a
/// double by one ulp in its last digit (the parser's, not the geometry's). A
/// tolerance of 1e-12 in normalised space is far below anything a real
/// extraction, pdfium update or layout change can produce, and far above a
/// round-trip artifact, so this compares geometry and not the seventeenth
/// digit of a decimal string. Text and offsets stay exact.
fn box_matches(left: Option<&NormalizedBox>, right: Option<&NormalizedBox>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => {
            const TOLERANCE: f64 = 1e-12;
            (left.x - right.x).abs() <= TOLERANCE
                && (left.y - right.y).abs() <= TOLERANCE
                && (left.w - right.w).abs() <= TOLERANCE
                && (left.h - right.h).abs() <= TOLERANCE
        }
        _ => false,
    }
}

#[derive(Debug, Deserialize)]
struct Pin {
    cases: Vec<PinCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PinCase {
    name: String,
    content: String,
    title: Option<String>,
    #[serde(rename = "mimeType")]
    mime_type: String,
    metadata: Option<serde_json::Value>,
    #[allow(dead_code)]
    pages: Vec<knownote_core::structure::PageInfo>,
}

fn load_pin() -> Pin {
    let path = pin_path();
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}\nRun: cargo run -p knownote-ingest --example gen-pdf-loader-fixture", path.display()));
    let pin: Pin = serde_json::from_str(&raw).expect("the pdf-loader pin is valid JSON");
    assert!(!pin.cases.is_empty(), "the pdf-loader pin has no cases");
    pin
}

/// The contract the rest of the pipeline reads, asserted against the pin so a
/// broken or hand-edited pin fails without a PDFium in the room.
#[test]
fn the_pin_records_both_synthetic_fixtures_and_holds_the_contract() {
    let pin = load_pin();

    let names: Vec<&str> = pin.cases.iter().map(|case| case.name.as_str()).collect();
    assert_eq!(
        names,
        ["sample.pdf", "multipage.pdf"],
        "the pin covers the synthetic fixtures in generation order"
    );

    // The tiny fixture the smoke test uses: one page, one title block.
    let sample = &pin.cases[0];
    assert_eq!(sample.pages.len(), 1, "sample.pdf has one page");
    assert_eq!(
        sample.content, "KnowNote Import Test",
        "sample.pdf's canonical text"
    );
    assert_eq!(sample.mime_type, "application/pdf");
    assert_eq!(sample.metadata, Some(serde_json::json!({ "pageCount": 1 })));

    // The multi-page fixture: one block per page, offsets ascending, every
    // block slicing back out of the canonical text.
    let multipage = &pin.cases[1];
    assert_eq!(multipage.pages.len(), 3, "multipage.pdf has three pages");
    assert_eq!(
        multipage.content,
        "Page One Text\n\nPage Two Text\n\nPage Three Text",
        "pages join with a blank line"
    );
    assert_eq!(multipage.metadata, Some(serde_json::json!({ "pageCount": 3 })));

    let mut previous_page_start = 0usize;
    let mut block_count = 0usize;

    for case in &pin.cases {
        for (page_index, page) in case.pages.iter().enumerate() {
            assert_eq!(page.page_number, page_index as i64 + 1, "1-based pages");
            assert!(page.start_offset >= previous_page_start, "pages ascend");
            assert!(
                page.end_offset <= case.content.len(),
                "a page's range stays inside the canonical text"
            );
            previous_page_start = page.start_offset;

            let mut page_cursor = page.start_offset;
            for block in &page.blocks {
                assert_eq!(
                    &case.content[block.start_offset..block.end_offset],
                    block.text,
                    "{} page {}: the block is not the slice it claims",
                    case.name,
                    page.page_number
                );
                assert_eq!(
                    block.start_offset, page_cursor,
                    "{} page {}: the block starts where the cursor is",
                    case.name, page.page_number
                );
                page_cursor = block.end_offset + 2;

                let box_value = block
                    .bbox
                    .expect("a paragraph block carries its box");
                assert!(
                    (0.0..=1.0).contains(&box_value.x)
                        && (0.0..=1.0).contains(&box_value.y)
                        && (0.0..=1.0).contains(&(box_value.w))
                        && (0.0..=1.0).contains(&(box_value.h)),
                    "boxes normalise to 0..1"
                );
                block_count += 1;
            }
        }
    }

    assert!(
        block_count >= 4,
        "only {block_count} blocks in the pin: too small to mean anything"
    );
}

/// No PDFium, no panic: the loader answers with a clean error in any
/// environment, whether the library is absent or the bytes are not a PDF.
#[test]
fn a_malformed_pdf_errors_instead_of_panicking() {
    let result = PdfLoader::new().load_from_buffer(b"this is not a PDF");
    let error = result.expect_err("non-PDF bytes cannot produce a document");
    let rendered = error.to_string();
    assert!(!rendered.is_empty(), "the error says something");
}

/// The engine gate: with a real PDFium loaded, the loader reproduces the pin
/// exactly — every character, every offset, every box.
#[test]
#[ignore = "needs KNOWNOTE_PDFIUM"]
fn the_rust_pdf_loader_reproduces_the_committed_pin() {
    if std::env::var("KNOWNOTE_PDFIUM")
        .map(|value| value.trim().is_empty())
        .unwrap_or(true)
    {
        eprintln!(
            "skipped: set KNOWNOTE_PDFIUM to the directory holding the PDFium \
             library to run this test"
        );
        return;
    }

    let pin = load_pin();
    let loader = PdfLoader::new();

    let mut compared = 0usize;
    let mut blocks_compared = 0usize;

    for case in &pin.cases {
        let path = fixture_pdf(&case.name);
        let raw = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));

        let result = loader
            .load_from_buffer(&raw)
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));

        assert_eq!(
            result.content, case.content,
            "{}: the canonical text differs",
            case.name
        );
        assert_eq!(result.title, case.title, "{}: the title differs", case.name);
        assert_eq!(result.mime_type, case.mime_type, "{}: the mime differs", case.name);
        assert_eq!(
            result.metadata, case.metadata,
            "{}: the metadata differs",
            case.name
        );

        let Some(DocumentStructure::Pages { pages }) = &result.structure else {
            panic!("{}: the loader produced no pages", case.name);
        };

        assert_eq!(pages.len(), case.pages.len(), "{}: the page count differs", case.name);
        for (index, (page, expected)) in pages.iter().zip(case.pages.iter()).enumerate() {
            assert_eq!(page.page_number, expected.page_number, "{}: page {} has a different number", case.name, index + 1);
            assert_eq!(page.content, expected.content, "{}: page {} has different text", case.name, index + 1);
            assert_eq!(page.start_offset, expected.start_offset, "{}: page {} starts elsewhere", case.name, index + 1);
            assert_eq!(page.end_offset, expected.end_offset, "{}: page {} ends elsewhere", case.name, index + 1);

            if !box_matches(page.bbox.as_ref(), expected.bbox.as_ref()) {
                panic!(
                    "{}: page {} has a different box ({:?} vs {:?})",
                    case.name,
                    index + 1,
                    page.bbox,
                    expected.bbox
                );
            }

            assert_eq!(
                page.blocks.len(),
                expected.blocks.len(),
                "{}: page {} has a different block count",
                case.name,
                index + 1
            );
            for (block_index, (block, expected_block)) in
                page.blocks.iter().zip(expected.blocks.iter()).enumerate()
            {
                assert_eq!(
                    block.text, expected_block.text,
                    "{}: page {} block {block_index} has different text",
                    case.name,
                    index + 1
                );
                assert_eq!(
                    block.start_offset, expected_block.start_offset,
                    "{}: page {} block {block_index} starts elsewhere",
                    case.name,
                    index + 1
                );
                assert_eq!(
                    block.end_offset, expected_block.end_offset,
                    "{}: page {} block {block_index} ends elsewhere",
                    case.name,
                    index + 1
                );
                if !box_matches(block.bbox.as_ref(), expected_block.bbox.as_ref()) {
                    panic!(
                        "{}: page {} block {block_index} has a different box",
                        case.name,
                        index + 1
                    );
                }
                blocks_compared += 1;
            }
        }

        // The invariant again, on live output rather than on the pin.
        for block in pages.iter().flat_map(|page| &page.blocks) {
            assert_eq!(
                &result.content[block.start_offset..block.end_offset],
                block.text
            );
        }

        compared += 1;
    }

    assert!(
        compared >= 2,
        "only {compared} documents were compared: the pin is too small to mean anything"
    );
    assert!(
        blocks_compared >= 4,
        "only {blocks_compared} blocks were compared"
    );
}
