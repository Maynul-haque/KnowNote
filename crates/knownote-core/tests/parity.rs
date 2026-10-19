//! Differential test: the Rust chunker against the TypeScript one.
//!
//! The fixture in `tests/fixtures/chunk-parity.json` is produced by
//! `scripts/gen-chunk-parity.mjs`, which runs the original TypeScript
//! implementation over the real eval corpus and a set of awkward inputs. This
//! test replays it and asserts the port produces the same result.
//!
//! This is the gate that makes the port trustworthy. A chunker that is
//! *nearly* right still satisfies its own contract while answering a
//! different question, and only a comparison against the reference catches
//! that. The committed eval baseline is the second gate; this one is the
//! first, and it is cheap enough to run on every change.

use std::collections::BTreeMap;
use std::path::PathBuf;

use knownote_core::structure::{DocumentStructure, SectionInfo};
use knownote_core::text::Utf16Index;
use knownote_core::types::BlockKind;
use knownote_core::{build_document_blocks, ChunkBlock, ChunkOptions, ChunkingService};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    content: String,
    #[serde(default)]
    loader: Option<LoaderFixture>,
    #[serde(default)]
    blocks: Vec<BlockFixture>,
    runs: Vec<Run>,
}

/// What the TypeScript loader produced.
#[derive(Debug, Deserialize)]
struct LoaderFixture {
    /// The raw document. The loader test in `knownote-ingest` reads this; the
    /// chunker and block tests do not need it.
    #[allow(dead_code)]
    source: String,
    /// The canonical text. The loader test in `knownote-ingest` reads this.
    #[allow(dead_code)]
    content: String,
    /// The document title. The loader test in `knownote-ingest` reads this.
    #[allow(dead_code)]
    title: Option<String>,
    sections: Vec<SectionFixture>,
}

/// One section of the document structure.
#[derive(Debug, Deserialize)]
struct SectionFixture {
    level: i64,
    title: String,
    /// The UTF-16 offset, for the loader test.
    #[allow(dead_code)]
    #[serde(rename = "startOffsetUtf16")]
    start_offset_utf16: usize,
    #[serde(rename = "startOffset")]
    start_offset: usize,
    #[serde(default)]
    children: Vec<SectionFixture>,
}

#[derive(Debug, Deserialize)]
struct BlockFixture {
    id: String,
    kind: String,
    page: Option<i64>,
    level: Option<i64>,
    text: String,
    #[serde(rename = "startOffset")]
    start_offset: usize,
    #[serde(rename = "endOffset")]
    end_offset: usize,
}

#[derive(Debug, Deserialize)]
struct Run {
    mode: String,
    name: String,
    options: OptionsFixture,
    chunks: Vec<ChunkFixture>,
}

#[derive(Debug, Default, Deserialize)]
struct OptionsFixture {
    #[serde(rename = "chunkSize")]
    chunk_size: Option<usize>,
    #[serde(rename = "chunkOverlap")]
    chunk_overlap: Option<usize>,
    #[serde(rename = "minChunkSize")]
    min_chunk_size: Option<usize>,
    #[serde(rename = "allowSpanPages")]
    allow_span_pages: Option<bool>,
    #[serde(rename = "respectHeadings")]
    respect_headings: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ChunkFixture {
    index: usize,
    content: String,
    #[serde(rename = "startOffsetUtf16")]
    start_offset_utf16: usize,
    #[serde(rename = "endOffsetUtf16")]
    end_offset_utf16: usize,
    #[serde(rename = "startOffset")]
    start_offset: usize,
    #[serde(rename = "endOffset")]
    end_offset: usize,
    #[serde(rename = "tokenCount")]
    token_count: usize,
    #[serde(rename = "pageStart")]
    page_start: Option<i64>,
    #[serde(rename = "pageEnd")]
    page_end: Option<i64>,
    #[serde(rename = "blockSpans")]
    block_spans: Vec<SpanFixture>,
}

#[derive(Debug, Deserialize)]
struct SpanFixture {
    #[serde(rename = "blockId")]
    block_id: String,
    #[serde(rename = "startInBlockUtf16")]
    start_in_block_utf16: usize,
    #[serde(rename = "endInBlockUtf16")]
    end_in_block_utf16: usize,
    #[serde(rename = "startInBlock")]
    start_in_block: usize,
    #[serde(rename = "endInBlock")]
    end_in_block: usize,
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("chunk-parity.json")
}

fn parse_kind(raw: &str) -> BlockKind {
    match raw {
        "heading" => BlockKind::Heading,
        "paragraph" => BlockKind::Paragraph,
        "list_item" => BlockKind::ListItem,
        "table" => BlockKind::Table,
        "figure" => BlockKind::Figure,
        "caption" => BlockKind::Caption,
        "code" => BlockKind::Code,
        other => panic!("unknown block kind in the fixture: {other}"),
    }
}

fn to_options(fixture: &OptionsFixture) -> ChunkOptions {
    ChunkOptions {
        chunk_size: fixture.chunk_size,
        chunk_overlap: fixture.chunk_overlap,
        separators: None,
        min_chunk_size: fixture.min_chunk_size,
        allow_span_pages: fixture.allow_span_pages,
        respect_headings: fixture.respect_headings,
    }
}

/// The fixture must exist, and it must not be empty: a parity test that
/// silently compares nothing is worse than no test at all.
fn load_fixture() -> Fixture {
    let path = fixture_path();
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "cannot read {}: {error}\nRun: node --experimental-transform-types --import ./test/ts-resolve.mjs scripts/gen-chunk-parity.mjs",
            path.display()
        )
    });

    let fixture: Fixture = serde_json::from_str(&raw).expect("the parity fixture is valid JSON");
    assert!(
        !fixture.cases.is_empty(),
        "the parity fixture has no cases: it was probably generated by a broken script"
    );
    fixture
}

#[test]
fn the_rust_chunker_matches_the_typescript_chunker() {
    let fixture = load_fixture();
    let service = ChunkingService::new();

    let mut total_runs = 0usize;
    let mut total_chunks = 0usize;

    for case in &fixture.cases {
        let blocks: Vec<ChunkBlock> = case
            .blocks
            .iter()
            .map(|block| ChunkBlock {
                id: block.id.clone(),
                kind: parse_kind(&block.kind),
                page: block.page,
                level: block.level,
                text: block.text.clone(),
                start_offset: block.start_offset,
                end_offset: block.end_offset,
            })
            .collect();

        // The block texts must be the exact slice of the document they claim.
        for block in &blocks {
            assert_eq!(
                &case.content[block.start_offset..block.end_offset],
                block.text,
                "case {}: block {} is not the slice of the document it claims",
                case.name,
                block.id
            );
        }

        let index = Utf16Index::new(&case.content);

        let block_texts: std::collections::HashMap<String, String> = case
            .blocks
            .iter()
            .map(|block| (block.id.clone(), block.text.clone()))
            .collect();

        for run in &case.runs {
            total_runs += 1;

            let options = to_options(&run.options);
            let chunks = match run.mode.as_str() {
                "blocks" => service.chunk_blocks(&case.content, &blocks, Some(&options)),
                "fallback" => service.chunk(&case.content, Some(&options)),
                other => panic!("unknown run mode in the fixture: {other}"),
            };

            let label = format!("{} / {} / {}", case.name, run.mode, run.name);

            assert_eq!(
                chunks.len(),
                run.chunks.len(),
                "{label}: chunk count differs"
            );

            for (actual, expected) in chunks.iter().zip(run.chunks.iter()) {
                total_chunks += 1;

                assert_eq!(actual.index, expected.index, "{label}: chunk index differs");
                assert_eq!(
                    actual.content, expected.content,
                    "{label}: chunk {} content differs",
                    actual.index
                );
                assert_eq!(
                    actual.start_offset, expected.start_offset,
                    "{label}: chunk {} start offset differs",
                    actual.index
                );
                assert_eq!(
                    actual.end_offset, expected.end_offset,
                    "{label}: chunk {} end offset differs",
                    actual.index
                );
                assert_eq!(
                    actual.token_count, expected.token_count,
                    "{label}: chunk {} token estimate differs",
                    actual.index
                );
                assert_eq!(
                    actual.page_start, expected.page_start,
                    "{label}: chunk {} page start differs",
                    actual.index
                );
                assert_eq!(
                    actual.page_end, expected.page_end,
                    "{label}: chunk {} page end differs",
                    actual.index
                );

                // The byte offsets and the UTF-16 offsets must describe the
                // same position. This is the invariant that lets the database
                // change unit without moving a citation.
                assert_eq!(
                    index.utf16_of_byte(actual.start_offset),
                    expected.start_offset_utf16,
                    "{label}: chunk {} byte and UTF-16 start offsets disagree",
                    actual.index
                );
                assert_eq!(
                    index.utf16_of_byte(actual.end_offset),
                    expected.end_offset_utf16,
                    "{label}: chunk {} byte and UTF-16 end offsets disagree",
                    actual.index
                );

                assert_eq!(
                    actual.block_spans.len(),
                    expected.block_spans.len(),
                    "{label}: chunk {} block span count differs",
                    actual.index
                );

                for (actual_span, expected_span) in
                    actual.block_spans.iter().zip(expected.block_spans.iter())
                {
                    assert_eq!(
                        actual_span.block_id, expected_span.block_id,
                        "{label}: chunk {} block span id differs",
                        actual.index
                    );
                    assert_eq!(
                        actual_span.start_in_block, expected_span.start_in_block,
                        "{label}: chunk {} span of {} starts elsewhere",
                        actual.index,
                        expected_span.block_id
                    );
                    assert_eq!(
                        actual_span.end_in_block, expected_span.end_in_block,
                        "{label}: chunk {} span of {} ends elsewhere",
                        actual.index,
                        expected_span.block_id
                    );
                    // The byte offset and the UTF-16 offset must point at the
                    // same character. They are different numbers whenever the
                    // block holds a character outside the Basic Multilingual
                    // Plane, so the check is a conversion rather than an
                    // equality.
                    let block_text = block_texts
                        .get(&expected_span.block_id)
                        .expect("the span points at a block the fixture knows");
                    let block_index = Utf16Index::new(block_text);
                    assert_eq!(
                        block_index.utf16_of_byte(expected_span.start_in_block),
                        expected_span.start_in_block_utf16,
                        "{label}: span of {} starts at a different character",
                        expected_span.block_id
                    );
                    assert_eq!(
                        block_index.utf16_of_byte(expected_span.end_in_block),
                        expected_span.end_in_block_utf16,
                        "{label}: span of {} ends at a different character",
                        expected_span.block_id
                    );
                }
            }
        }
    }

    assert!(total_runs > 0, "no runs were compared");
    assert!(
        total_chunks > 100,
        "only {total_chunks} chunks were compared: the fixture is too small to mean anything"
    );
    println!(
        "[parity] compared {total_chunks} chunks across {total_runs} runs in {} cases",
        fixture.cases.len()
    );
}

/// The block builder has to produce the same blocks as the original, not just
/// consume them.
///
/// The chunker test above feeds the *original's* blocks to the Rust chunker,
/// so it proves the chunker is exact given the right blocks. It does not prove
/// the Rust block builder produces those blocks — and the blocks are what a
/// citation resolves to, so a difference there moves the ground truth and
/// changes the retrieval.
#[test]
fn the_rust_block_builder_matches_the_typescript_one() {
    let fixture = load_fixture();
    let mut compared = 0usize;
    let mut blocks_compared = 0usize;

    for case in &fixture.cases {
        let Some(loader) = &case.loader else {
            continue;
        };
        compared += 1;

        let structure = DocumentStructure::Sections {
            sections: loader
                .sections
                .iter()
                .map(to_section)
                .collect(),
        };

        let drafts = build_document_blocks(&case.content, Some(&structure));

        assert_eq!(
            drafts.len(),
            case.blocks.len(),
            "{}: the block count differs",
            case.name
        );

        for (index, (actual, expected)) in drafts.iter().zip(case.blocks.iter()).enumerate() {
            assert_eq!(
                actual.kind.as_str(),
                expected.kind,
                "{}: block {index} has a different kind",
                case.name
            );
            assert_eq!(
                actual.page, expected.page,
                "{}: block {index} is on a different page",
                case.name
            );
            assert_eq!(
                actual.level, expected.level,
                "{}: block {index} has a different level",
                case.name
            );
            assert_eq!(
                actual.text, expected.text,
                "{}: block {index} has different text",
                case.name
            );
            assert_eq!(
                actual.start_offset, expected.start_offset,
                "{}: block {index} starts elsewhere",
                case.name
            );
            assert_eq!(
                actual.end_offset, expected.end_offset,
                "{}: block {index} ends elsewhere",
                case.name
            );
            blocks_compared += 1;
        }
    }

    assert!(
        compared >= 20,
        "only {compared} documents were compared"
    );
    assert!(
        blocks_compared > 100,
        "only {blocks_compared} blocks were compared"
    );
}

/// Rebuild a section from the fixture.
fn to_section(fixture: &SectionFixture) -> SectionInfo {
    SectionInfo {
        level: fixture.level,
        title: fixture.title.clone(),
        content: String::new(),
        start_offset: fixture.start_offset,
        end_offset: fixture.start_offset,
        children: fixture
            .children
            .iter()
            .map(to_section)
            .collect(),
    }
}

/// The fixture must cover the shapes that are easy to get wrong, or a green
/// run says very little.
#[test]
fn the_fixture_covers_the_shapes_that_are_easy_to_get_wrong() {
    let fixture = load_fixture();

    let names: BTreeMap<&str, ()> = fixture
        .cases
        .iter()
        .map(|case| (case.name.as_str(), ()))
        .collect();

    for required in [
        "synthetic/cjk",
        "synthetic/astral",
        "synthetic/code-and-table",
        "synthetic/empty",
        "synthetic/whitespace-only",
        "synthetic/repeated-blank-lines",
    ] {
        assert!(
            names.contains_key(required),
            "the parity fixture is missing the case {required}"
        );
    }

    // The corpus is what the eval baseline was measured on, so it has to be
    // part of the comparison.
    let corpus_cases = fixture
        .cases
        .iter()
        .filter(|case| case.name.starts_with("corpus/"))
        .count();
    assert!(
        corpus_cases >= 20,
        "only {corpus_cases} corpus documents are in the fixture"
    );

    // Both chunking paths have to be exercised.
    let modes: BTreeMap<&str, ()> = fixture
        .cases
        .iter()
        .flat_map(|case| case.runs.iter())
        .map(|run| (run.mode.as_str(), ()))
        .collect();
    assert!(modes.contains_key("blocks"), "the block path was not compared");
    assert!(modes.contains_key("fallback"), "the fallback path was not compared");
}
