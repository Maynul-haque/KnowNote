//! Differential test: the Rust Markdown loader against the TypeScript one.
//!
//! The fixture is the same one the chunker parity test uses. It is produced
//! by `scripts/gen-chunk-parity.mjs`, which runs the original TypeScript
//! loader over the eval corpus.
//!
//! What is compared is what the rest of the pipeline reads: the canonical
//! text, the title, and the section tree's shape, levels, titles and offsets.
//! The section *content* is not compared, because the original computes a
//! nested section's content by slicing the parent's content with the child's
//! absolute offsets, which is a mismatch of coordinate systems and comes back
//! empty. The offsets, which are the part that matters, are correct in both.

use std::path::PathBuf;

use knownote_core::structure::DocumentStructure;
use knownote_ingest::MarkdownLoader;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    #[serde(default)]
    loader: Option<LoaderFixture>,
}

#[derive(Debug, Deserialize)]
struct LoaderFixture {
    source: String,
    content: String,
    title: Option<String>,
    sections: Vec<SectionFixture>,
}

#[derive(Debug, Deserialize)]
struct SectionFixture {
    level: i64,
    title: String,
    #[serde(rename = "startOffset")]
    start_offset: usize,
    #[serde(default)]
    children: Vec<SectionFixture>,
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("knownote-core")
        .join("tests")
        .join("fixtures")
        .join("chunk-parity.json")
}

fn load_fixture() -> Fixture {
    let path = fixture_path();
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "cannot read {}: {error}\nRun: npm run chunk:parity",
            path.display()
        )
    });
    let fixture: Fixture = serde_json::from_str(&raw).expect("the parity fixture is valid JSON");
    assert!(
        !fixture.cases.is_empty(),
        "the parity fixture has no cases"
    );
    fixture
}

/// Compare two section trees.
fn compare_sections(
    actual: &[knownote_core::structure::SectionInfo],
    expected: &[SectionFixture],
    label: &str,
) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: the section count differs"
    );

    for (index, (actual_section, expected_section)) in
        actual.iter().zip(expected.iter()).enumerate()
    {
        assert_eq!(
            actual_section.level, expected_section.level,
            "{label}: section {index} has a different level"
        );
        assert_eq!(
            actual_section.title, expected_section.title,
            "{label}: section {index} has a different title"
        );
        assert_eq!(
            actual_section.start_offset, expected_section.start_offset,
            "{label}: section {index} starts elsewhere"
        );
        compare_sections(
            &actual_section.children,
            &expected_section.children,
            &format!("{label} > {}", expected_section.title),
        );
    }
}

#[test]
fn the_rust_markdown_loader_matches_the_typescript_one() {
    let fixture = load_fixture();
    let loader = MarkdownLoader::new();

    let mut compared = 0usize;
    let mut sections_compared = 0usize;

    for case in &fixture.cases {
        let Some(expected) = &case.loader else {
            continue;
        };
        compared += 1;

        let result = loader.parse(&expected.source);

        assert_eq!(
            result.content, expected.content,
            "{}: the canonical text differs",
            case.name
        );
        assert_eq!(
            result.title, expected.title,
            "{}: the title differs",
            case.name
        );

        let Some(DocumentStructure::Sections { sections }) = &result.structure else {
            panic!("{}: the loader produced no sections", case.name);
        };

        compare_sections(sections, &expected.sections, &case.name);
        sections_compared += expected.sections.len();
    }

    assert!(
        compared >= 20,
        "only {compared} documents were compared: the fixture is too small to mean anything"
    );
    assert!(
        sections_compared > 20,
        "only {sections_compared} top-level sections were compared"
    );
}

#[test]
fn a_heading_hierarchy_nests_the_way_the_original_nests_it() {
    let fixture = load_fixture();
    let loader = MarkdownLoader::new();

    // The first corpus document has a nested hierarchy; it is the one that
    // proves the tree is built the same way and not just flat.
    let Some(case) = fixture
        .cases
        .iter()
        .find(|case| case.loader.as_ref().is_some_and(|loader| {
            loader
                .sections
                .iter()
                .any(|section| !section.children.is_empty())
        }))
    else {
        panic!("the fixture has no document with a nested heading hierarchy");
    };

    let expected = case.loader.as_ref().expect("the case has a loader");
    let result = loader.parse(&expected.source);
    let Some(DocumentStructure::Sections { sections }) = &result.structure else {
        panic!("the loader produced no sections");
    };

    let nested = sections
        .iter()
        .find(|section| !section.children.is_empty())
        .expect("a section with children");

    assert!(
        !nested.children.is_empty(),
        "the nested section lost its children"
    );
    for child in &nested.children {
        assert!(
            child.level > nested.level,
            "a child is not deeper than its parent"
        );
        assert!(
            child.start_offset > nested.start_offset,
            "a child starts before its parent"
        );
    }
}
