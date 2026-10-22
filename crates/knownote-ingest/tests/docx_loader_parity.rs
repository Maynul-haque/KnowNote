//! Differential test: the Rust DOCX loader against the TypeScript one.
//!
//! The fixture is `crates/knownote-core/tests/fixtures/docx-loader.json`,
//! produced by `scripts/gen-docx-loader-fixture.mjs`.

use std::path::PathBuf;

use knownote_core::structure::DocumentStructure;
use knownote_ingest::DocxLoader;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    content: String,
    title: Option<String>,
    #[serde(rename = "mimeType")]
    mime_type: String,
    metadata: Option<serde_json::Value>,
    structure: Option<StructureFixture>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum StructureFixture {
    #[serde(rename = "sections")]
    Sections { sections: Vec<SectionFixture> },
}

#[derive(Debug, Deserialize)]
struct SectionFixture {
    level: i64,
    title: String,
    content: String,
    #[serde(rename = "startOffset")]
    start_offset: usize,
    #[serde(rename = "endOffset")]
    end_offset: usize,
    #[serde(default)]
    children: Vec<SectionFixture>,
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("knownote-core")
        .join("tests")
        .join("fixtures")
        .join("docx-loader.json")
}

fn fixture_doc(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("knownote-core")
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn load_fixture() -> Fixture {
    let path = fixture_path();
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!("cannot read {}: {error}", path.display())
    });
    serde_json::from_str(&raw).expect("the docx parity fixture is valid JSON")
}

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
            actual_section.content, expected_section.content,
            "{label}: section {index} has different content"
        );
        assert_eq!(
            actual_section.start_offset, expected_section.start_offset,
            "{label}: section {index} starts elsewhere"
        );
        assert_eq!(
            actual_section.end_offset, expected_section.end_offset,
            "{label}: section {index} ends elsewhere"
        );
        compare_sections(
            &actual_section.children,
            &expected_section.children,
            &format!("{label} > {}", expected_section.title),
        );
    }
}

#[test]
fn the_rust_docx_loader_matches_the_typescript_one() {
    let fixture = load_fixture();
    let loader = DocxLoader::new();

    let mut compared = 0usize;

    for case in &fixture.cases {
        compared += 1;

        let path = fixture_doc(&case.name);
        let raw = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));

        let result = loader.parse(&raw).unwrap_or_else(|error| panic!("parse error: {error}"));

        assert_eq!(
            result.content, case.content,
            "{}: the canonical text differs",
            case.name
        );
        assert_eq!(
            result.title, case.title,
            "{}: the title differs",
            case.name
        );
        assert_eq!(
            result.mime_type, case.mime_type,
            "{}: the mime type differs",
            case.name
        );
        assert_eq!(
            result.metadata, case.metadata,
            "{}: metadata differs",
            case.name
        );

        match (&result.structure, &case.structure) {
            (Some(DocumentStructure::Sections { sections: actual }), Some(StructureFixture::Sections { sections: expected })) => {
                compare_sections(&actual, &expected, &case.name);
            }
            (None, None) => {}
            _ => panic!("{}: structures differ", case.name),
        }
    }

    assert!(compared > 0, "no cases to compare");
}
