//! Writes `crates/knownote-core/tests/fixtures/pdf-loader.json`: what the Rust
//! PDF loader produced for the synthetic PDF fixtures.
//!
//! This is the *pin* for the loader: `cargo test --ignored --features-safe` —
//! specifically the `pdf_loader_parity` test in this crate, run with
//! `KNOWNOTE_PDFIUM` pointing at a directory holding the PDFium library —
//! asserts the loader still produces exactly this. A pdfium update, an
//! extraction change or a layout regression that moves one character, offset
//! or box shows up as a diff of one line.
//!
//! The chunk parity generator (`scripts/gen-chunk-parity.mjs`) also reads this
//! file to build the PDF cases of `chunk-parity.json`, where the TypeScript
//! chunker runs over these very blocks.
//!
//! Usage:
//!   $env:KNOWNOTE_PDFIUM = "<directory containing pdfium.dll>"
//!   cargo run -p knownote-ingest --example gen-pdf-loader-fixture
//!
//! An optional argument overrides the library directory for this run without
//! touching the environment.

use std::path::PathBuf;

use knownote_ingest::{DocumentLoader, PdfLoader};

fn main() {
    let fixtures_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("test").join("fixtures");

    let names: &[&str] = &["sample.pdf", "multipage.pdf"];

    let loader = PdfLoader::new();
    let mut cases: Vec<String> = Vec::new();

    for name in names {
        let path = fixtures_dir.join(name);
        let raw = std::fs::read(&path).unwrap_or_else(|error| {
            panic!("cannot read {}: {error}", path.display())
        });

        let result = loader
            .load_from_buffer(&raw)
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));

        let Some(knownote_core::structure::DocumentStructure::Pages { pages }) =
            &result.structure
        else {
            panic!("{}: the loader produced no pages", path.display());
        };

        let case = serde_json::json!({
            "name": name,
            // The canonical text, in the byte offsets the loader stores.
            "content": result.content,
            "title": result.title,
            "mimeType": result.mime_type,
            "metadata": result.metadata,
            "pages": pages,
        });

        cases.push(serde_json::to_string(&case).expect("the case serialises"));

        // A tiny proof of life on stdout: the contract the rest of the
        // pipeline depends on most.
        println!(
            "{}: {} bytes -> {} chars, {} pages, title={:?}",
            name,
            raw.len(),
            result.content.chars().count(),
            pages.len(),
            result.title,
        );
        for page in pages {
            println!(
                "  page {}: {} blocks: {:?}",
                page.page_number,
                page.blocks.len(),
                page
                    .blocks
                    .iter()
                    .map(|block| block.text.as_str())
                    .collect::<Vec<_>>()
            );
        }
    }

    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("knownote-core")
        .join("tests")
        .join("fixtures")
        .join("pdf-loader.json");

    // One case per line: a regenerated fixture is reviewed as a whole
    // document, but a change lands in exactly one line.
    let generated_by = "knownote-ingest/examples/gen-pdf-loader-fixture.rs (pdfium)";
    let note = "The Rust PDF loader's own output for the synthetic PDF fixtures. Offsets \
                are byte offsets into `content`; the boxes are normalised to the page. \
                pdf.js and Pdfium are different extraction engines, so this is the engine \
                gate for the loader, not a pdf.js comparison - the contract tests are in \
                tests/pdf_layout.rs.";

    let mut body = String::new();
    body.push_str("{\n");
    body.push_str(&format!(
        "  \"generatedBy\": {},\n",
        serde_json::to_string(generated_by).unwrap()
    ));
    body.push_str(&format!(
        "  \"note\": {},\n",
        serde_json::to_string(note).unwrap()
    ));
    body.push_str("  \"cases\": [\n");
    for (index, case) in cases.iter().enumerate() {
        if index > 0 {
            body.push_str(",\n");
        }
        body.push(' ');
        body.push_str(case);
    }
    body.push_str("\n  ]\n}\n");

    std::fs::write(&out, body).expect("the fixture writes");
    println!("wrote {}", out.display());
}
