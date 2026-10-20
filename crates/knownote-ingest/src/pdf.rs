//! The PDF loader.
//!
//! A port of `src/main/services/loaders/PdfLoader.ts` and
//! `src/main/services/loaders/pdfTextLayout.ts`, running on the real PDFium
//! through `pdfium-render` instead of pdf.js.
//!
//! # One engine, the same contract
//!
//! The original extracts text with pdf.js and this port extracts it with
//! Pdfium. Those are different engines, and a byte-identical extraction is
//! not the bar: what has to hold — and what the tests hold — is the
//! *contract*, every part of it:
//!
//! * `content` is the page contents joined by a blank line, trimmed, and
//!   every offset in it is a **byte** offset into that string. A block's
//!   text is always exactly `content[start_offset..end_offset]`.
//! * `layout_page_text` reproduces the original's geometry algorithm — the
//!   same constants, the same rules — so lines and paragraphs come out the
//!   way the citation layer expects them. `test/pdfTextLayout.test.ts` pins
//!   it with synthetic items in the original; `crates/knownote-ingest/tests/pdf_layout.rs`
//!   ports those tests item for item, engine-free.
//! * The committed pin fixture
//!   (`crates/knownote-core/tests/fixtures/pdf-loader.json`) records what
//!   this loader produced for the synthetic PDF fixtures, so an extraction,
//!   a pdfium update or a layout regression that changes any character,
//!   offset or box is caught by `cargo test --ignored` with `KNOWNOTE_PDFIUM`
//!   set. That is the honest differential gate for PDF: pitting this port
//!   against pdf.js would only prove the engines differ.
//!
//! The loader binds to a prebuilt PDFium (bblanchon/pdfium-binaries,
//! Apache-2.0) at runtime: the library is never compiled here and never ships
//! inside this crate.
//!
//! `PageInfo.metadata` (page width/height) of the original is deliberately
//! not carried over: nothing in the pipeline reads it (the block builder
//! reads `page.blocks` only), and the box is already normalised.

use std::env;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use knownote_core::structure::{
    DocumentStructure, NormalizedBox, PageBlockInfo, PageInfo,
};
use knownote_core::text::is_js_whitespace;
use pdfium_render::prelude::*;

use crate::loader::{DocumentLoadResult, DocumentLoader, IngestError};

/// The same tolerance the original groups baseline by, relative to font size.
const LINE_TOLERANCE: f64 = 0.5;
/// A vertical gap larger than this many line heights starts a paragraph.
const PARAGRAPH_GAP: f64 = 0.6;
/// A first-line indent larger than this many font sizes starts a paragraph.
const INDENT_RATIO: f64 = 1.0;
/// A gap between items larger than this many font sizes is a word space.
const SPACE_GAP: f64 = 0.25;

pub const PDF_MIME_TYPE: &str = "application/pdf";
pub const PDF_EXTENSION: &str = "pdf";

/// The name of the environment variable that points at the directory holding
/// the PDFium library (`pdfium.dll`, `libpdfium.so`, ...).
pub const PDFIUM_DIR_ENV: &str = "KNOWNOTE_PDFIUM";

// ---------------------------------------------------------------------------
// The pdfium binding, resolved once per process.
// ---------------------------------------------------------------------------

static PDFIUM: OnceLock<Option<Pdfium>> = OnceLock::new();
static PDFIUM_ERROR: OnceLock<String> = OnceLock::new();

/// The bound PDFium library, resolving it on first use.
fn pdfium() -> Result<&'static Pdfium, IngestError> {
    let resolved = PDFIUM.get_or_init(|| match try_to_bind_pdfium() {
        Ok(bindings) => Some(Pdfium::new(bindings)),
        Err(description) => {
            let _ = PDFIUM_ERROR.get_or_init(|| description);
            None
        }
    });

    match resolved {
        Some(pdfium) => Ok(pdfium),
        None => Err(IngestError::Parse(
            PDFIUM_ERROR
                .get()
                .cloned()
                .unwrap_or_else(|| "the PDFium library is not available".to_string()),
        )),
    }
}

/// Bind to the first PDFium that loads, or record why none did.
///
/// The order is the deployment order: an explicit directory override, then
/// the library the release bundles next to the executable, then whatever the
/// system library loader can find. All resolution happens once: `OnceLock`
/// also keeps a half-initialised PDFium process state out of a race.
fn try_to_bind_pdfium() -> Result<Box<dyn PdfiumLibraryBindings>, String> {
    let library_name = Pdfium::pdfium_platform_library_name();

    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Some(directory) = env::var(PDFIUM_DIR_ENV)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    {
        candidates.push(Path::new(&directory).join(&library_name));
    }

    if let Ok(executable) = env::current_exe() {
        if let Some(directory) = executable.parent() {
            candidates.push(directory.join(&library_name));
        }
    }

    let mut tried: Vec<String> = Vec::new();

    for path in &candidates {
        match Pdfium::bind_to_library(path) {
            Ok(bindings) => return Ok(bindings),
            Err(error) => tried.push(format!("{}: {error}", path.display())),
        }
    }

    match Pdfium::bind_to_system_library() {
        Ok(bindings) => Ok(bindings),
        Err(error) => {
            let looked = candidates
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let tried = format_error_history(&tried);
            let hint = library_name.to_string_lossy();

            Err(format!(
                "the PDFium library is not available: looked in {looked} and in \
                 the system library paths; set {PDFIUM_DIR_ENV} to the directory \
                 that holds {hint}. {tried} last error: {error}"
            ))
        }
    }
}

fn format_error_history(tried: &[String]) -> String {
    match tried {
        [] => "Nothing had to be tried from the explicit or built-in locations;".to_string(),
        entries => format!("These loads failed: [{}].", entries.join("], [")),
    }
}

/// Reads PDF documents.
pub struct PdfLoader;

impl PdfLoader {
    /// A new loader.
    pub fn new() -> Self {
        PdfLoader
    }

    /// Extracts the text of one PDF into the canonical object shape.
    pub fn extract(&self, buffer: &[u8]) -> Result<DocumentLoadResult, IngestError> {
        let pdfium = pdfium()?;
        let document = pdfium
            .load_pdf_from_byte_vec(buffer.to_vec(), None)
            .map_err(|error| IngestError::Parse(format!("failed to parse PDF: {error}")))?;

        let metadata = document.metadata();

        // `|| undefined` in the original: an empty title is no title.
        let title = metadata
            .get(PdfDocumentMetadataTagType::Title)
            .map(|tag| tag.value().to_string())
            .filter(|value| !value.is_empty());

        let mut info = serde_json::Map::new();
        info.insert(
            "pageCount".to_string(),
            serde_json::json!(document.pages().len() as i64),
        );
        for (key, tag) in [
            ("author", PdfDocumentMetadataTagType::Author),
            ("creator", PdfDocumentMetadataTagType::Creator),
            ("producer", PdfDocumentMetadataTagType::Producer),
            ("creationDate", PdfDocumentMetadataTagType::CreationDate),
            ("subject", PdfDocumentMetadataTagType::Subject),
            ("keywords", PdfDocumentMetadataTagType::Keywords),
        ] {
            if let Some(value) = metadata.get(tag) {
                info.insert(key.to_string(), serde_json::json!(value.value()));
            }
        }

        let mut pages: Vec<PageInfo> = Vec::new();
        let mut full_text = String::new();
        let mut current_offset = 0usize;

        // `PdfPage` does not hand out its own index, so the loop counts.
        let mut page_index = 0usize;

        for page in document.pages().iter() {
            let page_number = (page_index + 1) as i64;
            page_index += 1;

            let page_width = page.width().value as f64;
            let page_height = page.height().value as f64;

            let items = page_items(&page)?;
            let paragraphs = layout_page_text(&items, page_width, page_height);

            // The page content is the paragraphs joined by a blank line, and
            // the paragraph offsets are anchored to that string; the anchors
            // are then shifted into the whole document's coordinate space.
            let mut page_content = String::new();
            let mut page_blocks: Vec<PageBlockInfo> = Vec::new();
            let mut page_cursor = 0usize;

            for paragraph in &paragraphs {
                if !page_content.is_empty() {
                    page_content.push_str("\n\n");
                    page_cursor += 2;
                }

                let text = paragraph.text.clone();
                let start_offset = current_offset + page_cursor;
                page_blocks.push(PageBlockInfo {
                    text: text.clone(),
                    start_offset,
                    end_offset: start_offset + text.len(),
                    bbox: Some(paragraph.bbox),
                });

                page_content.push_str(&text);
                page_cursor += text.len();
            }

            let bounding_boxes: Vec<NormalizedBox> =
                paragraphs.iter().map(|paragraph| paragraph.bbox).collect();

            pages.push(PageInfo {
                page_number,
                content: page_content.clone(),
                start_offset: current_offset,
                end_offset: current_offset + page_content.len(),
                bbox: union_bbox(&bounding_boxes),
                blocks: page_blocks,
            });

            let page_text_with_newline = page_content + "\n\n";
            full_text.push_str(&page_text_with_newline);
            current_offset += page_text_with_newline.len();
        }

        Ok(DocumentLoadResult {
            // The same `.trim()` the original applies; it only ever removes
            // the final blank line, so the page offsets keep pointing at the
            // right characters.
            content: full_text.trim().to_string(),
            title,
            mime_type: PDF_MIME_TYPE.to_string(),
            structure: Some(DocumentStructure::Pages { pages }),
            metadata: Some(serde_json::Value::Object(info)),
        })
    }
}

impl Default for PdfLoader {
    fn default() -> Self {
        PdfLoader::new()
    }
}

impl DocumentLoader for PdfLoader {
    fn supported_mime_types(&self) -> &'static [&'static str] {
        &[PDF_MIME_TYPE]
    }

    fn supported_extensions(&self) -> &'static [&'static str] {
        &[PDF_EXTENSION]
    }

    fn load_from_path(&self, path: &Path) -> Result<DocumentLoadResult, IngestError> {
        let buffer = std::fs::read(path)
            .map_err(|error| IngestError::Io { path: path.display().to_string(), source: error })?;
        self.extract(&buffer)
    }

    fn load_from_buffer(&self, buffer: &[u8]) -> Result<DocumentLoadResult, IngestError> {
        self.extract(buffer)
    }
}

// ---------------------------------------------------------------------------
// Extraction: one page's text objects, in the shape the layout algorithm eats.
// ---------------------------------------------------------------------------

/// A text item converted to viewport coordinates (origin top left, y down).
///
/// The same shape the original's `toPositionedItems()` produces, with the
/// item's text the only string component.
pub struct PositionedTextItem {
    /// The item's text.
    pub text: String,
    /// The text's left edge.
    pub left: f64,
    /// The text's baseline, measured down from the top of the page.
    pub baseline: f64,
    /// The text's width.
    pub width: f64,
    /// The font size, in the same units as the page geometry.
    pub height: f64,
}

/// A paragraph a page's text items collapse into.
pub struct PageTextParagraph {
    pub text: String,
    pub bbox: NormalizedBox,
}

/// The text objects of one page, in layout order.
fn page_items(page: &PdfPage<'_>) -> Result<Vec<PositionedTextItem>, IngestError> {
    let page_height = page.height().value as f64;

    // Open the text page once and keep it open: `for_object()` re-uses it
    // instead of loading the page again for every object.
    let text_page = page
        .text()
        .map_err(|error| IngestError::Parse(format!("failed to read the PDF text: {error}")))?;

    let mut items = Vec::new();

    for object in page.objects().iter() {
        let Some(text_object) = object.as_text_object() else {
            continue;
        };

        let text = text_page.for_object(text_object);
        // `if (!item.str) continue` in the original.
        if text.is_empty() {
            continue;
        }

        // The text object's matrix carries what pdf.js' item transform carried:
        // translation e/f is the pen origin — the left edge and the baseline —
        // and the bounds give the advance width. The baseline is flipped into
        // viewport coordinates, because the layout algorithm is written
        // top-down.
        let left = text_object.get_horizontal_translation().value as f64;
        let baseline = page_height - text_object.get_vertical_translation().value as f64;
        let width = text_object
            .bounds()
            .map(|bounds| (bounds.width().value as f64).abs())
            .unwrap_or(0.0);

        // `fontHeight > 0 ? fontHeight : 1` in the original.
        let mut height = text_object.scaled_font_size().value as f64;
        if !height.is_finite() || height <= 0.0 {
            height = 1.0;
        }

        items.push(PositionedTextItem {
            text,
            left,
            baseline,
            width,
            height,
        });
    }

    Ok(items)
}

// ---------------------------------------------------------------------------
// The layout algorithm: a port of pdfTextLayout.ts.
// ---------------------------------------------------------------------------

/// Groups a page's text items into lines and the lines into paragraphs.
///
/// Order is reading order: lines run top to bottom, items in a line run left
/// to right. A page with no usable text produces no paragraphs.
pub fn layout_page_text(
    items: &[PositionedTextItem],
    page_width: f64,
    page_height: f64,
) -> Vec<PageTextParagraph> {
    let lines = group_lines(items);
    let paragraphs = group_paragraphs(&lines);

    paragraphs
        .into_iter()
        .map(|paragraph| PageTextParagraph {
            text: paragraph.text,
            bbox: normalize_box(&paragraph.rect, page_width, page_height),
        })
        .collect()
}

/// The unit one line is drafted from, before its text is rendered.
struct LineDraft {
    items: Vec<PositionedTextItem>,
    baseline: f64,
    height: f64,
    left: f64,
    right: f64,
    top: f64,
    bottom: f64,
}

/// A line of text on a page.
struct TextLine {
    text: String,
    left: f64,
    right: f64,
    top: f64,
    bottom: f64,
    height: f64,
}

/// Groups items that share a baseline into lines, and joins the items of one
/// line from left to right.
fn group_lines(items: &[PositionedTextItem]) -> Vec<TextLine> {
    // Whitespace-only items and items with unmeasurable geometry never take
    // part in a line. (`trim().length > 0` and `Number.isFinite`,
    // respectively.)
    let usable: Vec<&PositionedTextItem> = items
        .iter()
        .filter(|item| {
            item.text.chars().any(|c| !is_js_whitespace(c))
                && item.left.is_finite()
                && item.baseline.is_finite()
                && item.width.is_finite()
                && item.height.is_finite()
        })
        .collect();

    let mut sorted: Vec<&PositionedTextItem> = usable;
    sorted.sort_by(|a, b| {
        a.baseline
            .total_cmp(&b.baseline)
            .then_with(|| a.left.total_cmp(&b.left))
    });

    let mut drafts: Vec<LineDraft> = Vec::new();

    for item in sorted {
        let same_line = drafts.last().is_some_and(|last: &LineDraft| {
            let tolerance = (last.height.min(item.height) * LINE_TOLERANCE).max(1.0);
            (item.baseline - last.baseline).abs() <= tolerance
        });

        if same_line {
            let last = drafts
                .last_mut()
                .expect("a line draft exists when the item continues one");

            last.items.push(PositionedTextItem {
                text: item.text.clone(),
                left: item.left,
                baseline: item.baseline,
                width: item.width,
                height: item.height,
            });

            let count = last.items.len() as f64;
            // The original's running mean, in the original's operation order.
            last.baseline = (last.baseline * (count - 1.0) + item.baseline) / count;
            last.height = last.height.max(item.height);
            last.left = last.left.min(item.left);
            last.right = last.right.max(item.left + item.width);
            last.top = last.top.min(item.baseline - item.height);
            last.bottom = last.bottom.max(item.baseline);
        } else {
            drafts.push(LineDraft {
                items: vec![PositionedTextItem {
                    text: item.text.clone(),
                    left: item.left,
                    baseline: item.baseline,
                    width: item.width,
                    height: item.height,
                }],
                baseline: item.baseline,
                height: item.height,
                left: item.left,
                right: item.left + item.width,
                top: item.baseline - item.height,
                bottom: item.baseline,
            });
        }
    }

    drafts
        .into_iter()
        .filter(|draft| draft.items.iter().any(|item| !is_blank_text(&item.text)))
        .map(|draft| TextLine {
            text: line_text(&draft.items),
            left: draft.left,
            right: draft.right,
            top: draft.top,
            bottom: draft.bottom,
            height: draft.height,
        })
        .filter(|line| !line.text.is_empty())
        .collect()
}

fn is_blank_text(text: &str) -> bool {
    text.chars().all(is_js_whitespace)
}

/// Joins one line's items left to right, inserting a space where the gap says
/// there is one. A word that the producer split into several items is not
/// given an inner space.
fn line_text(items: &[PositionedTextItem]) -> String {
    let mut ordered: Vec<&PositionedTextItem> = items.iter().collect();
    ordered.sort_by(|a, b| a.left.total_cmp(&b.left));

    let mut text = String::new();
    let mut previous_right: Option<f64> = None;
    let mut previous_height = 0.0f64;

    for item in ordered {
        if let Some(previous_right) = previous_right {
            let gap = item.left - previous_right;
            if gap > SPACE_GAP * previous_height.max(item.height) {
                text.push(' ');
            }
        }

        text.push_str(&item.text);
        previous_right = Some(item.left + item.width);
        previous_height = item.height;
    }

    collapse_whitespace(&text)
}

/// A paragraph drafted line by line.
struct ParagraphDraft {
    lines: Vec<TextLine>,
    left: f64,
    top: f64,
    bottom: f64,
}

/// The box of a paragraph before it is normalised.
struct BoxDraft {
    left: f64,
    top: f64,
    right: f64,
    bottom: f64,
}

/// The layout's output before the box is normalised.
struct DraftParagraph {
    text: String,
    rect: BoxDraft,
}

/// Groups lines into paragraphs by vertical gap and first-line indent.
///
/// A line on the same `left` as its paragraph's first line continues it even
/// when the vertical gap is large, as long as it does not exceed the paragraph
/// gap; the indent rule is what keeps a list of individually indented lines
/// apart without gaps between them.
fn group_paragraphs(lines: &[TextLine]) -> Vec<DraftParagraph> {
    let mut paragraphs: Vec<ParagraphDraft> = Vec::new();
    let mut current: Option<ParagraphDraft> = None;

    // Start a new paragraph at `line` (after flushing the open one, if any).
    fn start(current: &mut Option<ParagraphDraft>, line: &TextLine) {
        *current = Some(ParagraphDraft {
            lines: vec![clone_line(line)],
            left: line.left,
            top: line.top,
            bottom: line.bottom,
        });
    }

    for line in lines {
        // The decision is taken against an immutable view of the open
        // paragraph; the mutation happens after it, without it.
        enum Action {
            Start,
            Continue,
            Break,
        }
        let action = match &current {
            None => Action::Start,
            Some(draft) => {
                let previous = draft
                    .lines
                    .last()
                    .expect("an open paragraph always has a last line");
                let gap = line.top - previous.bottom;
                let line_height = previous.height.max(line.height);
                let new_paragraph = gap > PARAGRAPH_GAP * line_height
                    || (gap > 0.0 && line.left - draft.left > INDENT_RATIO * line_height);
                if new_paragraph {
                    Action::Break
                } else {
                    Action::Continue
                }
            }
        };

        match action {
            Action::Start | Action::Break => {
                // `break` flushes the open paragraph before the new one opens.
                if let Some(draft) = current.take() {
                    if !draft.lines.is_empty() {
                        paragraphs.push(draft);
                    }
                }
                start(&mut current, line);
            }
            Action::Continue => {
                let draft = current.as_mut().expect("the paragraph is open");
                draft.lines.push(clone_line(line));
                draft.top = draft.top.min(line.top);
                draft.bottom = draft.bottom.max(line.bottom);
            }
        }
    }

    if let Some(draft) = current.take() {
        if !draft.lines.is_empty() {
            paragraphs.push(draft);
        }
    }

    paragraphs
        .into_iter()
        .map(|paragraph| {
            let mut left = f64::INFINITY;
            let mut right = f64::NEG_INFINITY;
            for line in &paragraph.lines {
                left = left.min(line.left);
                right = right.max(line.right);
            }

            DraftParagraph {
                text: collapse_whitespace(
                    &paragraph
                        .lines
                        .iter()
                        .map(|line| line.text.as_str())
                        .collect::<Vec<_>>()
                        .join(" "),
                ),
                rect: BoxDraft {
                    left,
                    top: paragraph.top,
                    right,
                    bottom: paragraph.bottom,
                },
            }
        })
        .filter(|paragraph| !paragraph.text.is_empty())
        .collect()
}

fn clone_line(line: &TextLine) -> TextLine {
    TextLine {
        text: line.text.clone(),
        left: line.left,
        right: line.right,
        top: line.top,
        bottom: line.bottom,
        height: line.height,
    }
}

/// Viewport coordinates to a box normalised to the page.
fn normalize_box(rect: &BoxDraft, page_width: f64, page_height: f64) -> NormalizedBox {
    let clamp = |value: f64| value.min(1.0).max(0.0);
    // `pageWidth || 1` in the original: a missing or non-finite page size
    // degrades every box to the full page rather than to NaN.
    let width = if page_width.is_finite() && page_width != 0.0 {
        page_width
    } else {
        1.0
    };
    let height = if page_height.is_finite() && page_height != 0.0 {
        page_height
    } else {
        1.0
    };

    let left = rect.left.min(rect.right);
    let right = rect.left.max(rect.right);
    let top = rect.top.min(rect.bottom);
    let bottom = rect.top.max(rect.bottom);

    NormalizedBox {
        x: clamp(left / width),
        y: clamp(top / height),
        w: clamp((right - left) / width),
        h: clamp((bottom - top) / height),
    }
}

/// Collapses runs of whitespace to one space and trims the ends.
///
/// `text.replace(/\s+/g, ' ').trim()` in the original, character for
/// character: [`is_js_whitespace`] is exactly the set JavaScript collapses.
fn collapse_whitespace(text: &str) -> String {
    let mut collapsed = String::with_capacity(text.len());

    let mut in_whitespace = false;
    for character in text.chars() {
        if is_js_whitespace(character) {
            if !in_whitespace {
                collapsed.push(' ');
                in_whitespace = true;
            }
        } else {
            collapsed.push(character);
            in_whitespace = false;
        }
    }

    collapsed.trim().to_string()
}

/// The union of several normalised boxes, or nothing when there are none.
fn union_bbox(boxes: &[NormalizedBox]) -> Option<NormalizedBox> {
    if boxes.is_empty() {
        return None;
    }

    let mut left = f64::INFINITY;
    let mut top = f64::INFINITY;
    let mut right = f64::NEG_INFINITY;
    let mut bottom = f64::NEG_INFINITY;
    for box_value in boxes {
        left = left.min(box_value.x);
        top = top.min(box_value.y);
        right = right.max(box_value.x + box_value.w);
        bottom = bottom.max(box_value.y + box_value.h);
    }

    Some(NormalizedBox {
        x: left,
        y: top,
        w: right - left,
        h: bottom - top,
    })
}
