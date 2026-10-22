//! Reading documents into the canonical text.
//!
//! A loader turns a file into the canonical text plus whatever structure it
//! found — pages, sections, or neither. The structure is what lets a citation
//! resolve to a page and a heading rather than to a paragraph of flat text.

pub mod docx;
pub mod loader;
pub mod markdown;
pub mod pdf;

pub use docx::DocxLoader;
pub use loader::{DocumentLoadResult, DocumentLoader, IngestError};
pub use markdown::MarkdownLoader;
pub use pdf::{layout_page_text, PdfLoader, PositionedTextItem};
