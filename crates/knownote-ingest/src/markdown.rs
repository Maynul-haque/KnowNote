//! The Markdown loader.
//!
//! A port of `src/main/services/loaders/MarkdownLoader.ts`.
//!
//! Markdown has no pages, but it does have a heading hierarchy, and that is
//! what a citation needs: a heading is the unit a reader recognises, and the
//! block builder turns each heading and the text under it into blocks with
//! the heading's offsets.
//!
//! # One deliberate difference
//!
//! The original fills a nested section's `content` by slicing the *parent's*
//! content string with the child's *absolute* offsets. Those two are in
//! different coordinate systems, so the slice runs past the end of the string
//! and comes back empty. The offsets, which are the part the rest of the
//! pipeline reads, are correct in the original, and they are correct here.
//! The content of a nested section is not.

use std::path::Path;

use knownote_core::structure::{DocumentStructure, SectionInfo};
use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag, TagEnd};

use crate::loader::{DocumentLoadResult, DocumentLoader, IngestError};

/// Reads Markdown documents.
pub struct MarkdownLoader;

impl MarkdownLoader {
    /// A new loader.
    pub fn new() -> Self {
        MarkdownLoader
    }

    /// Parse Markdown into the canonical text and its section structure.
    pub fn parse(&self, raw: &str) -> DocumentLoadResult {
        // The canonical text is the trimmed document, and the offsets are
        // taken from that same string, so a section's offsets always index
        // the text the store holds.
        let content = raw.trim().to_string();

        let mut options = Options::empty();
        options.insert(Options::ENABLE_TABLES);
        options.insert(Options::ENABLE_FOOTNOTES);
        options.insert(Options::ENABLE_STRIKETHROUGH);
        options.insert(Options::ENABLE_TASKLISTS);

        let mut sections: Vec<SectionInfo> = Vec::new();
        // The path of each open section, from the root, and its level.
        let mut stack: Vec<(Vec<usize>, i64)> = Vec::new();
        let mut title: Option<String> = None;

        let mut heading_text = String::new();
        let mut heading_level = 0i64;
        let mut heading_start = 0usize;
        let mut in_heading = false;

        let mut has_code_blocks = false;
        let mut has_tables = false;

        for (event, range) in Parser::new_ext(&content, options).into_offset_iter() {
            match event {
                Event::Start(Tag::Heading { level, .. }) => {
                    in_heading = true;
                    heading_level = heading_depth(level);
                    heading_text.clear();
                    heading_start = range.start;
                }
                Event::End(TagEnd::Heading(_)) => {
                    in_heading = false;
                    let heading_end = range.end;

                    if title.is_none() && heading_level == 1 {
                        title = Some(heading_text.clone());
                    }

                    // A heading of level N is a child of the most recent
                    // heading of a lower level.
                    while let Some((_, level)) = stack.last() {
                        if *level >= heading_level {
                            stack.pop();
                        } else {
                            break;
                        }
                    }

                    let section = SectionInfo {
                        level: heading_level,
                        title: heading_text.clone(),
                        content: String::new(),
                        start_offset: heading_start,
                        end_offset: heading_end,
                        children: Vec::new(),
                    };

                    let path = match stack.last() {
                        None => {
                            sections.push(section);
                            vec![sections.len() - 1]
                        }
                        Some((parent_path, _)) => {
                            let index = {
                                let parent = node_at(&mut sections, parent_path);
                                parent.children.push(section);
                                parent.children.len() - 1
                            };
                            let mut path = parent_path.clone();
                            path.push(index);
                            path
                        }
                    };

                    stack.push((path, heading_level));
                }
                Event::Text(text) | Event::Code(text) => {
                    if in_heading {
                        heading_text.push_str(&text);
                    }
                }
                Event::Start(Tag::CodeBlock(_)) => {
                    has_code_blocks = true;
                }
                Event::Start(Tag::Table(_)) => {
                    has_tables = true;
                }
                _ => {}
            }
        }

        fill_sections(&mut sections, &content, content.len());

        DocumentLoadResult {
            content,
            title,
            mime_type: "text/markdown".to_string(),
            structure: Some(DocumentStructure::Sections { sections }),
            metadata: Some(serde_json::json!({
                "lineCount": raw.split('\n').count(),
                "hasCodeBlocks": has_code_blocks,
                "hasTables": has_tables,
            })),
        }
    }
}

impl Default for MarkdownLoader {
    fn default() -> Self {
        MarkdownLoader::new()
    }
}

/// The depth of a heading, from 1 to 6.
fn heading_depth(level: HeadingLevel) -> i64 {
    match level {
        HeadingLevel::H1 => 1,
        HeadingLevel::H2 => 2,
        HeadingLevel::H3 => 3,
        HeadingLevel::H4 => 4,
        HeadingLevel::H5 => 5,
        HeadingLevel::H6 => 6,
    }
}

/// The section at a path, counting child indices from the root.
fn node_at<'a>(sections: &'a mut [SectionInfo], path: &[usize]) -> &'a mut SectionInfo {
    let mut current = &mut sections[path[0]];
    for index in &path[1..] {
        current = &mut current.children[*index];
    }
    current
}

/// Give every section the text that runs from its heading to the next one.
///
/// A section's content runs to the next section *at the same level*, so a
/// parent heading never contains its children.
fn fill_sections(sections: &mut Vec<SectionInfo>, document: &str, scope_end: usize) {
    let starts: Vec<usize> = sections.iter().map(|section| section.start_offset).collect();
    let count = sections.len();

    for index in 0..count {
        let start = starts[index];
        let end = if index + 1 < count {
            starts[index + 1]
        } else {
            scope_end
        };

        {
            let section = &mut sections[index];
            let from = start.min(document.len());
            let to = end.min(document.len()).max(from);
            section.content = document[from..to].trim().to_string();
            section.end_offset = end;
        }

        let children_end = sections[index].end_offset;
        fill_sections(&mut sections[index].children, document, children_end);
    }
}

impl DocumentLoader for MarkdownLoader {
    fn supported_mime_types(&self) -> &[&'static str] {
        &["text/markdown"]
    }

    fn supported_extensions(&self) -> &[&'static str] {
        &["md", "markdown"]
    }

    fn load_from_path(&self, path: &Path) -> Result<DocumentLoadResult, IngestError> {
        let bytes = std::fs::read(path).map_err(|source| IngestError::Io {
            path: path.display().to_string(),
            source,
        })?;
        self.load_from_buffer(&bytes)
    }

    fn load_from_buffer(&self, buffer: &[u8]) -> Result<DocumentLoadResult, IngestError> {
        let text = String::from_utf8(buffer.to_vec())
            .map_err(|error| IngestError::Encoding(error.to_string()))?;
        Ok(self.parse(&text))
    }
}
