//! The Word document loader.
//!
//! A port of `src/main/services/loaders/DocxLoader.ts`.
//!
//! Like Mammoth, it unzips the file and extracts text and headings from `word/document.xml`.
//! It reconstructs the same raw text block separation and builds a section hierarchy based on
//! heading paragraphs.

use std::io::{Cursor, Read};
use std::path::Path;

use knownote_core::structure::{DocumentStructure, SectionInfo};
use quick_xml::events::Event;
use quick_xml::reader::Reader;

use crate::loader::{DocumentLoadResult, DocumentLoader, IngestError};

/// Reads Word documents (.docx).
pub struct DocxLoader;

impl DocxLoader {
    /// A new loader.
    pub fn new() -> Self {
        DocxLoader
    }

    /// Parse a DOCX file from a buffer.
    pub fn parse(&self, raw: &[u8]) -> Result<DocumentLoadResult, IngestError> {
        let cursor = Cursor::new(raw);
        let mut archive = zip::ZipArchive::new(cursor)
            .map_err(|error| IngestError::Parse(format!("failed to unzip docx: {error}")))?;

        let mut file = archive
            .by_name("word/document.xml")
            .map_err(|error| IngestError::Parse(format!("missing word/document.xml: {error}")))?;

        let mut xml_data = String::new();
        file.read_to_string(&mut xml_data)
            .map_err(|error| IngestError::Parse(format!("failed to read document.xml: {error}")))?;

        let mut reader = Reader::from_str(&xml_data);
        reader.config_mut().trim_text(true);

        let mut content = String::new();
        let mut in_p = false;
        let mut current_p_text = String::new();
        let mut current_heading_level: Option<i64> = None;
        let mut in_t = false;

        let mut flat_sections: Vec<SectionInfo> = Vec::new();

        loop {
            match reader.read_event() {
                Ok(Event::Start(ref e)) => {
                    let name_bytes = e.name().into_inner();
                    let name_str = String::from_utf8_lossy(name_bytes.as_ref()).into_owned();
                    let name = name_str.trim_matches('\0');
                    if name == "w:p" {
                        in_p = true;
                        current_p_text.clear();
                        current_heading_level = None;
                    } else if name == "w:t" {
                        in_t = true;
                    }
                },
                Ok(Event::Empty(ref e)) => {
                    let name_bytes = e.name().into_inner();
                    let name_str = String::from_utf8_lossy(name_bytes.as_ref()).into_owned();
                    let name = name_str.trim_matches('\0');
                    if name == "w:pStyle" {
                        if in_p {
                            for attr in e.attributes() {
                                if let Ok(attr) = attr {
                                    let key_bytes = attr.key.into_inner();
                                    let key_str = String::from_utf8_lossy(key_bytes.as_ref()).into_owned();
                                    let key = key_str.trim_matches('\0');
                                    if key == "w:val" {
                                        let val_str = String::from_utf8_lossy(attr.value.as_ref().as_bytes()).into_owned();
                                        let val = val_str.trim_matches('\0');
                                        if val.starts_with("Heading") && val.len() == 8 {
                                            if let Ok(level) = val[7..8].parse::<i64>() {
                                                current_heading_level = Some(level);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(Event::Text(e)) => {
                    if in_t {
                        let text_bytes = e.into_inner();
                        let text_str = String::from_utf8_lossy(text_bytes.as_ref().as_bytes()).into_owned();
                        let text = text_str.trim_matches('\0');
                        let unescaped = quick_xml::escape::unescape(text).map_err(|e| IngestError::Parse(format!("XML unescape error: {e:?}")))?;
                        current_p_text.push_str(unescaped.as_ref());
                    }
                }
                Ok(Event::End(ref e)) => {
                    let name_bytes = e.name().into_inner();
                    let name_str = String::from_utf8_lossy(name_bytes.as_ref()).into_owned();
                    let name = name_str.trim_matches('\0');
                    if name == "w:p" {
                        in_p = false;
                        let text = current_p_text.trim();
                        if !text.is_empty() {
                            if !content.is_empty() {
                                content.push_str("\n\n");
                            }

                            // If this paragraph is a heading, record it for later offset resolution
                            if let Some(level) = current_heading_level {
                                flat_sections.push(SectionInfo {
                                    level,
                                    title: text.to_string(),
                                    content: String::new(),
                                    start_offset: 0,
                                    end_offset: 0,
                                    children: Vec::new(),
                                });
                            }

                            content.push_str(text);
                        }
                    } else if name == "w:t" {
                        in_t = false;
                    }
                },
                Ok(Event::Eof) => break,
                Err(e) => return Err(IngestError::Parse(format!("XML error at position {}: {e:?}", reader.error_position()))),
                _ => {}
            }
        }

        // Now find the offsets
        let mut sections: Vec<SectionInfo> = Vec::new();
        let num_headings = flat_sections.len();

        for i in 0..num_headings {
            let heading = &flat_sections[i];
            let title = &heading.title;

            let start_offset = match content.find(title) {
                Some(pos) => pos,
                None => continue,
            };

            let mut end_offset = content.len();
            for j in (i + 1)..num_headings {
                let next_heading = &flat_sections[j];
                if next_heading.level <= heading.level {
                    if let Some(next_start) = content[start_offset + title.len()..].find(&next_heading.title) {
                        end_offset = start_offset + title.len() + next_start;
                        break;
                    }
                }
            }

            let section_content = content[start_offset..end_offset].trim().to_string();

            sections.push(SectionInfo {
                level: heading.level,
                title: title.clone(),
                content: section_content,
                start_offset,
                end_offset,
                children: Vec::new(),
            });
        }

        let title = sections.iter().find(|s| s.level == 1).map(|s| s.title.clone());

        // Build the hierarchy tree
        let mut root: Vec<SectionInfo> = Vec::new();
        let mut stack: Vec<Vec<usize>> = Vec::new(); // paths

        for section in sections {
            let level = section.level;

            while let Some(parent_path) = stack.last() {
                let parent_level = node_at(&mut root, parent_path).level;
                if parent_level >= level {
                    stack.pop();
                } else {
                    break;
                }
            }

            let path = match stack.last() {
                None => {
                    root.push(section);
                    vec![root.len() - 1]
                }
                Some(parent_path) => {
                    let parent = node_at(&mut root, parent_path);
                    parent.children.push(section);
                    let mut new_path = parent_path.clone();
                    new_path.push(parent.children.len() - 1);
                    new_path
                }
            };
            stack.push(path);
        }

        let word_count = content.split_whitespace().count();

        Ok(DocumentLoadResult {
            content: content.clone(),
            title,
            mime_type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document".to_string(),
            structure: if !root.is_empty() {
                Some(DocumentStructure::Sections { sections: root })
            } else {
                None
            },
            metadata: Some(serde_json::json!({
                "warnings": [],
                "characterCount": content.chars().map(|c| if c.len_utf16() == 2 { 2 } else { 1 }).sum::<usize>(),
                "wordCount": word_count,
            })),
        })
    }
}

impl Default for DocxLoader {
    fn default() -> Self {
        DocxLoader::new()
    }
}

fn node_at<'a>(sections: &'a mut [SectionInfo], path: &[usize]) -> &'a mut SectionInfo {
    let mut current = &mut sections[path[0]];
    for index in &path[1..] {
        current = &mut current.children[*index];
    }
    current
}

impl DocumentLoader for DocxLoader {
    fn supported_mime_types(&self) -> &[&'static str] {
        &[
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        ]
    }

    fn supported_extensions(&self) -> &[&'static str] {
        &["docx", "doc"]
    }

    fn load_from_path(&self, path: &Path) -> Result<DocumentLoadResult, IngestError> {
        let bytes = std::fs::read(path).map_err(|source| IngestError::Io {
            path: path.display().to_string(),
            source,
        })?;
        self.load_from_buffer(&bytes)
    }

    fn load_from_buffer(&self, buffer: &[u8]) -> Result<DocumentLoadResult, IngestError> {
        self.parse(buffer)
    }
}
