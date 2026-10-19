//! Text helpers shared by the chunker.
//!
//! The original implementation is TypeScript, and a JavaScript string is
//! indexed by UTF-16 code unit while a Rust string is indexed by byte. These
//! helpers make the difference explicit instead of letting it hide in an
//! index:
//!
//! * [`utf16_len`] and [`utf16_len_range`] count UTF-16 code units, which is
//!   the unit the original used for every *size* comparison.
//! * [`is_whitespace`] and [`is_sentence_end`] are the original's own narrow
//!   character sets, not Rust's `char::is_whitespace`.
//! * [`is_js_whitespace`] reproduces the exact set that JavaScript's
//!   `String.prototype.trim` removes, which is wider than both.

/// The number of UTF-16 code units in `text`.
///
/// A character outside the Basic Multilingual Plane counts as two.
pub fn utf16_len(text: &str) -> usize {
    text.chars().map(char::len_utf16).sum()
}

/// The number of UTF-16 code units in `text[start..end]`.
///
/// `start` and `end` must be character boundaries.
pub fn utf16_len_range(text: &str, start: usize, end: usize) -> usize {
    utf16_len(&text[start..end])
}

/// The whitespace set the original chunker trims: space, tab, newline, carriage return.
///
/// This is deliberately narrower than [`is_js_whitespace`] and narrower than
/// Rust's `char::is_whitespace`.
pub fn is_whitespace(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r')
}

/// The set of characters that end a sentence in the original chunker.
pub fn is_sentence_end(c: char) -> bool {
    matches!(c, '\u{3002}' | '\u{ff01}' | '\u{ff1f}' | '.' | '!' | '?')
}

/// The exact set `String.prototype.trim` removes in JavaScript.
///
/// This is the ECMAScript `WhiteSpace` and `LineTerminator` sets. It is wider
/// than [`is_whitespace`], and wider than Rust's `char::is_whitespace` in one
/// place: it includes U+FEFF, the byte order mark.
pub fn is_js_whitespace(c: char) -> bool {
    matches!(
        c,
        '\u{9}'   // tab
            | '\u{a}' // line feed
            | '\u{b}' // vertical tab
            | '\u{c}' // form feed
            | '\u{d}' // carriage return
            | '\u{20}' // space
            | '\u{a0}' // no-break space
            | '\u{1680}' // ogham space mark
            | '\u{2000}'..='\u{200a}' // en quad .. hair space
            | '\u{2028}' // line separator
            | '\u{2029}' // paragraph separator
            | '\u{202f}' // narrow no-break space
            | '\u{205f}' // medium mathematical space
            | '\u{3000}' // ideographic space
            | '\u{feff}' // zero width no-break space
    )
}

/// Whether `text` is empty or contains nothing but JavaScript whitespace.
pub fn is_blank_js(text: &str) -> bool {
    text.chars().all(is_js_whitespace)
}

/// The character at byte offset `index`.
///
/// `index` must be a character boundary; the caller returns `'\0'` for an
/// empty range, which never compares equal to whitespace or a sentence end.
pub fn char_at(text: &str, index: usize) -> char {
    text.get(index..).and_then(|rest| rest.chars().next()).unwrap_or('\0')
}

/// Advance `start` past any leading whitespace, stopping at `end`.
///
/// Returns the offset of the first non-whitespace character, or `end` when the
/// whole range is whitespace.
pub fn trim_leading_whitespace(text: &str, start: usize, end: usize) -> usize {
    let mut cursor = end;
    for (offset, c) in text[start..end].char_indices() {
        if !is_whitespace(c) {
            return start + offset;
        }
        cursor = start + offset + c.len_utf8();
    }
    cursor
}

/// Retreat `end` past any trailing whitespace, stopping at `start`.
///
/// Returns the offset one past the last non-whitespace character, or `start`
/// when the whole range is whitespace.
pub fn trim_trailing_whitespace(text: &str, start: usize, end: usize) -> usize {
    let mut last_non_whitespace = start;
    for (offset, c) in text[start..end].char_indices() {
        if !is_whitespace(c) {
            last_non_whitespace = start + offset + c.len_utf8();
        }
    }
    last_non_whitespace
}

/// A map between UTF-16 code-unit indices and byte offsets.
///
/// A JavaScript string is indexed by UTF-16 code unit and a Rust string by
/// byte, so the character-window fallback needs both. The map is built once per
/// call and every lookup is a binary search.
///
/// A character outside the Basic Multilingual Plane occupies two UTF-16 code
/// units but one byte range. The first unit maps to the start of the
/// character and the second to its end, so a range that starts on the second
/// unit starts after the character rather than inside it.
pub struct Utf16Index {
    /// `offsets[i]` is the byte offset of UTF-16 code-unit index `i`.
    offsets: Vec<u32>,
}

impl Utf16Index {
    /// Build the index for `text`.
    pub fn new(text: &str) -> Self {
        let mut offsets: Vec<u32> = Vec::with_capacity(text.len() + 1);
        let mut byte = 0usize;

        for c in text.chars() {
            let units = c.len_utf16();
            offsets.push(byte as u32);
            byte += c.len_utf8();
            for _ in 1..units {
                offsets.push(byte as u32);
            }
        }
        offsets.push(byte as u32);

        Self { offsets }
    }

    /// The number of UTF-16 code units in the text.
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Whether the text is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The byte offset of a UTF-16 code-unit index.
    ///
    /// An index past the end clamps to the end of the text.
    pub fn byte_of(&self, index: usize) -> usize {
        match self.offsets.get(index) {
            Some(offset) => *offset as usize,
            None => *self.offsets.last().unwrap_or(&0) as usize,
        }
    }

    /// The UTF-16 code-unit index of a byte offset.
    ///
    /// The byte offset must be a character boundary. A byte offset that falls
    /// inside a character resolves to the index of the character that contains
    /// it.
    pub fn utf16_of_byte(&self, byte: usize) -> usize {
        let target = byte as u32;
        // The offsets are non-decreasing and a character outside the Basic
        // Multilingual Plane contributes two equal entries, so the answer is
        // the last index whose offset is at or below the target. A binary
        // search would find any of the equal entries, not the last.
        self.offsets
            .partition_point(|offset| *offset <= target)
            .saturating_sub(1)
    }
}

