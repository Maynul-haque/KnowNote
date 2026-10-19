//! Token estimation.
//!
//! A port of `ChunkingService.estimateTokens`. It is an estimate, not a
//! tokenizer: the number is used to budget a prompt, and the original
//! implementation's formula is the reference the eval baseline was measured
//! against.

/// Estimate how many tokens `text` costs.
///
/// The formula treats CJK as roughly 1.5 characters per token and everything
/// else as roughly 4 characters per token, which is the empirical shape the
/// original used for mixed Chinese and English text.
pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }

    let mut cjk_chars: usize = 0;
    let mut other_chars: usize = 0;

    for c in text.chars() {
        let code = c as u32;
        let is_cjk = (0x4e00..=0x9fff).contains(&code)  // CJK unified ideographs
            || (0x3400..=0x4dbf).contains(&code) // CJK extension A
            || (0xf900..=0xfaff).contains(&code) // CJK compatibility ideographs
            || (0x3040..=0x309f).contains(&code) // Hiragana
            || (0x30a0..=0x30ff).contains(&code) // Katakana
            || (0xac00..=0xd7af).contains(&code); // Hangul syllables

        if is_cjk {
            cjk_chars += 1;
        } else {
            other_chars += 1;
        }
    }

    (cjk_chars as f64 / 1.5 + other_chars as f64 / 4.0).ceil() as usize
}
