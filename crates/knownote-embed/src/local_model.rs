//! The pin for the built-in embedding model.
//!
//! The model version has to be pinned. One release of KnowNote corresponds to
//! one embedding space; if the app fetched "latest" on every start, the model
//! would change underneath the vectors and old and new embeddings would end up
//! compared against each other in the same table.
//!
//! The SHA256 values are real: the large files are Git LFS objects, so their
//! oid *is* the SHA256, and the small ones were computed locally. A model
//! upgrade has to update the revision, the sizes and the hashes together, or
//! the download refuses to write into the cache because the check fails.

/// The built-in local embedding model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalEmbeddingModel {
    /// The repository id on the model hub.
    pub id: &'static str,
    /// A short name for the settings screen.
    pub label: &'static str,
    /// The commit the files are fetched from.
    pub revision: &'static str,
    /// The quantisation of the ONNX weights.
    pub dtype: &'static str,
    /// The width of a vector.
    pub dimensions: usize,
    /// How token vectors are combined into one.
    pub pooling: &'static str,
    /// Whether the vector is unit length.
    pub normalize: bool,
    /// The prefix a question is embedded with.
    pub query_prefix: &'static str,
    /// The prefix a passage is embedded with.
    pub document_prefix: &'static str,
    /// The sentencepiece limit of the XLM-R tokenizer.
    pub max_length: usize,
    /// The model this one was quantised from.
    pub base_model: &'static str,
    /// The licence the weights ship under.
    pub license: &'static str,
}

/// The built-in local embedding model.
pub const LOCAL_EMBEDDING_MODEL: LocalEmbeddingModel = LocalEmbeddingModel {
    id: "Xenova/multilingual-e5-small",
    label: "multilingual-e5-small",
    revision: "761b726dd34fb83930e26aab4e9ac3899aa1fa78",
    dtype: "q8",
    dimensions: 384,
    pooling: "mean",
    normalize: true,
    query_prefix: "query: ",
    document_prefix: "passage: ",
    // The XLM-R sentencepiece limit.
    max_length: 512,
    base_model: "intfloat/multilingual-e5-small",
    license: "MIT",
};

/// One file of the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingModelFile {
    /// The path inside the repository, and inside the cache directory.
    pub path: &'static str,
    /// The size in bytes.
    pub size: u64,
    /// The SHA256 of the file.
    pub sha256: &'static str,
}

/// The files that have to be present for the model to load.
///
/// The `q8` dtype maps to the `_quantized` ONNX file.
pub const LOCAL_EMBEDDING_FILES: &[EmbeddingModelFile] = &[
    EmbeddingModelFile {
        path: "onnx/model_quantized.onnx",
        size: 118_308_185,
        sha256: "f80102d3f2a1229f387d3c81909990d8945513e347b0eab049f7de3c6f98c193",
    },
    EmbeddingModelFile {
        path: "tokenizer.json",
        size: 17_082_730,
        sha256: "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
    },
    EmbeddingModelFile {
        path: "sentencepiece.bpe.model",
        size: 5_069_051,
        sha256: "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
    },
    EmbeddingModelFile {
        path: "config.json",
        size: 658,
        sha256: "cb99455288675345e1a4f411438d5d0adbba5fbd3a67ea4fb03c015433b996c1",
    },
    EmbeddingModelFile {
        path: "tokenizer_config.json",
        size: 443,
        sha256: "a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b",
    },
    EmbeddingModelFile {
        path: "quant_config.json",
        size: 674,
        sha256: "59d175f15264115f18c698d76e443b5d49fc6c8c599911c421405ef4f236e87d",
    },
    EmbeddingModelFile {
        path: "special_tokens_map.json",
        size: 167,
        sha256: "d05497f1da52c5e09554c0cd874037a083e1dc1b9cfd48034d1c717f1afc07a7",
    },
];

/// The total size of the model files, about 140.5 MB.
pub fn local_embedding_total_bytes() -> u64 {
    LOCAL_EMBEDDING_FILES.iter().map(|file| file.size).sum()
}

/// The primary model hub.
pub const DEFAULT_HUB_BASE_URL: &str = "https://huggingface.co";

/// A mirror, used when the primary host is unreachable.
pub const MIRROR_HUB_BASE_URL: &str = "https://hf-mirror.com";

/// The URL that resolves one file of the model.
///
/// The mirror answers a small file with a 307 (a relative path) and the Xet
/// weights with a 302 to an absolute URL carrying a one-hour signed
/// signature. A resumed download therefore has to ask for this URL again
/// rather than reuse the 302 target, or the signature will have expired.
pub fn build_resolve_url(base_url: &str, revision: &str, file_path: &str) -> String {
    let base = base_url.trim_end_matches('/');
    let encoded = file_path
        .split('/')
        .map(|segment| {
            segment
                .bytes()
                .map(|byte| match byte {
                    b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                        (byte as char).to_string()
                    }
                    other => format!("%{other:02X}"),
                })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("/");

    format!("{base}/{}/resolve/{revision}/{encoded}", LOCAL_EMBEDDING_MODEL.id)
}

/// The identity of the space the built-in model produces.
///
/// The dtype is part of the identifier: the same model at a different
/// quantisation is a different space.
pub fn local_embedding_space() -> knownote_embedding::EmbeddingSpace {
    knownote_embedding::EmbeddingSpace::local(
        format!("{}#{}", LOCAL_EMBEDDING_MODEL.id, LOCAL_EMBEDDING_MODEL.dtype),
        LOCAL_EMBEDDING_MODEL.revision,
        LOCAL_EMBEDDING_MODEL.dimensions,
    )
}
