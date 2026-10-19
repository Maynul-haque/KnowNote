//! The dataset's types.
//!
//! A port of the parts of `src/main/eval/types.ts` the harness reads.
//!
//! The dataset is keyed on *corpus identity*, never on database identity: a
//! ground truth names a document by its corpus-relative path and a block by
//! its ordinal in the block sequence the ingestion produced. Runtime document
//! ids are random and block ids embed them, so a dataset keyed on those would
//! break the moment a parser changed — and would stop measuring anything.

use serde::{Deserialize, Serialize};

/// One question, with its ground truth.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalQuestion {
    pub id: String,
    pub question: String,
    /// The query class, used to group the report.
    #[serde(rename = "type", default)]
    pub question_type: Option<String>,
    /// Whether the question can be answered from the corpus.
    ///
    /// An unanswerable question carries no ground truth. The two conditions
    /// are checked against each other, because getting them the wrong way
    /// round is silent: an answerable question with no ground truth reads as
    /// a permanent miss.
    #[serde(default)]
    pub answerable: Option<bool>,
    #[serde(default)]
    pub relevant: Vec<EvalRelevantLocation>,
    #[serde(rename = "goldAnswer", default)]
    pub gold_answer: Option<String>,
}

/// One place in the corpus that answers a question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalRelevantLocation {
    /// The corpus-relative path.
    pub document: String,
    /// The page, for a paginated source.
    #[serde(default)]
    pub page: Option<i64>,
    /// The block's ordinal in the document's block sequence.
    pub block: usize,
    /// An excerpt that must still be in the block.
    #[serde(default)]
    pub quote: Option<String>,
}

impl EvalQuestion {
    /// Whether the question can be answered.
    pub fn is_answerable(&self) -> bool {
        self.answerable.unwrap_or(true)
    }
}

/// The question's class, or the untagged group.
pub fn question_type_of(question: &EvalQuestion) -> String {
    question
        .question_type
        .clone()
        .unwrap_or_else(|| "untagged".to_string())
}

/// Check the dataset's own contract.
///
/// An answerable question must have ground truth and an unanswerable one
/// must not. Getting either the wrong way round does not make the harness
/// fail — it makes the numbers quietly meaningless.
pub fn assert_question_shape(question: &EvalQuestion) -> Result<(), String> {
    if question.id.is_empty() {
        return Err("a question has no id".to_string());
    }
    if question.question.trim().is_empty() {
        return Err(format!("question {} has no text", question.id));
    }

    if question.is_answerable() {
        if question.relevant.is_empty() {
            return Err(format!(
                "question {} is answerable but has no ground truth",
                question.id
            ));
        }
    } else if !question.relevant.is_empty() {
        return Err(format!(
            "question {} is unanswerable but carries ground truth",
            question.id
        ));
    }

    Ok(())
}

/// The committed validation/test assignment.
pub type SplitAssignment = std::collections::BTreeMap<String, String>;

/// Which split to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EvalSplit {
    All,
    Validation,
    Test,
}

impl EvalSplit {
    /// The name used in the report.
    pub fn as_str(self) -> &'static str {
        match self {
            EvalSplit::All => "all",
            EvalSplit::Validation => "validation",
            EvalSplit::Test => "test",
        }
    }
}

/// Read the split assignment.
///
/// The manifest is committed to the repository, so a parse error names the
/// file it came from rather than failing somewhere else.
pub fn parse_split_assignment(raw: &str, label: &str) -> Result<SplitAssignment, String> {
    let parsed: SplitAssignment = serde_json::from_str(raw)
        .map_err(|error| format!("{label} could not be read as JSON: {error}"))?;

    for (question_id, split) in &parsed {
        if split != "validation" && split != "test" {
            return Err(format!(
                "{label} assigns {question_id} to {split:?}, which is neither validation nor test"
            ));
        }
    }

    Ok(parsed)
}

/// The questions of one split.
///
/// A question that the manifest does not mention is refused rather than
/// defaulted: the split is an experiment design, and a new question has to be
/// assigned on purpose.
pub fn select_split(
    questions: &[EvalQuestion],
    split: EvalSplit,
    assignment: &SplitAssignment,
) -> Result<Vec<EvalQuestion>, String> {
    if split == EvalSplit::All {
        return Ok(questions.to_vec());
    }

    let wanted = split.as_str();
    let mut selected = Vec::new();

    for question in questions {
        let assigned = assignment.get(&question.id).ok_or_else(|| {
            format!(
                "question {} is not in the split manifest, so it cannot be assigned to a split on purpose",
                question.id
            )
        })?;

        if assigned == wanted {
            selected.push(question.clone());
        }
    }

    Ok(selected)
}
