# How It Works

This is a conceptual overview of the pipeline. The README contains a minimal
end-to-end example for both the Python API and CLI.

## 1) Tokenizer Inspection

`treetok` takes a Hugging Face tokenizer and snapshots:

- Token strings (vocabulary)
- A decoded, user-visible form for each token
- The dominant "marker" convention, if any

The marker is typically one of:

- A leading-space marker used by byte-level BPE and SentencePiece tokenizers
- A continuation marker used by WordPiece tokenizers

This step also identifies tokens to skip during clustering (special tokens,
added tokens, and common reserved placeholders).

## 2) Feature Construction

For each token pair, `treetok` computes a compact feature vector combining:

- String similarity (multiple edit-distance/similarity metrics)
- Length statistics
- Common prefix/suffix overlap
- Equality checks on canonicalized forms (casefolding, normalization, stripped
  marker forms, decoded forms)
- Script/marker agreement
- A proximity signal derived from token ids
- Tokenizer-family indicators

Features are computed on a stripped form so marker conventions do not dominate
distance when tokens are otherwise identical.

## 3) Candidate Pair Generation

Scoring every possible pair in a vocabulary is too expensive. Instead,
`treetok` generates candidate pairs by:

- Stratifying tokens into coarse buckets (script, marker presence, length)
- Comparing only within nearby length bands
- Using batched edit-distance scoring to drop most pairs efficiently

The output of this stage is a stream of candidate `(i, j)` pairs.

## 4) Training Labels

The training dataset is a mixture of:

- Synthetic positives from controlled surface-form transforms (case toggles,
  normalization changes, marker toggles, simple edge punctuation stripping)
- Hard negatives mined from the candidate stream that are close in edit
  distance but do not match on canonical forms
- Easy negatives sampled across strata for balance

## 5) Classifier And Thresholds

An XGBoost binary classifier is trained on the feature vectors.

Two operating thresholds are tuned on a held-out validation split:

- An "edge" threshold used to decide whether a pair is merge-worthy
- A stricter "merge" threshold used when joining already-nontrivial
  components (to reduce chain merges)

## 6) Clustering

Clustering runs in three stages:

1. Canonical grouping using a conservative key (script + casefolded stripped
   form). This collapses obvious variants without a classifier call
2. Anchor expansion: score only anchor-to-unassigned candidates and assign each
   token to at most one anchor
3. Fallback union: for remaining tokens, score candidate edges and do a
   confidence-ordered union with safeguards that prevent runaway components

Scoring can optionally run in a thread pool; output is deterministic.
