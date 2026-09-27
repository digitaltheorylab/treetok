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

## Marker Policy

Whether marker toggles count as variants is a modeling decision, not a fact
about strings: `\u0120hello` is word-initial while `hello` is a continuation
fragment. `treetok` makes this an explicit, end-to-end knob:

- `marker_policy="merge"` (default): marker toggles (`hello` /
  `\u0120hello`, `World` / `\u2581World`, `the` / `##the`) are surface-form
  variants. The comparison surface drops the marker's leading space,
  canonical keys ignore marker presence, and candidate generation pairs
  tokens across the marker boundary. Marker toggles are synthetic *positives*
  during training
- `marker_policy="separate"`: marker variants must never merge. Canonical
  keys and candidate strata include the marker bit, so cross-marker pairs are
  structurally unreachable, and marker toggles are mined as hard *negatives*
  during training

The policy is recorded in each training dataset and in the trained model
artifact; clustering defaults to the model's recorded policy so features are
computed the same way at inference as at training time. Datasets and models
built under different policies should not be mixed.

## 2) Feature Construction

For each token pair, `treetok` computes a compact feature vector combining:

- String similarity (multiple edit-distance/similarity metrics)
- Length statistics
- Common prefix/suffix overlap
- Equality checks on canonicalized forms (casefolding, normalization, stripped
  marker forms, decoded forms)
- Script/marker agreement
- Tokenizer-family indicators

Distance features are computed on a comparison surface (marker-stripped, and
decoded for byte-level BPE) so marker conventions and byte-glyph encodings do
not dominate distance when tokens are otherwise identical. Under
`marker_policy="merge"`, that surface also drops the marker's leading space
for byte-level BPE tokens so marker toggles compare equal.

## 3) Candidate Pair Generation

Scoring every possible pair in a vocabulary is too expensive. Instead,
`treetok` generates candidate pairs by:

- Stratifying tokens into coarse buckets (script, length, and \u2014 under
  `marker_policy="separate"` \u2014 marker presence)
- Comparing only within nearby length bands
- Using batched edit-distance scoring to drop most pairs efficiently

The output of this stage is a stream of candidate `(i, j)` pairs.

## 4) Training Labels

The training dataset is a mixture of:

- Synthetic positives from controlled surface-form transforms (case toggles,
  normalization changes, simple edge punctuation stripping, and \u2014 under
  `marker_policy="merge"` \u2014 marker toggles)
- Hard negatives mined from the candidate stream that are close in edit
  distance but do not match on canonical forms (under
  `marker_policy="separate"`, marker toggles are added here)
- Easy negatives sampled across strata for balance

## 5) Classifier And Thresholds

An XGBoost binary classifier is trained on the feature vectors.

Two operating thresholds are tuned on a held-out validation split:

- An "edge" threshold used to decide whether a pair is merge-worthy
- A stricter "merge" threshold used when joining already-nontrivial
  components (to reduce chain merges)

## 6) Clustering

Clustering runs in three stages:

1. Canonical grouping using a conservative key (script + casefolded
   comparison form, plus the marker bit under `marker_policy="separate"`).
   This collapses obvious variants without a classifier call. It uses
   its own, lower length floor (`canonical_min_len`, default 2) than the
   scoring stages below, because it buckets on an exact key and scores no
   pairs, so it has none of the pair-count blowup or edit-distance concerns
   that keep the scoring floor at 3
2. Anchor expansion: score only anchor-to-unassigned candidates and assign each
   token to at most one anchor
3. Fallback union: for remaining tokens, score candidate edges and do a
   confidence-ordered union with safeguards that prevent runaway components

Scoring can optionally run in a thread pool; output is deterministic.
