# Overview

`treetok` clusters *surface-form variants* inside a tokenizer vocabulary.

In many Hugging Face tokenizers, the same user-visible text can appear in
multiple token strings due to conventions like leading-space markers
(byte-level BPE/SentencePiece), WordPiece continuation markers, case variants,
and Unicode normalization differences.

`treetok` learns a merge decision rule from labeled token pairs and then uses
that rule to cluster a vocabulary into small groups that represent "the same
surface form".

## What It Clusters

Examples of the kinds of variants `treetok` targets:

- Marker variants: e.g. leading-space token vs non-leading-space token
- Case variants: e.g. `username` vs `UserName`
- Unicode normalization variants (NFKC/NFKD)
- Simple edge punctuation differences (single leading or trailing punctuation)

## What It Does Not Cluster

`treetok` is intentionally *not* a lemmatizer or semantic grouper. It treats
these as negatives:

- Morphological variants (e.g. `cat` vs `cats`)
- Different words that are close in edit distance
- Subword relationships that change meaning (e.g. `ing` vs `sing`)

## Typical Workflow

1. Inspect a tokenizer to understand its marker convention
2. Build a labeled dataset of token pairs
3. Train a classifier
4. Cluster a tokenizer vocabulary and review the clusters

See the [training guide](training.md) for guidance on dataset sizing and
threshold tuning, and the [CLI instructions](cli.md) for the CLI surface.
