# Training

## Dataset Composition

Training data is built from three sources:

- Synthetic positives: surface-form transforms that should merge *if both forms
  exist in the vocabulary*
- Hard negatives: close-in-distance pairs that should *not* merge
- Easy negatives: cross-stratum pairs for class balance

The defaults are intended to be starting points, however, your tokenizer mix
can change what "hard" looks like.

## Recommended Starting Point

As a starting point for a single tokenizer:

```sh
python -m treetok build-dataset <model> -o data/<model>.parquet \
  --n-positives 2000 --n-hard-negatives 6000 --n-easy-negatives 1000
```

If you plan to cluster a tokenizer family you did not train on, it is usually
better to train across multiple tokenizers (different families and vocab
styles)

## Understanding The Two Thresholds

Training produces two thresholds:

- `edge_threshold`: used to admit individual merge edges
- `merge_threshold`: stricter cutoff used when bridging already-nontrivial
  components

They are tuned to hit high precision targets on the validation split, trading
off recall for fewer false merges.

## Troubleshooting

If you see thresholds pinned to floors (or generally lower than expected), it
often means your dataset is too easy:

- Add additional tokenizers to training
- Increase hard negatives relative to positives
- Prefer harder negatives (close edit distance, same script/marker/length)
