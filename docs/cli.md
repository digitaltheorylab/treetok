# CLI

The CLI is provided via `python -m treetok` and exposes four subcommands:

- `inspect`
- `build-dataset`
- `train`
- `cluster`

## inspect

```sh
python -m treetok inspect <model>
```

Prints tokenizer metadata as JSON (family, vocab size, marker stats).

## build-dataset

```sh
python -m treetok build-dataset <model> -o <out.parquet> \
  --n-positives 5000 --n-hard-negatives 5000 --n-easy-negatives 2000 --seed 0 \
  --marker-policy merge
```

Writes a Parquet file containing labeled token pairs and feature columns.

`--marker-policy` (default `merge`) selects marker-variant semantics: `merge`
labels marker toggles (`hello` / `\u0120hello` / `##hello`) as positives;
`separate` mines them as hard negatives. The policy is recorded in the
dataset and later in the trained model.

## train

```sh
python -m treetok train <data1.parquet> [<data2.parquet> ...] -o <model.json>
```

Trains a classifier and saves a single JSON artifact containing the booster,
feature metadata, tuned thresholds, and the marker policy inherited from the
datasets (all input datasets must share one policy).

Key flags (defaults shown):

- `--num-boost-round 400`
- `--early-stopping-rounds 30`
- `--val-size 0.2`
- `--seed 0`
- `--target-precision 0.99`
- `--threshold-floor 0.5`
- `--merge-target-precision 0.999`
- `--merge-threshold-floor 0.85`

## cluster

```sh
python -m treetok cluster <model> --classifier <model.json> \
  -k 25 -o clusters.json
```

Key flags (defaults shown):

- `-k, --top-k` (default: no limit)
- `--edge-threshold` (default: classifier tuned)
- `--merge-threshold` (default: classifier tuned)
- `--marker-policy` (default: the policy recorded in the classifier artifact)
- `--batch-size 50000`
- `-j, --n-jobs 1` (use `<= 0` to mean "all CPUs")
- `-q, --quiet` (skip stdout pretty-print)

## Output Format

When `-o` is provided, `cluster` writes a JSON list of objects:

```json
{
  "representative": "username",
  "representative_id": 5907,
  "count": 10,
  "source": "canonical",
  "tokens": ["username", "..."],
  "token_ids": [5907, "..."],
  "decoded": ["username", "..."]
}
```

`source` indicates how the cluster was formed:

- `canonical`: collapsed by a conservative canonical key
- `mixed`: a canonical group that gained extra members during anchor scoring
- `kruskal`: formed during the fallback union pass
