## treetok

Cluster surface-form variants in Hugging Face tokenizer vocabularies using a
learned merge classifier.

`treetok` does the following:

- Inspects a `transformers.AutoTokenizer`
- Builds ~30 string-based per-pair features
- Trains an XGBoost binary classifier
- Clusters the vocabulary

Clustering uses an anchor-based algorithm designed to avoid transitive chain
blow-ups.

### Install

This repo is set up as a `pixi` workspace:

```sh
pixi install
pixi run python -m treetok inspect gpt2
```

### Quickstart (Python API)

```python
from treetok import (
    MergeClassifier,
    build_dataset,
    cluster_vocab,
    feature_matrix,
    inspect,
    print_clusters,
    read_dataset,
    write_dataset,
)

# Inspect a tokenizer
view = inspect("gpt2")
print(view.family, view.vocab_size, view.prefix_marker, view.marker_kind)

# Build one or more labeled datasets
write_dataset(build_dataset("gpt2"), "data/gpt2.parquet")

# Train
table = read_dataset("data/gpt2.parquet")
X, y = feature_matrix(table)

clf = MergeClassifier().fit(X, y)
clf.save("model.json")

# Cluster
clusters = cluster_vocab("gpt2", clf, top_k=25)
print_clusters(clusters)
```

### Quickstart (CLI)

```sh
python -m treetok inspect gpt2
python -m treetok build-dataset gpt2 -o data/gpt2.parquet
python -m treetok train data/gpt2.parquet -o model.json
python -m treetok cluster gpt2 --classifier model.json -k 25 -o clusters.json
```

### Default Model

`treetok` ships with a model trained on the following tokenizer mix:

- [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)
- [Olmo 3 7B](https://huggingface.co/allenai/Olmo-3-1025-7B)
- [Gemma 4 EB](https://huggingface.co/google/gemma-4-E4B)
- [Ministral 3 8B](https://huggingface.co/mistralai/Ministral-3-8B-Base-2512)
- [Qwen 3.5 9B](https://huggingface.co/Qwen/Qwen3.5-9B)

Your mileage may vary with this model. For best performance, train one on your
own tokenizer(s). See `docs/training.md`.

### Docs

- `docs/index.md`
- `docs/overview.md`
- `docs/how-it-works.md`
- `docs/training.md`
- `docs/cli.md`
- `docs/file-tree.md`

### Limitations

- Training labels are synthetic + heuristic; you may need more tokenizers and/or
  harder negatives for robust thresholds
- The goal is surface-form variants, not morphology or semantics
