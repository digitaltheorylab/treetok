## treetok

Find and cluster surface-form token duplicates using Levensthein distance

**File tree**

```
├── pyproject.toml
├── README.md
└── src
    └── treetok
        ├── __init__.py             Package init
        ├── __main__.py             CLI entrypoint
        ├── bktree.py               Flat BK-tree and union-find
        ├── cluster.py              Stratified clusterer
        ├── find.py                 High-level API: find, print, save
        └── parallel.py             Multiprocessing functions
```

## Usage

### API

`treetok` may be called from other Python programs like so:

```py
from treetok import cluster_vocab, print_clusters

vocab = ["Hello", "hello", "world", "worlds"]
normalize_fn = lambda x : x.strip()

clusters = cluster_vocab(
    vocab, max_distance=2, normalize_fn=normalize_fn, top_k=10
)
print_clusters(clusters)
```

`normalize_fn` accepts any callable with a string input. The default
normalization strategy does the following:

1. NFKC normalize
2. Strip whitespace and convert to lowercase
3. Remove common edge punctuation

### Command line

`treetok` also work as a CLI for HuggingFace tokenizers. To use it for this,
install the `cli` environment with `pixi` (or install `transformers`)
separately:

```sh
pixi install -e cli
pixi s -e cli
```

Then call `treetok` like so:

1. Show all clusters for GPT-2

   ```sh
   python -m treetok gpt2
   ```

2. Show BERT's top 20 clusters with a big max distance, then save to JSON

   ```sh
   python -m treetok bert-base-uncased -d 4 -k 20 -o clusters.json
   ```

3. (Optional) Show all options

   ```sh
   python -m treetok -h
   ```
