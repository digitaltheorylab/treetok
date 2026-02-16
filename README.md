treetok
-------

Find and cluster near-duplicate tokens in HuggingFace tokenizer vocabularies
using Levenshtein distance

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
        └── find.py                 High-level API: find, print, save
```

**Usage**

To call from the command line:

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

To use in Python:

```py
from treetok import find_token_clusters, print_clusters

normalize_fn = lambda x : x.strip()
clusters = find_token_clusters(
    "gpt2", max_distance=2, normalize_fn=normalize_fn, top_k=10
)
print_clusters(clusters)
```

`normalize_fn` accepts any callable with a string input. The default
normalization strategy runs NFKC normalization and whitespace stripping on
tokens.

