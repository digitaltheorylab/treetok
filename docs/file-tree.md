# File Tree

```
treetok
├── data
│   └── model.json              # Default model
├── docs                        # Documentation pages
├── pixi.lock                   # pixi lockfile
├── pyproject.toml              # Python package metadata + pixi workspace config
├── README.md                   # Project README
├── scripts
│   └── train.sh                # Train the default model
└── src
    └── treetok
        ├── __init__.py         # Public API
        ├── __main__.py         # CLI entrypoint
        ├── candidates.py       # Edit-distance candidate generation
        ├── cluster.py          # Clustering pipeline
        ├── data.py             # Dataset construction
        ├── featurizer.py       # Convenience dataset->(X,y)
        ├── features.py         # Feature spec + feature building
        ├── hf.py               # Hugging Face tokenizer utilities
        ├── model.py            # XGBoost merge classifier training
        └── parallel.py         # Thread-pool utilities
```
