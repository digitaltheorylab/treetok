"""treetok: tokenizer surface-form clustering with a learned merge classifier."""

from .cluster import cluster_vocab, print_clusters
from .data import (
    DatasetConfig,
    build_dataset,
    feature_matrix,
    read_dataset,
    write_dataset,
)
from .featurizer import TreetokFeaturizer
from .hf import TokenizerView, inspect
from .model import MergeClassifier

__all__ = [
    "DatasetConfig",
    "MergeClassifier",
    "TreetokFeaturizer",
    "TokenizerView",
    "build_dataset",
    "cluster_vocab",
    "feature_matrix",
    "inspect",
    "print_clusters",
    "read_dataset",
    "write_dataset",
]
