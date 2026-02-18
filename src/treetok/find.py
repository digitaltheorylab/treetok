"""Clustering wrapper and utilities."""

import json
from pathlib import Path

from transformers import AutoTokenizer

from .clusterer import TokenClusterer


def find_token_clusters(
    name, max_distance=2, normalize_fn=None, top_k=None, n_jobs=1
):
    """Load a tokenizer and cluster its vocabulary.

    Parameters
    ----------
    name : str
        Model identifier
    max_distance : int
        Levenshtein distance threshold for clustering
    normalize_fn : callable or None
        Normalization function
    top_k : int or None
        If set, only return `top_k` largest clusters
    n_jobs : int
        Number of worker processes. If > 0, use exactly that many workers. If
        <= 0, use all available CPUs

    Returns
    -------
    list[dict]
        List of cluster info dictionaries, sorted by cluster size
    """
    tokenizer = AutoTokenizer.from_pretrained(name)

    vocab_dict = tokenizer.get_vocab()
    vocab_items = sorted(vocab_dict.items(), key=lambda kv: kv[1])
    vocab = [t for t, _ in vocab_items]
    token_ids = [i for _, i in vocab_items]

    bktree = TokenClusterer(
        vocab, token_ids, normalize_fn, max_distance, n_jobs
    )
    bktree.cluster()

    info = bktree.get_cluster_info()
    if top_k is not None:
        info = info[:top_k]

    return info


def print_clusters(clusters, max_tokens_per_cluster=10):
    """Pretty-print cluster results.

    Parameters
    ----------
    clusters : list[dict]
        Cluster info dictionaries
    max_tokens_per_cluster : int
        Max number of token variants to print to screen
    """
    print("\n" + "=" * 100)
    print(f"{'#':<3} {'Representative':20} {'Count':>6} {'Tokens':<70}")
    print("=" * 100)

    for idx, cluster in enumerate(clusters, 1):
        rep = cluster["representative"]
        count = cluster["count"]
        tokens = cluster["tokens"]

        if len(tokens) > max_tokens_per_cluster:
            tokens = tokens[:max_tokens_per_cluster]
            tokens_str = ", ".join(repr(t) for t in tokens) + ", ..."
        else:
            tokens_str = ", ".join(repr(t) for t in tokens)

        print(f"{idx:<3} {rep:20} {count:>6}   {tokens_str}")

    print("=" * 100)


def save_clusters(clusters, output_path):
    """Save clusters to a JSON file.

    Parameters
    ----------
    clusters : list[dict]
        Cluster info dictionaries
    output_path : str or Path
        Path to JSON file
    """
    data = []
    for cluster in clusters:
        data.append(
            {
                "representative": cluster["representative"],
                "representative_id": int(cluster["representative_id"]),
                "count": int(cluster["count"]),
                "tokens": [repr(t) for t in cluster["tokens"]],
                "token_ids": [int(tid) for tid in cluster["token_ids"]],
                "normalized": [str(n) for n in cluster["normalized"]],
            }
        )

    output_path = Path(output_path)
    with output_path.open("w") as f:
        json.dump(data, f, indent=2)
