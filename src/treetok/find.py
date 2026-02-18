"""Clustering wrapper and utilities."""

import json
from pathlib import Path

from .clusterer import TokenClusterer


def cluster_vocab(
    vocab,
    token_ids=None,
    max_distance=2,
    normalize_fn=None,
    top_k=None,
    n_jobs=1,
):
    """Cluster tokenizer vocabulary.

    Parameters
    ----------
    vocab : sequence[str]
        Tokenizer vocabulary
    token_ids : sequence[int] or None
        Token IDs
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
    if token_ids is None:
        token_ids = list(range(len(vocab)))

    clusterer = TokenClusterer(
        vocab, token_ids, normalize_fn, max_distance, n_jobs
    )
    clusterer.cluster()

    info = clusterer.get_cluster_info()
    if top_k is not None:
        info = info[:top_k]

    return info


def load_hf_vocab(model_name):
    """Load vocab and token IDs from a HuggingFace tokenizer.

    Parameters
    ----------
    model_name : str
        Tokenizer's model name

    Returns
    -------
    tuple[list[str], list[int]]
        Vocab and token IDs

    Raises
    ------
    ImportError
        If `transformers` isn't available
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "HuggingFace support requires the optional dependency "
            "transformers'. Install it separately or use the pixi "
            "'huggingface' environment"
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_dict = sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])
    vocab = [t for t, _ in vocab_dict]
    token_ids = [idx for _, idx in vocab_dict]

    return vocab, token_ids


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
