#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

from .find import cluster_vocab, load_hf_vocab, print_clusters, save_clusters


def main(argv=None):
    """Cluster tokens."""
    parser = argparse.ArgumentParser(
        description="Find token variants in a HuggingFace tokenizer",
    )
    parser.add_argument(
        "model",
        type=str,
        help="HuggingFace model name",
    )
    parser.add_argument(
        "-d",
        "--max-distance",
        type=int,
        default=2,
        help="Levenshtein distance threshold",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=None,
        help="Only return top-k largest clusters",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path for cluster results",
    )
    parser.add_argument(
        "-j",
        "--n-jobs",
        type=int,
        default=4,
        help="Number of workers for parallelism",
    )

    args = parser.parse_args()

    vocab, token_ids = load_hf_vocab(args.model)
    clusters = cluster_vocab(
        vocab,
        token_ids,
        max_distance=args.max_distance,
        top_k=args.top_k,
        n_jobs=args.n_jobs,
    )

    print_clusters(clusters)
    if args.output:
        save_clusters(clusters, args.output)


if __name__ == "__main__":
    main()
