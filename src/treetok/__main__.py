#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

from .find import find_token_clusters, print_clusters, save_clusters


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

    args = parser.parse_args()

    clusters = find_token_clusters(
        args.model,
        max_distance=args.max_distance,
        normalize_fn=None,
        top_k=args.top_k,
    )

    print_clusters(clusters)
    if args.output:
        save_clusters(clusters, args.output)


if __name__ == "__main__":
    main()
