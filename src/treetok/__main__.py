"""treetok CLI."""

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Run the `inspect` subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments
    """
    from .hf import inspect

    view = inspect(args.model)
    payload = {
        "model_name": view.model_name,
        "family": view.family,
        "vocab_size": view.vocab_size,
        "prefix_marker": view.prefix_marker,
        "marker_kind": view.marker_kind,
        "n_with_marker": int(view.has_marker.sum()),
        "n_special": len(view.special_token_ids),
        "n_added": len(view.added_token_ids),
    }
    json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


def _cmd_build_dataset(args: argparse.Namespace) -> int:
    """Run the `build-dataset` subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments
    """
    from .data import DatasetConfig, build_dataset, write_dataset

    cfg = DatasetConfig(
        n_synthetic_positives=args.n_positives,
        n_hard_negatives=args.n_hard_negatives,
        n_easy_negatives=args.n_easy_negatives,
        seed=args.seed,
    )
    table = build_dataset(args.model, cfg)
    write_dataset(table, args.output)
    print(f"wrote {table.num_rows} rows to {args.output} (model={args.model})")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """Run the `train` subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments
    """
    from .data import feature_matrix, read_dataset
    from .model import MergeClassifier

    tables = [read_dataset(p) for p in args.data]
    table = (
        pa.concat_tables(tables, promote_options="default")
        if len(tables) > 1
        else tables[0]
    )
    print(f"loaded {table.num_rows} rows across {len(tables)} file(s)")

    X, y = feature_matrix(table)
    print(
        f"X shape: {X.shape}, positives: {int((y == 1).sum())}, "
        f"negatives: {int((y == 0).sum())}"
    )

    clf = MergeClassifier()
    clf.fit(
        X,
        y,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        val_size=args.val_size,
        random_state=args.seed,
        target_precision=args.target_precision,
        threshold_floor=args.threshold_floor,
        merge_target_precision=args.merge_target_precision,
        merge_threshold_floor=args.merge_threshold_floor,
    )

    print(
        f"trained: edge_threshold={clf.report_.edge_threshold:.4f} "
        f"merge_threshold={clf.report_.merge_threshold:.4f}"
    )
    print(
        f"  edge:  val_f1={clf.report_.val_f1:.4f} "
        f"P={clf.report_.val_edge_precision:.4f} "
        f"R={clf.report_.val_edge_recall:.4f}"
    )
    print(
        f"  merge: P={clf.report_.val_merge_precision:.4f} "
        f"R={clf.report_.val_merge_recall:.4f}"
    )
    print(
        f"  (train_f1={clf.report_.train_f1:.4f}, "
        f"n_train={clf.report_.n_train}, n_val={clf.report_.n_val})"
    )

    clf.save(args.output)
    print(f"saved classifier to {args.output}")
    return 0


def _cmd_cluster(args: argparse.Namespace) -> int:
    """Run the `cluster` subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments
    """
    from .cluster import cluster_vocab, clusters_to_jsonable, print_clusters
    from .model import MergeClassifier

    clf = MergeClassifier.load(args.classifier)
    clusters = cluster_vocab(
        args.model,
        clf,
        edge_threshold=args.edge_threshold,
        merge_threshold=args.merge_threshold,
        top_k=args.top_k,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
    )

    if not args.quiet:
        print_clusters(clusters)

    if args.output:
        Path(args.output).write_text(
            json.dumps(
                clusters_to_jsonable(clusters), indent=2, ensure_ascii=False
            )
        )
        print(f"saved {len(clusters)} clusters to {args.output}")

    return 0


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for the `treetok` CLI.

    Parameters
    ----------
    argv : list[str] or None
        CLI arguments (excluding program name). Uses `sys.argv` when None

    Returns
    -------
    int
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="treetok",
        description=(
            "Cluster tokenizer surface-form variants with a learned classifier."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_inspect = sub.add_parser("inspect", help="Show tokenizer metadata")
    p_inspect.add_argument("model", type=str)
    p_inspect.set_defaults(func=_cmd_inspect)

    p_data = sub.add_parser(
        "build-dataset",
        help="Construct a labeled training dataset from a tokenizer",
    )
    p_data.add_argument("model", type=str)
    p_data.add_argument(
        "-o", "--output", type=Path, required=True, help="Parquet output path"
    )
    p_data.add_argument("--n-positives", type=int, default=5000)
    p_data.add_argument("--n-hard-negatives", type=int, default=5000)
    p_data.add_argument("--n-easy-negatives", type=int, default=2000)
    p_data.add_argument("--seed", type=int, default=0)
    p_data.set_defaults(func=_cmd_build_dataset)

    p_train = sub.add_parser("train", help="Train the merge classifier")
    p_train.add_argument(
        "data", nargs="+", type=Path, help="One or more dataset .parquet files"
    )
    p_train.add_argument(
        "-o", "--output", type=Path, required=True, help="Model output path"
    )
    p_train.add_argument("--num-boost-round", type=int, default=400)
    p_train.add_argument("--early-stopping-rounds", type=int, default=30)
    p_train.add_argument("--val-size", type=float, default=0.2)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument(
        "--target-precision",
        type=float,
        default=0.99,
        help="Tune threshold for the lowest cutoff hitting this precision",
    )
    p_train.add_argument(
        "--threshold-floor",
        type=float,
        default=0.5,
        help="Lower bound applied to the tuned edge threshold",
    )
    p_train.add_argument(
        "--merge-target-precision",
        type=float,
        default=0.999,
        help="Precision target for the stricter merge threshold",
    )
    p_train.add_argument(
        "--merge-threshold-floor",
        type=float,
        default=0.85,
        help="Lower bound applied to the tuned merge threshold",
    )
    p_train.set_defaults(func=_cmd_train)

    p_cluster = sub.add_parser(
        "cluster", help="Cluster a tokenizer's vocabulary"
    )
    p_cluster.add_argument("model", type=str)
    p_cluster.add_argument(
        "--classifier", type=Path, required=True, help="Path to model.json"
    )
    p_cluster.add_argument("-k", "--top-k", type=int, default=None)
    p_cluster.add_argument(
        "--edge-threshold",
        type=float,
        default=None,
        help="Override the classifier's tuned edge threshold",
    )
    p_cluster.add_argument(
        "--merge-threshold",
        type=float,
        default=None,
        help="Override the classifier's tuned merge threshold",
    )
    p_cluster.add_argument("--batch-size", type=int, default=50_000)
    p_cluster.add_argument(
        "-j",
        "--n-jobs",
        type=int,
        default=1,
        help="Threads for the classifier-scoring passes",
    )
    p_cluster.add_argument(
        "-o", "--output", type=Path, default=None, help="JSON output path"
    )
    p_cluster.add_argument(
        "-q", "--quiet", action="store_true", help="Skip stdout pretty-print"
    )
    p_cluster.set_defaults(func=_cmd_cluster)

    args = parser.parse_args(argv)

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
