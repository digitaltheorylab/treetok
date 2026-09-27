#!/usr/bin/env python

"""Grid-search dataset + training policy and score on an OOD tokenizer.

Scoring uses a smooth size-based penalty modulated by source ("mixed" source
clusters cost more) and pairwise internal Levenshtein spread. We report top
runs sorted by score with Pareto-frontier markers on `(score, coverage)`.
"""

import argparse
import itertools
import json
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path

import pyarrow as pa
from rapidfuzz.distance import Levenshtein

from treetok import (
    DatasetConfig,
    MergeClassifier,
    build_dataset,
    cluster_vocab,
    feature_matrix,
    read_dataset,
    write_dataset,
)
from treetok.cluster import clusters_to_jsonable

# Family balance: 4x byte-level BPE, 2x SentencePiece (gemma + multilingual
# xlm-roberta for script coverage), 1x WordPiece
TRAIN_MODELS: list[tuple[str, str]] = [
    ("answerdotai/ModernBERT-base", "modernbert.parquet"),
    ("allenai/Olmo-3-1025-7B", "olmo.parquet"),
    ("google/gemma-4-E4B", "gemma.parquet"),
    ("mistralai/Ministral-3-8B-Base-2512", "mistral.parquet"),
    ("Qwen/Qwen3.5-9B", "qwen.parquet"),
    ("google-bert/bert-base-uncased", "bert-wordpiece.parquet"),
    ("facebookAI/xlm-roberta-base", "xlmr.parquet"),
]


DIST_MIN_COUNT = 6  # only compute pairwise spread for clusters this big
DIST_SAMPLE_K = 8  # decoded members sampled per cluster (C(8, 2) = 28 pairs)


@dataclass(frozen=True)
class GridConfig:
    """One grid-search configuration.

    Parameters
    ----------
    n_pos : int
        Number of synthetic positives per tokenizer
    n_hard : int
        Number of hard negatives per tokenizer
    n_easy : int
        Number of easy negatives per tokenizer
    seed : int
        Seed for both dataset construction and the train/val split
    val_size : float
        Validation fraction in (0, 1)
    target_precision : float
        Edge-threshold precision target
    threshold_floor : float
        Lower bound for the tuned edge threshold
    merge_target_precision : float
        Merge-threshold precision target
    merge_threshold_floor : float
        Lower bound for the tuned merge threshold
    marker_policy : str
        Marker-variant semantics ("merge" or "separate") used for both
        dataset construction and the persisted classifier
    """

    n_pos: int
    n_hard: int
    n_easy: int
    seed: int
    val_size: float
    target_precision: float
    threshold_floor: float
    merge_target_precision: float
    merge_threshold_floor: float
    marker_policy: str = "merge"

    def config_id(self) -> str:
        """Return a identifier for this config.

        Returns
        -------
        str
            Identifier with preserved decimals
        """
        return (
            f"pos{self.n_pos}_hard{self.n_hard}_easy{self.n_easy}"
            f"__tp{self.target_precision}_mtp{self.merge_target_precision}"
            f"__mp-{self.marker_policy}"
        )


def _load_json(path: Path):
    """Read a JSON file.

    Parameters
    ----------
    path : Path
        File path

    Returns
    -------
    Any
        Parsed JSON content
    """
    return json.loads(path.read_text(encoding="utf-8"))


def _percentile(values: list[int | float], p: float) -> float:
    """Compute a nearest-rank percentile.

    Parameters
    ----------
    values : list[int | float]
        Samples
    p : float
        Percentile in [0, 1]

    Returns
    -------
    float
        Percentile value, or 0.0 if `values` is empty
    """
    if not values:
        return 0.0

    v = sorted(values)
    k = int(p * (len(v) - 1))

    return float(v[k])


def _sample_decoded_members(cluster: dict, k: int) -> list[str]:
    """Return up to `k` decoded members of a cluster, deterministically.

    Parameters
    ----------
    cluster : dict
        Cluster record produced by `clusters_to_jsonable`
    k : int
        Maximum number of members to return

    Returns
    -------
    list[str]
        Sorted decoded members truncated to `k` items
    """
    dec = cluster.get("decoded") or []
    if not isinstance(dec, list):
        return []

    members = [str(x) for x in dec if x]
    if not members:
        return []

    members.sort()

    return members[: min(len(members), k)]


def _pairwise_spread(members: list[str]) -> float:
    """Compute mean pairwise normalized Levenshtein distance over `members`.

    Returns 0 in `[0, 1]` for identical strings and 1 for fully dissimilar.
    Requires at least 2 members; returns 0.0 otherwise

    Parameters
    ----------
    members : list[str]
        Decoded member strings

    Returns
    -------
    float
        Mean of `1 - normalized_similarity` across all pairs
    """
    if len(members) < 2:
        return 0.0

    total = 0.0
    n_pairs = 0
    for a, b in combinations(members, 2):
        sim = Levenshtein.normalized_similarity(a, b)
        total += 1.0 - sim
        n_pairs += 1

    if n_pairs == 0:
        return 0.0

    return total / n_pairs


def score_clusters(
    *,
    clusters: list[dict],
    vocab_size: int,
    edge_threshold: float | None,
    merge_threshold: float | None,
) -> tuple[float, dict]:
    """Compute distribution-based metrics for a clustering.

    Returns a composite score (`coverage / (1 + p95_size)`) and a metrics dict
    containing cluster size distribution percentiles. Pareto ranking should use
    `(coverage, p95_size)` directly.

    Parameters
    ----------
    clusters : list[dict]
        Cluster records produced by `clusters_to_jsonable`
    vocab_size : int
        Total vocabulary size (for computing coverage percentage)
    edge_threshold : float or None
        Tuned edge threshold from the trained classifier
    merge_threshold : float or None
        Tuned merge threshold from the trained classifier

    Returns
    -------
    tuple[float, dict]
        Composite score (higher is better) and metrics dict
    """

    def c_count(c: dict) -> int:
        return int(c.get("count", 0))

    sizes = [c_count(c) for c in clusters if c_count(c) >= 2]
    coverage = sum(sizes)
    coverage_pct = 100.0 * coverage / vocab_size if vocab_size > 0 else 0.0

    size_p50 = _percentile(sizes, 0.50)
    size_p75 = _percentile(sizes, 0.75)
    size_p90 = _percentile(sizes, 0.90)
    size_p95 = _percentile(sizes, 0.95)
    size_max = float(max(sizes)) if sizes else 0.0

    spread_terms: list[float] = []
    for c in clusters:
        n = c_count(c)
        if n >= DIST_MIN_COUNT:
            members = _sample_decoded_members(c, DIST_SAMPLE_K)
            spread_terms.append(_pairwise_spread(members))

    spread_p95 = _percentile(spread_terms, 0.95)

    score = coverage / (1.0 + size_p95) if size_p95 >= 0 else float(coverage)

    metrics = {
        "coverage": coverage,
        "coverage_pct": round(coverage_pct, 2),
        "vocab_size": vocab_size,
        "n_clusters": len(sizes),
        "size_distribution": {
            "p50": size_p50,
            "p75": size_p75,
            "p90": size_p90,
            "p95": size_p95,
            "max": size_max,
        },
        "spread_p95": round(spread_p95, 3),
        "edge_threshold": edge_threshold,
        "merge_threshold": merge_threshold,
    }

    return float(score), metrics


def pareto_frontier_ids(rows: list[dict]) -> set[str]:
    """Return the set of `config_id`s on the (coverage, p95_size) Pareto front.

    A row is Pareto-optimal when no other row has both higher coverage and
    lower p95_size (with at least one strict inequality). Ties on both
    objectives keep all tied rows on the frontier.

    Parameters
    ----------
    rows : list[dict]
        Result rows (each must contain `config_id` and `ood.coverage`,
        `ood.size_distribution.p95`)

    Returns
    -------
    set[str]
        Config ids on the Pareto frontier
    """
    pts: list[tuple[int, float, str]] = []
    for r in rows:
        cid = r.get("config_id")
        if not isinstance(cid, str):
            continue

        try:
            ood = r.get("ood") or {}
            coverage = int(ood.get("coverage", 0))
            size_dist = ood.get("size_distribution") or {}
            p95_size = float(size_dist.get("p95", float("inf")))
        except Exception:
            continue

        pts.append((coverage, p95_size, cid))

    frontier: set[str] = set()
    for cov_i, p95_i, cid_i in pts:
        dominated = False
        for cov_j, p95_j, _ in pts:
            if (cov_j >= cov_i and p95_j <= p95_i) and (
                cov_j > cov_i or p95_j < p95_i
            ):
                dominated = True
                break

        if not dominated:
            frontier.add(cid_i)

    return frontier


def read_existing_ids(results_path: Path) -> set[str]:
    """Collect already-recorded `config_id` values from a JSONL file.

    Parameters
    ----------
    results_path : Path
        Path to a results JSONL file

    Returns
    -------
    set[str]
        Config ids present in the file (used to skip already-scored runs)
    """
    if not results_path.exists():
        return set()

    out: set[str] = set()
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            cid = obj.get("config_id")
            if isinstance(cid, str):
                out.add(cid)

    return out


def build_datasets(cfg: GridConfig, data_dir: Path, *, resume: bool) -> None:
    """Build per-tokenizer parquet datasets for a config.

    Parameters
    ----------
    cfg : GridConfig
        Grid configuration providing dataset composition
    data_dir : Path
        Directory to write parquet files into
    resume : bool
        If True, skip per-tokenizer builds whose parquet already exists
    """
    for model_name, filename in TRAIN_MODELS:
        out_path = data_dir / filename
        if resume and out_path.exists():
            continue

        ds_cfg = DatasetConfig(
            n_synthetic_positives=cfg.n_pos,
            n_hard_negatives=cfg.n_hard,
            n_easy_negatives=cfg.n_easy,
            seed=cfg.seed,
            marker_policy=cfg.marker_policy,
        )
        table = build_dataset(model_name, ds_cfg)
        write_dataset(table, out_path)
        print(
            f"wrote {table.num_rows} rows to {out_path} (model={model_name})"
        )


def train_classifier(
    cfg: GridConfig, data_dir: Path, model_json: Path, *, resume: bool
) -> tuple[MergeClassifier, object | None]:
    """Train (or load) a `MergeClassifier` for a config.

    Parameters
    ----------
    cfg : GridConfig
        Grid configuration providing training knobs
    data_dir : Path
        Directory containing parquet datasets to concatenate
    model_json : Path
        Path to read/write the saved classifier
    resume : bool
        If True and `model_json` exists, load instead of retraining

    Returns
    -------
    tuple[MergeClassifier, TrainReport or None]
        Fitted (or loaded) classifier and the training report when freshly
        trained; `None` when the classifier was loaded from disk
    """
    if resume and model_json.exists():
        return MergeClassifier.load(model_json), None

    parquets = sorted(data_dir.glob("*.parquet"))
    tables = [read_dataset(p) for p in parquets]
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

    clf = MergeClassifier(marker_policy=cfg.marker_policy)
    clf.fit(
        X,
        y,
        num_boost_round=400,
        early_stopping_rounds=30,
        val_size=cfg.val_size,
        random_state=cfg.seed,
        target_precision=cfg.target_precision,
        threshold_floor=cfg.threshold_floor,
        merge_target_precision=cfg.merge_target_precision,
        merge_threshold_floor=cfg.merge_threshold_floor,
    )
    clf.save(model_json)
    print(
        f"trained: edge_threshold={clf.edge_threshold_:.4f} "
        f"merge_threshold={clf.merge_threshold_:.4f}"
    )

    return clf, clf.report_


def cluster_ood(
    classifier: MergeClassifier,
    ood_model: str,
    ood_json: Path,
    *,
    resume: bool,
    n_jobs: int = -1,
) -> tuple[list[dict], int]:
    """Cluster the OOD tokenizer's vocabulary and persist the result.

    Parameters
    ----------
    classifier : MergeClassifier
        Fitted classifier
    ood_model : str
        Hugging Face model identifier for the OOD tokenizer
    ood_json : Path
        Path to read/write the clustered vocabulary
    resume : bool
        If True and `ood_json` exists, return the cached clusters
    n_jobs : int
        Threads for the classifier-scoring passes

    Returns
    -------
    tuple[list[dict], int]
        Clusters in the same shape as `clusters_to_jsonable` produces, and
        the vocabulary size
    """
    from treetok.hf import inspect

    view = inspect(ood_model)
    vocab_size = view.vocab_size

    if resume and ood_json.exists():
        return _load_json(ood_json), vocab_size

    clusters = cluster_vocab(ood_model, classifier, n_jobs=n_jobs)
    clusters_data = clusters_to_jsonable(clusters)
    ood_json.write_text(
        json.dumps(clusters_data, ensure_ascii=False), encoding="utf-8"
    )

    return clusters_data, vocab_size


def _report_to_dict(report) -> dict | None:
    """Convert a `TrainReport` dataclass into a JSON-serializable dict.

    Parameters
    ----------
    report : TrainReport or None
        Report produced by `MergeClassifier.fit`

    Returns
    -------
    dict or None
        Serializable mapping of report fields, or `None` when the classifier
        was loaded from disk (no fresh report available)
    """
    if report is None:
        return None

    return {
        "train_f1": float(report.train_f1),
        "val_f1": float(report.val_f1),
        "val_edge_precision": float(report.val_edge_precision),
        "val_edge_recall": float(report.val_edge_recall),
        "val_merge_precision": float(report.val_merge_precision),
        "val_merge_recall": float(report.val_merge_recall),
        "n_train": int(report.n_train),
        "n_val": int(report.n_val),
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point for the grid-search runner.

    Parameters
    ----------
    argv : list[str] or None
        CLI arguments (excluding program name). Uses `sys.argv` when None

    Returns
    -------
    int
        Process exit code
    """
    ap = argparse.ArgumentParser(prog="grid_search")
    ap.add_argument("--outroot", type=Path, default=Path("data/grid"))
    ap.add_argument("--ood-model", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument(
        "--marker-policy",
        choices=("merge", "separate"),
        default="merge",
        help="Marker-variant semantics for datasets, training, and clustering",
    )
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)

    resume = not args.no_resume

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.outroot / timestamp
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "configs").mkdir(parents=True, exist_ok=True)
    results_path = run_root / "results.jsonl"

    print("# grid-search")
    print(f"run_root={run_root}")
    print(f"OOD_MODEL={args.ood_model}")
    print(f"results={results_path}")

    # Grid axes
    n_positions = [2500, 4000, 5000]
    n_hards = [18000, 24000]
    n_easies = [2000]
    target_precisions = [0.95, 0.99]
    merge_target_precisions = [0.95, 0.999, 0.9995]
    seed = 0

    # Fixed parameters
    fixed = {
        "val_size": 0.5,
        "threshold_floor": 0.5,
        "merge_threshold_floor": 0.5,
    }

    grid: list[GridConfig] = []
    for n_pos, n_hard, n_easy, tp, mtp in itertools.product(
        n_positions,
        n_hards,
        n_easies,
        target_precisions,
        merge_target_precisions,
    ):
        grid.append(
            GridConfig(
                n_pos=n_pos,
                n_hard=n_hard,
                n_easy=n_easy,
                seed=seed,
                val_size=fixed["val_size"],
                target_precision=tp,
                threshold_floor=fixed["threshold_floor"],
                merge_target_precision=mtp,
                merge_threshold_floor=fixed["merge_threshold_floor"],
                marker_policy=args.marker_policy,
            )
        )

    print(f"Grid size: {len(grid)} configs")

    existing = set() if not resume else read_existing_ids(results_path)

    for cfg in grid:
        cid = cfg.config_id()
        cfg_dir = run_root / "configs" / cid
        data_dir = cfg_dir / "data"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        model_json = cfg_dir / "model.json"
        ood_json = cfg_dir / "ood.clusters.json"

        if cid in existing:
            continue

        # 1) Build per-tokenizer parquet datasets
        build_datasets(cfg, data_dir, resume=resume)

        # 2) Train classifier (or load from disk)
        classifier, train_report = train_classifier(
            cfg, data_dir, model_json, resume=resume
        )

        # 3) OOD evaluation
        clusters_data, vocab_size = cluster_ood(
            classifier, args.ood_model, ood_json, resume=resume
        )

        # 4) Score
        score, metrics = score_clusters(
            clusters=clusters_data,
            vocab_size=vocab_size,
            edge_threshold=float(classifier.edge_threshold_),
            merge_threshold=float(classifier.merge_threshold_),
        )

        size_dist = metrics["size_distribution"]
        row = {
            "config_id": cid,
            "params": {
                "n_pos": cfg.n_pos,
                "n_hard": cfg.n_hard,
                "n_easy": cfg.n_easy,
                "seed": cfg.seed,
                "threshold_floor": cfg.threshold_floor,
                "marker_policy": cfg.marker_policy,
            },
            "train": {
                "args": {
                    "val_size": cfg.val_size,
                    "seed": cfg.seed,
                    "target_precision": cfg.target_precision,
                    "threshold_floor": cfg.threshold_floor,
                    "merge_target_precision": cfg.merge_target_precision,
                    "merge_threshold_floor": cfg.merge_threshold_floor,
                },
                "edge_threshold": metrics["edge_threshold"],
                "merge_threshold": metrics["merge_threshold"],
                "report": _report_to_dict(train_report),
            },
            "ood": {
                "model": args.ood_model,
                "vocab_size": metrics["vocab_size"],
                "n_clusters": metrics["n_clusters"],
                "coverage": metrics["coverage"],
                "coverage_pct": metrics["coverage_pct"],
                "size_distribution": size_dist,
                "spread_p95": metrics["spread_p95"],
            },
            "score": score,
            "paths": {
                "run_dir": str(cfg_dir.resolve()),
                "model": str(model_json.resolve()),
                "ood_clusters": str(ood_json.resolve()),
            },
        }

        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

        existing.add(cid)

        print(
            f"[{cid}] score={score:.1f} cov={metrics['coverage']} ({metrics['coverage_pct']}%) "
            f"p95={size_dist['p95']:.0f} max={size_dist['max']:.0f}"
        )

    # Print top-k for convenience, with Pareto markers on (coverage, p95_size)
    try:
        all_rows = []
        with results_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                all_rows.append(json.loads(line))

        frontier = pareto_frontier_ids(all_rows)
        all_rows.sort(key=lambda r: -float(r.get("score", 0)))
        top = all_rows[: max(0, int(args.top_k))]
        if top:
            print(
                "\nTop by score (higher is better; [*] marks Pareto frontier "
                "on coverage x p95_size):"
            )
            print(
                f"{'score':>10}  {'coverage':>8}  {'cov%':>6}  "
                f"{'p50':>4}  {'p75':>4}  {'p90':>4}  {'p95':>4}  {'max':>4}  config_id"
            )
            for r in top:
                ood = r.get("ood") or {}
                size_dist = ood.get("size_distribution") or {}
                cid = r.get("config_id")
                marker = " [*]" if cid in frontier else ""
                print(
                    f"{float(r.get('score', 0)):>10.1f}  "
                    f"{int(ood.get('coverage', 0)):>8}  "
                    f"{float(ood.get('coverage_pct', 0)):>5.1f}%  "
                    f"{int(size_dist.get('p50', 0)):>4}  "
                    f"{int(size_dist.get('p75', 0)):>4}  "
                    f"{int(size_dist.get('p90', 0)):>4}  "
                    f"{int(size_dist.get('p95', 0)):>4}  "
                    f"{int(size_dist.get('max', 0)):>4}  "
                    f"{cid}{marker}"
                )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
