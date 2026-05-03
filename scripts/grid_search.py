#!/usr/bin/env python

"""Grid-search dataset + training policy and score on an OOD tokenizer.

This is the orchestration layer for `treetok` model selection:

- Build per-tokenizer training datasets
- Train an XGBoost merge classifier
- Evaluate clustering on an OOD tokenizer (default: Qwen/Qwen3-8B)
- Score runs with a smooth size-based penalty modulated by source ("mixed"
  source clusters cost more) and pairwise internal Levenshtein spread
- Report top runs sorted by score with Pareto-frontier markers on
  `(score, coverage)`

Artifacts are written under `data/grid/<timestamp>/configs/<config_id>/` with a
single `results.jsonl` at the run root. Re-running into the same root resumes
where it left off; pass `--no-resume` to force recomputation.
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


TRAIN_MODELS: list[tuple[str, str]] = [
    ("answerdotai/ModernBERT-base", "bert.parquet"),
    ("allenai/Olmo-3-1025-7B", "olmo.parquet"),
    ("google/gemma-4-E4B", "gemma.parquet"),
    ("mistralai/Ministral-3-8B-Base-2512", "mistral.parquet"),
    ("Qwen/Qwen3.5-9B", "qwen.parquet"),
]


# Scoring constants (smooth, single-pass)
SIZE_FREE = 6  # clusters this size or smaller incur no size penalty
SIZE_EXP = 1.5  # super-linear growth above SIZE_FREE
MIXED_MULT = 3.0  # multiplier for source == "mixed"
DIST_AMP = 4.0  # 1 + DIST_AMP * dist_term scales the per-cluster term
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

    def config_id(self) -> str:
        """Return a stable filesystem-safe identifier for this config.

        Returns
        -------
        str
            Identifier with preserved decimals
        """
        return (
            f"pos{self.n_pos}_hard{self.n_hard}_easy{self.n_easy}_seed{self.seed}"
            f"__tp{self.target_precision}_tf{self.threshold_floor}"
            f"_mf{self.merge_threshold_floor}"
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


def _p95(values: list[float]) -> float:
    """Compute a nearest-rank 95th percentile.

    Parameters
    ----------
    values : list[float]
        Samples

    Returns
    -------
    float
        95th percentile, or 0.0 if `values` is empty
    """
    if not values:
        return 0.0

    v = sorted(values)
    k = int((0.95 * (len(v) - 1)))

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
    edge_threshold: float | None,
    merge_threshold: float | None,
) -> tuple[float, dict]:
    """Compute the OOD badness score and metric breakdown for a clustering.

    Per-cluster penalty is `size_term * mixed_mult * (1 + DIST_AMP * dist)`
    where `size_term = max(0, count - SIZE_FREE) ** SIZE_EXP`. Clusters with
    `count < DIST_MIN_COUNT` use `dist = 0`. The aggregate score (lower is
    better) is the sum of per-cluster penalties

    Parameters
    ----------
    clusters : list[dict]
        Cluster records produced by `clusters_to_jsonable`
    edge_threshold : float or None
        Tuned edge threshold from the trained classifier
    merge_threshold : float or None
        Tuned merge threshold from the trained classifier

    Returns
    -------
    tuple[float, dict]
        Scalar badness and a metrics dict for the JSONL row
    """

    def c_count(c: dict) -> int:
        return int(c.get("count", 0))

    def c_source(c: dict) -> str:
        return str(c.get("source", ""))

    n_clusters = len(clusters)
    coverage = sum(1 for c in clusters if c_count(c) >= 2)
    max_count = max((c_count(c) for c in clusters), default=0)

    # Diagnostic-only counts (not used in score)
    n_big16 = sum(1 for c in clusters if c_count(c) > 16)
    n_bad32 = sum(1 for c in clusters if c_count(c) > 32)
    n_verybad64 = sum(1 for c in clusters if c_count(c) > 64)
    n_mixed = sum(1 for c in clusters if c_source(c) == "mixed")
    max_mixed_count = max(
        (c_count(c) for c in clusters if c_source(c) == "mixed"),
        default=0,
    )
    mixed_mass = sum(c_count(c) for c in clusters if c_source(c) == "mixed")

    badness = 0.0
    spread_terms: list[float] = []
    spread_terms_mixed: list[float] = []
    n_dist_clusters = 0
    n_dist_clusters_mixed = 0
    for c in clusters:
        n = c_count(c)
        if n < 2:
            continue

        size_term = max(0.0, n - SIZE_FREE) ** SIZE_EXP
        mixed_mult = MIXED_MULT if c_source(c) == "mixed" else 1.0

        if n >= DIST_MIN_COUNT:
            members = _sample_decoded_members(c, DIST_SAMPLE_K)
            dist_term = _pairwise_spread(members)
            spread_terms.append(dist_term)
            n_dist_clusters += 1
            if c_source(c) == "mixed":
                spread_terms_mixed.append(dist_term)
                n_dist_clusters_mixed += 1
        else:
            dist_term = 0.0

        badness += size_term * mixed_mult * (1.0 + DIST_AMP * dist_term)

    spread_p95 = _p95(spread_terms)
    spread_max = float(max(spread_terms) if spread_terms else 0.0)
    spread_p95_mixed = _p95(spread_terms_mixed)
    spread_max_mixed = float(
        max(spread_terms_mixed) if spread_terms_mixed else 0.0
    )

    metrics = {
        "n_clusters": n_clusters,
        "coverage": coverage,
        "max_count": max_count,
        "n_big16": n_big16,
        "n_bad32": n_bad32,
        "n_verybad64": n_verybad64,
        "n_mixed": n_mixed,
        "max_mixed_count": max_mixed_count,
        "mixed_mass": mixed_mass,
        "spread_p95": spread_p95,
        "distance": {
            "min_count": DIST_MIN_COUNT,
            "sample_k": DIST_SAMPLE_K,
            "n_clusters": n_dist_clusters,
            "n_mixed_clusters": n_dist_clusters_mixed,
            "pairwise_p95": spread_p95,
            "pairwise_max": spread_max,
            "pairwise_p95_mixed": spread_p95_mixed,
            "pairwise_max_mixed": spread_max_mixed,
        },
        "params": {
            "size_free": SIZE_FREE,
            "size_exp": SIZE_EXP,
            "mixed_mult": MIXED_MULT,
            "dist_amp": DIST_AMP,
        },
        "edge_threshold": edge_threshold,
        "merge_threshold": merge_threshold,
    }
    return float(badness), metrics


def pareto_frontier_ids(rows: list[dict]) -> set[str]:
    """Return the set of `config_id`s on the (badness, coverage) Pareto front.

    A row is on the frontier when no other row has both a smaller `score` and
    a larger `coverage`, with at least one strict inequality. Ties on both
    objectives keep all tied rows on the frontier

    Parameters
    ----------
    rows : list[dict]
        Result rows (each must contain `score`, `config_id`, and
        `ood.coverage`)

    Returns
    -------
    set[str]
        Config ids on the Pareto frontier
    """
    pts: list[tuple[float, int, str]] = []
    for r in rows:
        cid = r.get("config_id")
        if not isinstance(cid, str):
            continue

        try:
            score = float(r.get("score", float("inf")))
            coverage = int((r.get("ood") or {}).get("coverage", 0))
        except Exception:
            continue

        pts.append((score, coverage, cid))

    frontier: set[str] = set()
    for s_i, c_i, cid_i in pts:
        dominated = False
        for s_j, c_j, _ in pts:
            if (s_j <= s_i and c_j >= c_i) and (s_j < s_i or c_j > c_i):
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

    clf = MergeClassifier()
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
    n_jobs: int = 4,
) -> list[dict]:
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
    list[dict]
        Clusters in the same shape as `clusters_to_jsonable` produces
    """
    if resume and ood_json.exists():
        return _load_json(ood_json)

    clusters = cluster_vocab(ood_model, classifier, n_jobs=n_jobs)
    clusters_data = clusters_to_jsonable(clusters)
    ood_json.write_text(
        json.dumps(clusters_data, ensure_ascii=False), encoding="utf-8"
    )

    return clusters_data


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

    # Narrow grid defaults
    n_pos = 2500
    n_hards = [12000, 18000, 24000]
    n_easies = [1000, 2000]
    merge_floors = [0.85, 0.9, 0.95]
    seed = 0

    fixed = {
        "val_size": 0.5,
        "target_precision": 0.99,
        "threshold_floor": 0.8,
        "merge_target_precision": 0.999,
    }

    grid: list[GridConfig] = []
    for n_hard, n_easy, mf in itertools.product(
        n_hards, n_easies, merge_floors
    ):
        grid.append(
            GridConfig(
                n_pos=n_pos,
                n_hard=int(n_hard),
                n_easy=int(n_easy),
                seed=seed,
                val_size=float(fixed["val_size"]),
                target_precision=float(fixed["target_precision"]),
                threshold_floor=float(fixed["threshold_floor"]),
                merge_target_precision=float(fixed["merge_target_precision"]),
                merge_threshold_floor=float(mf),
            )
        )

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
        clusters_data = cluster_ood(
            classifier, args.ood_model, ood_json, resume=resume
        )

        # 4) Score
        score, metrics = score_clusters(
            clusters=clusters_data,
            edge_threshold=float(classifier.edge_threshold_),
            merge_threshold=float(classifier.merge_threshold_),
        )

        row = {
            "config_id": cid,
            "params": {
                "n_pos": cfg.n_pos,
                "n_hard": cfg.n_hard,
                "n_easy": cfg.n_easy,
                "seed": cfg.seed,
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
                "n_clusters": metrics["n_clusters"],
                "coverage": metrics["coverage"],
                "max_count": metrics["max_count"],
                "n_big16": metrics["n_big16"],
                "n_bad32": metrics["n_bad32"],
                "n_verybad64": metrics["n_verybad64"],
                "n_mixed": metrics["n_mixed"],
                "max_mixed_count": metrics["max_mixed_count"],
                "mixed_mass": metrics["mixed_mass"],
                "spread_p95": metrics["spread_p95"],
                "distance": metrics["distance"],
            },
            "score": score,
            "score_params": metrics["params"],
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
            f"[{cid}] score={score:.2f} coverage={metrics['coverage']} "
            f"spread_p95={metrics['spread_p95']:.3f} max={metrics['max_count']}"
        )

    # Print top-k for convenience, with Pareto markers on (score, coverage)
    try:
        all_rows = []
        with results_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                all_rows.append(json.loads(line))

        frontier = pareto_frontier_ids(all_rows)
        all_rows.sort(key=lambda r: float(r.get("score", 1e30)))
        top = all_rows[: max(0, int(args.top_k))]
        if top:
            print(
                "\nTop by score (lower is better; [*] marks Pareto frontier):"
            )
            print(
                f"{'score':>10}  {'coverage':>8}  {'spread_p95':>10}  "
                f"{'max':>4}  config_id"
            )
            for r in top:
                ood = r.get("ood") or {}
                cid = r.get("config_id")
                marker = " [*]" if cid in frontier else ""
                print(
                    f"{float(r.get('score')):>10.2f}  "
                    f"{int(ood.get('coverage', 0)):>8}  "
                    f"{float(ood.get('spread_p95', 0.0)):>10.3f}  "
                    f"{int(ood.get('max_count', 0)):>4}  "
                    f"{cid}{marker}"
                )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
