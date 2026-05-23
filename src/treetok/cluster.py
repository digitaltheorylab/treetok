"""Representative-anchored clustering with Kruskal fallback."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .candidates import (
    _eligible_mask,
    iter_anchor_pair_batches,
    iter_pair_batches,
)
from .features import TokenFeatures
from .hf import inspect
from .model import MergeClassifier
from .parallel import score_pairs_batched

DEFAULT_BATCH_SIZE = 50_000
DEFAULT_N_JOBS = 1


@dataclass
class ClusterInfo:
    """One merged cluster.

    `source` reports which step produced the cluster:

    - "canonical": step 1
    - "anchor": step 2
    - "kruskal": step 3
    - "mixed": multiple steps contributed members
    """

    representative: str
    representative_id: int
    tokens: list[str]
    token_ids: list[int]
    decoded: list[str]
    count: int
    source: str = "canonical"


class _UnionFind:
    """Weighted quick-union with path compression."""

    __slots__ = ("parent", "rank", "size")

    def __init__(self, n):
        """Initialize a union-find over `n` elements.

        Parameters
        ----------
        n
            Number of elements
        """
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n

    def find(self, x) -> int:
        """Return the representative for element `x`.

        Parameters
        ----------
        x
            Element index

        Returns
        -------
        int
            Root representative
        """
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]

        return x

    def comp_size(self, x) -> int:
        """Return the size of the component containing `x`.

        Parameters
        ----------
        x
            Element index

        Returns
        -------
        int
            Component size
        """
        return self.size[self.find(x)]

    def union(self, x, y) -> int:
        """Return root after union; raises nothing if already joined."""
        rx = self.find(x)
        ry = self.find(y)

        if rx == ry:
            return rx

        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx

        self.parent[ry] = rx
        self.size[rx] += self.size[ry]

        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

        return rx


def _canonical_key(tf: TokenFeatures, idx: int) -> tuple[int, str]:
    """Return `(script_bucket, casefold(stripped))` for a token."""
    return int(tf.script[idx]), tf.casefold[idx]


def _anchor_groups(
    tf: TokenFeatures,
) -> tuple[dict[int, list[int]], np.ndarray]:
    """Group eligible tokens by canonical key.

    Parameters
    ----------
    tf : TokenFeatures
        Precomputed per-token cache

    Returns
    -------
    tuple[dict[int, list[int]], np.ndarray]
        - `groups` maps anchor index -> list of member indices (anchor +
          others)
        - `anchored_mask` is True for every token that ended up in a
          non-singleton canonical bucket (the anchor and its co-members)
    """
    elig = _eligible_mask(tf)
    n = tf.view.vocab_size

    buckets = defaultdict(list)
    for idx in range(n):
        if not elig[idx]:
            continue

        key = _canonical_key(tf, idx)

        # Skip empty casefolds; those would collapse non-related tokens
        if not key[1]:
            continue

        buckets[key].append(idx)

    groups = {}
    anchored = np.zeros(n, dtype=bool)

    ids = tf.view.ids
    for members in buckets.values():
        if len(members) < 2:
            continue

        # Anchor is the member with the lowest token id
        members_sorted = sorted(members, key=lambda i: int(ids[i]))
        anchor = members_sorted[0]
        groups[anchor] = members_sorted

        for m in members_sorted:
            anchored[m] = True

    return groups, anchored


def _anchor_pass(
    tf: TokenFeatures,
    classifier: MergeClassifier,
    groups: dict[int, list[int]],
    anchored: np.ndarray,
    anchor_admit_threshold: float,
    batch_size: int,
    n_jobs: int = 1,
) -> dict[int, list[int]]:
    """Extend each anchor's group with classifier-scored neighbors.

    Generates candidate pairs via edit-distance candidate generation, restricted
    to tokens that are either an anchor or unanchored (so anchored co-members
    are not re-evaluated). For each pair, only edges connecting an anchor to an
    unanchored token are scored; admitted candidates join the anchor

    A single false admission here pollutes the canonical group's identity, so
    this pass uses `anchor_admit_threshold` (typically `merge_threshold`) rather
    than the looser edge gate

    Each unanchored token is assigned to at most one anchor: the
    highest-probability anchor encountered wins. Ties are broken by anchor id

    Parameters
    ----------
    tf : TokenFeatures
        Precomputed per-token cache
    classifier : MergeClassifier
        Fitted classifier
    groups : dict[int, list[int]]
        Mapping of anchor index -> member indices
    anchored : np.ndarray
        True for every token that ended up in a non-singleton bucket
    anchor_admit_threshold : float
        Threshold for admitting a token to an anchor group using the classifier
        scores
    batch_size : int
        Batch size for generating candidate pairs
    n_jobs : int
        Number of threads to use

    Returns
    -------
    dict[int, list[int]]
        Updated mapping of anchor index -> member indices
    """
    if not groups:
        return groups

    anchor_set = set(groups.keys())

    # Candidate mask: all unanchored eligible tokens. The iterator adds anchors
    # back internally so a single per-stratum candidate-generation pass sees
    # both
    unanchored_mask = ~anchored

    # Track best (probability, anchor_idx) per unanchored candidate; final
    # assignment goes to the strongest anchor link
    best = {}

    batch_iter = iter_anchor_pair_batches(
        tf, anchor_set, unanchored_mask, batch_size=batch_size
    )

    for kept_pairs, kept_probs in score_pairs_batched(
        tf,
        classifier,
        batch_iter,
        threshold=anchor_admit_threshold,
        n_jobs=n_jobs,
    ):
        if kept_pairs.size == 0:
            continue

        # iter_anchor_pair_batches yields (anchor, candidate) in that order
        for k in range(len(kept_pairs)):
            anchor = int(kept_pairs[k, 0])
            other = int(kept_pairs[k, 1])
            prob = float(kept_probs[k])
            prev = best.get(other)

            # Tie-break by lower anchor id for determinism across thread orders
            if (
                prev is None
                or prob > prev[0]
                or (prob == prev[0] and anchor < prev[1])
            ):
                best[other] = (prob, anchor)

    for other, (_, anchor) in best.items():
        groups[anchor].append(other)

    return groups


def _kruskal_unanchored(
    tf: TokenFeatures,
    classifier: MergeClassifier,
    unanchored_mask: np.ndarray,
    edge_threshold: float,
    merge_threshold: float,
    batch_size: int,
    n_jobs: int,
    max_cluster_size: int = 4,
) -> dict[int, list[int]]:
    """Confidence-ordered union over the unanchored token subgraph.

    Edges with `P >= edge_threshold` are sorted by descending probability. Each
    edge is processed in order:

    - Both endpoints singleton: union admits at `edge_threshold`
    - Exactly one endpoint non-singleton: union still at `edge_threshold`, as
      long as the resulting cluster size stays <= `max_cluster_size`
    - Both endpoints non-singleton: union only when `P >= merge_threshold` AND
      the resulting size stays <= `max_cluster_size`

    The size cap is the primary chain defense: surface-form variants that
    escape the canonical anchor pass should appear as small groups, while
    anything larger is almost certainly a spurious transitive chain through
    intermediate tokens

    Parameters
    ----------
    tf : TokenFeatures
        Precomputed per-token cache
    classifier : MergeClassifier
        Fitted classifier
    unanchored_mask : np.ndarray
        True for every token that is not yet in a bucket
    edge_threshold : float
        Cutoff above which a pair is considered a positive merge candidate
    merge_threshold : float
        Stricter cutoff for bridging two non-singleton components. Always >=
        `edge_threshold`
    batch_size : int
        Batch size for generating candidate pairs
    n_jobs : int
        Number of threads to use
    max_cluster_size : int
        Maximum cluster size

    Returns
    -------
    dict[int, list[int]]
        Updated mapping of anchor index -> member indices
    """
    n = tf.view.vocab_size

    # Score all candidate pairs first, keep those above edge_threshold
    edges = []
    batch_iter = iter_pair_batches(tf, unanchored_mask, batch_size=batch_size)
    for kept_pairs, kept_probs in score_pairs_batched(
        tf,
        classifier,
        batch_iter,
        threshold=edge_threshold,
        n_jobs=n_jobs,
    ):
        if kept_pairs.size == 0:
            continue

        for k in range(len(kept_pairs)):
            edges.append(
                (
                    float(kept_probs[k]),
                    int(kept_pairs[k, 0]),
                    int(kept_pairs[k, 1]),
                )
            )

    # Sort by descending probability for Kruskal-style growth. We tie-break
    # deterministically by (i, j) to match sequential output regardless of
    # thread completion order
    edges.sort(key=lambda e: (-e[0], e[1], e[2]))

    uf = _UnionFind(n)
    for prob, i, j in edges:
        a_size = uf.comp_size(i)
        b_size = uf.comp_size(j)

        # Size cap: never grow a component beyond max_cluster_size
        if a_size + b_size > max_cluster_size:
            continue

        # Two-tier gate: bridging two non-singleton components demands
        # stricter probability than singleton-to-component additions
        if a_size > 1 and b_size > 1 and prob < merge_threshold:
            continue

        uf.union(i, j)

    # Collect components touching at least one initially-unanchored token
    groups = defaultdict(list)
    for idx in np.flatnonzero(unanchored_mask):
        groups[uf.find(idx)].append(int(idx))

    # Drop singletons; relabel anchor to lowest-id member
    out = {}
    ids = tf.view.ids
    for members in groups.values():
        if len(members) < 2:
            continue

        members_sorted = sorted(members, key=lambda i: ids[i])
        out[members_sorted[0]] = members_sorted

    return out


def cluster_view(
    tf: TokenFeatures,
    classifier: MergeClassifier,
    *,
    edge_threshold: float | None = None,
    merge_threshold: float | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_jobs: int = DEFAULT_N_JOBS,
) -> list[ClusterInfo]:
    """Run the three-step clustering pipeline on a TokenFeatures pack.

    Steps:

    1. Canonical anchors: group eligible tokens by `(script_bucket,
       casefold(stripped))`. Members of a non-singleton bucket form a cluster
       anchored on the lowest-id token. No classifier call required: the
       canonical key already proves they're surface-form variants

    2. Anchor pass: for each anchor, generate edit-distance candidates from
       nearby length bands of the same `(script, has_marker)` stratum. Score
       `classifier(anchor, candidate)`, admitting candidates whose probability
       meets `edge_threshold`

    3. Kruskal fallback: tokens still unanchored after step 2 are sometimes
       genuinely chained (orthographic variants that don't share a canonical).
       Score the edit-distance candidate pairs restricted to those tokens, sort
       by descending P, and run a confidence-ordered union with a stricter
       `merge_threshold` gate when bridging two non-singleton components

    `n_jobs` controls thread-pool parallelism for the classifier-scoring
    passes. `n_jobs == 1` runs sequentially; `n_jobs <= 0` resolves to
    `os.cpu_count()`

    Parameters
    ----------
    tf : TokenFeatures
        Precomputed per-token cache
    classifier : MergeClassifier
        Fitted classifier
    edge_threshold : float
        Cutoff above which a pair is considered a positive merge candidate
    merge_threshold : float
        Stricter cutoff for bridging two non-singleton components. Always >=
        `edge_threshold`
    batch_size : int
        Batch size for generating candidate pairs
    n_jobs : int
        Number of threads to use

    Returns
    -------
    list[ClusterInfo]
        Clusters
    """
    edge_t = (
        float(edge_threshold)
        if edge_threshold is not None
        else classifier.edge_threshold
    )
    merge_t = (
        float(merge_threshold)
        if merge_threshold is not None
        else classifier.merge_threshold
    )

    view = tf.view

    # Step 1: canonical anchor groups
    groups, anchored = _anchor_groups(tf)

    # Step 2: extend anchors with classifier-scored neighbors
    groups = _anchor_pass(
        tf,
        classifier,
        groups,
        anchored,
        anchor_admit_threshold=merge_t,
        batch_size=batch_size,
        n_jobs=n_jobs,
    )

    # Mark anchored set after step 2
    anchored_after_step2 = anchored.copy()
    for members in groups.values():
        for m in members:
            anchored_after_step2[m] = True

    # Step 3: Kruskal fallback over still-unanchored tokens
    elig = _eligible_mask(tf)
    unanchored_mask = elig & ~anchored_after_step2
    kruskal_groups = _kruskal_unanchored(
        tf,
        classifier,
        unanchored_mask,
        edge_threshold=edge_t,
        merge_threshold=merge_t,
        batch_size=batch_size,
        n_jobs=n_jobs,
    )

    # Materialize ClusterInfo
    clusters = []
    for anchor, members in groups.items():
        if len(members) < 2:
            continue

        # Source: "canonical" if every member shares the canonical key with
        # anchor; otherwise "mixed" (anchor pass added at least one)
        canon = _canonical_key(tf, anchor)
        all_canonical = all(_canonical_key(tf, m) == canon for m in members)

        clusters.append(
            ClusterInfo(
                representative=view.vocab[anchor],
                representative_id=view.ids[anchor],
                tokens=[view.vocab[m] for m in members],
                token_ids=[view.ids[m] for m in members],
                decoded=[view.decoded[m] for m in members],
                count=len(members),
                source="canonical" if all_canonical else "mixed",
            )
        )

    for anchor, members in kruskal_groups.items():
        if len(members) < 2:
            continue

        # Source: "kruskal" if added during Kruskal fallback
        clusters.append(
            ClusterInfo(
                representative=view.vocab[anchor],
                representative_id=view.ids[anchor],
                tokens=[view.vocab[m] for m in members],
                token_ids=[view.ids[m] for m in members],
                decoded=[view.decoded[m] for m in members],
                count=len(members),
                source="kruskal",
            )
        )

    clusters.sort(key=lambda c: c.count, reverse=True)

    return clusters


def cluster_vocab(
    model_name: str,
    classifier: MergeClassifier,
    *,
    edge_threshold: float | None = None,
    merge_threshold: float | None = None,
    top_k: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_jobs: int = DEFAULT_N_JOBS,
) -> list[ClusterInfo]:
    """Inspect a tokenizer, build features, and cluster.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    classifier : MergeClassifier
        Fitted classifier.
    edge_threshold : float | None
        Override for the classifier's tuned edge threshold.
    merge_threshold : float | None
        Override for the classifier's tuned merge threshold.
    top_k : int | None
        Return only the top-k clusters by member count.
    batch_size : int
        Batch size for generating candidate pairs.
    n_jobs : int
        Number of threads to use.

    Returns
    -------
    list[ClusterInfo]
        Clusters.
    """
    view = inspect(model_name)
    tf = TokenFeatures.from_view(view)
    clusters = cluster_view(
        tf,
        classifier,
        edge_threshold=edge_threshold,
        merge_threshold=merge_threshold,
        batch_size=batch_size,
        n_jobs=n_jobs,
    )
    if top_k is not None:
        clusters = clusters[:top_k]

    return clusters


def clusters_to_jsonable(clusters: list[ClusterInfo]) -> list[dict]:
    """Convert ClusterInfo list to a JSON-serializable structure."""
    return [
        {
            "representative": c.representative,
            "representative_id": int(c.representative_id),
            "count": int(c.count),
            "source": c.source,
            "tokens": c.tokens,
            "token_ids": [int(x) for x in c.token_ids],
            "decoded": c.decoded,
        }
        for c in clusters
    ]


def print_clusters(
    clusters: list[ClusterInfo], max_tokens_per_cluster: int = 10
) -> None:
    """Pretty-print clusters to stdout."""
    print("\n" + "=" * 110)
    print(f"{'#':<3} {'Representative':18} {'Count':>6} {'Src':>9}   Tokens")
    print("=" * 110)
    for n, c in enumerate(clusters, 1):
        toks = c.tokens
        if len(toks) > max_tokens_per_cluster:
            shown = ", ".join(repr(t) for t in toks[:max_tokens_per_cluster])
            shown += ", ..."
        else:
            shown = ", ".join(repr(t) for t in toks)
        print(
            f"{n:<3} {c.representative:18} {c.count:>6} {c.source:>9}   "
            f"{shown}"
        )
    print("=" * 110)
