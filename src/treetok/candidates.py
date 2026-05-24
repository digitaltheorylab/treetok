"""Candidate-pair generation via batched edit-distance scoring.

This module implements edit-distance candidate generation over stratified token
buckets.

Our classifier is precise but expensive per pair, so we never run it on the
entire V*(V-1)/2 cross product. Instead, this module stratifies the vocabulary
by `(scripts, has_marker, length)` and uses `rapidfuzz.process.cidst` to
compute Levenshtein distance for every (query, choice) pair within a stratum
band.
"""

import math
from collections import defaultdict
from typing import Iterator

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from .features import TokenFeatures

MIN_LEN = 3
MAX_DIST_CEILING = 3
DEFAULT_BATCH_SIZE = 50_000

# Cap the per-stratum cdist matrix to bound peak memory; the matrix is
# (Q_CHUNK, |choices|) of uint8. 1024 * 50000 = 50 MB upper bound which is
# comfortable. Strata rarely have more than a few thousand choices anyway.
_Q_CHUNK = 1024


def _length_aware_max_dist(token_len: int) -> int:
    """Return the max edit distance allowed for a given token length.

    `min(MAX_DIST_CEILING, ceil(0.4 * len))`: short tokens only match
    near-exact candidates while longer tokens get more slack

    Parameters
    ----------
    token_len : int
        Length of the token on the comparison surface

    Returns
    -------
    int
        Maximum Levenshtein distance cutoff for this length
    """
    if token_len < MIN_LEN:
        return 0

    return min(MAX_DIST_CEILING, max(1, math.ceil(0.4 * token_len)))


def _eligible_mask(tf: TokenFeatures) -> np.ndarray:
    """Return a boolean mask of tokens eligible for clustering.

    Excluded:
    - Comparison-surface length below `MIN_LEN`
    - Special tokens, added tokens
    - Bracket-wrapped reserved slots (e.g. BERT's `[unused0]`..`[unusedN]`).
      Detected on `view.stripped` because the bracketing convention is part
      of the raw token surface, independent of the byte-level decode pass
    """
    view = tf.view
    mask = tf.compare_len >= MIN_LEN

    ids = view.ids
    skip_ids = view.special_token_ids | view.added_token_ids
    if skip_ids:
        skip = np.array([int(i) in skip_ids for i in ids], dtype=bool)
        mask &= ~skip

    bracketed = np.array(
        [
            (s.startswith("[") and s.endswith("]"))
            or (s.startswith("<") and s.endswith(">"))
            for s in view.stripped
        ],
        dtype=bool,
    )
    mask &= ~bracketed

    return mask


def _stratify(
    tf: TokenFeatures, base_mask: np.ndarray
) -> dict[tuple[int, bool, int], list[int]]:
    """Group token indices by (script, has_marker, compare_length).

    Parameters
    ----------
    tf : TokenFeatures
        Precomputed per-token cache
    base_mask : np.ndarray
        Boolean mask selecting which token indices to include

    Returns
    -------
    dict[tuple[int, bool, int], list[int]]
        Map from `(script, has_marker, length)` to a list of token indices
    """
    has_marker = tf.has_marker
    script = tf.script
    compare_len = tf.compare_len

    strata = defaultdict(list)
    for idx in np.flatnonzero(base_mask):
        key = (
            int(script[idx]),
            bool(has_marker[idx]),
            int(compare_len[idx]),
        )
        strata[key].append(idx)

    return strata


def _cdist_pairs(
    tf: TokenFeatures,
    queries: list[int],
    choices: list[int],
    max_dist: int,
    *,
    same_bucket: bool,
    enforce_lt: bool,
) -> np.ndarray | None:
    """Score (queries x choices) with rapidfuzz.cdist; return kept (i, j) pairs.

    Parameters
    ----------
    queries, choices : list[int]
        Global token indices for the query / choice sides
    max_dist : int
        Levenshtein cutoff. Cells > `max_dist` are silently dropped
    same_bucket : bool
        When True, also drop self-pairs (query and choice refer to the same
        global index)
    enforce_lt : bool
        When True, additionally drop pairs where global `i >= j`. Used when
        the caller wants symmetric (i < j) output

    Returns
    -------
    np.ndarray or None
        Shape (k, 2) int64 array of kept pairs. `None` if empty
    """
    if not queries or not choices:
        return None

    compare = tf.compare
    q_strs = [compare[g] for g in queries]
    c_strs = [compare[g] for g in choices]
    q_arr = np.asarray(queries, dtype=np.int64)
    c_arr = np.asarray(choices, dtype=np.int64)

    parts: list[np.ndarray] = []
    for start in range(0, len(q_strs), _Q_CHUNK):
        end = min(start + _Q_CHUNK, len(q_strs))
        qchunk_strs = q_strs[start:end]
        qchunk_idx = q_arr[start:end]

        D = process.cdist(
            qchunk_strs,
            c_strs,
            scorer=Levenshtein.distance,
            score_cutoff=max_dist,
            dtype=np.uint8,
        )

        # Anything beyond cutoff is filled with score_cutoff + 1 by rapidfuzz
        rows, cols = np.where(D <= max_dist)
        if rows.size == 0:
            continue

        gi = qchunk_idx[rows]
        gj = c_arr[cols]

        # Self-pair filter: same global index
        if same_bucket:
            keep = gi != gj
            if not keep.all():
                gi = gi[keep]
                gj = gj[keep]

        if enforce_lt:
            keep = gi < gj
            if not keep.all():
                gi = gi[keep]
                gj = gj[keep]

        if gi.size == 0:
            continue

        parts.append(np.column_stack([gi, gj]))

    if not parts:
        return None

    return np.concatenate(parts, axis=0)


def _yield_in_batches(
    arrays: Iterator[np.ndarray], batch_size: int
) -> Iterator[np.ndarray]:
    """Re-batch a stream of (k_i, 2) arrays into chunks of `batch_size`."""
    buf: list[np.ndarray] = []
    held = 0
    for arr in arrays:
        if arr is None or arr.size == 0:
            continue

        buf.append(arr)
        held += len(arr)
        while held >= batch_size:
            cat = np.concatenate(buf, axis=0)
            yield cat[:batch_size]

            rem = cat[batch_size:]
            buf = [rem] if rem.size else []
            held = len(rem)

    if held:
        yield np.concatenate(buf, axis=0) if len(buf) > 1 else buf[0]


def iter_pair_batches(
    tf: TokenFeatures,
    mask: np.ndarray | None = None,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[np.ndarray]:
    """Yield batches of candidate (i, j) pairs with i < j.

    Pairs come from cdist within `(script, has_marker)` strata, scanning each
    length bucket against bands at `L..L + q_max` (the upper half; the
    symmetric lower half would produce duplicates, so we skip it)

    Parameters
    ----------
    tf : TokenFeatures
        Precomputed per-token cache
    mask : np.ndarray or None
        Optional extra boolean mask to apply on top of `eligible_mask`
    batch_size : int
        Target number of pairs per batch

    Returns
    -------
    Iterator[np.ndarray]
        Iterator of `(k, 2)` int64 arrays
    """
    elig = _eligible_mask(tf)
    if mask is not None:
        elig = elig & np.asarray(mask, dtype=bool)

    strata = _stratify(tf, elig)

    def gen() -> Iterator[np.ndarray]:
        # Group strata by (script, has_marker) so we know which length
        # buckets coexist.
        by_sm: dict[tuple[int, bool], dict[int, list[int]]] = defaultdict(dict)
        for (s, m, L), members in strata.items():
            by_sm[(s, m)][L] = members

        for (s, m), length_buckets in by_sm.items():
            for L, q_members in length_buckets.items():
                q_max = _length_aware_max_dist(L)
                if q_max == 0:
                    continue
                for off in range(0, q_max + 1):
                    other_L = L + off
                    c_members = length_buckets.get(other_L)
                    if not c_members:
                        continue
                    same_bucket = off == 0
                    # Only the same-bucket case can produce mirrored
                    # (i, j) / (j, i) duplicates from a single cdist call,
                    # so enforce_lt only there. Cross-bucket pairs are
                    # naturally unique because each bucket pair is visited
                    # exactly once by scanning off in [0, q_max].
                    pairs = _cdist_pairs(
                        tf,
                        q_members,
                        c_members,
                        q_max,
                        same_bucket=same_bucket,
                        enforce_lt=same_bucket,
                    )
                    if pairs is not None:
                        yield pairs

    yield from _yield_in_batches(gen(), batch_size)


def iter_pairs(
    tf: TokenFeatures, mask: np.ndarray | None = None
) -> Iterator[tuple[int, int]]:
    """Yield candidate (i, j) pairs one at a time.

    Convenience adapter around `iter_pair_batches` for callers that want a flat
    tuple stream (e.g. dataset construction in `data.py`)
    """
    for batch in iter_pair_batches(tf, mask):
        for k in range(len(batch)):
            yield int(batch[k, 0]), int(batch[k, 1])


def iter_anchor_pair_batches(
    tf: TokenFeatures,
    anchor_idx: set[int],
    candidate_mask: np.ndarray,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[np.ndarray]:
    """Yield batches of (anchor, candidate) pairs.

    Column 0 is always an anchor; column 1 is never an anchor. Designed for the
    clustering anchor pass: only anchor vs. non-anchor edges are scored, so the
    classifier never re-evaluates pairs already collapsed by the canonical key
    step

    Parameters
    ----------
    tf : TokenFeatures
        Precomputed per-token cache
    anchor_idx : set[int]
        Set of token indices that are anchors
    candidate_mask : np.ndarray
        Boolean mask selecting which tokens are eligible candidates
    batch_size : int
        Target number of pairs per batch

    Returns
    -------
    Iterator[np.ndarray]
        Iterator of `(k, 2)` int64 arrays, where column 0 is an anchor
    """
    if not anchor_idx:
        return

    n = tf.view.vocab_size
    elig = _eligible_mask(tf) & candidate_mask

    is_anchor = np.zeros(n, dtype=bool)
    for a in anchor_idx:
        elig[a] = True
        is_anchor[a] = True

    strata = _stratify(tf, elig)

    # Only strata that contain at least one anchor are relevant
    anchor_sm = set()
    for a in anchor_idx:
        anchor_sm.add((tf.script[a], tf.has_marker[a]))

    def gen() -> Iterator[np.ndarray]:
        by_sm = defaultdict(dict)
        for (s, m, L), members in strata.items():
            if (s, m) not in anchor_sm:
                continue

            by_sm[(s, m)][L] = members

        for (_s, _m), length_buckets in by_sm.items():
            for L, members in length_buckets.items():
                q_max = _length_aware_max_dist(L)
                if q_max == 0:
                    continue

                # Queries: anchors in this length bucket only
                queries = [g for g in members if is_anchor[g]]
                if not queries:
                    continue

                for off in range(-q_max, q_max + 1):
                    other_L = L + off
                    c_members = length_buckets.get(other_L)
                    if not c_members:
                        continue

                    # Choices are non-anchors only
                    choices = [g for g in c_members if not is_anchor[g]]
                    if not choices:
                        continue

                    pairs = _cdist_pairs(
                        tf,
                        queries,
                        choices,
                        q_max,
                        same_bucket=False,
                        enforce_lt=False,
                    )
                    if pairs is not None:
                        yield pairs

    yield from _yield_in_batches(gen(), batch_size)
