"""Thread-pool parallelism for the cluster-pass scoring loop."""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Iterator

import numpy as np

from .features import TokenFeatures, pair_features_batch
from .model import MergeClassifier


def resolve_n_jobs(n_jobs: int) -> int:
    """Resolve a user-supplied `n_jobs` to an actual worker count.

    `n_jobs <= 0` resolves to `os.cpu_count()` (or 1 if that's None).
    Otherwise the value is returned as-is, clipped to at least 1

    Parameters
    ----------
    n_jobs : int
        Worker count requested by the caller

    Returns
    -------
    int
        Resolved worker count
    """
    if n_jobs <= 0:
        return max(1, os.cpu_count() or 1)

    return max(1, int(n_jobs))


def _score_one_batch(
    tf: TokenFeatures,
    classifier: MergeClassifier,
    threshold: float,
    batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Score one batch and return the kept pairs.

    Parameters
    ----------
    tf : TokenFeatures
        Precomputed per-token cache
    classifier : MergeClassifier
        Fitted classifier
    threshold : float
        Probability cutoff
    batch : np.ndarray
        Shape `(k, 2)` batch of token-index pairs

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `(kept_pairs, kept_probs)` where both are length-k filtered
    """
    X = pair_features_batch(tf, batch)
    p = classifier.predict_proba(X)
    keep = p >= threshold

    return batch[keep], p[keep]


def score_pairs_batched(
    tf: TokenFeatures,
    classifier: MergeClassifier,
    batch_iter: Iterable[np.ndarray],
    threshold: float,
    *,
    n_jobs: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield `(kept_pairs, kept_probs)` for each input batch.

    `batch_iter` is an iterator of numpy `(k, 2)` int64 arrays; the upstream
    candidate iterator already pre-sized them.

    `n_jobs == 1` runs entirely on the main thread. `n_jobs > 1` dispatches
    scoring to a thread pool while the candidate iterator is still consumed
    sequentially on the main thread.

    Output ordering matches the order the candidate iterator emits batches.

    Parameters
    ----------
    tf : TokenFeatures
        Precomputed per-token cache
    classifier : MergeClassifier
        Fitted classifier
    batch_iter : Iterable[np.ndarray]
        Iterator of `(k, 2)` int64 pair batches
    threshold : float
        Probability cutoff
    n_jobs : int
        Thread count. `<= 0` resolves to all available CPUs

    Returns
    -------
    Iterator[tuple[np.ndarray, np.ndarray]]
        Iterator yielding filtered batches
    """
    workers = resolve_n_jobs(n_jobs)

    if workers == 1:
        for batch in batch_iter:
            yield _score_one_batch(tf, classifier, threshold, batch)
        return

    # Buffered prefetch: maintain up to `workers * 2` batches so the producer
    # never starves the pool, but bounded so we don't materialize the whole
    # candidate stream up front
    in_flight = []
    max_in_flight = max(2 * workers, 4)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        bi = iter(batch_iter)

        for batch in bi:
            in_flight.append(
                pool.submit(_score_one_batch, tf, classifier, threshold, batch)
            )
            if len(in_flight) >= max_in_flight:
                break

        while in_flight:
            fut = in_flight.pop(0)
            yield fut.result()

            try:
                batch = next(bi)
            except StopIteration:
                continue
            in_flight.append(
                pool.submit(_score_one_batch, tf, classifier, threshold, batch)
            )
