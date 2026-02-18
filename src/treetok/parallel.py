"""Fork-based multiprocessing backend."""

import sys
import multiprocessing as mp

_WORKER_DATA = None
_WORKER_CORE_FN = None


def supports_fork():
    """Check if the platform supports fork-based multiprocessing.

    Returns
    -------
    bool
        True if fork is available and safe to use
    """
    # Windows won't support
    if sys.platform == "win32":
        return False

    # Check if fork is available in multiprocessing
    if not hasattr(mp, "get_start_method"):
        return False

    # Check available start methods
    try:
        methods = mp.get_all_start_methods()
        return "fork" in methods
    except Exception:
        return False


def _init_worker(data, core_fn):
    """Initialize a worker.

    Parameters
    ----------
    data : dict
        Clusterer data
    """
    global _WORKER_DATA, _WORKER_CORE_FN
    _WORKER_DATA = data
    _WORKER_CORE_FN = core_fn


def _search_neighbors_batch(args):
    """Worker function for parallel neighbor search.

    Parameters
    ----------
    args : tuple[int]
        Start index and end indices

    Returns
    -------
    list[tuple[int, list[int]]]
        List of (idx, neighbors) tuples for indices in range
    """
    start, end = args
    data = _WORKER_DATA
    core_fn = _WORKER_CORE_FN

    out = []
    for idx in range(start, end):
        neighbors = core_fn(
            idx,
            data["normalized"],
            data["norm_len"],
            data["prefix_marker"],
            data["trees"],
            data["strata_normalized"],
            data["strata"],
            data["global_to_local"],
            data["max_distance"],
            data["min_cluster_length"],
        )
        out.append((idx, neighbors))

    return out


def iter_neighbors_fork(*, vocab_size, n_jobs, data, core_fn, chunk_size=None):
    """Yield (idx, neighbors) using a fork-based multiprocessing pool.

    Returns
    -------
    vocab_size : int
        Total number of vocabulary items
    n_jobs : int
        Number of worker processes. If > 0, use exactly that many workers. If
        <= 0, use all available CPUs
    data : dict
        Payload for `core_fn`
    core_fn : callable
        Core neighbor-search function
    chunk_size : int or None
        Size of contiguous index ranges assigned to each index. If None, a
        default is chosen

    Yields
    ------
    tuple[int, list[int]]
        Global vocabulary index and its list of neighbors

    Raises
    ------
    RunTimeError
        If fork isn't available
    """
    if not supports_fork():
        raise RuntimeError("Fork unavailable; cannot use parallel backend")

    n_workers = n_jobs if n_jobs > 0 else mp.cpu_count()
    n_workers = max(1, min(n_workers, vocab_size))

    if chunk_size is None:
        chunk_size = max(1, vocab_size // n_workers)

    work_items = [
        (idx, min(idx + chunk_size, vocab_size))
        for idx in range(0, vocab_size, chunk_size)
    ]

    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(data, core_fn),
    ) as pool:
        for chunk in pool.imap(_search_neighbors_batch, work_items, 1):
            for idx, neighbors in chunk:
                yield idx, neighbors
