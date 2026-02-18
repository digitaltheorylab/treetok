"""Token clusterer."""

import unicodedata
import warnings
from collections import Counter, defaultdict

import numpy as np

from .bktree import _FlatBKTree, _UnionFind
from . import parallel


def _search_neighbors_core(
    idx,
    normalized,
    norm_len,
    prefix_marker,
    trees,
    strata_normalized,
    strata,
    global_to_local,
    max_distance,
    min_cluster_length,
):
    """Core neighbor search.

    Parameters
    ----------
    idx : int
        Global vocabulary index
    normalized : list[str]
        Normalized tokens
    norm_len : np.ndarray
        Token lengths
    prefix_marker : list[str]
        Prefix markers per token
    trees : dict
        BK-trees per stratum
    strata : dict
        Global indices per stratum
    global_to_local : list
        Mapping from global to (key, local) indices
    max_distance : int
        Maximum Levenshtein distance
    min_cluster_length : int
        Minimum token length for clustering

    Returns
    -------
    list[int]
        Global indices of neighbor tokens
    """
    tok = normalized[idx]
    tok_len = norm_len[idx]
    marker = prefix_marker[idx]

    # Compute effective max distance
    if tok_len < min_cluster_length:
        max_dist = 0
    else:
        max_dist = min(max_distance, tok_len // 2)

    if max_dist == 0:
        return []

    neighbors = []
    for offset in range(-max_dist, max_dist + 1):
        key = (marker, tok_len + offset)
        tree = trees.get(key)
        if tree is None:
            continue

        norms = strata_normalized[key]
        members = strata[key]

        query_local = (
            global_to_local[idx][1] if key == global_to_local[idx][0] else -1
        )

        for h in tree.search(tok, query_local, max_dist, norms):
            neighbors.append(members[h])

    return neighbors


class TokenClusterer:
    """Cluster transformer-tokenizer tokens by Levenshtein distance.

    Internally the vocabulary is stratified by (prefix_marker, length) into
    many small `_FlatBKTree` instances so that:

    - The prefix-marker constraint is enforced by construction
    - The length constraint is enforced by querying only adjacent length bands
    - Each individual tree is small

    Clustering uses union-find with optional multiprocessing support.
    """

    # Minimum token length (after normalization) to be eligible for clustering
    MIN_CLUSTER_LENGTH = 3

    def __init__(
        self, vocab, ids=None, normalize_fn=None, max_distance=1, n_jobs=1
    ):
        """Initialize the clusterer.

        Parameters
        ----------
        vocab : list[str]
            Tokens from the tokenizer vocabulary
        ids : list[int] or None
            Corresponding token IDs
        normalize_fn : callable or None
            Token normalizer applied before distance computation. Defaults to
            NFKC + strip
        max_distance : int
            Maximum Levenshtein distance for clustering
        n_jobs : int
            Number of worker processes. If > 0, use exactly that many workers.
            If <= 0, use all available CPUs
        """
        self.vocab = list(vocab)
        self.vocab_size = len(vocab)

        ids = list(range(self.vocab_size)) if ids is None else ids
        self.token_ids = np.array(ids, dtype=np.int32)

        fn = self._basic_normalize if normalize_fn is None else normalize_fn
        self.normalize_fn = fn
        self.max_distance = max_distance

        # Multiprocessing configuration
        fork_ok = parallel.supports_fork()
        self.n_jobs = n_jobs
        self._use_fork = fork_ok and n_jobs != 1
        if not fork_ok and n_jobs not in (0, 1):
            warnings.warn(
                "Fork-based multiprocessing not available on this platform. "
                "Falling back to single-threaded execution.",
                RuntimeWarning,
            )

        # Precompute normalized forms and their lengths
        self.normalized = [self.normalize_fn(t) for t in vocab]
        self.norm_len = np.array(
            [len(s) for s in self.normalized], dtype=np.int32
        )

        # Infer prefix markers and assign one per token
        markers = self._infer_prefix_markers(vocab)
        self.prefix_marker = [
            self._assign_prefix_marker(self.normalized[i], markers)
            for i in range(self.vocab_size)
        ]

        # Strata keyed by (prefix_marker, length)
        self._strata = defaultdict(list)
        self._trees = {}
        self._strata_normalized = {}
        self._global_to_local = [None] * self.vocab_size

        self._built = False
        self.clusters = None

    def build(self):
        """Build per-stratum BK-trees."""
        if self.vocab_size == 0:
            return

        # Assign every token to its stratum
        for idx in range(self.vocab_size):
            key = (self.prefix_marker[idx], int(self.norm_len[idx]))
            local = len(self._strata[key])
            self._strata[key].append(idx)
            self._global_to_local[idx] = (key, local)

        # Build a flat BK-tree per stratum
        max_edge = 2 * self.max_distance + 1
        for key, members in self._strata.items():
            norms = [self.normalized[g] for g in members]
            self._strata_normalized[key] = norms

            tree = _FlatBKTree(len(members), max_edge)
            for local_idx in range(len(members)):
                tree.insert(local_idx, norms)

            self._trees[key] = tree

        self._built = True

    def cluster(self):
        """Cluster tokens into variant groups using union-find.

        Two tokens are in the same cluster when they are connected by a chain
        of pairwise distances <= `max_distance`. Only clusters with >= 2
        members are retained

        Returns
        -------
        list[list[int]]
            Each inner list contains global vocab indices belonging to one
            cluster
        """
        if not self._built:
            self.build()

        uf = _UnionFind(self.vocab_size)

        # Are we doing parallel or sequential neighbor search?
        if self._use_fork and self.vocab_size > 100:
            found = self._search_all_neighbors_parallel()
        else:
            found = self._search_all_neighbors_sequential()

        # Union-find merging
        for idx, neighbors in found:
            neighbors = sorted(neighbors)
            for neigh in neighbors:
                # Avoid doing each undirected edge twice
                if neigh > idx:
                    uf.union(idx, neigh)

        groups = defaultdict(list)
        for idx in range(self.vocab_size):
            groups[uf.find(idx)].append(idx)

        self.clusters = [g for g in groups.values() if len(g) >= 2]

        return self.clusters

    def _search_all_neighbors_sequential(self):
        """Sequential neighbor search.

        Returns
        -------
        iterator[tuple[int, list[int]]]
            Global vocabulary index and its list of neighbors
        """
        for idx in range(self.vocab_size):
            yield idx, self._search_neighbors(idx)

    def _search_all_neighbors_parallel(self):
        """Parallel neighbor search using fork.

        Returns
        -------
        iterator[tuple[int, list[int]]]
            Global vocabulary index and its list of neighbors
        """
        data = {
            "normalized": self.normalized,
            "norm_len": self.norm_len,
            "prefix_marker": self.prefix_marker,
            "trees": self._trees,
            "strata": self._strata,
            "strata_normalized": self._strata_normalized,
            "global_to_local": self._global_to_local,
            "max_distance": self.max_distance,
            "min_cluster_length": self.MIN_CLUSTER_LENGTH,
        }

        yield from parallel.iter_neighbors_fork(
            vocab_size=self.vocab_size,
            n_jobs=self.n_jobs,
            data=data,
            core_fn=_search_neighbors_core,
        )

    def _search_neighbors(self, idx):
        """Return global indices of all neighbors of `idx`.

        Parameters
        ----------
        idx : int
            Global vocabulary index

        Returns
        -------
        list[int]
            Global indices of neighbor tokens
        """
        return _search_neighbors_core(
            idx,
            self.normalized,
            self.norm_len,
            self.prefix_marker,
            self._trees,
            self._strata_normalized,
            self._strata,
            self._global_to_local,
            self.max_distance,
            self.MIN_CLUSTER_LENGTH,
        )

    def get_cluster_info(self):
        """Return cluster metadata sorted by descending size.

        Returns
        -------
        list[dict]
            One dict per cluster with keys 'tokens', 'token_ids',
            'normalized', 'count', 'representative', 'representative_id'
        """
        if self.clusters is None:
            self.cluster()

        info = []
        for cluster in self.clusters:
            tokens = [self.vocab[i] for i in cluster]
            ids = [int(self.token_ids[i]) for i in cluster]
            norms = [self.normalized[i] for i in cluster]
            info.append(
                {
                    "tokens": tokens,
                    "token_ids": ids,
                    "normalized": norms,
                    "count": len(tokens),
                    "representative": tokens[0],
                    "representative_id": ids[0],
                }
            )

        return sorted(info, key=lambda x: x["count"], reverse=True)

    @staticmethod
    def _assign_prefix_marker(normalized_tok, markers):
        """Only assign a marker when the token has content beyond it.

        Parameters
        ----------
        normalized_tok : str
            Normalized token
        markers : list[str]
            Prefix markers, sorted longest-first

        Returns
        -------
        str
            Prefix marker for the token
        """
        for p in markers:
            if normalized_tok.startswith(p) and len(normalized_tok) > len(p):
                return p

        return ""

    @staticmethod
    def _basic_normalize(tok):
        """NFKC-normalize and strip a token.

        Parameters
        ----------
        tok : str
            Raw token

        Returns
        -------
        str
            Normalized token
        """
        return unicodedata.normalize("NFKC", str(tok)).strip()

    @staticmethod
    def _infer_prefix_markers(
        vocab,
        max_prefix_len=2,
        min_fraction=0.01,
        max_markers=8,
    ):
        """Infer non-alphanumeric prefix markers common in the vocabulary.

        Parameters
        ----------
        vocab : list[str]
            Vocabulary tokens
        max_prefix_len : int
            Maximum prefix length to consider
        min_fraction : float
            A prefix must appear in at least this fraction of the vocab
        max_markers : int
            Maximum number of markers to return

        Returns
        -------
        list[str]
            Inferred prefix strings, sorted longest-first so that the
            most specific marker is matched first
        """
        counts = Counter()
        n = len(vocab)

        for tok in vocab:
            if not tok:
                continue

            s = str(tok)
            for k in range(1, max_prefix_len + 1):
                if len(s) < k:
                    break

                p = s[:k]
                if any(unicodedata.category(ch)[0] in ("L", "N") for ch in p):
                    continue

                counts[p] += 1

        candidates = [
            (p, c) for p, c in counts.items() if c / n >= min_fraction
        ]
        candidates.sort(key=lambda x: (-len(x[0]), -x[1]))

        return [p for p, _ in candidates[:max_markers]]
