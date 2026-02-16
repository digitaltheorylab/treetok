"""BK-Tree clusterer for token variants."""

import numpy as np
from rapidfuzz.distance import Levenshtein


def bounded_levenshtein(s1, s2, max_dist):
    """Compute bounded Levenshtein distance.

    Uses length-based rejection followed by rapidfuzz's cutoff-enabled
    implementation to avoid unnecessary work.

    Parameters
    ----------
    s1 : str
        First string
    s2 : str
        Second string
    max_dist : int
        Maximum distance of interest

    Returns
    -------
    int
        Levenshtein distance if <= max_dist, else max_dist + 1
    """
    if abs(len(s1) - len(s2)) > max_dist:
        return max_dist + 1

    d = Levenshtein.distance(s1, s2, score_cutoff=max_dist)

    return d if d <= max_dist else max_dist + 1


class _FlatBKTree:
    """BK-tree backed by a flat numpy children array."""

    __slots__ = ("children", "root", "size", "max_edge")

    def __init__(self, capacity, max_edge):
        """Intialize the tree.

        Parameters
        ----------
        capacity : int
            Maximum number of nodes
        max_edge : int
            Maximum edge label (Levenshtein distance) stored. Edges with
            distance > max_edge are silently dropped
        """
        self.max_edge = max_edge
        self.children = np.full((capacity, max_edge + 1), -1)
        self.root = -1
        self.size = 0

    def insert(self, idx, normalized_tokens):
        """Insert `idx` into the tree.

        Parameters
        ----------
        idx : int
            Index into `normalized_tokens` for the token being inserted
        normalized_tokens : list[str]
            All normalized tokens addressable by local index
        """
        if self.root == -1:
            self.root = idx
            self.size = 1
            return

        tok = normalized_tokens[idx]
        node = self.root
        while True:
            d = Levenshtein.distance(tok, normalized_tokens[node])
            if d > self.max_edge:
                return

            child = self.children[node, d]
            if child == -1:
                self.children[node, d] = idx
                self.size += 1
                return

            node = child

    def search(self, tok, idx, max_dist, normalized_tokens):
        """Return local indices within `max_dist` of `tok`.

        Parameters
        ----------
        tok : str
            Normalized query token
        idx : int
            Local index of the query token
        max_dist : int
            Maximum Levenshtein distance
        normalized_tokens : list[str]
            All normalized tokens addressable by local index

        Returns
        -------
        list[int]
            Local indices of neighbors
        """
        if self.root == -1:
            return []

        neighbors = []
        stack = [self.root]
        me = self.max_edge

        while stack:
            node = stack.pop()
            d = bounded_levenshtein(tok, normalized_tokens[node], max_dist)

            if d <= max_dist and node != idx:
                neighbors.append(node)

            lo = max(0, d - max_dist)
            hi = min(me, d + max_dist)

            children = self.children[node]
            for ed in range(lo, hi + 1):
                child = children[ed]
                if child != -1:
                    stack.append(child)

        return neighbors


class _UnionFind:
    """Weighted quick-union with path compression."""

    __slots__ = ("parent", "rank")

    def __init__(self, n):
        """Initialize the union.

        Parameters
        ----------
        n : int
            Rank size
        """
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]

        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return

        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx

        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
