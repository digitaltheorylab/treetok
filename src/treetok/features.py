"""Feature builders for the merge classifier."""

import unicodedata
from dataclasses import dataclass

import numpy as np
from rapidfuzz.distance import DamerauLevenshtein, JaroWinkler, Levenshtein

from .hf import TokenizerView
from .script import script_bucket

FEATURE_SPEC = {
    "version": 1,
    "names": [
        # Edit-distance family
        "lev_dist",
        "lev_norm",
        "damerau_dist",
        "jaro_winkler",
        # Length
        "len_diff",
        "len_ratio",
        "min_len",
        "max_len",
        # Affix overlap
        "lcp_len",
        "lcs_suffix_len",
        "lcp_ratio",
        # Equality on canonical forms
        "casefold_eq",
        "nfkc_eq",
        "nfkd_eq",
        "stripped_eq",
        "decoded_eq",
        "decoded_casefold_eq",
        # Marker / script agreement
        "same_marker",
        "both_have_marker",
        "neither_has_marker",
        "same_script",
        # Id proximity
        "id_diff_log",
        # Tokenizer family one-hot (5 keys, mirrors TokenizerView)
        "family_byte_level_bpe",
        "family_sentencepiece",
        "family_wordpiece",
        "family_bpe",
        "family_other",
    ],
}

_FEATURE_NAMES = tuple(FEATURE_SPEC["names"])
_FEAT_IDX = {name: k for k, name in enumerate(_FEATURE_NAMES)}
_FAMILY_FEATURE_NAMES = tuple(
    n for n in _FEATURE_NAMES if n.startswith("family_")
)
_FAMILY_FEATURE_IDXS = np.array(
    [_FEAT_IDX[n] for n in _FAMILY_FEATURE_NAMES], dtype=np.int64
)


def _lcp_len(a: str, b: str) -> int:
    """Return the length of the longest common prefix.

    Parameters
    ----------
    a, b : str
        Input strings

    Returns
    -------
    int
        Prefix length
    """
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1

    return i


def _lcs_suffix_len(a: str, b: str) -> int:
    """Return the length of the longest common suffix.

    Parameters
    ----------
    a, b : str
        Input strings

    Returns
    -------
    int
        Suffix length
    """
    n = min(len(a), len(b))
    i = 0
    while i < n and a[-1 - i] == b[-1 - i]:
        i += 1

    return i


def _hash_array(strings: list[str]) -> np.ndarray:
    """Build an int64 fingerprint array for fast equality compares.

    Parameters
    ----------
    strings : list[str]
        Strings to fingerprint

    Returns
    -------
    np.ndarray
        Shape (n,) int64 array of fingerprints
    """
    out = np.empty(len(strings), dtype=np.int64)
    for i, s in enumerate(strings):
        out[i] = hash(s)

    return out


@dataclass
class TokenFeatures:
    """Per-token feature pack precomputed for a single tokenizer.

    Pair features include:

    - Multiple edit distances (Levenshtein, Damerau, Jaro-Winkler) and ratios
    - Length difference + ratio
    - Longest common prefix / suffix lengths
    - Marker agreement, decoded-form equality, casefold/NFKC/NFKD/stripped
      equality
    - Script agreement
    - Log-id-distance (proxy for BPE merge-rank proximity)
    - Tokenizer family one-hot (broadcast from TokenizerView)
    """

    view: TokenizerView

    # Compare keys (strings) used for pair-equality features
    casefold: list[str]
    nfkc: list[str]
    nfkd: list[str]
    stripped: list[str]
    decoded: list[str]
    decoded_casefold: list[str]

    # Integer fingerprint arrays paralleling the string lists above
    casefold_hash: np.ndarray
    nfkc_hash: np.ndarray
    nfkd_hash: np.ndarray
    stripped_hash: np.ndarray
    decoded_hash: np.ndarray
    decoded_casefold_hash: np.ndarray

    # Numeric features
    len_chars: np.ndarray
    stripped_len: np.ndarray
    script: np.ndarray
    has_marker: np.ndarray

    # Tokenizer family one-hot, broadcast across all pairs
    family_one_hot: np.ndarray

    @classmethod
    def from_view(cls, view: TokenizerView) -> "TokenFeatures":
        """Build a per-token feature pack for one tokenizer.

        Parameters
        ----------
        view : TokenizerView
            Tokenizer snapshot

        Returns
        -------
        TokenFeatures
            Precomputed per-token cache

        Raises
        ------
        ValueError
            If shape of the family one-hot array mismatches family feature
            names size
        """
        vocab = view.vocab
        stripped = view.stripped

        casefold = [s.casefold() for s in stripped]
        nfkc = [unicodedata.normalize("NFKC", s) for s in stripped]
        nfkd = [unicodedata.normalize("NFKD", s) for s in stripped]
        decoded = list(view.decoded)
        decoded_casefold = [s.casefold() for s in decoded]

        len_chars = np.array([len(t) for t in vocab], dtype=np.int32)
        stripped_len = np.array([len(s) for s in stripped], dtype=np.int32)
        script = np.array(
            [script_bucket(s, view.family) for s in stripped], dtype=np.int8
        )

        family_one_hot = view.family_one_hot()
        if family_one_hot.shape[0] != len(_FAMILY_FEATURE_NAMES):
            raise ValueError(
                f"family_one_hot has {family_one_hot.shape[0]} dims but "
                f"FEATURE_SPEC expects {len(_FAMILY_FEATURE_NAMES)} family_* "
                f"features: {_FAMILY_FEATURE_NAMES}"
            )

        return cls(
            view=view,
            casefold=casefold,
            nfkc=nfkc,
            nfkd=nfkd,
            stripped=list(stripped),
            decoded=decoded,
            decoded_casefold=decoded_casefold,
            casefold_hash=_hash_array(casefold),
            nfkc_hash=_hash_array(nfkc),
            nfkd_hash=_hash_array(nfkd),
            stripped_hash=_hash_array(stripped),
            decoded_hash=_hash_array(decoded),
            decoded_casefold_hash=_hash_array(decoded_casefold),
            len_chars=len_chars,
            stripped_len=stripped_len,
            script=script,
            has_marker=view.has_marker.copy(),
            family_one_hot=family_one_hot,
        )


def _alloc_out(n: int) -> np.ndarray:
    """Allocate an output feature matrix.

    Parameters
    ----------
    n : int
        Number of rows (pairs)

    Returns
    -------
    np.ndarray
        Shape (n, n_features) float32 matrix
    """
    return np.empty((n, len(_FEATURE_NAMES)), dtype=np.float32)


def _set_col(out: np.ndarray, name: str, values) -> None:
    """Assign a full column by feature name.

    Parameters
    ----------
    out : np.ndarray
        Feature matrix to mutate in-place
    name : str
        Feature name from FEATURE_SPEC["names"]
    values : array-like
        Column values, broadcastable to (n,)

    Returns
    -------
    None
        Mutates `out` in-place
    """
    out[:, _FEAT_IDX[name]] = values


def pair_features_batch(tf: TokenFeatures, pairs: np.ndarray) -> np.ndarray:
    """Vectorized batch builder for FEATURE_SPEC-ordered feature matrices.

    Parameters
    ----------
    tf : TokenFeatures
        Precomputed per-token cache
    pairs : np.ndarray
        Shape (n, 2) array of token-index pairs

    Returns
    -------
    np.ndarray
        Shape (n, n_features) float32 matrix in FEATURE_SPEC order
    """
    pairs = np.asarray(pairs, dtype=np.int64)
    n = len(pairs)
    out = _alloc_out(n)
    if n == 0:
        return out

    i = pairs[:, 0]
    j = pairs[:, 1]

    # Length stats
    li = tf.stripped_len[i].astype(np.int32, copy=False)
    lj = tf.stripped_len[j].astype(np.int32, copy=False)
    min_len = np.minimum(li, lj)
    max_len = np.maximum(li, lj)
    len_diff = np.abs(li - lj)

    # Avoid /0: when both lengths are 0, ratio convention is 1.0
    safe_max = np.where(max_len > 0, max_len, 1).astype(np.float32)
    len_ratio = np.where(max_len > 0, min_len / safe_max, 1.0)

    # Equality features
    casefold_eq = tf.casefold_hash[i] == tf.casefold_hash[j]
    nfkc_eq = tf.nfkc_hash[i] == tf.nfkc_hash[j]
    nfkd_eq = tf.nfkd_hash[i] == tf.nfkd_hash[j]
    stripped_eq = tf.stripped_hash[i] == tf.stripped_hash[j]
    decoded_eq = tf.decoded_hash[i] == tf.decoded_hash[j]
    decoded_casefold_eq = (
        tf.decoded_casefold_hash[i] == tf.decoded_casefold_hash[j]
    )

    # Marker/script
    has_i = tf.has_marker[i]
    has_j = tf.has_marker[j]
    same_marker = has_i == has_j
    both = has_i & has_j
    neither = (~has_i) & (~has_j)
    same_script = tf.script[i] == tf.script[j]

    # Id distance
    ids_i = tf.view.ids[i].astype(np.int64, copy=False)
    ids_j = tf.view.ids[j].astype(np.int64, copy=False)
    id_diff_log = np.log1p(np.abs(ids_i - ids_j))

    # Per-pair edit distances + affix overlaps
    stripped = tf.stripped
    lev_dist = np.empty(n, dtype=np.float32)
    damerau_dist = np.empty(n, dtype=np.float32)
    jaro_winkler = np.empty(n, dtype=np.float32)
    lcp_arr = np.empty(n, dtype=np.float32)
    lcs_arr = np.empty(n, dtype=np.float32)

    _lev = Levenshtein.distance
    _dam = DamerauLevenshtein.distance
    _jw = JaroWinkler.similarity
    _lcp = _lcp_len
    _lcs = _lcs_suffix_len

    max_len_list = max_len.tolist()
    for k, (ii, jj) in enumerate(pairs.tolist()):
        a, b = stripped[ii], stripped[jj]

        lev_dist[k] = _lev(a, b)
        damerau_dist[k] = _dam(a, b)
        jaro_winkler[k] = _jw(a, b) if max_len_list[k] > 0 else 0.0
        lcp_arr[k] = _lcp(a, b)
        lcs_arr[k] = _lcs(a, b)

    lev_norm = np.where(max_len > 0, lev_dist / safe_max, 0.0)
    lcp_ratio = np.where(max_len > 0, lcp_arr / safe_max, 0.0)

    # Assemble in FEATURE_SPEC order
    _set_col(out, "lev_dist", lev_dist)
    _set_col(out, "lev_norm", lev_norm.astype(np.float32))
    _set_col(out, "damerau_dist", damerau_dist)
    _set_col(out, "jaro_winkler", jaro_winkler)

    _set_col(out, "len_diff", len_diff.astype(np.float32))
    _set_col(out, "len_ratio", len_ratio.astype(np.float32))
    _set_col(out, "min_len", min_len.astype(np.float32, copy=False))
    _set_col(out, "max_len", max_len.astype(np.float32, copy=False))

    _set_col(out, "lcp_len", lcp_arr)
    _set_col(out, "lcs_suffix_len", lcs_arr)
    _set_col(out, "lcp_ratio", lcp_ratio.astype(np.float32))

    _set_col(out, "casefold_eq", casefold_eq.astype(np.float32))
    _set_col(out, "nfkc_eq", nfkc_eq.astype(np.float32))
    _set_col(out, "nfkd_eq", nfkd_eq.astype(np.float32))
    _set_col(out, "stripped_eq", stripped_eq.astype(np.float32))
    _set_col(out, "decoded_eq", decoded_eq.astype(np.float32))
    _set_col(
        out, "decoded_casefold_eq", decoded_casefold_eq.astype(np.float32)
    )

    _set_col(out, "same_marker", same_marker.astype(np.float32))
    _set_col(out, "both_have_marker", both.astype(np.float32))
    _set_col(out, "neither_has_marker", neither.astype(np.float32))
    _set_col(out, "same_script", same_script.astype(np.float32))

    _set_col(out, "id_diff_log", id_diff_log.astype(np.float32))

    # Family one-hot is identical for every pair from the same TokenizerView
    fam = tf.family_one_hot.astype(np.float32, copy=False)
    out[:, _FAMILY_FEATURE_IDXS] = fam[np.newaxis, :]

    return out


def pair_features(tf: TokenFeatures, i: int, j: int) -> np.ndarray:
    """Build the FEATURE_SPEC-ordered feature vector for pair (i, j).

    Distance features are computed on the stripped form so leading-space and/or
    continuation markers don't dominate the edit distance when tokens are
    otherwise identical

    Parameters
    ----------
    tf : TokenFeatures
        Precomputed per-token cache
    i, j : int
        Token indices into the underlying TokenizerView vocabulary

    Returns
    -------
    np.ndarray
        1-D float32 vector of length len(FEATURE_SPEC['names'])
    """
    return pair_features_batch(tf, np.array([[i, j]], dtype=np.int64))[0]
