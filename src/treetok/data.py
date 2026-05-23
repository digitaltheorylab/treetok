"""Training-data construction for the merge classifier."""

import random
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .candidates import _eligible_mask, iter_pairs
from .features import FEATURE_SPEC, TokenFeatures, pair_features
from .hf import TokenizerView, inspect
from .script import CANONICAL_CORES, char_script, is_alphabetic


@dataclass
class DatasetConfig:
    """Configuration for dataset construction."""

    n_synthetic_positives: int = 5_000
    n_hard_negatives: int = 5_000
    n_easy_negatives: int = 2_000
    seed: int = 0


def _toggle_marker(s: str, view: TokenizerView) -> list[str]:
    """Return marker-toggled variants of `s`.

    Parameters
    ----------
    s : str
        Token string
    view : TokenizerView
        Tokenizer snapshot providing the dominant marker

    Returns
    -------
    list[str]
        Candidate variants obtained by removing or prepending the marker
    """
    out = []
    marker = view.prefix_marker
    if not marker:
        return out

    if s.startswith(marker) and len(s) > len(marker):
        out.append(s[len(marker) :])
    else:
        out.append(marker + s)

    return out


def _case_variants(s: str) -> list[str]:
    """Return common case variants of `s` distinct from the input.

    Parameters
    ----------
    s : str
        Token string

    Returns
    -------
    list[str]
        Distinct case variants
    """
    out = set()
    out.add(s.lower())
    out.add(s.upper())
    out.add(s.capitalize())

    # Toggle case of the first character only (provided `s[0].isalpha()`)
    if s and s[0].isalpha():
        out.add(s[0].swapcase() + s[1:])

    out.discard(s)

    return list(out)


def _norm_variants(s: str) -> list[str]:
    """Return NFKC/NFKD normalization variants distinct from the input.

    Parameters
    ----------
    s : str
        Token string

    Returns
    -------
    list[str]
        Distinct normalized variants
    """
    out = set()
    out.add(unicodedata.normalize("NFKC", s))
    out.add(unicodedata.normalize("NFKD", s))
    out.discard(s)

    return list(out)


def _strip_edge_punct(s: str) -> list[str]:
    """Drop a single edge-punctuation character if present.

    Parameters
    ----------
    s : str
        Token string

    Returns
    -------
    list[str]
        Variants obtained by stripping one leading and/or trailing
        punctuation character
    """
    if not s:
        return []

    out = []
    head, tail = s[0], s[-1]

    if not head.isalnum() and len(s) > 1:
        out.append(s[1:])
    if not tail.isalnum() and len(s) > 1:
        out.append(s[:-1])

    return out


def _candidate_partners(s: str, view: TokenizerView) -> list[str]:
    """Generate candidate surface-form variant strings for `s`.

    Restricted to true surface-form variants:

    - Case
    - NFKC/NFKD normalization
    - Edge-punctuation stripping
    - Marker-toggling

    We exclude typos for this step and use them as hard negatives instead

    The caller filters to partners that actually appear in the tokenizer's
    vocabulary

    Parameters
    ----------
    s : str
        Token string.
    view : TokenizerView
        Tokenizer snapsho.

    Returns
    -------
    list[str]
        Candidate variant strings.
    """
    seeds = [s]
    seeds.extend(_toggle_marker(s, view))
    seeds.extend(_case_variants(s))

    pool = set()
    for seed in seeds:
        pool.update(_case_variants(seed))
        pool.update(_norm_variants(seed))
        pool.update(_strip_edge_punct(seed))
        pool.update(_toggle_marker(seed, view))

    pool.discard(s)

    return list(pool)


def _vocab_index(view: TokenizerView) -> dict[str, int]:
    """Build a token -> index lookup for `view.vocab`.

    Parameters
    ----------
    view : TokenizerView
        Tokenizer snapshot

    Returns
    -------
    dict[str, int]
        Mapping from token string to its index in `view.vocab`
    """
    return {t: i for i, t in enumerate(view.vocab)}


def _derive_alphabets(
    view: TokenizerView, final_cap: int = 64
) -> dict[int, str]:
    """Build per-script noising alphabets for hard-negative generation.

    For each alphabetic script, the output alphabet is the script's canonical
    core (see `CANONICAL_CORES`) followed by the most frequent additional
    characters observed in the tokenizer vocabulary, truncated to `final_cap`.
    The core is always included so low-frequency core letters are never 
    dropped when a tokenizer happens to undersample them

    Non-alphabetic scripts (CJK, OTHER) are omitted from the result. Callers
    should treat a missing key as "skip the character-insertion loop" rather
    than fall back to ASCII

    Parameters
    ----------
    view : TokenizerView
        Tokenizer snapshot
    final_cap : int
        Maximum number of characters per alphabet, including the core. Bounds
        the per-token partner-generation work in the hard-negative miner

    Returns
    -------
    dict[int, str]
        Mapping from script bucket id to alphabet string. For each alphabetic
        script the string starts with that script's canonical core (in
        canonical order), followed by vocabulary-derived augmentations in
        descending frequency
    """
    family = view.family
    freq = defaultdict(Counter)
    for s in view.stripped:
        for c in s:
            if not c.isalpha():
                continue

            sid = char_script(c, family)
            if not is_alphabetic(sid):
                continue

            # Skip characters whose casefold expands to multiple code points
            # (e.g. German `ß` -> "ss"); we want strictly single-char alphabet
            # entries so the partner-generation loop emits well-formed strings
            folded = c.casefold()
            if len(folded) != 1:
                continue

            freq[sid][folded] += 1

    result = {}
    for sid, core in CANONICAL_CORES.items():
        core_set = set(core)
        # Append vocabulary chars not already in the core, in frequency-
        # descending order. `counts.most_common()` returns an empty list when
        # `sid` is absent from `freq`, so scripts the tokenizer doesn't use
        # still get the core
        counts = freq.get(sid, Counter())
        tail = [c for c, _ in counts.most_common() if c not in core_set]
        alphabet = core + "".join(tail)
        result[sid] = alphabet[:final_cap]

    return result


def synthetic_positives(
    tf: TokenFeatures, cfg: DatasetConfig, rng: random.Random
) -> list[tuple[int, int]]:
    """Yield up to `cfg.n_synthetic_positives` (i, j) pairs.

    Walks the eligible vocab in random order, generates partners per token via
    `_candidate_partners`, and keeps pairs where the partner exists in the
    vocabulary

    Parameters
    ----------
    tf : TokenFeatures
        Precomputed per-token cache
    cfg : DatasetConfig
        Dataset construction settings
    rng : random.Random
        Random generator

    Returns
    -------
    list[tuple[int, int]]
        List of `(i, j)` index pairs with `i < j`
    """
    view = tf.view
    eligible = _eligible_mask(tf)
    idx_pool = [i for i, ok in enumerate(eligible) if ok]
    rng.shuffle(idx_pool)

    vocab_idx = _vocab_index(view)
    pairs = []
    seen_pairs = set()

    target = cfg.n_synthetic_positives
    for i in idx_pool:
        if len(pairs) >= target:
            break

        s = view.vocab[i]
        partners = _candidate_partners(s, view)
        for p in partners:
            j = vocab_idx.get(p)
            if j is None or j == i:
                continue

            key = (min(i, j), max(i, j))
            if key in seen_pairs:
                continue

            seen_pairs.add(key)
            pairs.append(key)
            if len(pairs) >= target:
                break

    return pairs


def hard_negatives(
    tf: TokenFeatures,
    cfg: DatasetConfig,
    rng: random.Random,
    positives: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Mine pairs that look close but are not surface-form variants.

    These negatives share stratum and have low edit distance, but their
    stripped/casefold/decoded forms differ. This regime is most likely to fool
    naive thresholds

    We generate three sub-streams, then balance:

    1. `iter_pairs`-derived: pairs whose stripped, casefold, and decoded forms
       all differ; the "1-dit but not a variant" case
    2. Single-char prefix/suffix insertion: explicitly construct short-token
       negatives like "ing" <-> "Sing". The synthetic positive generator can't
       produce these, and the classifier badly needs them
    3. Single-char prefix/surfix substitution at short lengths

    Returns a shuffled, deduplicated subsample up to `cfg.n_hard_negatives`

    Parameters
    ----------
    tf : TokenFeatures
        Precomputed per-token cache
    cfg : DatasetConfig
        Dataset construction settings
    rng : random.Random
        Random generator
    positives : set[tuple[int, int]]
        Positive pairs to exclude

    Returns
    -------
    list[tuple[int, int]]
        List of `(i, j)` index pairs with `i < j`
    """
    target = cfg.n_hard_negatives
    pool_cap = max(target * 8, 50_000)

    cf = tf.casefold
    stripped = tf.stripped
    decoded = tf.decoded

    pool = []
    seen = set()

    # Sub-stream 1: stratum-based mined negatives
    for i, j in iter_pairs(tf):
        key = (i, j)
        if key in positives or key in seen:
            continue

        if (
            stripped[i] == stripped[j]
            or cf[i] == cf[j]
            or decoded[i] == decoded[j]
        ):
            continue

        seen.add(key)
        pool.append(key)
        if len(pool) >= pool_cap:
            break

    # Sub-stream 2 + 3: explicit short-token edit negatives
    explicit = _explicit_edit_negatives(tf, positives, seen, rng)
    for key in explicit:
        if key in seen:
            continue

        seen.add(key)
        pool.append(key)

    rng.shuffle(pool)

    return pool[:target]


def _explicit_edit_negatives(
    tf: TokenFeatures,
    positives: set[tuple[int, int]],
    already_seen: set[tuple[int, int]],
    rng: random.Random,
    max_len: int = 12,
) -> list[tuple[int, int]]:
    """Construct explicit single-edit negatives spanning short and mid-length
    tokens.

    For each eligible token, look up vocab partners produced by:

    - Single-char insertion/deletion/substitution at head or tail
    - Prefixed by common code/text punctuation
    - Same prefix-N or suffix-N as the seed token

    These prefix/suffix-edit negatives fool a naive edit-distance threshold;
    they're especially observable on byte-level BPE tokenizers

    Parameters
    ----------
    tf : TokenFeatures
        Precomputed per-token cache
    positives : set[tuple[int, int]]
        Positive pairs to exclude
    already_seen : set[tuple[int, int]]
        Pairs already sampled for negatives
    rng : random.Random
        Random generator
    max_len : int
        Maximum stripped token length to consider

    Returns
    -------
    list[tuple[int, int]]
        List of `(i, j)` index pairs with `i < j`
    """
    view = tf.view
    vi = {t: i for i, t in enumerate(view.vocab)}
    alphabets = _derive_alphabets(view)

    out = []

    cf = tf.casefold

    eligible_idx = [
        i for i, n in enumerate(tf.stripped_len) if 3 <= int(n) <= max_len
    ]
    rng.shuffle(eligible_idx)

    # Common punctuation prefixes that frequently appear in code-tokenizer
    # vocabs and create high-Levenshtein-overlap false positives
    punct_prefixes = ("-", "_", ".", "<", "\\", "$", "[", "=", "/", "(", ",")

    # Build a suffix index used for shared-suffix negatives, e.g. "traction" /
    # "<Action>" / "ivation" all share suffix "tion"/"ction"
    suffix_idx = defaultdict(list)
    for i, s in enumerate(view.stripped):
        if not (3 <= len(s) <= max_len):
            continue

        # Use the last 4 chars as the bucket key when available; the "traction
        # <-> <Action" family clusters by suffix "ction"
        suffix_idx[s[-4:].lower() if len(s) >= 4 else s.lower()].append(i)

    for i in eligible_idx[:8_000]:
        s = view.stripped[i]
        partners: list[str] = []

        # Single-char head/tail edits
        if len(s) >= 2:
            partners.append(s[1:])
            partners.append(s[:-1])

        # Single-character insertions/substitutions only apply to alphabetic
        # scripts; for CJK/OTHER tokens we skip this loop and fall through to
        # the punctuation-prefix and suffix-bucket passes
        alphabet = alphabets.get(int(tf.script[i]), "")
        for c in alphabet:
            partners.append(c + s)
            partners.append(s + c)

            if s and c != s[0]:
                partners.append(c + s[1:])
            if s and c != s[-1]:
                partners.append(s[:-1] + c)

        # Punctuation-prefixed counterparts
        for p in punct_prefixes:
            partners.append(p + s)
            if len(s) >= 2:
                partners.append(p + s[1:])  # punct prefix replacing head char

        # Marker-prepended variants
        if view.prefix_marker:
            partners = [view.prefix_marker + p for p in partners] + partners

        for p in partners:
            j = vi.get(p)
            if j is None or j == i:
                continue

            key = (min(i, j), max(i, j))
            if key in positives or key in already_seen:
                continue

            # Require casefold inequality on stripped forms; otherwise it's
            # actually a positive (e.g. case-only edit)
            if cf[i] == cf[j]:
                continue

            out.append(key)

    # Shared-suffix negatives: pairs sharing the trailing 4 chars but otherwise
    # different (e.g. "traction"/"<Action"/"ivation"). Cap how otherwise
    # different; cap how many pairs each suffix bucket contributes to keep the
    # dataset balanced
    suffix_pairs_per_bucket = 6
    for bucket_members in suffix_idx.values():
        if len(bucket_members) < 2:
            continue

        rng.shuffle(bucket_members)
        cap = min(len(bucket_members), suffix_pairs_per_bucket + 1)

        for a in range(cap):
            for b in range(a + 1, cap):
                i, j = bucket_members[a], bucket_members[b]
                key = (min(i, j), max(i, j))
                if key in positives or key in already_seen:
                    continue

                if cf[i] == cf[j]:
                    continue

                out.append(key)

    return out


def easy_negatives(
    tf: TokenFeatures,
    cfg: DatasetConfig,
    rng: random.Random,
    exclude: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Sample random cross-stratum pairs.

    The classifier learns "different lengths/scripts/marker = not a merge"
    trivially from these. Mostly a class-balance buffer

    Parameters
    ----------
    tf : TokenFeatures
        Precomputed per-token cache
    cfg : DatasetConfig
        Dataset construction settings
    rng : random.Random
        Random generator
    exclude : set[tuple[int, int]]
        Pairs to exclude (typically positives and mined hard negatives)

    Returns
    -------
    list[tuple[int, int]]
        List of `(i, j)` index pairs with `i < j`
    """
    eligible = _eligible_mask(tf)
    idx_pool = [i for i, ok in enumerate(eligible) if ok]
    if len(idx_pool) < 2:
        return []

    target = cfg.n_easy_negatives
    out = []
    tries = 0
    max_tries = target * 20

    while len(out) < target and tries < max_tries:
        tries += 1
        i, j = rng.sample(idx_pool, 2)
        key = (min(i, j), max(i, j))
        if key in exclude:
            continue

        # Reject pairs that are too similar (length within 1 char and share
        # script + marker). Those belong to the hard-neg pool
        if (
            abs(int(tf.stripped_len[i]) - int(tf.stripped_len[j])) <= 1
            and tf.script[i] == tf.script[j]
            and tf.has_marker[i] == tf.has_marker[j]
        ):
            continue

        out.append(key)

    return out


def build_dataset(
    model_name: str, cfg: DatasetConfig | None = None
) -> pa.Table:
    """Construct a labeled training table for one tokenizer.

    Output schema (Parquet):

        model_name : str
        family     : str
        token_a    : str
        token_b    : str
        id_a       : int32
        id_b       : int32
        label      : int8       (1 positive, 0 negative)
        source     : str        ("synthetic_pos" | "hard_neg" | "easy_neg")
        f0..f{N-1} : float32    feature columns matching FEATURE_SPEC

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier
    cfg : DatasetConfig or None
        Dataset construction settings

    Returns
    -------
    pa.Table
        Table containing token pairs, labels, and feature columns
    """
    cfg = cfg or DatasetConfig()
    rng = random.Random(cfg.seed)

    view = inspect(model_name)
    tf = TokenFeatures.from_view(view)

    pos = synthetic_positives(tf, cfg, rng)
    pos_set = set(pos)
    hard = hard_negatives(tf, cfg, rng, pos_set)
    hard_set = set(hard)
    easy = easy_negatives(tf, cfg, rng, pos_set | hard_set)

    rows = []
    rows.extend((i, j, 1, "synthetic_pos") for i, j in pos)
    rows.extend((i, j, 0, "hard_neg") for i, j in hard)
    rows.extend((i, j, 0, "easy_neg") for i, j in easy)

    n = len(rows)
    n_feat = len(FEATURE_SPEC["names"])
    feat = np.empty((n, n_feat), dtype=np.float32)
    for k, (i, j, _, _) in enumerate(rows):
        feat[k] = pair_features(tf, i, j)

    cols = {
        "model_name": pa.array([model_name] * n, type=pa.string()),
        "family": pa.array([view.family] * n, type=pa.string()),
        "token_a": pa.array([view.vocab[i] for i, *_ in rows]),
        "token_b": pa.array([view.vocab[j] for _, j, *_ in rows]),
        "id_a": pa.array([view.ids[i] for i, *_ in rows], type=pa.int32()),
        "id_b": pa.array([view.ids[j] for _, j, *_ in rows], type=pa.int32()),
        "label": pa.array([lbl for *_, lbl, _ in rows], type=pa.int8()),
        "source": pa.array([src for _, _, _, src in rows]),
    }
    for k, name in enumerate(FEATURE_SPEC["names"]):
        cols[f"f_{name}"] = pa.array(feat[:, k], type=pa.float32())

    return pa.table(cols)


def write_dataset(table: pa.Table, output_path) -> None:
    """Write a labeled dataset to Parquet.

    Parameters
    ----------
    table : pa.Table
        Training dataset
    output_path
        Parquet output path
    """
    pq.write_table(table, str(output_path))


def read_dataset(path) -> pa.Table:
    """Read a labeled dataset from Parquet.

    Parameters
    ----------
    path
        Parquet file path

    Returns
    -------
    pa.Table
        Training dataset
    """
    return pq.read_table(str(path))


def feature_matrix(table: pa.Table) -> tuple[np.ndarray, np.ndarray]:
    """Extract `(X, y)` arrays from a dataset table.

    Parameters
    ----------
    table : pa.Table
        Table written by `write_dataset`

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `X` feature matrix (float32) and `y` labels (int8)

    Raises
    ------
    KeyError
        If columns are missing
    """
    feat_cols = [f"f_{n}" for n in FEATURE_SPEC["names"]]
    required = feat_cols + ["label"]
    missing = [c for c in required if c not in table.column_names]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    n = table.num_rows
    m = len(feat_cols)

    X = np.empty((n, m), dtype=np.float32)
    for k, col in enumerate(feat_cols):
        X[:, k] = table.column(col).to_numpy(zero_copy_only=False)

    y = (
        table.column("label")
        .to_numpy(zero_copy_only=False)
        .astype(np.int8, copy=False)
    )

    return X, y
