"""Inspect an AutoTokenizer."""

import os

os.environ.setdefault("TRANSFORMERS_NO_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from transformers import AutoTokenizer

# Canonical marker strings recognized across tokenizer families
LEADING_SPACE_MARKERS = ("\u0120", "\u2581")  # 'Ġ' (byte-level BPE), '▁' (SP)
CONTINUATION_MARKERS = ("##",)  # WordPiece


@dataclass
class TokenizerView:
    """Read-only snapshot of an AutoTokenizer.

    Attributes
    ----------
    model_name : str
        Model identifier passed to `AutoTokenizer.from_pretrained`
    family : str
        One of "byte_level_bpe", "sentencepiece", "wordpiece", "bpe", "other"
    vocab : list[str]
        Token strings, ordered by token id ascending
    ids : np.ndarray
        Token ids matching `vocab`, dtype int32
    decoded : list[str]
        `tokenizer.convert_tokens_to_string([tok])` per token
    special_token_ids : set[int]
        Ids reported via `tokenizer.all_special_ids`
    added_token_ids : set[int]
        Ids of tokens added on top of the base vocabulary
    prefix_marker : str
        Dominant marker string in the vocab ("Ġ", "▁", "##", or "")
    marker_kind : str
        One of "leading_space", "continuation", "none"
    has_marker : np.ndarray[bool]
        Per-token flag indicating presence of the dominant marker
    stripped : list[str]
        Token with the marker stripped (or the original string when absent)
    """

    model_name: str
    family: str
    vocab: list[str]
    ids: np.ndarray
    decoded: list[str]
    special_token_ids: set[int]
    added_token_ids: set[int]
    prefix_marker: str
    marker_kind: str
    has_marker: np.ndarray
    stripped: list[str]
    family_one_hot_keys: tuple[str, ...] = field(
        default=(
            "byte_level_bpe",
            "sentencepiece",
            "wordpiece",
            "bpe",
            "other",
        )
    )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def family_one_hot(self) -> np.ndarray:
        """Return a 1-D one-hot encoding of `family` over the canonical keys."""
        oh = np.zeros(len(self.family_one_hot_keys), dtype=np.int8)
        try:
            oh[self.family_one_hot_keys.index(self.family)] = 1
        except ValueError:
            oh[self.family_one_hot_keys.index("other")] = 1

        return oh


def inspect(model_name: str, **tokenizer_kwargs: dict | None) -> TokenizerView:
    """Load a tokenizer and return a `TokenizerView`.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier
    tokenizer_kwargs : dict or None
        Tokenizer keywords arguments

    Returns
    -------
    TokenizerView
    """
    tokenizer_kwargs = tokenizer_kwargs or {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

    # Sort by id so list position == id position
    items = sorted(tokenizer.get_vocab().items(), key=lambda kv: kv[1])
    vocab = [t for t, _ in items]
    ids = np.array([i for _, i in items], dtype=np.int32)

    prefix_marker, marker_kind = _detect_marker(vocab)
    has_marker = np.zeros(len(vocab), dtype=bool)
    stripped = [""] * len(vocab)

    if prefix_marker and marker_kind == "leading_space":
        for i, t in enumerate(vocab):
            if t.startswith(prefix_marker) and len(t) > len(prefix_marker):
                has_marker[i] = True
                stripped[i] = t[len(prefix_marker) :]
            else:
                stripped[i] = t
    elif prefix_marker and marker_kind == "continuation":
        for i, t in enumerate(vocab):
            if t.startswith(prefix_marker) and len(t) > len(prefix_marker):
                has_marker[i] = True
                stripped[i] = t[len(prefix_marker) :]
            else:
                stripped[i] = t
    else:
        stripped = list(vocab)

    family = _detect_family(tokenizer, prefix_marker, marker_kind)
    decoded = _decode_each(tokenizer, vocab)

    special_token_ids = set(int(i) for i in tokenizer.all_special_ids)
    added_token_ids = _added_token_ids(tokenizer)

    return TokenizerView(
        model_name=model_name,
        family=family,
        vocab=vocab,
        ids=ids,
        decoded=decoded,
        special_token_ids=special_token_ids,
        added_token_ids=added_token_ids,
        prefix_marker=prefix_marker,
        marker_kind=marker_kind,
        has_marker=has_marker,
        stripped=stripped,
    )


def _detect_marker(vocab: Iterable[str]) -> tuple[str, str]:
    """Find the dominant prefix marker by frequency.

    Parameters
    ----------
    vocab : Iterable[str]
        Tokenizer vocabulary

    Returns
    -------
    (marker, kind) : tuple[str, str]
        `kind` is "leading_space", "continuation", or "none"
    """
    counts = Counter()
    n = 0

    def _count_first_match(
        token: str, kind: str, markers: tuple[str, ...]
    ) -> bool:
        for m in markers:
            if token.startswith(m) and len(token) > len(m):
                counts[(kind, m)] += 1
                return True

        return False

    for t in vocab:
        n += 1
        if not t:
            continue
        if _count_first_match(t, "leading_space", LEADING_SPACE_MARKERS):
            continue
        _count_first_match(t, "continuation", CONTINUATION_MARKERS)

    if not counts:
        return "", "none"

    # Require the marker to be reasonably common (>= 5% of vocab)
    (kind, marker), c = max(counts.items(), key=lambda kv: kv[1])
    if c / max(n, 1) < 0.05:
        return "", "none"

    return marker, kind


def _detect_family(tokenizer, prefix_marker: str, marker_kind: str) -> str:
    """Classify a tokenizer into a coarse family using best-effort heuristics.

    Classification is based on (in priority order): the fast tokenizer backend
    model type, the tokenizer class name, and common prefix-marker conventions

    Parameters
    ----------
    tokenizer : AutoTokenizer
        A Hugging Face tokenizer
    prefix_marker : str
        The dominant prefix marker, such as "▁" or "Ġ". Use "" if none desired
    marker_kind : str
         One of: "leading_space", "continuation", or "none"

    Returns
    -------
    str
        One of: "wordpiece", "sentencepiece", "byte_level_bpe", "bpe", "other"
    """
    cls_name = type(tokenizer).__name__.lower()

    backend_model = ""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    model = getattr(backend, "model", None) if backend is not None else None
    if model is not None:
        backend_model = type(model).__name__.lower()

    # Strong signals first
    if backend_model in {"wordpiece", "unigram"}:
        return "wordpiece" if backend_model == "wordpiece" else "sentencepiece"

    if "bert" in cls_name and "roberta" not in cls_name:
        return "wordpiece"

    if "gpt2" in cls_name or "roberta" in cls_name:
        return "byte_level_bpe"

    # Marker-based heuristics
    match (marker_kind, prefix_marker):
        case ("leading_space", "\u2581"):
            return "sentencepiece"
        case ("leading_space", "\u0120"):
            return "byte_level_bpe"
        case ("continuation", _):
            return "wordpiece"

    if backend_model == "bpe":
        return "bpe"

    return "other"


def _decode_each(tokenizer, vocab: list[str]) -> list[str]:
    """Decode every token to its user-visible string form.

    Uses `tokenizer.convert_tokens_to_string([token])` when available and falls
    back to returning the raw token string if conversion fails

    Parameters
    ----------
    tokenizer : AutoTokenizer
        A Hugging Face tokenizer
    vocab : list[str]
        Vocabulary tokens as returned by the tokenizer

    Returns
    -------
    list[str]
        Decoded, user-visible strings corresponding to each entry in `vocab`
    """
    out = []
    convert = getattr(tokenizer, "convert_tokens_to_string", None)

    for t in vocab:
        if convert is None:
            out.append(t)
            continue

        try:
            d = convert([t])
        except Exception:
            d = t

        if not isinstance(d, str):
            d = t if isinstance(t, str) else str(t)

        out.append(d)

    return out


def _added_token_ids(tokenizer) -> set[int]:
    """Collect ids of added tokens across Hugging Face tokenizer variants.

    Parameters
    ----------
    tokenizer : AutoTokenizer
        A Hugging Face tokenizer

    Returns
    -------
    set[int]
        Token ids corresponding to added tokens
    """
    ids = set()

    decoder = getattr(tokenizer, "added_tokens_decoder", None)
    if isinstance(decoder, dict):
        ids.update(i for i in decoder.keys())

    encoder = getattr(tokenizer, "added_tokens_encoder", None)
    if isinstance(encoder, dict):
        ids.update(v for v in encoder.values())

    return ids
