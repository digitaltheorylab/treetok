"""Writing-system classification for tokens and characters."""

import string
import unicodedata

# Script bucket ids
SCRIPT_OTHER = 0
SCRIPT_LATIN = 1
SCRIPT_CYRILLIC = 2
SCRIPT_GREEK = 3
SCRIPT_CJK = 4
SCRIPT_ARABIC = 5
SCRIPT_HEBREW = 6
SCRIPT_DEVANAGARI = 7
SCRIPT_BYTE_GLYPH = 8


def _build_byte_glyph_set() -> frozenset[str]:
    """Return the set of code points GPT-2 byte-level BPE uses for non-ASCII
    bytes and control bytes.

    Mirrors `bytes_to_unicode()` from OpenAI's GPT-2 encoder and the Hugging
    Face port: bytes 33..126, 161..172, 174..255 map to themselves; the
    remaining control/whitespace bytes (0..32, 127..160, 173) are remapped to
    Latin Extended-A code points starting at U+0100

    We return the image of every byte that is NOT an ASCII printable
    (0x21..0x7E). Concretely this is:

    - Bytes 0x80..0xFF, which map into the Latin-1 supplement (e.g. byte 0xC3
      -> "Ã"); these are typical mojibake glyphs
    - Control / whitespace bytes (0x00..0x20, 0x7F, 0xA0, 0xAD), which map into
      the remapped block at U+0100..U+0143; this includes the leading-space
      marker `Ġ` (from byte 0x20) and `ġ` (from byte 0x7F)

    Returns
    -------
    frozenset[str]
        Single-character strings spanning the byte-glyph alphabet
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    # Exclude the ASCII-printable range (0x21..0x7E); those map to themselves
    # and overlap with genuine Latin tokens. Keep everything else
    return frozenset(
        chr(c) for b, c in zip(bs, cs) if not (0x21 <= b <= 0x7E)
    )


# Characters that GPT-2 / Qwen3-style byte-level BPE uses to encode raw bytes
# 0x80..0xFF. These render as Latin-1 supplement / Latin Extended glyphs but
# carry no script meaning of their own; they're the surface form of arbitrary
# byte sequences. We bucket them separately to avoid polluting the Latin
# noising alphabet and to prevent mojibake tokens from clustering with real
# Latin tokens
BYTE_GLYPH_CHARS = _build_byte_glyph_set()

# The tokenizer family whose vocabulary uses BYTE_GLYPH_CHARS to represent raw
# bytes. Used to gate byte-glyph detection in `script_bucket` and `char_script1
BYTE_LEVEL_FAMILY = "byte_level_bpe"


# Scripts with individual letters; for these, single-character head/tail edits
# work for "near miss" and the noising alphabet should always include the base
# letter set
ALPHABETIC_SCRIPTS = frozenset(
    {
        SCRIPT_LATIN,
        SCRIPT_CYRILLIC,
        SCRIPT_GREEK,
        SCRIPT_ARABIC,
        SCRIPT_HEBREW,
        SCRIPT_DEVANAGARI,
    }
)

# Per-script canonical core alphabets. These act as a floor for the noising
# alphabet: any script-appropriate single-character edit should be reachable
# even when a tokenizer's vocabulary undersamples low-frequency letters
#
# Note: we skip CJK and OTHER. Character insertion noising isn't a meaningful
# "near miss" model for those buckets, so the dataset builder falls back to its
# punctuation/suffix-bucket passes instead
CANONICAL_CORES = {
    SCRIPT_LATIN: string.ascii_lowercase,
    SCRIPT_CYRILLIC: "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
    SCRIPT_GREEK: "αβγδεζηθικλμνξοπρστυφχψω",
    SCRIPT_ARABIC: "ابتثجحخدذرزسشصضطظعغفقكلمنهوي",
    SCRIPT_HEBREW: "אבגדהוזחטיכלמנסעפצקרשת",
    SCRIPT_DEVANAGARI: "अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह",
}


def is_alphabetic(script_id: int) -> bool:
    """Return True if `script_id` is an alphabetic script.

    Parameters
    ----------
    script_id : int
        Script bucket id

    Returns
    -------
    bool
        True for Latin/Cyrillic/Greek/Arabic/Hebrew/Devanagari; False for CJK
        and OTHER
    """
    return script_id in ALPHABETIC_SCRIPTS


def script_bucket(s: str, family: str = "") -> int:
    """Return a coarse script bucket id for the dominant script in `s`.

    When `family="byte_levelbpe"`, characters in `BYTE_GLYPH_CHARS` are
    bucketed as `SCRIPT_BYTE_GLYPH` rather than their Unicode-name script. For
    other families (WordPiece, SentencePiece), those characters are real Latin
    diacritics and keep their natural script id

    Parameters
    ----------
    s : str
        Input string
    family : str
        Tokenizer family (see `TokenizerView.family`)

    Returns
    -------
    int
        Script bucket id
    """
    byte_level = family == BYTE_LEVEL_FAMILY
    for ch in s:
        # In byte-level BPE vocabularies, byte-glyph code points carry no
        # script meaning of their own; bucket them separately so they don't
        # pollute the Latin alphabet
        if byte_level and ch in BYTE_GLYPH_CHARS:
            return SCRIPT_BYTE_GLYPH

        cat = unicodedata.category(ch)
        if cat[0] not in ("L",):
            continue

        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue

        if "LATIN" in name:
            return SCRIPT_LATIN
        if "CYRILLIC" in name:
            return SCRIPT_CYRILLIC
        if "GREEK" in name:
            return SCRIPT_GREEK
        if (
            "CJK" in name
            or "HIRAGANA" in name
            or "KATAKANA" in name
            or "HANGUL" in name
        ):
            return SCRIPT_CJK
        if "ARABIC" in name:
            return SCRIPT_ARABIC
        if "HEBREW" in name:
            return SCRIPT_HEBREW
        if "DEVANAGARI" in name:
            return SCRIPT_DEVANAGARI

        return SCRIPT_OTHER

    return SCRIPT_OTHER


def char_script(c: str, family: str = "") -> int:
    """Return the script bucket id for a single character.

    Parameters
    ----------
    c : str
        Single character
    family : str
        Tokenizer family. See `script_bucket` for the byte-level semantics

    Returns
    -------
    int
        Script bucket id
    """
    if family == BYTE_LEVEL_FAMILY and c in BYTE_GLYPH_CHARS:
        return SCRIPT_BYTE_GLYPH

    cat = unicodedata.category(c)
    if cat[0] != "L":
        return SCRIPT_OTHER

    try:
        name = unicodedata.name(c)
    except ValueError:
        return SCRIPT_OTHER

    if "LATIN" in name:
        return SCRIPT_LATIN
    if "CYRILLIC" in name:
        return SCRIPT_CYRILLIC
    if "GREEK" in name:
        return SCRIPT_GREEK
    if (
        "CJK" in name
        or "HIRAGANA" in name
        or "KATAKANA" in name
        or "HANGUL" in name
    ):
        return SCRIPT_CJK
    if "ARABIC" in name:
        return SCRIPT_ARABIC
    if "HEBREW" in name:
        return SCRIPT_HEBREW
    if "DEVANAGARI" in name:
        return SCRIPT_DEVANAGARI

    return SCRIPT_OTHER
