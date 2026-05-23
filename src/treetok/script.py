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


def script_bucket(s: str) -> int:
    """Return a coarse script bucket id for the dominant script in `s`.

    Parameters
    ----------
    s : str
        Input string

    Returns
    -------
    int
        Script bucket id
    """
    for ch in s:
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


def char_script(c: str) -> int:
    """Return the script bucket id for a single character.

    Parameters
    ----------
    c : str
        Single character

    Returns
    -------
    int
        Script bucket id
    """
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
