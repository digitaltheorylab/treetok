"""Writing-system classification for tokens and characters."""

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
