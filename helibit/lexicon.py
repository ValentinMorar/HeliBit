"""
HeliBit-AI: Closed Lexicon & Semantic Role Mapping Module
Defines semantic roles, word-to-value mappings, and bitboard encoding for time expressions.
"""

from typing import Optional, Tuple, Dict
from helibit.bitboard import Bitboard64

# Closed vocabulary dictionaries
_ONES: Dict[str, int] = {
    "zero": 0, "oh": 0,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19
}

_TENS: Dict[str, int] = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50
}

NUMBER_WORDS: Dict[str, int] = {}
NUMBER_WORDS.update(_ONES)
NUMBER_WORDS.update(_TENS)
for t_str, t_val in _TENS.items():
    for o_str, o_val in _ONES.items():
        if 1 <= o_val <= 9:
            NUMBER_WORDS[f"{t_str}-{o_str}"] = t_val + o_val
            NUMBER_WORDS[f"{t_str} {o_str}"] = t_val + o_val
            NUMBER_WORDS[f"{t_str}{o_str}"] = t_val + o_val

DIRECTION_WORDS: Dict[str, int] = {
    "past": Bitboard64.ROLE_PAST,
    "after": Bitboard64.ROLE_PAST,
    "to": Bitboard64.ROLE_TO,
    "before": Bitboard64.ROLE_TO,
    "till": Bitboard64.ROLE_TO,
    "until": Bitboard64.ROLE_TO,
}

MODIFIER_WORDS: Dict[str, Tuple[int, int]] = {
    "quarter": (Bitboard64.ROLE_QUARTER, 15),
    "half": (Bitboard64.ROLE_HALF, 30),
}

HOUR_UNIT_WORDS: Dict[str, int] = {
    "o'clock": Bitboard64.ROLE_HOUR_UNIT,
    "oclock": Bitboard64.ROLE_HOUR_UNIT,
    "clock": Bitboard64.ROLE_HOUR_UNIT,
    "sharp": Bitboard64.ROLE_HOUR_UNIT,
}

MERIDIEM_WORDS: Dict[str, int] = {
    "am": Bitboard64.ROLE_AM,
    "a.m.": Bitboard64.ROLE_AM,
    "morning": Bitboard64.ROLE_AM,
    "pm": Bitboard64.ROLE_PM,
    "p.m.": Bitboard64.ROLE_PM,
    "afternoon": Bitboard64.ROLE_PM,
    "evening": Bitboard64.ROLE_PM,
    "night": Bitboard64.ROLE_PM,
}

SPECIAL_TIME_WORDS: Dict[str, Tuple[int, int]] = {
    "noon": (12, 0),
    "midday": (12, 0),
    "midnight": (0, 0),
}


def compute_token_fingerprint(token: str) -> int:
    """Computes deterministic 54-bit lexical fingerprint using the bitboard lattice hash."""
    return Bitboard64.from_string(token).value & Bitboard64.FINGERPRINT_MASK


def parse_numeric_token(token: str) -> Optional[int]:
    """Attempts to parse a token as an integer number (0-63)."""
    clean = token.strip().lower()
    if clean.isdigit():
        val = int(clean)
        if 0 <= val <= 63:
            return val
        return None
    if clean in NUMBER_WORDS:
        return NUMBER_WORDS[clean]
    return None


def encode_token_to_bitboard(token: str) -> Bitboard64:
    """
    Encodes a lexical token into a structured 64-bit bitboard:
      [ROLE: 4 bits | VALUE: 6 bits | FINGERPRINT: 54 bits]
    """
    clean = token.strip().lower()
    fingerprint = compute_token_fingerprint(clean)

    # Check numeric tokens
    num_val = parse_numeric_token(clean)
    if num_val is not None:
        return Bitboard64.from_role_value(Bitboard64.ROLE_NUMBER, num_val, fingerprint)

    # Check direction words
    if clean in DIRECTION_WORDS:
        role = DIRECTION_WORDS[clean]
        return Bitboard64.from_role_value(role, 0, fingerprint)

    # Check modifiers (quarter, half)
    if clean in MODIFIER_WORDS:
        role, val = MODIFIER_WORDS[clean]
        return Bitboard64.from_role_value(role, val, fingerprint)

    # Check hour units (o'clock)
    if clean in HOUR_UNIT_WORDS:
        role = HOUR_UNIT_WORDS[clean]
        return Bitboard64.from_role_value(role, 0, fingerprint)

    # Check meridiem (am, pm, etc.)
    if clean in MERIDIEM_WORDS:
        role = MERIDIEM_WORDS[clean]
        return Bitboard64.from_role_value(role, 0, fingerprint)

    # Unknown / unassigned role
    return Bitboard64.from_role_value(Bitboard64.ROLE_UNKNOWN, 0, fingerprint)
