"""
HeliBit-AI: Semantic Slot Parser Module
Extracts semantic time slots (hour, minute_offset, direction, meridiem) from
structured bitboard tokens and lexical patterns.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from helibit.bitboard import Bitboard64
from helibit.lexicon import (
    encode_token_to_bitboard,
    SPECIAL_TIME_WORDS,
)


class TimeSlotParser:
    """
    Parses natural language time expressions into structured semantic slots:
      - hour: int (0-23)
      - minute_offset: int (0-59)
      - direction: 'PAST' | 'TO' | 'EXACT' | None
      - meridiem: 'AM' | 'PM' | None
      - is_24h: bool
    """

    DIGITAL_TIME_REGEX = re.compile(r'\b(\d{1,2}):(\d{2})\b')

    @classmethod
    def parse_digital(cls, text: str) -> Optional[Dict[str, Any]]:
        """Extracts direct digital time representations (e.g. '14:30', '4:07')."""
        match = cls.DIGITAL_TIME_REGEX.search(text)
        if not match:
            return None

        h_str, m_str = match.groups()
        hour = int(h_str)
        minute = int(m_str)

        if hour < 0 or hour > 23 or minute < 0 or minute > 59:
            return None

        meridiem = None
        lower = text.lower()
        if "am" in lower or "a.m." in lower or "morning" in lower:
            meridiem = "AM"
        elif "pm" in lower or "p.m." in lower or "afternoon" in lower or "evening" in lower or "night" in lower:
            meridiem = "PM"

        is_24h = hour >= 13 or (len(h_str) == 2 and h_str.startswith("0"))

        return {
            "hour": hour,
            "minute_offset": minute,
            "direction": "EXACT",
            "meridiem": meridiem,
            "is_24h": is_24h,
        }

    @classmethod
    def tokenize_and_encode(cls, text: str) -> List[Tuple[str, Bitboard64]]:
        """Tokenizes text and produces structured Bitboard64 tokens."""
        # Normalize text
        norm = text.lower().replace("a.m.", "am").replace("p.m.", "pm")
        norm = norm.replace("o' clock", "o'clock").replace("o ’ clock", "o'clock")
        cleaned = re.sub(r'([^\w\s\':-])', r' \1 ', norm)
        raw_tokens = [t.strip() for t in cleaned.split() if t.strip()]

        encoded: List[Tuple[str, Bitboard64]] = []
        i = 0
        while i < len(raw_tokens):
            tok = raw_tokens[i]

            # Check for hyphenated compound number split like ['twenty', '-', 'five']
            if i + 2 < len(raw_tokens) and raw_tokens[i + 1] == '-':
                combined = f"{tok}-{raw_tokens[i + 2]}"
                bb = encode_token_to_bitboard(combined)
                if bb.role() == Bitboard64.ROLE_NUMBER:
                    encoded.append((combined, bb))
                    i += 3
                    continue

            # Check for space compound number like ['twenty', 'five']
            if i + 1 < len(raw_tokens):
                combined = f"{tok} {raw_tokens[i + 1]}"
                bb = encode_token_to_bitboard(combined)
                if bb.role() == Bitboard64.ROLE_NUMBER:
                    encoded.append((combined, bb))
                    i += 2
                    continue

            encoded.append((tok, encode_token_to_bitboard(tok)))
            i += 1

        return encoded

    @classmethod
    def extract_slots(cls, text: str) -> Dict[str, Any]:
        """
        Extracts semantic slots from text using structured bitboard role queries.
        Returns a dictionary of slots. If syntax is invalid, returns hour=None.
        """
        # First check digital format
        digital_slots = cls.parse_digital(text)
        if digital_slots is not None:
            return digital_slots

        # Tokenize and encode into bitboards
        token_bbs = cls.tokenize_and_encode(text)
        if not token_bbs:
            return {
                "hour": None,
                "minute_offset": 0,
                "direction": None,
                "meridiem": None,
                "is_24h": False,
            }

        # Check for special time words (noon, midnight, midday)
        for tok, bb in token_bbs:
            clean = tok.strip().lower()
            if clean in SPECIAL_TIME_WORDS:
                h, m = SPECIAL_TIME_WORDS[clean]
                # Check if preceded by direction, e.g. "ten to midnight"
                # Handled below by assigning this as a number/hour
                pass

        direction: Optional[str] = None
        dir_idx: Optional[int] = None
        meridiem: Optional[str] = None
        modifier_offset: Optional[int] = None
        modifier_idx: Optional[int] = None

        # Scan for direction, meridiem, and modifiers via bitboard role
        for idx, (tok, bb) in enumerate(token_bbs):
            role = bb.role()

            if role == Bitboard64.ROLE_PAST:
                direction = "PAST"
                dir_idx = idx
            elif role == Bitboard64.ROLE_TO:
                direction = "TO"
                dir_idx = idx
            elif role == Bitboard64.ROLE_QUARTER:
                modifier_offset = 15
                modifier_idx = idx
            elif role == Bitboard64.ROLE_HALF:
                modifier_offset = 30
                modifier_idx = idx
            elif role == Bitboard64.ROLE_AM:
                meridiem = "AM"
            elif role == Bitboard64.ROLE_PM:
                meridiem = "PM"

        # Check special words in token sequence
        special_hour: Optional[int] = None
        special_idx: Optional[int] = None
        for idx, (tok, bb) in enumerate(token_bbs):
            clean = tok.strip().lower()
            if clean in SPECIAL_TIME_WORDS:
                h, _ = SPECIAL_TIME_WORDS[clean]
                special_hour = h
                special_idx = idx
                if clean == "midnight":
                    meridiem = "PM" if direction == "TO" else "AM"
                elif clean in ("noon", "midday"):
                    meridiem = "PM"
                break

        # Collect number tokens
        numbers: List[Tuple[int, int]] = []  # (index, value)
        for idx, (tok, bb) in enumerate(token_bbs):
            if bb.role() == Bitboard64.ROLE_NUMBER:
                numbers.append((idx, bb.numeric_value()))

        # If special hour exists (e.g. midnight, noon) and no explicit number
        if special_hour is not None:
            if direction is None and modifier_offset is None and not numbers:
                # Standalone "noon" or "midnight"
                return {
                    "hour": special_hour,
                    "minute_offset": 0,
                    "direction": "EXACT",
                    "meridiem": meridiem,
                    "is_24h": False,
                }
            # Part of directional phrase, e.g. "ten to midnight"
            numbers.append((special_idx, special_hour))
            numbers.sort(key=lambda x: x[0])

        hour: Optional[int] = None
        minute_offset: Optional[int] = None

        if direction is not None:
            # Directional phrase: requires minute/modifier BEFORE direction, and hour AFTER direction
            # If direction is the first token, it is malformed (e.g. "past four quarter")
            if dir_idx == 0:
                return {
                    "hour": None,
                    "minute_offset": None,
                    "direction": None,
                    "meridiem": None,
                    "is_24h": False,
                }

            # Minutes before direction
            mins_before = [val for idx, val in numbers if idx < dir_idx]
            # Hours after direction
            hours_after = [val for idx, val in numbers if idx > dir_idx]

            if modifier_idx is not None and modifier_idx < dir_idx:
                minute_offset = modifier_offset
            elif mins_before:
                minute_offset = mins_before[-1]

            if hours_after:
                hour = hours_after[0]

            # Structural syntax validation: both hour and minute must be correctly placed
            if hour is None or minute_offset is None:
                return {
                    "hour": None,
                    "minute_offset": None,
                    "direction": None,
                    "meridiem": None,
                    "is_24h": False,
                }

        else:
            # No direction word (EXACT)
            direction = "EXACT"
            if len(numbers) >= 2:
                # "four twenty", "seven forty five"
                hour = numbers[0][1]
                minute_offset = numbers[1][1]
            elif len(numbers) == 1:
                # "five o'clock", "five", "five pm"
                hour = numbers[0][1]
                minute_offset = 0
            elif modifier_offset is not None:
                # modifier without direction and without hour -> invalid
                hour = None
                minute_offset = modifier_offset

        # Validate extracted hour (0-23) and minute_offset (0-59)
        if hour is not None and (hour < 0 or hour > 23):
            hour = None
        if minute_offset is not None and (minute_offset < 0 or minute_offset > 59):
            minute_offset = None

        return {
            "hour": hour,
            "minute_offset": minute_offset if minute_offset is not None else 0,
            "direction": direction,
            "meridiem": meridiem,
            "is_24h": False,
        }
