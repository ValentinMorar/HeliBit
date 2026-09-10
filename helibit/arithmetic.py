"""
HeliBit-AI: Integer Arithmetic Module
Implements pure integer arithmetic (zero-FPU) to compute 24-hour HH:MM time
strings from extracted semantic slots.
"""

from typing import Optional, Dict, Any


def resolve_time(slots: Dict[str, Any]) -> Optional[str]:
    """
    Computes time string 'HH:MM' from semantic slots:
      slots = {
          "hour": Optional[int],          # 0-23
          "minute_offset": Optional[int], # 0-59
          "direction": Optional[str],     # "PAST", "TO", "EXACT", or None
          "meridiem": Optional[str],      # "AM", "PM", or None
          "is_24h": Optional[bool],       # True if directly parsed from 24h format
      }

    Returns 'HH:MM' or None if essential slots are missing or invalid (abstention).
    """
    raw_hour = slots.get("hour")
    if raw_hour is None:
        return None  # Abstention: no hour identified

    if not isinstance(raw_hour, int) or raw_hour < 0 or raw_hour > 23:
        return None

    raw_offset = slots.get("minute_offset")
    minute_offset = 0 if raw_offset is None else raw_offset
    if not isinstance(minute_offset, int) or minute_offset < 0 or minute_offset >= 60:
        return None

    direction = slots.get("direction") or "EXACT"
    meridiem = slots.get("meridiem")
    is_24h = slots.get("is_24h", False) or raw_hour >= 13

    if is_24h:
        # Direct 24-hour representation
        hour = raw_hour
        if direction == "TO":
            minute = (60 - minute_offset) % 60
            if minute_offset != 0:
                hour = (hour - 1) % 24
        else:
            minute = minute_offset
        return f"{hour:02d}:{minute:02d}"

    # 12-hour base arithmetic
    hour = raw_hour % 12
    if direction == "TO":
        minute = (60 - minute_offset) % 60
        if minute_offset != 0:
            hour = (hour - 1) % 12
    else:
        # EXACT or PAST
        minute = minute_offset

    # Meridiem resolution
    if meridiem == "PM":
        display_hour = (hour + 12) if hour != 12 else 12
    elif meridiem == "AM":
        display_hour = 0 if hour == 0 or raw_hour == 12 else hour
    else:
        # When meridiem is None, preserve 12h clock convention (1-12)
        display_hour = hour if hour != 0 else 12

    return f"{display_hour:02d}:{minute:02d}"
