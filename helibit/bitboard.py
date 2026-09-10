"""
HeliBit-AI: Topological Bitboard Module
Implements 64-bit cellular bitboard representations, structured role/value bitfields,
bitwise operators, and distance metrics as specified in the HeliBit-AI v2 architecture.
"""

from typing import Union


class Bitboard64:
    """
    Represents an 8x8 boolean matrix (64-bit cellular bitboard) as a single uint64 scalar.
    In v2, the 64-bit scalar is structured into explicit semantic fields:
      - Bits 63–60: ROLE (0=NUMBER, 1=UNIT_HOUR, 2=DIRECTION_PAST, 3=DIRECTION_TO,
                          4=MODIFIER_QUARTER, 5=MODIFIER_HALF, 6=MERIDIEM_AM, 7=MERIDIEM_PM,
                          8=UNKNOWN)
      - Bits 59–54: VALUE (0–63 integer value, used when ROLE=NUMBER)
      - Bits 53–0:  LEXICAL_FINGERPRINT (hash for POPCNT/Hamming resonance and disambiguation)
    """

    MASK_64 = 0xFFFFFFFFFFFFFFFF
    ROLE_SHIFT = 60
    ROLE_MASK = 0xF
    VALUE_SHIFT = 54
    VALUE_MASK = 0x3F
    FINGERPRINT_MASK = (1 << 54) - 1

    # Standard Role Codes
    ROLE_NUMBER = 0
    ROLE_HOUR_UNIT = 1
    ROLE_PAST = 2
    ROLE_TO = 3
    ROLE_QUARTER = 4
    ROLE_HALF = 5
    ROLE_AM = 6
    ROLE_PM = 7
    ROLE_UNKNOWN = 8

    def __init__(self, value: int = 0):
        """Initialize a 64-bit bitboard with an integer scalar value."""
        self.value = value & self.MASK_64

    @classmethod
    def from_role_value(cls, role: int, value: int = 0, fingerprint: int = 0) -> "Bitboard64":
        """
        Creates a structured Bitboard64 encoding explicit role, numeric value, and lexical fingerprint.
        """
        encoded = (
            ((role & cls.ROLE_MASK) << cls.ROLE_SHIFT)
            | ((value & cls.VALUE_MASK) << cls.VALUE_SHIFT)
            | (fingerprint & cls.FINGERPRINT_MASK)
        )
        return cls(encoded)

    def role(self) -> int:
        """Extracts the semantic ROLE field (bits 63–60)."""
        return (self.value >> self.ROLE_SHIFT) & self.ROLE_MASK

    def numeric_value(self) -> int:
        """Extracts the numeric VALUE field (bits 59–54)."""
        return (self.value >> self.VALUE_SHIFT) & self.VALUE_MASK

    def fingerprint(self) -> int:
        """Extracts the lexical fingerprint field (bits 53–0)."""
        return self.value & self.FINGERPRINT_MASK

    @classmethod
    def from_pattern(cls, pattern: str) -> "Bitboard64":
        """
        Creates a Bitboard64 from a 64-character binary string (e.g. '0101...').
        Newlines and spaces are automatically stripped.
        """
        clean_pat = pattern.replace("\n", "").replace(" ", "").strip()
        if len(clean_pat) != 64:
            raise ValueError(f"Pattern must be exactly 64 bits long, got {len(clean_pat)}")
        val = int(clean_pat, 2)
        return cls(val)

    @classmethod
    def from_string(cls, text: str) -> "Bitboard64":
        """
        Deterministic topological compiler helper: maps a string token to a 64-bit bitboard.
        Uses multi-hash feature bit placement across the 8x8 lattice.
        """
        val = 0
        text_bytes = text.encode("utf-8")
        # Generate 64-bit feature lattice using prime step hashing
        for i, byte in enumerate(text_bytes):
            bit_pos1 = (byte * 31 + i * 17) % 64
            bit_pos2 = (byte * 59 + i * 43) % 64
            val |= (1 << bit_pos1) | (1 << bit_pos2)
        
        # Ensure non-zero bitboard for valid tokens
        if val == 0:
            val = 0x0123456789ABCDEF
            
        return cls(val)

    def intersect(self, other: "Bitboard64") -> "Bitboard64":
        """Concept Intersection (BA & BB) - PAND / AND instruction (1 cycle)."""
        return Bitboard64(self.value & other.value)

    def diverge(self, other: "Bitboard64") -> "Bitboard64":
        """Structural Divergence (BA ^ BB) - PXOR / XOR instruction (1 cycle)."""
        return Bitboard64(self.value ^ other.value)

    def popcount(self) -> int:
        """Energy / Resonance Score (||B||1) - POPCNT instruction (1 cycle)."""
        return self.value.bit_count()

    def hamming_distance(self, other: "Bitboard64") -> float:
        """
        Normalized Hamming Metric:
        DH(BA, BB) = POPCNT(BA ^ BB) / 64
        """
        diff = self.diverge(other)
        return diff.popcount() / 64.0

    def resonance_score(self, other: "Bitboard64") -> int:
        """Computes overlap resonance score: POPCNT(BA & BB)."""
        return self.intersect(other).popcount()

    def rotate_left(self, shift: int = 1) -> "Bitboard64":
        """Axial pitch translation / state rotation (SHL/ROR instruction)."""
        shift = shift % 64
        rotated = ((self.value << shift) | (self.value >> (64 - shift))) & self.MASK_64
        return Bitboard64(rotated)

    def to_grid(self) -> str:
        """Returns visual representation as an 8x8 boolean grid string."""
        binary_str = f"{self.value:064b}"
        rows = [binary_str[i : i + 8] for i in range(0, 64, 8)]
        return "\n".join(" ".join('.' if c == '0' else '#' for c in row) for row in rows)

    def __and__(self, other: "Bitboard64") -> "Bitboard64":
        return self.intersect(other)

    def __xor__(self, other: "Bitboard64") -> "Bitboard64":
        return self.diverge(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Bitboard64):
            return False
        return self.value == other.value

    def __repr__(self) -> str:
        return (
            f"Bitboard64(0x{self.value:016X}, role={self.role()}, "
            f"val={self.numeric_value()}, bits={self.popcount()})"
        )
