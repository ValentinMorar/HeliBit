"""
HeliBit-AI: Topological Bitboard Module
Implements 64-bit cellular bitboard representations, bitwise operators, 
and distance metrics as specified in the HeliBit-AI architecture.
"""

from typing import Union


class Bitboard64:
    """
    Represents an 8x8 boolean matrix (64-bit cellular bitboard) as a single uint64 scalar.
    Executes concept operations in native register instructions (AND, XOR, POPCNT).
    """

    MASK_64 = 0xFFFFFFFFFFFFFFFF

    def __init__(self, value: int = 0):
        """Initialize a 64-bit bitboard with an integer scalar value."""
        self.value = value & self.MASK_64

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
        return f"Bitboard64(0x{self.value:016X}, bits={self.popcount()})"
