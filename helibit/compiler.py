"""
HeliBit-AI: Topological Bitboard Compiler & Vocabulary Module
Maps natural language tokens onto 64-bit boolean bitboards and continuous transition paths dynamically.
"""

import re
from typing import Dict, List, Tuple, Optional, Iterable
from helibit.bitboard import Bitboard64


class TopologicalCompiler:
    """
    Topological compiler mapping textual tokens to 8x8 (64-bit) cellular bitboard structures
    and constructing trajectory co-occurrence associations.
    """

    def __init__(self):
        self.vocabulary: Dict[str, Bitboard64] = {}
        self.co_occurrence_weights: Dict[Tuple[str, str], float] = {}

    def get_token_bitboard(self, token: str) -> Bitboard64:
        """Returns the 64-bit bitboard signature for a given token, compiling on-the-fly if needed."""
        norm_token = token.lower().strip()
        if norm_token not in self.vocabulary:
            self.vocabulary[norm_token] = Bitboard64.from_string(norm_token)
        return self.vocabulary[norm_token]

    def tokenize(self, text: str) -> List[str]:
        """Normalizes and tokenizes input natural language stream into lexical tokens."""
        norm_text = text.lower().replace("a.m.", "am").replace("p.m.", "pm")
        cleaned = re.sub(r'([^\w\s:])', r' \1 ', norm_text)
        tokens = [t.strip() for t in cleaned.split() if t.strip()]
        return tokens

    def compile_sequence(self, text: str) -> List[Tuple[str, Bitboard64]]:
        """Compiles an input sequence string into an ordered list of (token, Bitboard64) pairs."""
        tokens = self.tokenize(text)
        return [(tok, self.get_token_bitboard(tok)) for tok in tokens]

    def register_transition(self, token1: str, token2: str, weight: float = 1.0) -> None:
        """Updates graph co-occurrence physical spring tension between two sequential tokens."""
        key = (token1.lower(), token2.lower())
        self.co_occurrence_weights[key] = self.co_occurrence_weights.get(key, 0.0) + weight

    def clear(self) -> None:
        """Clears compiled vocabulary and transition weights."""
        self.vocabulary.clear()
        self.co_occurrence_weights.clear()
