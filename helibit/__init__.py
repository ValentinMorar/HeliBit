"""
HeliBit-AI: Universal Neuromorphic Compositional Architecture based on
Parametric Helical Trajectories and Topological 64-Bitboard Representations.
Version 2.0.0
"""

from helibit.bitboard import Bitboard64
from helibit.trajectory import HelicalTrajectory
from helibit.dynamics import HebbianOscillator, TokenState
from helibit.compiler import TopologicalCompiler
from helibit.engine import HeliBitEngine, InferenceResult
from helibit.trainer import HeliBitTrainer
from helibit.lexicon import encode_token_to_bitboard
from helibit.parser import TimeSlotParser
from helibit.arithmetic import resolve_time

__version__ = "2.0.0"
__all__ = [
    "Bitboard64",
    "HelicalTrajectory",
    "HebbianOscillator",
    "TokenState",
    "TopologicalCompiler",
    "HeliBitEngine",
    "InferenceResult",
    "HeliBitTrainer",
    "encode_token_to_bitboard",
    "TimeSlotParser",
    "resolve_time",
]
