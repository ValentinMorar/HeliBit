"""
HeliBit-AI: Universal Neuromorphic Computing Architecture based on Parametric Helical Trajectories
and Topological 64/256-Bitboard Representations.
"""

from helibit.bitboard import Bitboard64
from helibit.trajectory import HelicalTrajectory
from helibit.dynamics import HebbianOscillator, TokenState
from helibit.compiler import TopologicalCompiler
from helibit.engine import HeliBitEngine, InferenceResult
from helibit.trainer import HeliBitTrainer

__version__ = "1.0.0"
__all__ = [
    "Bitboard64",
    "HelicalTrajectory",
    "HebbianOscillator",
    "TokenState",
    "TopologicalCompiler",
    "HeliBitEngine",
    "InferenceResult",
    "HeliBitTrainer",
]
