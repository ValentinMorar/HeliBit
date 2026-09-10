"""
HeliBit-AI: Local Hebbian Trainer & Prototype Calibration Module
Performs online local resonance training, token mass consolidation,
role prototype accumulation, and model serialization for HeliBit-AI v2.
"""

import os
import json
import time
import tempfile
from typing import Dict, List, Tuple, Any
from helibit.bitboard import Bitboard64
from helibit.parser import TimeSlotParser
from helibit.engine import HeliBitEngine


class HeliBitTrainer:
    """
    Online Local Hebbian Trainer for HeliBit-AI v2.
    Calibrates token dynamics, accumulates role prototypes from training bitboards,
    and validates compositional slot resolution.
    """

    def __init__(self, engine: HeliBitEngine):
        self.engine = engine
        self.compiler = engine.compiler
        self.oscillator = engine.oscillator
        self.training_history: List[Dict[str, float]] = []

    def train_epoch(self, dataset: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Executes one local training epoch over the dataset.
        Updates Hebbian token masses and builds role prototype bitboard centroids.
        """
        start_time = time.perf_counter()
        total_samples = len(dataset)
        correct_predictions = 0
        total_resonance_force = 0.0
        total_hamming_dist = 0.0
        pair_count = 0

        # Accumulators for role prototype centroids: role -> list of fingerprint bit values
        role_bits: Dict[int, List[int]] = {}

        for text, target in dataset:
            # 1. Tokenize and encode input sequence
            token_bbs = TimeSlotParser.tokenize_and_encode(text)
            tokens = [t for t, _ in token_bbs]
            bitboards = [b for _, b in token_bbs]

            # Register tokens in engine token_states and compiler
            for tok, bb in token_bbs:
                state = self.engine._get_or_create_state(tok)
                state.reinforce_mass(delta=0.05)
                self.compiler.vocabulary[tok.lower()] = bb

                role = bb.role()
                if role != Bitboard64.ROLE_UNKNOWN:
                    if role not in role_bits:
                        role_bits[role] = []
                    role_bits[role].append(bb.fingerprint())

            # Pairwise resonance and Hebbian oscillator relaxation
            for i in range(len(tokens) - 1):
                t1, t2 = tokens[i], tokens[i + 1]
                b1, b2 = bitboards[i], bitboards[i + 1]

                resonance = float(b1.resonance_score(b2))
                dist = b1.hamming_distance(b2)

                total_resonance_force += resonance
                total_hamming_dist += dist
                pair_count += 1

                self.compiler.register_transition(t1, t2, weight=1.0)
                state1 = self.engine._get_or_create_state(t1)
                state2 = self.engine._get_or_create_state(t2)

                self.oscillator.step_dynamics(state1, resonance_force=resonance)
                self.oscillator.step_dynamics(state2, resonance_force=resonance)

            # 2. Compositional inference evaluation
            result = self.engine.predict(text)
            if result.predicted_target == target:
                correct_predictions += 1

        # Synthesize role prototypes via majority-voting / bitwise OR
        for role, fps in role_bits.items():
            if fps:
                combined_fp = 0
                for bit_pos in range(54):
                    bit_count = sum((fp >> bit_pos) & 1 for fp in fps)
                    if bit_count * 2 >= len(fps):
                        combined_fp |= (1 << bit_pos)
                self.engine.role_prototypes[role] = combined_fp

        accuracy = (correct_predictions / total_samples) * 100.0 if total_samples > 0 else 0.0
        mean_resonance = (total_resonance_force / pair_count) if pair_count > 0 else 0.0
        mean_dh = (total_hamming_dist / pair_count) if pair_count > 0 else 0.0
        elapsed_sec = time.perf_counter() - start_time

        epoch_stats = {
            "accuracy": accuracy,
            "mean_resonance_force": mean_resonance,
            "mean_hamming_distance": mean_dh,
            "elapsed_sec": elapsed_sec,
            "sample_count": total_samples,
        }

        self.training_history.append(epoch_stats)
        return epoch_stats

    def train(self, dataset: List[Tuple[str, str]], epochs: int = 5) -> List[Dict[str, float]]:
        """Runs multiple training epochs."""
        history = []
        for ep in range(1, epochs + 1):
            stats = self.train_epoch(dataset)
            stats["epoch"] = ep
            history.append(stats)
        return history

    def save_checkpoint(self, filepath: str) -> str:
        """Serializes engine state, role prototypes, token masses, and dynamics parameters to JSON."""
        state_data: Dict[str, Any] = {
            "version": "2.0.0",
            "architecture": "HeliBit-AI v2 Compositional Neuro-Symbolic",
            "role_prototypes": {str(k): v for k, v in self.engine.role_prototypes.items()},
            "token_states": {},
            "co_occurrence_weights": {},
        }

        for token, state in self.engine.token_states.items():
            state_data["token_states"][token] = {
                "mass": state.mass,
                "pitch": state.pitch,
                "velocity": state.velocity,
                "activation_count": state.activation_count,
            }

        for (t1, t2), weight in self.compiler.co_occurrence_weights.items():
            key = f"{t1}->{t2}"
            state_data["co_occurrence_weights"][key] = weight

        json_str = json.dumps(state_data, indent=2)

        target_path = os.path.abspath(filepath)
        try:
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(json_str)
            return target_path
        except (OSError, FileNotFoundError, PermissionError):
            fallback_path = os.path.join(tempfile.gettempdir(), os.path.basename(filepath))
            with open(fallback_path, "w", encoding="utf-8") as f:
                f.write(json_str)
            return fallback_path

    def load_checkpoint(self, filepath: str) -> None:
        """Loads trained topological state parameters from a JSON checkpoint file."""
        temp_path = os.path.join(tempfile.gettempdir(), os.path.basename(filepath))
        target_path = os.path.abspath(filepath)

        if os.path.exists(temp_path):
            if not os.path.exists(target_path) or os.path.getmtime(temp_path) >= os.path.getmtime(target_path):
                target_path = temp_path

        with open(target_path, "r", encoding="utf-8") as f:
            state_data = json.load(f)

        # Load role prototypes
        if "role_prototypes" in state_data:
            self.engine.role_prototypes = {
                int(k): int(v) for k, v in state_data["role_prototypes"].items()
            }

        # Load token states
        for token, data in state_data.get("token_states", {}).items():
            state = self.engine._get_or_create_state(token)
            state.mass = data.get("mass", 1.0)
            state.pitch = data.get("pitch", 1.0)
            state.velocity = data.get("velocity", 0.0)
            state.activation_count = data.get("activation_count", 0)

        # Load transition weights
        for key, weight in state_data.get("co_occurrence_weights", {}).items():
            if "->" in key:
                t1, t2 = key.split("->", 1)
                self.compiler.co_occurrence_weights[(t1, t2)] = weight
