"""
HeliBit-AI: Local Hebbian Trainer Module
Performs online local resonance training, token mass consolidation, and spring relaxation over training corpora.
Persists trained topological network checkpoints.
"""

import os
import json
import time
import tempfile
from typing import Dict, List, Tuple, Any
from helibit.bitboard import Bitboard64
from helibit.dynamics import HebbianOscillator, TokenState
from helibit.compiler import TopologicalCompiler
from helibit.engine import HeliBitEngine


class HeliBitTrainer:
    """
    Online Local Hebbian Trainer operating without global Backpropagation.
    Refines token bitboard feature overlaps, updates masses, and relaxes transition pitch springs.
    """

    def __init__(self, engine: HeliBitEngine):
        self.engine = engine
        self.compiler = engine.compiler
        self.oscillator = engine.oscillator
        self.training_history: List[Dict[str, float]] = []

    def train_epoch(self, dataset: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Executes one local Hebbian training epoch over the dataset.
        Returns loss and resonance metrics.
        """
        start_time = time.perf_counter()
        total_samples = len(dataset)
        correct_predictions = 0
        total_resonance_force = 0.0
        total_hamming_dist = 0.0
        pair_count = 0

        # Register all targets in engine
        for _, target in dataset:
            self.engine.register_target(target)

        candidate_targets = list(self.engine.known_targets)

        for text, target in dataset:
            # 1. Register input sequence tokens in engine token_states
            input_tokens = [t for t, _ in self.compiler.compile_sequence(text)]
            for tok in input_tokens:
                self.compiler.register_transition(tok, target, weight=2.0)
                state_tok = self.engine._get_or_create_state(tok)
                state_tok.reinforce_mass(delta=0.05)

            full_seq = f"{text} {target}"
            token_pairs = self.compiler.compile_sequence(full_seq)
            tokens = [t for t, _ in token_pairs]
            bitboards = [b for _, b in token_pairs]

            for i in range(len(tokens) - 1):
                t1, t2 = tokens[i], tokens[i + 1]
                b1, b2 = bitboards[i], bitboards[i + 1]

                # Compute register overlap resonance
                resonance = float(b1.resonance_score(b2))
                dist = b1.hamming_distance(b2)
                
                total_resonance_force += resonance
                total_hamming_dist += dist
                pair_count += 1

                # Update graph tension and token mass consolidation
                self.compiler.register_transition(t1, t2, weight=2.0)
                state1 = self.engine._get_or_create_state(t1)
                state2 = self.engine._get_or_create_state(t2)
                
                state1.reinforce_mass(delta=0.05)
                state2.reinforce_mass(delta=0.05)

                # Driven damped harmonic oscillator step
                self.oscillator.step_dynamics(state1, resonance_force=resonance)
                self.oscillator.step_dynamics(state2, resonance_force=resonance)

            # 2. Evaluate accuracy post Hebbian adaptation
            result = self.engine.predict(text, candidate_targets=candidate_targets)
            if result.predicted_target == target:
                correct_predictions += 1

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

    def train(self, dataset: List[Tuple[str, str]], epochs: int = 10) -> List[Dict[str, float]]:
        """Runs multiple local Hebbian training epochs."""
        history = []
        for ep in range(1, epochs + 1):
            stats = self.train_epoch(dataset)
            stats["epoch"] = ep
            history.append(stats)
        return history

    def save_checkpoint(self, filepath: str) -> str:
        """Serializes trained topological token states, masses, and pitch parameters to JSON."""
        state_data: Dict[str, Any] = {
            "token_states": {},
            "co_occurrence_weights": {},
            "known_targets": list(self.engine.known_targets),
            "version": "1.0.0",
        }

        # Ensure all vocabulary tokens in compiler are registered in engine.token_states
        for token in self.compiler.vocabulary.keys():
            self.engine._get_or_create_state(token)

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
        
        # Try direct write, fallback to temp directory if restricted by OS sandbox
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
        
        # Use temp_path if it exists and was updated more recently
        if os.path.exists(temp_path):
            if not os.path.exists(target_path) or os.path.getmtime(temp_path) >= os.path.getmtime(target_path):
                target_path = temp_path

        with open(target_path, "r", encoding="utf-8") as f:
            state_data = json.load(f)

        for tgt in state_data.get("known_targets", []):
            self.engine.register_target(tgt)

        for token, data in state_data.get("token_states", {}).items():
            state = self.engine._get_or_create_state(token)
            state.mass = data.get("mass", 1.0)
            state.pitch = data.get("pitch", 1.0)
            state.velocity = data.get("velocity", 0.0)
            state.activation_count = data.get("activation_count", 0)

        for key, weight in state_data.get("co_occurrence_weights", {}).items():
            if "->" in key:
                t1, t2 = key.split("->", 1)
                self.compiler.co_occurrence_weights[(t1, t2)] = weight
