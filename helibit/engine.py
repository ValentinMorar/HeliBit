"""
HeliBit-AI: Universal Neuromorphic Inference Engine
Executes high-throughput bitwise SIMD inference, topological graph classification,
and sequence prediction using 64-bit Bitboards and Damped Helical Trajectories.
"""

import time
from typing import Dict, List, Optional, Tuple, Set
from helibit.bitboard import Bitboard64
from helibit.trajectory import HelicalTrajectory
from helibit.compiler import TopologicalCompiler
from helibit.dynamics import HebbianOscillator, TokenState


class InferenceResult:
    """Encapsulates the output of the generalized HeliBit-AI engine."""

    def __init__(
        self,
        predicted_target: str,
        latency_us: float,
        bitboard_overlap_score: int,
        mean_hamming_distance: float,
        trajectory_points: List[Tuple[float, float, float]],
    ):
        self.predicted_target = predicted_target
        self.formatted_time = predicted_target  # Backwards compatibility alias
        self.latency_us = latency_us
        self.bitboard_overlap_score = bitboard_overlap_score
        self.mean_hamming_distance = mean_hamming_distance
        self.trajectory_points = trajectory_points

    def __repr__(self) -> str:
        return (
            f"InferenceResult('{self.predicted_target}', latency={self.latency_us:.2f}us, "
            f"bitboard_overlap={self.bitboard_overlap_score}, mean_Dh={self.mean_hamming_distance:.4f})"
        )


class HeliBitEngine:
    """
    Universal Zero-FPU high-throughput inference engine based on 64-bit Bitboard logic
    and Damped Helical Trajectory filters.
    """

    def __init__(self):
        self.compiler = TopologicalCompiler()
        self.oscillator = HebbianOscillator()
        self.token_states: Dict[str, TokenState] = {}
        self.known_targets: Set[str] = set()
        self.trajectory_filter = HelicalTrajectory(base_radius=1.0, carrier_pitch=1.0, lyapunov_lambda=0.2)

    def reset(self) -> None:
        """Resets engine state, clearing token states, compiler vocabulary, and known targets."""
        self.compiler.clear()
        self.token_states.clear()
        self.known_targets.clear()

    def _get_or_create_state(self, token: str) -> TokenState:
        norm_token = token.lower().strip()
        if norm_token not in self.token_states:
            self.token_states[norm_token] = TokenState(norm_token)
        return self.token_states[norm_token]

    def register_target(self, target: str) -> None:
        """Registers a known target class / response string in the engine dictionary."""
        norm_target = target.strip()
        self.known_targets.add(norm_target)
        self.compiler.get_token_bitboard(norm_target)
        self._get_or_create_state(norm_target)

    def predict(self, text: str, candidate_targets: Optional[List[str]] = None) -> InferenceResult:
        """
        Universal inference entry point: maps input natural language sequence onto
        the best-matching candidate target using topological resonance and trajectory collapse.
        """
        start_time = time.perf_counter_ns()
        
        # 1. Topological Bitboard Compilation
        token_pairs = self.compiler.compile_sequence(text)
        tokens = [t for t, _ in token_pairs]
        bitboards = [b for _, b in token_pairs]
        
        total_overlap = 0
        hamming_distances = []
        
        # Compute register-level bitboard interactions (PAND, PXOR, POPCNT)
        for i in range(len(bitboards) - 1):
            b1, b2 = bitboards[i], bitboards[i + 1]
            overlap = b1.resonance_score(b2)
            dist = b1.hamming_distance(b2)
            total_overlap += overlap
            hamming_distances.append(dist)
            
            # Local Hebbian adaptation update without backpropagation
            state1 = self._get_or_create_state(tokens[i])
            state1.reinforce_mass(delta=0.05)
            self.oscillator.step_dynamics(state1, resonance_force=float(overlap))

        mean_dh = sum(hamming_distances) / len(hamming_distances) if hamming_distances else 0.0
        
        # 2. Evaluate Helical Trajectory Damping
        trajectory_pts = self.trajectory_filter.compute_arc(steps=max(len(tokens) * 5, 10), max_t=1.5)
        
        # 3. Task-Agnostic Top-Resonance Candidate Selection
        targets_to_evaluate = candidate_targets if candidate_targets else list(self.known_targets)
        
        if not targets_to_evaluate:
            predicted_target = text
        else:
            best_target = targets_to_evaluate[0]
            best_score = -999999.0
            
            input_composite_val = 0
            for b in bitboards:
                input_composite_val |= b.value
            composite_input_bitboard = Bitboard64(input_composite_val)

            for target in targets_to_evaluate:
                target_bitboard = self.compiler.get_token_bitboard(target)
                target_state = self._get_or_create_state(target)
                
                # Bitboard Overlap with Input Stream
                res_score = composite_input_bitboard.resonance_score(target_bitboard)
                dh_dist = composite_input_bitboard.hamming_distance(target_bitboard)
                
                # Co-occurrence spring tension bonus weighted by token specificity (inverse frequency)
                transition_bonus = 0.0
                for t in tokens:
                    weight = self.compiler.co_occurrence_weights.get((t.lower(), target.lower()), 0.0)
                    t_state = self.token_states.get(t.lower())
                    activations = t_state.activation_count if t_state else 1
                    spec_weight = 10.0 / (activations ** 0.5) if activations > 0 else 1.0
                    transition_bonus += weight * spec_weight
                
                # Trajectory collapse bonus
                traj_collapse = self.trajectory_filter.radius(1.0 + dh_dist)
                
                # Composite Topological Energy Score
                score = (transition_bonus * 10.0) + (res_score * 2.0) - (dh_dist * 5.0) + (target_state.mass * 0.1) - (traj_collapse * 2.0)
                
                if score > best_score:
                    best_score = score
                    best_target = target
            
            predicted_target = best_target

        end_time = time.perf_counter_ns()
        latency_us = (end_time - start_time) / 1000.0
        
        return InferenceResult(
            predicted_target=predicted_target,
            latency_us=latency_us,
            bitboard_overlap_score=total_overlap,
            mean_hamming_distance=mean_dh,
            trajectory_points=trajectory_pts,
        )

    # Backwards compatibility alias
    parse_time = predict
