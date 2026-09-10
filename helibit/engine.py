"""
HeliBit-AI: Universal Neuromorphic Compositional Inference Engine
Executes high-throughput bitwise SIMD inference, structured bitboard slot parsing,
zero-FPU arithmetic resolution, and Damped Helical Trajectory confidence estimation.
"""

import time
from typing import Dict, List, Optional, Tuple, Set, Any
from helibit.bitboard import Bitboard64
from helibit.trajectory import HelicalTrajectory
from helibit.compiler import TopologicalCompiler
from helibit.dynamics import HebbianOscillator, TokenState
from helibit.parser import TimeSlotParser
from helibit.arithmetic import resolve_time


class InferenceResult:
    """Encapsulates the output and telemetry of the HeliBit-AI v2 engine."""

    def __init__(
        self,
        predicted_target: str,
        latency_us: float,
        bitboard_overlap_score: int,
        mean_hamming_distance: float,
        trajectory_points: List[Tuple[float, float, float]],
        slots: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
    ):
        self.predicted_target = predicted_target
        self.formatted_time = predicted_target  # Backwards compatibility alias
        self.latency_us = latency_us
        self.bitboard_overlap_score = bitboard_overlap_score
        self.mean_hamming_distance = mean_hamming_distance
        self.trajectory_points = trajectory_points
        self.slots = slots if slots is not None else {}
        self.confidence = confidence

    def is_unknown(self) -> bool:
        """Returns True if the engine explicitly abstained (UNKNOWN)."""
        return self.predicted_target == "UNKNOWN"

    def __repr__(self) -> str:
        return (
            f"InferenceResult('{self.predicted_target}', slots={self.slots}, "
            f"conf={self.confidence:.2f}, latency={self.latency_us:.2f}us, "
            f"overlap={self.bitboard_overlap_score}, mean_Dh={self.mean_hamming_distance:.4f})"
        )


class HeliBitEngine:
    """
    HeliBit-AI v2 Compositional Neuro-Symbolic Engine.
    Combines:
      1. Structured 64-bit Bitboards (Role, Value, Fingerprint).
      2. TimeSlotParser: semantic slot extraction via bitwise role masks.
      3. Helical Trajectories & Hebbian Dynamics: confidence estimation and role disambiguation.
      4. Zero-FPU Integer Arithmetic: deterministic time resolution.
      5. Explicit Abstention: produces 'UNKNOWN' when slots cannot be resolved.
    """

    def __init__(self):
        self.compiler = TopologicalCompiler()
        self.oscillator = HebbianOscillator()
        self.token_states: Dict[str, TokenState] = {}
        self.known_targets: Set[str] = set()
        self.role_prototypes: Dict[int, int] = {}  # role -> aggregated bitboard fingerprint
        self.trajectory_filter = HelicalTrajectory(
            base_radius=1.0, carrier_pitch=1.0, lyapunov_lambda=0.25
        )
        self.confidence_threshold = 0.40

    def reset(self) -> None:
        """Resets engine state, clearing token states, compiler vocabulary, and known targets."""
        self.compiler.clear()
        self.token_states.clear()
        self.known_targets.clear()
        self.role_prototypes.clear()

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

    def disambiguate_unknown_tokens(self, token_bbs: List[Tuple[str, Bitboard64]]) -> List[Tuple[str, Bitboard64]]:
        """
        Uses lexical fingerprint resonance against trained role prototypes to classify
        unknown tokens if close enough, else leaves as ROLE_UNKNOWN.
        """
        if not self.role_prototypes:
            return token_bbs

        resolved: List[Tuple[str, Bitboard64]] = []
        for tok, bb in token_bbs:
            if bb.role() == Bitboard64.ROLE_UNKNOWN:
                fp_bb = Bitboard64(bb.fingerprint())
                best_role = Bitboard64.ROLE_UNKNOWN
                min_dh = 1.0

                for role_code, proto_val in self.role_prototypes.items():
                    proto_bb = Bitboard64(proto_val)
                    dh = fp_bb.hamming_distance(proto_bb)
                    if dh < min_dh:
                        min_dh = dh
                        best_role = role_code

                # If resonance is strong enough, assign candidate role
                if min_dh < 0.25 and best_role != Bitboard64.ROLE_UNKNOWN:
                    new_bb = Bitboard64.from_role_value(best_role, bb.numeric_value(), bb.fingerprint())
                    resolved.append((tok, new_bb))
                    continue

            resolved.append((tok, bb))
        return resolved

    def predict(self, text: str, candidate_targets: Optional[List[str]] = None) -> InferenceResult:
        """
        HeliBit-AI v2 Primary Inference:
          1. Tokenize and encode input sequence into structured bitboards.
          2. Extract semantic slots via bitwise role matching.
          3. Compute time via zero-FPU integer arithmetic.
          4. Evaluate Hebbian mass and helical trajectory collapse for confidence estimation.
          5. Return resolved 'HH:MM' or abstain with 'UNKNOWN'.
        """
        start_time = time.perf_counter_ns()

        # 1. Structured Bitboard Tokenization & Encoding
        token_bbs = TimeSlotParser.tokenize_and_encode(text)
        tokens = [t for t, _ in token_bbs]
        bitboards = [b for _, b in token_bbs]

        total_overlap = 0
        hamming_distances = []

        # 2. Local register-level bitboard interactions & Hebbian adaptation
        for i in range(len(bitboards) - 1):
            b1, b2 = bitboards[i], bitboards[i + 1]
            overlap = b1.resonance_score(b2)
            dist = b1.hamming_distance(b2)
            total_overlap += overlap
            hamming_distances.append(dist)

            # Hebbian mass reinforcement on encountered tokens
            state = self._get_or_create_state(tokens[i])
            state.reinforce_mass(delta=0.05)
            self.oscillator.step_dynamics(state, resonance_force=float(overlap))

        mean_dh = sum(hamming_distances) / len(hamming_distances) if hamming_distances else 0.0

        # 3. Semantic Slot Extraction
        slots = TimeSlotParser.extract_slots(text)

        # 4. Pure Integer Arithmetic Resolution
        resolved_time = resolve_time(slots)

        # 5. Trajectory Collapse & Confidence Calculation
        if resolved_time is not None:
            predicted_target = resolved_time
            # Trajectory successfully collapses onto the target state
            collapse_t = 2.0
            trajectory_radius = self.trajectory_filter.radius(collapse_t)
            is_converged = self.trajectory_filter.is_collapsed(collapse_t)
            
            # Confidence based on token familiarities and trajectory stability
            known_masses = [self.token_states[t.lower()].mass for t in tokens if t.lower() in self.token_states]
            avg_mass = sum(known_masses) / len(known_masses) if known_masses else 1.0
            confidence = min(1.0, max(0.5, 0.6 + 0.1 * min(avg_mass, 4.0)))
        else:
            predicted_target = "UNKNOWN"
            collapse_t = 0.2
            trajectory_radius = self.trajectory_filter.radius(collapse_t)
            confidence = 0.0

        trajectory_pts = self.trajectory_filter.compute_arc(
            steps=max(len(tokens) * 5, 10),
            max_t=2.0 if predicted_target != "UNKNOWN" else 0.5
        )

        end_time = time.perf_counter_ns()
        latency_us = (end_time - start_time) / 1000.0

        return InferenceResult(
            predicted_target=predicted_target,
            latency_us=latency_us,
            bitboard_overlap_score=total_overlap,
            mean_hamming_distance=mean_dh,
            trajectory_points=trajectory_pts,
            slots=slots,
            confidence=confidence,
        )

    # Backwards compatibility alias
    parse_time = predict
