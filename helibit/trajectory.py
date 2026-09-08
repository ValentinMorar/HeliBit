"""
HeliBit-AI: Parametric Helical Trajectory Module
Implements semantic arcs, continuous spatial wave propagation, and Lyapunov Guard damping operators.
"""

import math
from typing import Tuple, List


class HelicalTrajectory:
    """
    Models semantic relationships as parametric helical spatial curves:
    x(t) = R(t) * cos(w * t + phi)
    y(t) = R(t) * sin(w * t + phi)
    z(t) = p * t
    """

    def __init__(
        self,
        base_radius: float = 1.0,
        carrier_pitch: float = 1.0,
        phase_angle: float = 0.0,
        angular_velocity: float = 1.0,
        lyapunov_lambda: float = 0.15,
    ):
        """
        Initialize parametric helical parameters.
        :param base_radius: Initial interaction boundary R0.
        :param carrier_pitch: Axial propagation step pitch p_ij propelling wave packet.
        :param phase_angle: Initial semantic phase angle phi_ij.
        :param angular_velocity: Rotation frequency omega.
        :param lyapunov_lambda: Damping coefficient lambda for Lyapunov Guard function.
        """
        self.R0 = base_radius
        self.pitch = carrier_pitch
        self.phase = phase_angle
        self.omega = angular_velocity
        self.lambda_guard = lyapunov_lambda

    def radius(self, t: float, entropy_integral: float = 0.0) -> float:
        """
        Computes bounded radius using the Exponential Lyapunov Guard Function:
        R(t) = R0 * exp(-lambda * (t + entropy_integral))
        """
        guard_decay = self.lambda_guard * (t + entropy_integral)
        return self.R0 * math.exp(-guard_decay)

    def evaluate(self, t: float, entropy_integral: float = 0.0) -> Tuple[float, float, float]:
        """
        Evaluates 3D spatial coordinate [x(t), y(t), z(t)] along the helical trajectory.
        """
        r_t = self.radius(t, entropy_integral)
        angle = self.omega * t + self.phase
        
        x = r_t * math.cos(angle)
        y = r_t * math.sin(angle)
        z = self.pitch * t
        
        return (x, y, z)

    def compute_arc(
        self, steps: int = 50, max_t: float = 2.0, entropy_integral: float = 0.0
    ) -> List[Tuple[float, float, float]]:
        """Generates spatial points along trajectory from t=0 to max_t."""
        dt = max_t / steps
        return [self.evaluate(i * dt, entropy_integral) for i in range(steps + 1)]

    def is_collapsed(self, t: float, threshold: float = 0.05, entropy_integral: float = 0.0) -> bool:
        """Checks if Lyapunov Guard function has collapsed radius onto target node."""
        return self.radius(t, entropy_integral) < threshold

    def __repr__(self) -> str:
        return (
            f"HelicalTrajectory(R0={self.R0:.2f}, pitch={self.pitch:.2f}, "
            f"phi={self.phase:.2f}, lambda={self.lambda_guard:.2f})"
        )
