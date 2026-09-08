"""
HeliBit-AI: Local Resonance Dynamics Module
Implements on-the-fly local synaptic adaptation governed by driven, damped harmonic oscillator equations.
Replaces global Backpropagation with local Hebbian spring relaxation.
"""

from typing import Dict


class TokenState:
    """Stores local dynamical state (mass, pitch, velocity) for a semantic token."""

    def __init__(self, token_id: str, initial_mass: float = 1.0, default_pitch: float = 1.0):
        self.token_id = token_id
        self.mass = initial_mass
        self.pitch = default_pitch
        self.velocity = 0.0
        self.default_pitch = default_pitch
        self.activation_count = 0

    def reinforce_mass(self, delta: float = 0.1) -> None:
        """Consolidates token mass via recurrence activation: mi <- mi + delta * I(activation)."""
        self.activation_count += 1
        self.mass += delta

    def __repr__(self) -> str:
        return f"TokenState('{self.token_id}', mass={self.mass:.3f}, pitch={self.pitch:.3f})"


class HebbianOscillator:
    """
    Solves local harmonic oscillator equations for updating semantic step pitch p_ij without Backprop:
    m_i * (d^2 p / dt^2) + gamma * (dp / dt) + k * (p - p0) = F_resonance(t)
    """

    def __init__(
        self,
        damping_gamma: float = 0.5,
        stiffness_k: float = 2.0,
        dt: float = 0.1,
    ):
        """
        :param damping_gamma: Damping coefficient gamma.
        :param stiffness_k: Elastic spring coefficient k.
        :param dt: Time step delta for Euler integration.
        """
        self.gamma = damping_gamma
        self.k = stiffness_k
        self.dt = dt

    def step_dynamics(
        self,
        state: TokenState,
        resonance_force: float,
    ) -> float:
        """
        Performs one integration step of the forced damped oscillator equation.
        Updates pitch and velocity of the token state in-place.
        Returns the updated pitch.
        """
        # Restoring spring force: -k * (p - p0)
        spring_force = -self.k * (state.pitch - state.default_pitch)
        
        # Damping friction force: -gamma * v
        damping_force = -self.gamma * state.velocity
        
        # Net force equation: F_total = F_resonance + spring_force + damping_force
        total_force = resonance_force + spring_force + damping_force
        
        # Acceleration: a = F / m
        acceleration = total_force / max(state.mass, 1e-5)
        
        # Integration: v <- v + a * dt, p <- p + v * dt
        state.velocity += acceleration * self.dt
        state.pitch += state.velocity * self.dt
        
        return state.pitch
