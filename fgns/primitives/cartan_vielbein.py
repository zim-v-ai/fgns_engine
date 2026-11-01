"""
Location: // fgns_engine/fgns/primitives/cartan_vielbein.py
Description: SOTA Cartan Moving Frames (Vielbein / Tetrad) Differential Geometry Engine.
             Core Architecture for FGNS v0.6.0 (Unified Causal Manifold):
               1. Exact isometric isomorphism between flat Euclidean neural features R^D
                  and the Riemannian tangent bundle T_x H^D_kappa in R^(1+D).
               2. Strict Minkowski orthogonality preservation: <x, Lift_x(c)>_L == 0.000000.
               3. Exact metric norm preservation: ||Lift_x(c)||_L^2 == ||c||_2^2.
               4. Analytical closed-form roundtrip bijectivity: Descent_x(Lift_x(c)) == c.
               5. Zero-drift Gram-Schmidt projection eliminating multi-layer floating-point drift.
Language: Python 3.12+ / PyTorch 2.5+ / Differential Manifold Gauge Theory
"""

from __future__ import annotations

import math
from typing import Final, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fgns.primitives.lorentz_manifold import (
    decompose_minkowski_coordinates,
    lorentz_inner_product,
    lorentz_norm_squared,
    project_to_hyperboloid,
    project_to_tangent_space,
)

DEFAULT_CARTAN_EPS: Final[float] = 1e-7


class CartanVielbeinFrame(nn.Module):
    """
    SOTA Cartan Vielbein Orthonormal Moving Frame Operator.
    Provides isometric lifting from Euclidean space R^D to the tangent space T_x H^D_kappa,
    and isometric descent back to R^D, preserving exact metric norms and directional angles.
    Supports arbitrary leading polymorphic batch dimensions [..., D] and [..., 1+D].
    """

    def __init__(
        self,
        curvature: float = 1.0,
        eps: float = DEFAULT_CARTAN_EPS,
    ) -> None:
        super().__init__()
        self.curvature = max(float(curvature), eps)
        self.eps = eps

        self.sqrt_curvature = math.sqrt(self.curvature)
        self.inv_sqrt_curvature = 1.0 / self.sqrt_curvature

    def lift_to_tangent(
        self,
        x_base: torch.Tensor,
        c_euclidean: torch.Tensor,
    ) -> torch.Tensor:
        """
        Isometric Cartan Lifting Operator:
            Lift_x: R^D -> T_x H^D_kappa in R^(1+D).
        Maps flat neural features c in R^D into a tangent vector v in R^(1+D) satisfying:
            1. <x_base, v>_L == 0.0 (Strict Minkowski orthogonality).
            2. ||v||_L^2 == ||c||_2^2 (Exact isometric preservation of Euclidean norm).
        Closed-form analytical formula:
            v_0 = sqrt(kappa) * (x_spatial . c)
            v_spatial = c + (kappa * (x_spatial . c) / (1 + sqrt(kappa) * x_0)) * x_spatial.
        """
        target_dtype = c_euclidean.dtype
        x_f32 = x_base.to(torch.float32)
        c_f32 = c_euclidean.to(torch.float32)

        x_time, x_spatial = decompose_minkowski_coordinates(x_f32)

        """Scalar dot product (x_spatial . c) across feature channels"""
        dot_sc = torch.sum(x_spatial * c_f32, dim=-1, keepdim=True)

        """Timelike component v_0 derived from Levi-Civita parallel transport"""
        v_time = self.sqrt_curvature * dot_sc

        """Spacelike components v_spatial with metric dilation factor"""
        denominator = 1.0 + (self.sqrt_curvature * x_time)
        safe_denom = torch.clamp_min(denominator, self.eps)

        factor = (self.curvature * dot_sc) / safe_denom
        v_spatial = c_f32 + (factor * x_spatial)

        v_tangent = torch.cat([v_time, v_spatial], dim=-1)

        """Zero-drift Gram-Schmidt projection ensuring sub-epsilon orthogonality"""
        v_clean = project_to_tangent_space(
            x_base=x_f32,
            v=v_tangent,
            curvature=self.curvature,
            eps=self.eps,
        )
        return v_clean.to(target_dtype)

    def descent_to_euclidean(
        self,
        x_base: torch.Tensor,
        v_tangent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Isometric Cartan Descent Operator:
            Descent_x: T_x H^D_kappa in R^(1+D) -> R^D.
        Inverse projection converting tangent vectors back into flat Euclidean representations:
            c = v_spatial - (sqrt(kappa) * v_0 / (1 + sqrt(kappa) * x_0)) * x_spatial.
        Guarantees exact roundtrip bijectivity:
            Descent_x(Lift_x(c)) == c.
        """
        target_dtype = v_tangent.dtype
        x_f32 = x_base.to(torch.float32)
        v_f32 = v_tangent.to(torch.float32)

        x_time, x_spatial = decompose_minkowski_coordinates(x_f32)
        v_time, v_spatial = decompose_minkowski_coordinates(v_f32)

        denominator = 1.0 + (self.sqrt_curvature * x_time)
        safe_denom = torch.clamp_min(denominator, self.eps)

        factor = (self.sqrt_curvature * v_time) / safe_denom
        c_euclidean = v_spatial - (factor * x_spatial)

        return c_euclidean.to(target_dtype)

    def verify_orthogonality(
        self,
        x_base: torch.Tensor,
        c_euclidean: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluates absolute Minkowski inner product error: |<x, Lift_x(c)>_L|.
        Theoretical invariant: strictly 0.000000.
        """
        v = self.lift_to_tangent(x_base, c_euclidean)
        dots = lorentz_inner_product(x_base, v, keepdim=False)
        return torch.abs(dots)

    def verify_isometry(
        self,
        x_base: torch.Tensor,
        c_euclidean: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Verifies metric isometry by comparing Euclidean norm squared against Minkowski norm squared:
            euc_norm_sq = ||c||_2^2
            mink_norm_sq = ||v||_L^2 = - v_0^2 + ||v_spatial||_2^2.
        Theoretical invariant: euc_norm_sq == mink_norm_sq.
        """
        v = self.lift_to_tangent(x_base, c_euclidean)

        euc_norm_sq = torch.sum(c_euclidean.to(torch.float32).pow(2), dim=-1)
        mink_norm_sq = lorentz_norm_squared(v, keepdim=False)

        return euc_norm_sq, mink_norm_sq

    def forward(
        self,
        x_base: torch.Tensor,
        c_euclidean: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience forward execution mapping flat Euclidean features into the tangent bundle"""
        return self.lift_to_tangent(x_base, c_euclidean)


def verify_cartan_vielbein_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for Unit 8 of Layer 1.
    Validates:
      1. Strict Minkowski orthogonality <x, Lift_x(c)>_L == 0.0 across random manifold points.
      2. Exact isometry: Euclidean norm squared equals Minkowski pseudo-norm squared.
      3. Complete roundtrip bijectivity: Descent_x(Lift_x(c)) == c.
      4. Canonical pole behavior: At origin o = [1/sqrt(kappa), 0..0], v_0 == 0.0 and v_spatial == c.
      5. Full autograd backpropagation through lifting and descent operations.
    """
    torch.manual_seed(42)
    dim_space = 32
    curvature = 1.0
    num_samples = 16

    frame = CartanVielbeinFrame(curvature=curvature)

    """1. Generate points on the upper sheet of H^32_kappa"""
    raw_x = torch.randn(num_samples, dim_space + 1, dtype=torch.float32)
    x_manifold = project_to_hyperboloid(raw_x, curvature=curvature)

    c_initial = torch.randn(num_samples, dim_space, dtype=torch.float32, requires_grad=True)

    """2. Strict Orthogonality Invariant: <x, Lift_x(c)>_L == 0.0"""
    ortho_errors = frame.verify_orthogonality(x_manifold, c_initial)
    max_ortho_err = float(torch.max(ortho_errors).item())
    if max_ortho_err > 1e-5:
        return False

    """3. Metric Isometry Invariant: ||Lift_x(c)||_L^2 == ||c||_2^2"""
    euc_norm_sq, mink_norm_sq = frame.verify_isometry(x_manifold, c_initial)
    max_isometry_diff = float(torch.max(torch.abs(euc_norm_sq - mink_norm_sq)).item())
    if max_isometry_diff > 1e-4:
        return False

    """4. Roundtrip Bijectivity Invariant: c == Descent_x(Lift_x(c))"""
    v_lifted = frame.lift_to_tangent(x_manifold, c_initial)
    c_recovered = frame.descent_to_euclidean(x_manifold, v_lifted)

    max_roundtrip_err = float(torch.max(torch.abs(c_initial - c_recovered)).item())
    if max_roundtrip_err > 1e-5:
        return False

    """5. Specific Verification at the Origin Pole o"""
    o_pole = torch.zeros(1, dim_space + 1, dtype=torch.float32)
    o_pole[0, 0] = 1.0 / math.sqrt(curvature)

    c_test = torch.randn(1, dim_space, dtype=torch.float32)
    v_origin = frame.lift_to_tangent(o_pole, c_test)

    """At origin, timelike component v_0 must be strictly 0.0 and v_spatial == c"""
    if abs(float(v_origin[0, 0].item())) > 1e-7:
        return False

    origin_spatial_diff = float(torch.max(torch.abs(v_origin[0, 1:] - c_test[0])).item())
    if origin_spatial_diff > 1e-6:
        return False

    """6. Autograd Gradient Propagation Test"""
    loss = torch.sum(v_lifted.pow(2)) + torch.sum(c_recovered.pow(2))
    loss.backward()

    if c_initial.grad is None or torch.isnan(c_initial.grad).any():
        return False
    if float(torch.norm(c_initial.grad).item()) <= 0.0:
        return False

    return True