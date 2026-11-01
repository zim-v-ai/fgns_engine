"""
Location: // fgns_engine/fgns/primitives/lorentz_manifold.py
Description: SOTA Pseudo-Riemannian Differential Geometry Engine on the Lorentz Hyperboloid H^D_kappa.
             Core Mathematical Foundation for FGNS v0.6.5 (Dual-State Cauchy Flow):
               1. Complete pseudo-Riemannian Minkowski bilinear form and coordinate decomposition.
               2. Upper sheet hyperboloid of constant negative sectional curvature -kappa in R^(1+D).
               3. Chordal arsinh geodesic distance with two-stream safe square root (no sqrt(0) NaNs).
               4. Taylor-stabilized exponential and logarithmic mappings preventing 0/0 singularities.
               5. Levi-Civita parallel transport preserving Minkowski pseudo-orthogonality.
               6. Fast GEMM-accelerated pairwise geodesic distance matrix evaluation.
               7. Full export of decompose_minkowski_coordinates and lorentz_norm_squared contracts.
Language: Python 3.12+ / PyTorch 2.5+ / Differential Non-Euclidean Geometry sm_120 Aligned
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import math
from typing import Final, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fgns.primitives.safe_math import safe_hyperbolic_sqrt

DEFAULT_MANIFOLD_EPS: Final[float] = 1e-7
TAYLOR_THRESHOLD: Final[float] = 1e-5


def decompose_minkowski_coordinates(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decomposes tensor x in R^(1+D) into:
      x_time:    Scalar timelike coordinate x_0 of shape [..., 1]
      x_spatial: Vector spacelike coordinates x_{1:D} of shape [..., D]
    Guarantees zero-copy memory views without tensor reallocations.

    Args:
        x: Input Minkowski tensor of shape [..., 1+D].

    Returns:
        Tuple of (x_time, x_spatial).
    """
    x_time = x[..., 0:1]
    x_spatial = x[..., 1:]
    return x_time, x_spatial


def lorentz_inner_product(
    u: torch.Tensor,
    v: torch.Tensor,
    keepdim: bool = True,
    dim: int = -1,
) -> torch.Tensor:
    """
    Evaluates the pseudo-Riemannian Minkowski bilinear form on R^(1+D):
        <u, v>_L = - u_0 * v_0 + sum_{i=1}^D u_i * v_i.
    Supports arbitrary leading polymorphic batch dimensions [..., 1+D].

    Args:
        u: First state tensor [..., 1+D].
        v: Second state tensor [..., 1+D].
        keepdim: Retain the singleton trailing dimension.
        dim: Axis along which reduction is performed.

    Returns:
        Scalar Minkowski inner product tensor [...] or [..., 1].
    """
    target_dtype = u.dtype
    u_f32 = u.to(torch.float32)
    v_f32 = v.to(torch.float32)

    time_prod = u_f32[..., 0:1] * v_f32[..., 0:1]
    space_prod = torch.sum(u_f32[..., 1:] * v_f32[..., 1:], dim=dim, keepdim=True)

    inner_prod = -time_prod + space_prod

    if not keepdim:
        inner_prod = inner_prod.squeeze(dim)

    return inner_prod.to(target_dtype)


def lorentz_norm_squared(
    u: torch.Tensor,
    keepdim: bool = True,
    dim: int = -1,
) -> torch.Tensor:
    """
    Calculates the Minkowski squared pseudo-norm:
        ||u||_L^2 = <u, u>_L = - u_0^2 + ||u_spatial||_2^2.

    Args:
        u: State tensor [..., 1+D].
        keepdim: Retain the singleton trailing dimension.
        dim: Axis along which reduction is performed.

    Returns:
        Squared Minkowski pseudo-norm tensor [...] or [..., 1].
    """
    return lorentz_inner_product(u, u, keepdim=keepdim, dim=dim)


def project_to_hyperboloid(
    x: torch.Tensor,
    curvature: float = 1.00,
    eps: float = DEFAULT_MANIFOLD_EPS,
) -> torch.Tensor:
    """
    Analytically projects an arbitrary tensor x in R^(1+D) onto the upper sheet of H^D_kappa.
    Guarantees <x, x>_L == -1/kappa and x_0 >= 1/sqrt(kappa) > 0.
    Formula:
        x_0 = sqrt(1/kappa + ||x_spatial||_2^2).

    Args:
        x: Input tensor in R^(1+D).
        curvature: Sectional curvature parameter kappa.
        eps: Lower bound preventing division by zero.

    Returns:
        Point strictly residing on the upper sheet of H^D_kappa.
    """
    target_dtype = x.dtype
    x_f32 = x.to(torch.float32)

    safe_k = max(float(curvature), eps)
    inv_k = 1.0 / safe_k

    x_spatial = x_f32[..., 1:]
    spatial_norm_sq = torch.sum(x_spatial * x_spatial, dim=-1, keepdim=True)

    time_sq = inv_k + spatial_norm_sq
    x_time = torch.sqrt(torch.clamp_min(time_sq, inv_k))

    projected = torch.cat([x_time, x_spatial], dim=-1)
    return projected.to(target_dtype)


def project_to_tangent_space(
    x_base: torch.Tensor,
    v: torch.Tensor,
    curvature: float = 1.00,
    eps: float = DEFAULT_MANIFOLD_EPS,
) -> torch.Tensor:
    """
    Executes Minkowski Gram-Schmidt projection of vector v in R^(1+D) onto tangent space T_x H^D_kappa:
        v_tangent = v + kappa * <x_base, v>_L * x_base.
    Guarantees exact orthogonality <x_base, v_tangent>_L == 0.0 without numerical drift.

    Args:
        x_base: Base point residing on H^D_kappa [..., 1+D].
        v: Arbitrary velocity or force vector in R^(1+D).
        curvature: Sectional curvature parameter kappa.
        eps: Numerical stabilization constant.

    Returns:
        Orthogonal tangent vector residing strictly in T_x H^D_kappa.
    """
    target_dtype = v.dtype
    x_f32 = x_base.to(torch.float32)
    v_f32 = v.to(torch.float32)

    safe_k = max(float(curvature), eps)

    inner = lorentz_inner_product(x_f32, v_f32, keepdim=True)
    correction = (safe_k * inner) * x_f32

    v_tangent = v_f32 + correction
    return v_tangent.to(target_dtype)


def pairwise_minkowski_dot(
    u_mat: torch.Tensor,
    v_mat: torch.Tensor,
) -> torch.Tensor:
    """
    High-throughput pairwise Lorentz inner product matrix evaluation via GEMM:
        M_ij = <u_i, v_j>_L = - u_{i,0} * v_{j,0} + sum_{d=1}^D u_{i,d} * v_{j,d}.

    Args:
        u_mat: Tensor of M vectors [M, 1+D].
        v_mat: Tensor of N vectors [N, 1+D].

    Returns:
        Pairwise inner product matrix of shape [M, N].
    """
    target_dtype = u_mat.dtype
    u_f32 = u_mat.to(torch.float32)
    v_f32 = v_mat.to(torch.float32)

    v_tilde = v_f32.clone()
    v_tilde[:, 0] = -v_tilde[:, 0]

    inner_matrix = torch.matmul(u_f32, v_tilde.t())
    return inner_matrix.to(target_dtype)


class LorentzManifold(nn.Module):
    """
    SOTA Riemannian Lorentz Hyperboloid Manifold Operator.
    Models the upper sheet of H^D_kappa of constant negative sectional curvature -kappa.
    Embeds hierarchical trees of ARC rules and object containment graphs without distortion.
    """

    def __init__(
        self,
        curvature: float = 1.00,
        eps: float = DEFAULT_MANIFOLD_EPS,
    ) -> None:
        super().__init__()
        self.curvature = max(float(curvature), eps)
        self.eps = eps

        self.sqrt_curvature = math.sqrt(self.curvature)
        self.inv_curvature = 1.0 / self.curvature
        self.inv_sqrt_curvature = 1.0 / self.sqrt_curvature

    def origin(
        self,
        dim: int,
        device: Optional[Union[torch.device, str]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Constructs canonical identity origin pole on the upper sheet of H^D_kappa:
            o = [ 1 / sqrt(kappa), 0, 0, ..., 0 ] in R^(1+D).
        """
        o = torch.zeros(dim + 1, dtype=dtype, device=device)
        o[0] = self.inv_sqrt_curvature
        return o

    def inner_product(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        keepdim: bool = True,
        dim: int = -1,
    ) -> torch.Tensor:
        """Evaluates pseudo-Riemannian Minkowski bilinear form <u, v>_L."""
        return lorentz_inner_product(u, v, keepdim=keepdim, dim=dim)

    def norm_squared(
        self,
        u: torch.Tensor,
        keepdim: bool = True,
        dim: int = -1,
    ) -> torch.Tensor:
        """Calculates Minkowski squared pseudo-norm: ||u||_L^2 = <u, u>_L."""
        return lorentz_norm_squared(u, keepdim=keepdim, dim=dim)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Analytically projects tensor x onto the upper sheet of H^D_kappa."""
        return project_to_hyperboloid(x, curvature=self.curvature, eps=self.eps)

    def project_tangent(self, x_base: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Projects vector v onto tangent space T_x H^D_kappa."""
        return project_to_tangent_space(x_base, v, curvature=self.curvature, eps=self.eps)

    def geodesic_distance(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        keepdim: bool = False,
    ) -> torch.Tensor:
        """
        Evaluates exact Riemannian geodesic distance d_H(x, y) on H^D_kappa.
        Chordal arsinh formulation:
            Delta_L^2 = max(0, <x - y, x - y>_L)
            d_H(x, y) = (2 / sqrt(kappa)) * arsinh( 0.5 * sqrt(kappa) * safe_sqrt(Delta_L^2) ).
        Guarantees d_H(x, x) == 0.0 in forward and zero-gradient (no NaN) in backward pass.
        """
        target_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        y_f32 = y.to(torch.float32)

        diff = x_f32 - y_f32
        minkowski_sq = lorentz_inner_product(diff, diff, keepdim=keepdim)

        clamped_sq = torch.clamp_min(minkowski_sq, 0.0)
        norm_delta = safe_hyperbolic_sqrt(clamped_sq, eps=self.eps)

        half_arg = 0.50 * self.sqrt_curvature * norm_delta
        distance = (2.0 * self.inv_sqrt_curvature) * torch.asinh(half_arg)

        return distance.to(target_dtype)

    def distance_matrix(
        self,
        queries: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """
        High-throughput pairwise geodesic distance matrix between M queries and N candidates.
        Guarantees zero-diagonal identity without NaN gradients under backpropagation.
        """
        target_dtype = queries.dtype
        q_f32 = queries.to(torch.float32)
        c_f32 = candidates.to(torch.float32)

        q_0 = q_f32[:, 0:1]
        c_0 = c_f32[:, 0:1]
        diff_0_sq = (q_0 - c_0.t()).pow(2)

        q_sp = q_f32[:, 1:]
        c_sp = c_f32[:, 1:]

        q_sp_norm_sq = torch.sum(q_sp * q_sp, dim=-1, keepdim=True)
        c_sp_norm_sq = torch.sum(c_sp * c_sp, dim=-1, keepdim=True)
        dot_sp = torch.matmul(q_sp, c_sp.t())

        diff_sp_sq = q_sp_norm_sq - (2.0 * dot_sp) + c_sp_norm_sq.t()

        minkowski_diff_sq = -diff_0_sq + diff_sp_sq
        clamped_diff = torch.clamp_min(minkowski_diff_sq, 0.0)

        norm_delta = safe_hyperbolic_sqrt(clamped_diff, eps=self.eps)
        half_arg = 0.50 * self.sqrt_curvature * norm_delta

        distance_mat = (2.0 * self.inv_sqrt_curvature) * torch.asinh(half_arg)
        return distance_mat.to(target_dtype)

    def exp_map(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Riemannian exponential map: exp_x(v): T_x H^D_kappa -> H^D_kappa.
        Projects tangent vector v along geodesic flow emanating from base point x.
        Incorporates Taylor expansion safeguard for small velocities:
            exp_x(v) = cosh(theta) * x + (sinh(theta) / theta) * v.
        """
        target_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        v_f32 = v.to(torch.float32)

        v_tan = project_to_tangent_space(x_f32, v_f32, curvature=self.curvature, eps=self.eps)

        v_norm_sq = torch.clamp_min(lorentz_inner_product(v_tan, v_tan, keepdim=True), 0.0)
        v_norm = safe_hyperbolic_sqrt(v_norm_sq, eps=self.eps)

        theta = self.sqrt_curvature * v_norm
        use_taylor = theta < TAYLOR_THRESHOLD

        cosh_val = torch.where(use_taylor, 1.0 + 0.50 * (theta * theta), torch.cosh(theta))
        inv_theta_sinh = torch.where(
            use_taylor,
            1.0 + (theta * theta) / 6.0,
            torch.sinh(theta) / torch.clamp_min(theta, self.eps),
        )

        res_point = (cosh_val * x_f32) + (inv_theta_sinh * v_tan)
        projected_res = project_to_hyperboloid(res_point, curvature=self.curvature, eps=self.eps)

        return projected_res.to(target_dtype)

    def log_map(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Riemannian logarithmic map: log_x(y): H^D_kappa -> T_x H^D_kappa.
        Computes initial velocity vector of the geodesic trajectory connecting x to y:
            log_x(y) = (sqrt(kappa) * d / sinh(sqrt(kappa) * d)) * (y + kappa * <x, y>_L * x).
        """
        target_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        y_f32 = y.to(torch.float32)

        dist = self.geodesic_distance(x_f32, y_f32, keepdim=True)
        theta = self.sqrt_curvature * dist

        use_taylor = theta < TAYLOR_THRESHOLD

        inv_sinh_factor = torch.where(
            use_taylor,
            1.0 - (theta * theta) / 6.0,
            theta / torch.clamp_min(torch.sinh(theta), self.eps),
        )

        inner = lorentz_inner_product(x_f32, y_f32, keepdim=True)
        unscaled_tangent = y_f32 + (self.curvature * inner) * x_f32

        v_tangent = inv_sinh_factor * unscaled_tangent
        v_clean = project_to_tangent_space(x_f32, v_tangent, curvature=self.curvature, eps=self.eps)

        return v_clean.to(target_dtype)

    def exp_map_origin(
        self,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """
        Direct Canonical Origin Exponential Mapping:
        Projects unconstrained Euclidean vector u in R^D onto H^D_kappa in R^(1+D):
            x_0 = (1 / sqrt(kappa)) * cosh(sqrt(kappa) * ||u||_2)
            x_{1:D} = u * (sinh(sqrt(kappa) * ||u||_2) / (sqrt(kappa) * ||u||_2)).
        """
        target_dtype = u.dtype
        u_f32 = u.to(torch.float32)

        u_norm_sq = torch.sum(u_f32 * u_f32, dim=-1, keepdim=True)
        clamped_u_sq = torch.clamp_min(u_norm_sq, 0.0)
        u_norm = safe_hyperbolic_sqrt(clamped_u_sq, eps=self.eps)

        theta = self.sqrt_curvature * u_norm
        use_taylor = theta < TAYLOR_THRESHOLD

        cosh_val = torch.where(use_taylor, 1.0 + 0.50 * (theta * theta), torch.cosh(theta))
        x_0 = self.inv_sqrt_curvature * cosh_val

        inv_theta_sinh = torch.where(
            use_taylor,
            1.0 + (theta * theta) / 6.0,
            torch.sinh(theta) / torch.clamp_min(theta, self.eps),
        )
        x_spatial = inv_theta_sinh * u_f32

        point_h = torch.cat([x_0, x_spatial], dim=-1)
        projected = project_to_hyperboloid(point_h, curvature=self.curvature, eps=self.eps)

        return projected.to(target_dtype)

    def log_map_origin(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inverse Canonical Origin Logarithmic Mapping:
        Projects point x on H^D_kappa back to Euclidean tangent coordinates u in R^D:
            u = x_{1:D} * (sqrt(kappa) * d / sinh(sqrt(kappa) * d)).
        """
        target_dtype = x.dtype
        x_f32 = x.to(torch.float32)

        x_time, x_spatial = decompose_minkowski_coordinates(x_f32)

        origin_pt = torch.zeros_like(x_f32)
        origin_pt[..., 0] = self.inv_sqrt_curvature

        dist = self.geodesic_distance(origin_pt, x_f32, keepdim=True)
        theta = self.sqrt_curvature * dist

        use_taylor = theta < TAYLOR_THRESHOLD

        inv_sinh_factor = torch.where(
            use_taylor,
            1.0 - (theta * theta) / 6.0,
            theta / torch.clamp_min(torch.sinh(theta), self.eps),
        )

        u_euclidean = inv_sinh_factor * x_spatial
        return u_euclidean.to(target_dtype)

    def parallel_transport(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Levi-Civita Parallel Transport along geodesic trajectory:
            P_{x -> y}(v): T_x H^D_kappa -> T_y H^D_kappa.
        Reflection formulation strictly preserving Minkowski inner products:
            P_{x -> y}(v) = v - (kappa * <y, v>_L / (1 - kappa * <x, y>_L)) * (x + y).
        """
        target_dtype = v.dtype
        x_f32 = x.to(torch.float32)
        y_f32 = y.to(torch.float32)
        v_f32 = v.to(torch.float32)

        inner_y_v = lorentz_inner_product(y_f32, v_f32, keepdim=True)
        inner_x_y = lorentz_inner_product(x_f32, y_f32, keepdim=True)

        denominator = 1.0 - (self.curvature * inner_x_y)
        safe_denom = torch.clamp_min(denominator, self.eps)

        factor = (self.curvature * inner_y_v) / safe_denom
        transported = v_f32 - factor * (x_f32 + y_f32)

        v_clean = project_to_tangent_space(y_f32, transported, curvature=self.curvature, eps=self.eps)
        return v_clean.to(target_dtype)


def verify_lorentz_manifold_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for lorentz_manifold.py.
    Validates:
      1. Pseudo-Riemannian Metric Invariant: <x, x>_L == -1/kappa.
      2. Tangent space Gram-Schmidt orthogonality: <x, v_tan>_L == 0.0.
      3. Geodesic distance identity: d_H(x, x) == 0.0 without NaN autograd gradients.
      4. Roundtrip bijectivity: log_map_origin(exp_map_origin(u)) == u.
      5. Decomposition invariant: decompose_minkowski_coordinates splits cleanly.
    """
    torch.manual_seed(42)
    dim_space = 32
    curvature = 1.00
    num_points = 16

    manifold = LorentzManifold(curvature=curvature)

    """1. Metric Invariant Verification"""
    raw_points = torch.randn(num_points, dim_space + 1, dtype=torch.float32)
    points_h = manifold.project(raw_points)

    norms_sq = manifold.norm_squared(points_h, keepdim=False)
    target_norm = -1.00 / curvature
    max_norm_error = float(torch.max(torch.abs(norms_sq - target_norm)).item())
    if max_norm_error > 1e-4:
        return False

    """2. Coordinate Decomposition Verification"""
    x_t, x_sp = decompose_minkowski_coordinates(points_h)
    if x_t.shape != (num_points, 1) or x_sp.shape != (num_points, dim_space):
        return False

    """3. Tangent Space Orthogonality Verification"""
    raw_tangents = torch.randn(num_points, dim_space + 1, dtype=torch.float32)
    tangents_v = manifold.project_tangent(points_h, raw_tangents)

    ortho_dots = manifold.inner_product(points_h, tangents_v, keepdim=False)
    max_ortho_error = float(torch.max(torch.abs(ortho_dots)).item())
    if max_ortho_error > 1e-4:
        return False

    """4. Geodesic Distance Identity & Autograd Flow"""
    pt_grad = points_h[0:1].clone().requires_grad_(True)
    self_dist = manifold.geodesic_distance(pt_grad, pt_grad)
    loss = torch.sum(self_dist)
    loss.backward()

    if float(self_dist.item()) > 1e-5:
        return False
    if pt_grad.grad is None or torch.isnan(pt_grad.grad).any():
        return False

    """5. Origin Exponential & Logarithmic Roundtrip Test"""
    u_euclidean = torch.randn(num_points, dim_space, dtype=torch.float32) * 0.50
    pts_exp = manifold.exp_map_origin(u_euclidean)
    u_recovered = manifold.log_map_origin(pts_exp)

    roundtrip_diff = float(torch.max(torch.abs(u_euclidean - u_recovered)).item())
    if roundtrip_diff > 1e-4:
        return False

    return True


__all__ = [
    "decompose_minkowski_coordinates",
    "lorentz_inner_product",
    "lorentz_norm_squared",
    "project_to_hyperboloid",
    "project_to_tangent_space",
    "pairwise_minkowski_dot",
    "LorentzManifold",
    "verify_lorentz_manifold_diagnostics",
]