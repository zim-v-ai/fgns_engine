"""
Location: // fgns_engine/fgns/primitives/wavelet_kan.py
Description: SOTA High-Level Factored Morlet Wavelet KAN Layer Facade.
             Encapsulates parameter management, variance-preserving initialization,
             and dual-highway residual routing while delegating low-level GPU compute
             directly to the Native CUDA C++17 kernel in fgns.kernels.cuda_fgns_ops.
Architecture: FGNS v4.0.0 Finsler-Gauge Sheaf Neural Foundation.
Language: Python 3.12+ / PyTorch 2.5+ / Native CUDA C++17 sm_120 & sm_89 Optimized
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fgns.kernels.cuda_fgns_ops import cuda_morlet_kan_op


class FactoredMorletKANLinear(nn.Module):
    """
    SOTA Kolmogorov-Arnold Network Layer on Factored Continuous Morlet Wavelets.
    Combines a linear residual highway with memory-recomputing non-linear wavelets.
    Guarantees O(1) memory graph overhead during recurrent multi-step reasoning
    by executing evaluation through Native CUDA C++17 GPU compute shaders.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_wavelets: int = 4,
        omega_0: float = 5.0,
        sigma_min: float = 1e-4,
        residual_gate_init: float = 0.25,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.num_wavelets = num_wavelets
        self.omega_0 = omega_0
        self.sigma_min = sigma_min

        """1. Base linear highway preserving identity and quasi-linear gradients"""
        self.w_base = nn.Linear(d_in, d_out, bias=False)

        """2. Non-linear wavelet projection mapping mixture manifold into d_out"""
        self.w_wavelet_proj = nn.Linear(d_in, d_out, bias=True)

        """3. Learnable continuous Morlet wavelet functional parameters"""
        self.centers = nn.Parameter(torch.empty(d_in, num_wavelets))
        self.raw_scales = nn.Parameter(torch.empty(d_in, num_wavelets))
        self.channel_weights = nn.Parameter(torch.empty(d_in, num_wavelets))
        self.phases = nn.Parameter(torch.empty(d_in, num_wavelets))

        """4. Adaptive gate controlling non-linear topological curvature"""
        self.wavelet_gate = nn.Parameter(torch.tensor(residual_gate_init, dtype=torch.float32))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """
        Initializes Morlet wavelets across spectral frequency bands and spatial centers.
        Linear projections are initialized via variance-preserving truncated normal distributions.
        """
        std_base = 1.0 / math.sqrt(self.d_in)
        nn.init.trunc_normal_(
            self.w_base.weight,
            mean=0.0,
            std=std_base,
            a=-2.0 * std_base,
            b=2.0 * std_base,
        )
        nn.init.trunc_normal_(
            self.w_wavelet_proj.weight,
            mean=0.0,
            std=std_base,
            a=-2.0 * std_base,
            b=2.0 * std_base,
        )
        nn.init.zeros_(self.w_wavelet_proj.bias)

        """Evenly space wavelet centers across the active canonical manifold [-1.5, +1.5]"""
        grid_centers = torch.linspace(-1.5, 1.5, self.num_wavelets).unsqueeze(0).expand(self.d_in, self.num_wavelets)
        self.centers.data.copy_(grid_centers)

        """Initialize scales to cover adjacent wavelet bandwidths without spectral dead zones"""
        init_bandwidth = 3.0 / float(max(1, self.num_wavelets - 1))
        self.raw_scales.data.fill_(math.log(math.exp(init_bandwidth) - 1.0))

        """Initialize mixture weights to uniform unity and phases to zero"""
        self.channel_weights.data.fill_(1.0 / float(self.num_wavelets))
        self.phases.data.zero_()

    def get_effective_scales(self) -> torch.Tensor:
        """
        Evaluates strictly positive non-zero wavelet bandwidths:
        sigma = softplus(raw_scales) + sigma_min.
        Guarantees zero-division safety during coordinate scaling.
        """
        return F.softplus(self.raw_scales) + self.sigma_min

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes rank-polymorphic KAN transformation via Native CUDA C++17 shader.
        Args:
            x: Input tensor of shape [..., D_in]
        Returns:
            Output tensor of shape [..., D_out]
        """
        orig_shape = x.shape
        d_in = orig_shape[-1]
        if d_in != self.d_in:
            raise ValueError(f"Channel dimension mismatch: expected {self.d_in}, got {d_in}")

        batch_dims = orig_shape[:-1]
        total_tokens = 1
        for dim_size in batch_dims:
            total_tokens *= dim_size

        """Flatten arbitrary leading batch dimensions into continuous 2D matrix of tokens"""
        x_flat = x.reshape(total_tokens, d_in)

        """1. Evaluate linear highway branch in parallel"""
        y_base = self.w_base(x_flat)

        """2. Execute memory-recomputing Morlet KAN operator via Native CUDA C++ shader"""
        effective_scales = self.get_effective_scales()

        kan_features_flat = cuda_morlet_kan_op(
            x=x_flat,
            centers=self.centers,
            scales=effective_scales,
            weights=self.channel_weights,
            phases=self.phases,
            omega_0=self.omega_0,
        )

        """3. Project non-linear wave mixtures into target output manifold"""
        y_wavelet = self.w_wavelet_proj(kan_features_flat)

        """4. Dual-Highway residual combination with adaptive gating"""
        y_fused_flat = y_base + (y_wavelet * self.wavelet_gate)

        """Restore original polymorphic tensor rank [..., D_out]"""
        target_shape = list(batch_dims) + [self.d_out]
        return y_fused_flat.view(*target_shape)


def verify_wavelet_kan_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for wavelet_kan.py.
    """
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = FactoredMorletKANLinear(d_in=32, d_out=32, num_wavelets=4).to(device)
    x = torch.randn((2, 6, 6, 32), dtype=torch.float32, device=device, requires_grad=True)
    y = layer(x)
    if y.shape != (2, 6, 6, 32) or torch.isnan(y).any():
        return False
    loss = torch.sum(y.pow(2))
    loss.backward()
    if x.grad is None or layer.centers.grad is None or torch.isnan(x.grad).any():
        return False
    return True