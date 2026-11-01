"""
Location: // fgns_engine/fgns/models/octo_ssm.py
Description: SOTA 8-Axis Isotropic OctoWave SSM Deep Field Processor with Vectorized Hodge
             Boundary Normal Guidance and Native CUDA C++ Kernel Integration.
Language: Python 3.12+ / PyTorch 2.5+ / High-Throughput Isotropic Wavefront Backbone sm_120 Aligned
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import math
from typing import Final, List, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from fgns.config import FGNSConfig
from fgns.kernels.cuda_fgns_ops import OctoWaveSSMKernel
from fgns.primitives.swiglu import RMSNorm, TopologicalBypassFeedForward

DEFAULT_STENCIL_EPS: Final[float] = 1e-7
DEFAULT_RESIDUAL_SCALE: Final[float] = 0.20
DEFAULT_SWIGLU_EXPANSION: Final[float] = 2.6666666666666665


class BoundaryNormalGuidanceStencil(nn.Module):
    """
    SOTA Vectorized Spatial Boundary Normal and Hodge Dual Tangent Field Extractor.
    Evaluates all 8 Moore tangent alignments in a single fused tensor contraction without loops.
    """

    def __init__(self, eps: float = DEFAULT_STENCIL_EPS) -> None:
        super().__init__()
        self.eps = eps

        moore_tangent_weights = torch.tensor(
            [
                [-1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
                [-0.70710678, 0.70710678],
                [0.70710678, -0.70710678],
                [0.70710678, 0.70710678],
                [-0.70710678, -0.70710678],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("tangent_basis", moore_tangent_weights, persistent=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_f32 = mask.to(x.dtype).unsqueeze(-1)

        y_pad = F.pad(x, (0, 0, 0, 0, 1, 1), mode="replicate")
        grad_y = (y_pad[:, 2:, :, :] - y_pad[:, :-2, :, :]) * 0.50

        x_pad = F.pad(x, (0, 0, 1, 1, 0, 0), mode="replicate")
        grad_x = (x_pad[:, :, 2:, :] - x_pad[:, :, :-2, :]) * 0.50

        energy_y = torch.mean(grad_y * grad_y, dim=-1, keepdim=True)
        energy_x = torch.mean(grad_x * grad_x, dim=-1, keepdim=True)
        edge_energy = (energy_y + energy_x) * mask_f32

        norm_grad_y = F.normalize(grad_y, p=2, dim=-1, eps=self.eps)
        norm_grad_x = F.normalize(grad_x, p=2, dim=-1, eps=self.eps)

        t_basis = cast(torch.Tensor, self.tangent_basis).to(dtype=x.dtype, device=x.device)
        wy = t_basis[:, 0].view(1, 1, 1, 1, 8)
        wx = t_basis[:, 1].view(1, 1, 1, 1, 8)

        proj_all = (norm_grad_y.unsqueeze(-1) * wy + norm_grad_x.unsqueeze(-1) * wx).abs()
        directional_alignment = proj_all.mean(dim=-2) * mask_f32

        return directional_alignment, edge_energy


class OctoWaveBlock(nn.Module):
    """
    Fundamental Deep Mamba-style Field Reasoning Block.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_axes: int = 8,
        normal_router_weight: float = 0.35,
        expansion_factor: float = DEFAULT_SWIGLU_EXPANSION,
        eps: float = DEFAULT_STENCIL_EPS,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_axes = num_axes
        self.eps = eps

        self.norm_spatial = RMSNorm(d_model)
        self.stencil = BoundaryNormalGuidanceStencil(eps=eps)
        self.normal_router = nn.Linear(num_axes + 1, num_axes, bias=True)

        self.octo_kernel = OctoWaveSSMKernel(
            d_model=d_model,
            num_axes=num_axes,
            normal_router_weight=normal_router_weight,
        )

        self.norm_channel = RMSNorm(d_model)
        self.ffn = TopologicalBypassFeedForward(
            d_model=d_model,
            expansion_factor=expansion_factor,
            alignment_multiple=32,
        )

        self.scale_spatial = nn.Parameter(torch.tensor(DEFAULT_RESIDUAL_SCALE, dtype=torch.float32))
        self.scale_channel = nn.Parameter(torch.tensor(DEFAULT_RESIDUAL_SCALE, dtype=torch.float32))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.normal_router.weight)
        nn.init.zeros_(self.normal_router.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f32 = mask.to(x.dtype).unsqueeze(-1)

        directional_alignment, edge_energy = self.stencil(x, mask)
        normal_features = torch.cat([directional_alignment, edge_energy], dim=-1)
        direction_bias = torch.tanh(self.normal_router(normal_features))

        normed_spatial = self.norm_spatial(x)
        wave_field = self.octo_kernel(normed_spatial, mask, normal_bias=direction_bias)

        x1 = x + (wave_field * self.scale_spatial) * mask_f32

        normed_channel = self.norm_channel(x1)
        ffn_out = self.ffn(normed_channel)

        x2 = (x1 + (ffn_out * self.scale_channel)) * mask_f32
        return x2


class OctoWaveSSM(nn.Module):
    """
    SOTA High-Level Multi-Layer Isotropic Field Processor.
    """

    def __init__(self, config: FGNSConfig, num_layers: int = 2) -> None:
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        d_model = config.octowave.d_model
        num_axes = config.octowave.num_axes
        normal_router_weight = config.octowave.normal_router_weight

        self.blocks = nn.ModuleList([
            OctoWaveBlock(
                d_model=d_model,
                num_axes=num_axes,
                normal_router_weight=normal_router_weight,
            )
            for _ in range(num_layers)
        ])

        self.norm_final = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        H, W = orig_shape[-3], orig_shape[-2]
        d_model = orig_shape[-1]

        x_flat = x.reshape(-1, H, W, d_model)
        mask_flat = mask.reshape(-1, H, W)
        mask_f32 = mask_flat.to(x.dtype).unsqueeze(-1)

        h_current = x_flat
        for block in self.blocks:
            h_current = block(h_current, mask_flat)

        h_out_flat = self.norm_final(h_current) * mask_f32
        return h_out_flat.view(*orig_shape)


def verify_octo_ssm_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for octo_ssm.py.
    """
    torch.manual_seed(42)
    b, k, h, w, d = 2, 3, 4, 4, 32

    config = FGNSConfig()
    ssm = OctoWaveSSM(config=config, num_layers=2)

    ssm.blocks = nn.ModuleList([
        OctoWaveBlock(d_model=d, num_axes=8, normal_router_weight=0.35)
        for _ in range(2)
    ])
    ssm.norm_final = RMSNorm(d)

    x_4d = torch.randn(b, h, w, d, dtype=torch.float32, requires_grad=True)
    mask_4d = torch.ones(b, h, w, dtype=torch.bool)
    mask_4d[:, :, 2:] = False

    out_4d = ssm(x_4d, mask_4d)
    if out_4d.shape != (b, h, w, d) or not torch.all(out_4d[:, :, 2:, :] == 0.0):
        return False

    x_5d = torch.randn(b, k, h, w, d, dtype=torch.float32, requires_grad=True)
    mask_5d = torch.ones(b, k, h, w, dtype=torch.bool)
    mask_5d[:, :, :, 2:] = False

    out_5d = ssm(x_5d, mask_5d)
    if out_5d.shape != (b, k, h, w, d) or not torch.all(out_5d[:, :, :, 2:, :] == 0.0):
        return False

    loss = torch.sum(out_4d.pow(2))
    loss.backward()

    first_block = cast(OctoWaveBlock, ssm.blocks[0])
    router = cast(nn.Linear, first_block.normal_router)
    router_weight = cast(torch.Tensor, router.weight)
    if router_weight.grad is None or torch.isnan(router_weight.grad).any():
        return False

    return True