"""
Location: // fgns_engine/fgns/models/patch_projector.py
Description: SOTA Unified Spatial Patch Projector with Cartan Principal Fiber Modulation,
             S10-Equivariant 8D Color Structural Role Injection, and Zero-Redundancy Cached
             NA3D-Finsler-Lie v2.0 3D Simplex Cell Synthesis.
Language: Python 3.12+ / PyTorch 2.5+ / Gauge Field Theory sm_120 Aligned
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import math
from typing import Final, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fgns.config import FGNSConfig
from fgns.kernels.spectral_laplacian import (
    PoissonDirichletCavitySolver,
    SpectralGraphTopology,
)
from fgns.primitives.lie_coords import MultiscaleLieResonator
from fgns.primitives.potts_latching import OrthonormalPaletteFrame
from fgns.primitives.swiglu import RMSNorm, SwiGLU

DEFAULT_RESIDUAL_COLOR_GATE: Final[float] = 0.50
DEFAULT_BRANCH_GATE: Final[float] = 0.25
NA3D_CELL_STRIDE: Final[int] = 16


class TopologicalCoordinateEncoder(nn.Module):
    """
    Unified Spatial Field Projector & NA3D-v2 Simplex Synthesizer for FGNS v4.0.0.
    """

    def __init__(self, config: FGNSConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.octowave.d_model
        vocab_size = config.geometry.vocab_size

        """1. Orthonormal Grassmannian Palette Manifold on unit hypersphere S^(D-1)"""
        self.palette = OrthonormalPaletteFrame(vocab_size=vocab_size, d_model=d_model)

        """2. Differentiable Matrix-Free Hodge-Laplacian & Cached CUDA Finsler-Chordal Topology"""
        self.spectral_topology = SpectralGraphTopology(
            num_eigenmodes=config.topology.num_eigenmodes,
            chebyshev_order=config.topology.chebyshev_order,
            d_out=d_model,
        )

        """3. Asymmetric Kronecker-Factored Multiscale Lie Coordinate Resonator"""
        self.lie_resonator = MultiscaleLieResonator(
            d_model=d_model,
            scales=config.lie.supported_scale_factors,
            num_harmonics=config.lie.num_harmonic_phases,
            resonance_temperature=config.lie.resonance_temperature,
        )

        """4. S10-Equivariant 8D Color Structural Role Fiber Projector (Theorem 3)"""
        self.role_fiber_proj = nn.Linear(8, d_model, bias=False)
        self.role_gate = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

        """Branch normalizations enforcing metric preservation across manifolds"""
        self.norm_color = RMSNorm(d_model)
        self.norm_topo = RMSNorm(d_model)
        self.norm_coord = RMSNorm(d_model)

        """5. Cartan Principal Fiber-Modulation Gate"""
        self.fiber_modulator = SwiGLU(
            d_in=2 * d_model,
            d_out=2 * d_model,
            d_inner=4 * d_model,
            alignment_multiple=32,
        )

        """Learnable branch gates for dynamic manifold calibration"""
        self.gate_color_mod = nn.Parameter(torch.tensor(DEFAULT_RESIDUAL_COLOR_GATE, dtype=torch.float32))
        self.gate_topo = nn.Parameter(torch.tensor(DEFAULT_BRANCH_GATE, dtype=torch.float32))
        self.gate_coord = nn.Parameter(torch.tensor(DEFAULT_BRANCH_GATE, dtype=torch.float32))
        self.na3d_genesis_gate = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

        """Final unified field normalization"""
        self.norm_final = RMSNorm(d_model)

        nn.init.xavier_uniform_(self.role_fiber_proj.weight, gain=0.20)

    def _align_scale_ratios(
        self,
        scale_ratio: Optional[torch.Tensor],
        target_flat_batch: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if scale_ratio is None:
            return None

        ratio_flat = scale_ratio.float().view(-1)
        num_ratios = ratio_flat.numel()

        if num_ratios == 0:
            return None

        if num_ratios == target_flat_batch:
            return ratio_flat.to(device=device)

        if target_flat_batch % num_ratios == 0:
            reps = target_flat_batch // num_ratios
            return ratio_flat.repeat_interleave(reps).to(device=device)

        return ratio_flat[0].expand(target_flat_batch).to(device=device)

    def _synthesize_na3d_cell_alignment(
        self,
        color_field: torch.Tensor,
        topo_field: torch.Tensor,
        coord_field: torch.Tensor,
        grid_flat: torch.Tensor,
        mask_flat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Synthesizes structured 16-channel NA3D-v2 Finsler-Lie cells reusing cached CUDA chordal bundle in O(1).
        """
        B_flat, H, W, D = color_field.shape
        if D % NA3D_CELL_STRIDE != 0:
            return torch.zeros_like(color_field)

        num_cells = D // NA3D_CELL_STRIDE
        mask_5d = mask_flat.float().view(B_flat, H, W, 1, 1)

        cached = getattr(self.spectral_topology, "latest_chordal_bundle", None)
        if cached is not None and cached[0].shape[:3] == (B_flat, H, W):
            m_cav, h_span, w_span, _, dipole_yx, seed_f = cached
        else:
            (
                m_cav,
                h_span,
                w_span,
                _,
                dipole_yx,
                seed_f,
            ) = PoissonDirichletCavitySolver.compute_finsler_chordal_bundle(
                grid_flat, mask_flat, num_iterations=10
            )

        topo_cells = topo_field.to(torch.float32).view(B_flat, H, W, num_cells, NA3D_CELL_STRIDE)
        col_cells = color_field.to(torch.float32).view(B_flat, H, W, num_cells, NA3D_CELL_STRIDE)
        crd_cells = coord_field.to(torch.float32).view(B_flat, H, W, num_cells, NA3D_CELL_STRIDE)

        l_face = topo_cells[..., 0:3]
        c_core = col_cells[..., 3:6]
        r_face = crd_cells[..., 6:9]

        dipole_exp = dipole_yx.unsqueeze(3).expand(B_flat, H, W, num_cells, 2)
        cav_exp = m_cav.unsqueeze(3).expand(B_flat, H, W, num_cells, 1)
        raw_beta = torch.cat([dipole_exp, cav_exp], dim=-1) + (0.25 * topo_cells[..., 9:12])
        b_norm = torch.sqrt(torch.sum(raw_beta * raw_beta, dim=-1, keepdim=True) + 1e-7)
        beta_wind = (torch.tanh(b_norm) / b_norm) * raw_beta

        chiral_v = torch.tanh(
            torch.sum(l_face * torch.cross(c_core, r_face, dim=-1), dim=-1, keepdim=True)
        )
        log_span_h = (torch.log(h_span.clamp_min(1.0)) / 3.40).unsqueeze(3).expand(B_flat, H, W, num_cells, 1)
        log_span_w = (torch.log(w_span.clamp_min(1.0)) / 3.40).unsqueeze(3).expand(B_flat, H, W, num_cells, 1)
        seed_exp = seed_f.unsqueeze(3).expand(B_flat, H, W, num_cells, 1)

        meta_4 = torch.cat([chiral_v, log_span_h, log_span_w, seed_exp], dim=-1)

        na3d_cells = torch.cat([l_face, c_core, r_face, beta_wind, meta_4], dim=-1) * mask_5d
        return na3d_cells.view(B_flat, H, W, D).to(color_field.dtype)

    def forward(
        self,
        grid: torch.Tensor,
        mask: torch.Tensor,
        scale_ratio_h: Optional[torch.Tensor] = None,
        scale_ratio_w: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        orig_shape = grid.shape
        H, W = orig_shape[-2], orig_shape[-1]
        device = grid.device
        prototypes = self.palette.get_normalized_prototypes()
        d_model = prototypes.shape[-1]

        grid_flat = grid.reshape(-1, H, W)
        mask_flat = mask.reshape(-1, H, W)
        b_flat = grid_flat.shape[0]

        mask_f32 = mask_flat.float().unsqueeze(-1)

        """1. Extract Orthonormal Color Manifold Prototypes + S10-Equivariant 8D Structural Roles"""
        flat_tokens = grid_flat.view(-1).clamp(0, self.config.geometry.vocab_size - 1)
        raw_colors = F.embedding(flat_tokens, prototypes).view(b_flat, H, W, d_model)

        if d_model == self.role_fiber_proj.out_features:
            role_state = PoissonDirichletCavitySolver.extract_color_structural_roles(grid_flat, mask_flat)
            roles_padded = F.pad(role_state.role_vectors, (0, 0, 0, 1), mode="constant", value=0.0)
            b_idx = torch.arange(b_flat, device=device).view(b_flat, 1, 1).expand(b_flat, H, W)
            pixel_roles = roles_padded[b_idx, grid_flat.clamp(0, 10)]
            role_emb = self.role_fiber_proj(pixel_roles.to(raw_colors.dtype))
            raw_colors = raw_colors + (torch.tanh(self.role_gate.to(raw_colors.dtype)) * role_emb)

        color_field = self.norm_color(raw_colors) * mask_f32

        """2. Evaluate Matrix-Free Hodge-Laplacian & Cached CUDA Finsler-Chordal Topology"""
        topo_field_raw = self.spectral_topology(grid_flat, mask_flat)
        topo_field = self.norm_topo(topo_field_raw) * mask_f32

        """3. Evaluate Kronecker-Factored Multiscale Lie Coordinate Resonator"""
        r_h = self._align_scale_ratios(scale_ratio_h, b_flat, device)
        r_w = self._align_scale_ratios(scale_ratio_w, b_flat, device)

        coord_field_raw = self.lie_resonator(
            batch_size=b_flat,
            height=H,
            width=W,
            device=device,
            scale_ratio_h=r_h,
            scale_ratio_w=r_w,
        )
        coord_field = self.norm_coord(coord_field_raw) * mask_f32

        """4. Cartan Principal Fiber-Modulation"""
        geometric_joint = torch.cat([topo_field, coord_field], dim=-1)
        modulation_bundle = self.fiber_modulator(geometric_joint)
        gamma, beta = modulation_bundle.chunk(2, dim=-1)

        modulated_color = color_field * (1.0 + torch.tanh(gamma)) + beta

        """5. Canonical NA3D-v2 3D-Simplex Cell Genesis"""
        na3d_structured = self._synthesize_na3d_cell_alignment(
            color_field=modulated_color,
            topo_field=topo_field,
            coord_field=coord_field,
            grid_flat=grid_flat,
            mask_flat=mask_flat,
        )

        """6. Residual Unified Field Assembly with Calibrated Gating"""
        fused_field = (
            color_field
            + (self.gate_color_mod * modulated_color)
            + (self.gate_topo * topo_field)
            + (self.gate_coord * coord_field)
            + (torch.tanh(self.na3d_genesis_gate) * na3d_structured)
        )
        unified_field_flat = self.norm_final(fused_field) * mask_f32

        final_shape = list(orig_shape) + [d_model]
        unified_field = unified_field_flat.view(*final_shape)

        return unified_field


def verify_patch_projector_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for patch_projector.py.
    """
    torch.manual_seed(42)
    b_size, k_pairs, h, w, d_model = 2, 6, 6, 6, 32

    config = FGNSConfig()
    projector = TopologicalCoordinateEncoder(config=config)

    projector.palette.raw_prototypes = nn.Parameter(torch.randn(config.geometry.vocab_size, d_model))
    projector.spectral_topology.d_out = d_model
    projector.spectral_topology.swiglu_proj = SwiGLU(d_in=config.topology.num_eigenmodes + 3, d_out=d_model)
    projector.spectral_topology.chordal_modulator = nn.Linear(6, d_model, bias=False)
    projector.spectral_topology.norm = RMSNorm(d_model)
    projector.lie_resonator.d_model = d_model
    projector.lie_resonator.swiglu_compress = SwiGLU(d_in=projector.lie_resonator.total_raw_dim, d_out=d_model)
    projector.lie_resonator.norm = RMSNorm(d_model)
    projector.role_fiber_proj = nn.Linear(8, d_model, bias=False)
    projector.fiber_modulator = SwiGLU(d_in=2 * d_model, d_out=2 * d_model, d_inner=2 * d_model)

    projector.norm_color = RMSNorm(d_model)
    projector.norm_topo = RMSNorm(d_model)
    projector.norm_coord = RMSNorm(d_model)
    projector.norm_final = RMSNorm(d_model)

    q_grid = torch.randint(0, 10, (b_size, h, w), dtype=torch.int64)
    q_mask = torch.ones(b_size, h, w, dtype=torch.bool)
    q_mask[:, :, 3:] = False

    scale_h = torch.tensor([3.0, 1.0], dtype=torch.float32)
    scale_w = torch.tensor([3.0, 1.0], dtype=torch.float32)

    q_proj = projector(q_grid, q_mask, scale_ratio_h=scale_h, scale_ratio_w=scale_w)

    if q_proj.shape != (b_size, h, w, d_model):
        return False
    if torch.isnan(q_proj).any() or torch.isinf(q_proj).any():
        return False
    if not torch.all(q_proj[:, :, 3:, :] == 0.0):
        return False

    loss = torch.sum(q_proj.pow(2))
    loss.backward()

    if projector.palette.raw_prototypes.grad is None:
        return False
    if torch.isnan(projector.palette.raw_prototypes.grad).any():
        return False

    return True

