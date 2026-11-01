"""
Location: // fgns_engine/fgns/models/transition_lie/lie_extractor.py
Description: SOTA Non-Local Cross-Spatial Lie-Algebraic Transition Tensor Extractor
             with Optimal Transport Pre-Image Alignment, NA3D-v2 so(3)^C Torsion Extraction,
             and Cartan Decomposition on H^D_kappa.
Language: Python 3.12+ / PyTorch 2.5+ / Lie Group Differential Geometry sm_120 Aligned
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import math
from typing import Final, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fgns.config import FGNSConfig
from fgns.models.transition_lie.power_iteration import (
    DifferentiableRayleighPowerIteration,
)
from fgns.primitives.cartan_vielbein import CartanVielbeinFrame
from fgns.primitives.lorentz_manifold import LorentzManifold
from fgns.primitives.safe_math import (
    safe_divide,
    safe_hyperbolic_sqrt,
    safe_reciprocal,
)
from fgns.primitives.swiglu import RMSNorm, SwiGLU
from fgns.primitives.wavelet_kan import FactoredMorletKANLinear
from fgns.primitives.xor_involution import AnalyticXORInvolutionEngine
from fgns.types import (
    HyperbolicRuleState,
    LieTransitionState,
    XORParityState,
)

DEFAULT_LIE_EPS: Final[float] = 1e-7
NA3D_STRIDE: Final[int] = 16


class LieAlgebraTransitionExtractor(nn.Module):
    """
    SOTA Non-Local Cross-Spatial Lie-Algebraic Transition Tensor and Hyperbolic Rule Synthesizer.
    """

    def __init__(self, config: FGNSConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.octowave.d_model
        self.curvature = config.lorentz.curvature
        self.eps = DEFAULT_LIE_EPS

        self.manifold = LorentzManifold(curvature=self.curvature, eps=self.eps)
        self.cartan_frame = CartanVielbeinFrame(curvature=self.curvature, eps=self.eps)

        self.xor_engine = AnalyticXORInvolutionEngine(
            d_model=self.d_model,
            parity_alpha=config.xor.parity_alpha,
            eps=self.eps,
        )

        self.kan_warper = FactoredMorletKANLinear(
            d_in=self.d_model,
            d_out=self.d_model,
            num_wavelets=4,
            omega_0=5.0,
        )

        self.cross_align_gate = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
        self.na3d_torsion_gate = nn.Parameter(torch.tensor(0.30, dtype=torch.float32))

        self.probe_query = nn.Parameter(torch.empty(1, self.d_model))

        self.norm_in = RMSNorm(self.d_model)
        self.norm_out = RMSNorm(self.d_model)
        self.norm_xor = RMSNorm(self.d_model)

        raw_feature_dim = (5 * self.d_model) + 4

        self.synthesizer_swiglu = SwiGLU(
            d_in=raw_feature_dim,
            d_out=self.d_model,
            d_inner=2 * self.d_model,
            alignment_multiple=32,
        )

        self.norm_final_euclidean = RMSNorm(self.d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.d_model)
        nn.init.trunc_normal_(
            self.probe_query,
            mean=0.0,
            std=std,
            a=-2.0 * std,
            b=2.0 * std,
        )

    def _align_input_to_output_manifold(
        self,
        normed_in: torch.Tensor,
        in_mask: torch.Tensor,
        normed_out: torch.Tensor,
        out_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, K, H, W, D = normed_in.shape
        hw = H * W
        N = B * K

        q_out = normed_out.view(N, hw, D)
        k_in = normed_in.view(N, hw, D)
        in_m_flat = in_mask.view(N, hw)

        has_in = (in_m_flat.sum(dim=-1, keepdim=True) > 0).unsqueeze(-1)
        attn_bias = torch.where(
            in_m_flat.view(N, 1, hw),
            torch.zeros((N, 1, hw), dtype=q_out.dtype, device=q_out.device),
            torch.full((N, 1, hw), -10000.0, dtype=q_out.dtype, device=q_out.device),
        )

        h_cross_flat = F.scaled_dot_product_attention(
            q_out, k_in, k_in, attn_mask=attn_bias, dropout_p=0.0, is_causal=False
        )
        h_cross_flat = torch.where(has_in, h_cross_flat, torch.zeros_like(h_cross_flat))
        h_cross = h_cross_flat.view(B, K, H, W, D)

        alpha = torch.sigmoid(self.cross_align_gate.to(normed_in.dtype)) * 0.45
        in_m_f32 = in_mask.to(normed_in.dtype).unsqueeze(-1)
        out_m_f32 = out_mask.to(normed_in.dtype).unsqueeze(-1)

        local_anch = normed_in * in_m_f32
        aligned_in = ((1.0 - alpha) * local_anch + (alpha * h_cross)) * in_m_f32 + (
            (1.0 - in_m_f32) * h_cross
        )
        aligned_in = aligned_in * out_m_f32

        active_domain_mask = out_m_f32 * (
            in_mask.any(dim=-1, keepdim=True).any(dim=-2, keepdim=True).to(normed_in.dtype).unsqueeze(-1)
        )
        return aligned_in, active_domain_mask

    @staticmethod
    def _extract_na3d_torsion_embedding(
        aligned_in: torch.Tensor,
        normed_out: torch.Tensor,
        active_mask: torch.Tensor,
        eff_pair_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extracts the non-abelian 3D cross-product torsion moments between aligned input
        and target output NA3D cells across all C channels.
        """
        B, K, H, W, D = aligned_in.shape
        if D % NA3D_STRIDE != 0:
            return torch.zeros((B, D), dtype=aligned_in.dtype, device=aligned_in.device)

        c_cells = D // NA3D_STRIDE
        in_cells = (aligned_in.to(torch.float32) * active_mask.to(torch.float32)).view(
            B, K, H * W, c_cells, NA3D_STRIDE
        )
        out_cells = (normed_out.to(torch.float32) * active_mask.to(torch.float32)).view(
            B, K, H * W, c_cells, NA3D_STRIDE
        )

        tor_l = torch.cross(in_cells[..., 0:3], out_cells[..., 0:3], dim=-1)
        tor_c = torch.cross(in_cells[..., 3:6], out_cells[..., 3:6], dim=-1)
        tor_r = torch.cross(in_cells[..., 6:9], out_cells[..., 6:9], dim=-1)
        tor_gap = torch.cross(in_cells[..., 6:9], out_cells[..., 0:3], dim=-1)
        meta_delta = out_cells[..., 12:16] - in_cells[..., 12:16]

        packed_tor = torch.cat([tor_l, tor_c, tor_r, tor_gap, meta_delta], dim=-1).view(B, K, H * W, D)
        denom = active_mask.to(torch.float32).view(B, K, H * W, 1).sum(dim=2).clamp_min(1.0)
        pair_tor = packed_tor.sum(dim=2) / denom

        task_tor = torch.sum(pair_tor * eff_pair_weights.to(torch.float32).unsqueeze(-1), dim=1)
        return task_tor.to(aligned_in.dtype)

    def forward(
        self,
        h_in: torch.Tensor,
        in_mask: torch.Tensor,
        h_out: torch.Tensor,
        out_mask: torch.Tensor,
        exemplar_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, HyperbolicRuleState, LieTransitionState, XORParityState]:
        B, K, H, W, D = h_in.shape
        hw = H * W
        device = h_in.device
        dtype = h_in.dtype

        normed_in = self.norm_in(h_in)
        normed_out = self.norm_out(h_out)

        aligned_in, active_mask = self._align_input_to_output_manifold(
            normed_in=normed_in,
            in_mask=in_mask,
            normed_out=normed_out,
            out_mask=out_mask,
        )

        valid_pixels_per_pair = active_mask.view(B, K, hw).sum(dim=-1).clamp_min(1.0)
        pair_has_data = (valid_pixels_per_pair > 1.0).to(torch.float32)

        if exemplar_weights is not None:
            raw_w = exemplar_weights.to(device=device, dtype=torch.float32).view(B, K) * pair_has_data
            w_sum = raw_w.sum(dim=1, keepdim=True)
            uniform_w = pair_has_data / pair_has_data.sum(dim=1, keepdim=True).clamp_min(1.0)
            eff_pair_weights = torch.where(w_sum > 1e-6, raw_w / w_sum.clamp_min(1e-6), uniform_w).to(dtype)
        else:
            eff_pair_weights = (pair_has_data / pair_has_data.sum(dim=1, keepdim=True).clamp_min(1.0)).to(dtype)

        flat_h_in_pairs = aligned_in.view(B * K, H, W, D)
        flat_h_out_pairs = normed_out.view(B * K, H, W, D)

        xor_pairwise_flat, _ = self.xor_engine.forward_pairwise(
            u=flat_h_in_pairs,
            v=flat_h_out_pairs,
        )

        xor_pairwise_masked = xor_pairwise_flat.view(B, K, hw, D) * active_mask.view(B, K, hw, 1)
        xor_pooled_pairs = xor_pairwise_masked.sum(dim=2) / valid_pixels_per_pair.unsqueeze(-1).to(dtype)

        phi_xor_task = torch.sum(xor_pooled_pairs * eff_pair_weights.unsqueeze(-1), dim=1)
        na3d_tor = self._extract_na3d_torsion_embedding(aligned_in, normed_out, active_mask, eff_pair_weights)
        g_tor = torch.tanh(self.na3d_torsion_gate.to(dtype))
        phi_xor_normed = self.norm_xor(phi_xor_task + (g_tor * na3d_tor))

        flat_in = (aligned_in * active_mask).view(B, K, hw, D)
        in_mean = flat_in.sum(dim=2, keepdim=True) / valid_pixels_per_pair.view(B, K, 1, 1).to(dtype)
        centered_in = (flat_in - in_mean) * active_mask.view(B, K, hw, 1)

        delta_h = (normed_out - aligned_in) * active_mask
        flat_delta_pairs = delta_h.view(B * K, hw, D)
        flat_in_pairs = centered_in.view(B * K, hw, D)

        sigma_pairs = torch.bmm(flat_delta_pairs.transpose(1, 2), flat_in_pairs)
        sigma_pairs = sigma_pairs.view(B, K, D, D) / valid_pixels_per_pair.view(B, K, 1, 1).to(dtype)

        sigma_trans = torch.sum(sigma_pairs * eff_pair_weights.view(B, K, 1, 1), dim=1)

        sigma_t = sigma_trans.transpose(-2, -1)
        s_matrix = 0.50 * (sigma_trans + sigma_t)
        a_matrix = 0.50 * (sigma_trans - sigma_t)

        sa_prod = torch.bmm(s_matrix, a_matrix)
        as_prod = torch.bmm(a_matrix, s_matrix)
        k_commutator = 0.50 * ((sa_prod - as_prod) + (sa_prod - as_prod).transpose(-2, -1))

        frob_s = safe_hyperbolic_sqrt(torch.sum(s_matrix * s_matrix, dim=(-2, -1), keepdim=True), eps=self.eps)
        frob_a = safe_hyperbolic_sqrt(torch.sum(a_matrix * a_matrix, dim=(-2, -1), keepdim=True), eps=self.eps)
        frob_k = safe_hyperbolic_sqrt(torch.sum(k_commutator * k_commutator, dim=(-2, -1), keepdim=True), eps=self.eps)

        chirality_ratio = safe_divide(frob_k, (frob_s + frob_a + 1e-6), eps=self.eps).view(B, 1)

        lambda_s, v_s = DifferentiableRayleighPowerIteration.extract_dominant_eigenpair(
            matrix=s_matrix,
            num_iterations=4,
            eps=self.eps,
        )
        lambda_k, v_k = DifferentiableRayleighPowerIteration.extract_dominant_eigenpair(
            matrix=k_commutator,
            num_iterations=4,
            eps=self.eps,
        )

        q_probe = self.probe_query.expand(B, D).unsqueeze(-1).to(dtype)

        s_probe = torch.bmm(s_matrix, q_probe).squeeze(-1)
        k_probe = torch.bmm(k_commutator, q_probe).squeeze(-1)
        v_s_scaled = v_s * lambda_s
        v_k_scaled = v_k * lambda_k

        energy_meta = torch.cat([frob_s.squeeze(-1), frob_a.squeeze(-1), frob_k.squeeze(-1), chirality_ratio], dim=-1)

        raw_bundle = torch.cat(
            [s_probe, k_probe, v_s_scaled, v_k_scaled, phi_xor_normed, energy_meta],
            dim=-1,
        )
        z_synthesized = self.synthesizer_swiglu(raw_bundle)

        z_warped = self.kan_warper(z_synthesized)
        z_rule_euclidean = self.norm_final_euclidean(z_synthesized + z_warped)

        pole_origin = self.manifold.origin(dim=D, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1)
        v_tangent = self.cartan_frame.lift_to_tangent(x_base=pole_origin, c_euclidean=z_rule_euclidean)
        z_hyperbolic = self.manifold.exp_map(pole_origin, v_tangent)

        hyperbolic_state = HyperbolicRuleState(
            point_hyperboloid=z_hyperbolic.to(dtype),
            tangent_velocity=z_rule_euclidean.to(dtype),
            symmetric_invariant=s_matrix.to(dtype),
            anti_symmetric_rot=a_matrix.to(dtype),
            commutator=k_commutator.to(dtype),
            chirality_ratio=chirality_ratio.squeeze(-1).to(dtype),
        )

        lie_state = LieTransitionState(
            symmetric_invariant=s_matrix.to(dtype),
            anti_symmetric_rot=a_matrix.to(dtype),
            commutator=k_commutator.to(dtype),
            chirality_ratio=chirality_ratio.squeeze(-1).to(dtype),
        )

        xor_task_state = XORParityState(
            vorticity_flux=phi_xor_task.to(dtype),
            parity_selector=z_rule_euclidean.to(dtype),
            involution_norm=frob_k.squeeze(-1).to(dtype),
        )

        return z_rule_euclidean.to(dtype), hyperbolic_state, lie_state, xor_task_state


def verify_lie_extractor_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for lie_extractor.py.
    """
    torch.manual_seed(42)
    b, k, h, w, d = 2, 2, 4, 4, 32

    config = FGNSConfig()
    extractor = LieAlgebraTransitionExtractor(config=config)

    extractor.d_model = d
    extractor.manifold = LorentzManifold(curvature=1.0)
    extractor.cartan_frame = CartanVielbeinFrame(curvature=1.0)
    extractor.xor_engine = AnalyticXORInvolutionEngine(d_model=d, parity_alpha=0.25)
    extractor.kan_warper = FactoredMorletKANLinear(d_in=d, d_out=d, num_wavelets=4, omega_0=5.0)
    extractor.probe_query = nn.Parameter(torch.randn(1, d) / math.sqrt(d))
    extractor.norm_in = RMSNorm(d)
    extractor.norm_out = RMSNorm(d)
    extractor.norm_xor = RMSNorm(d)
    extractor.synthesizer_swiglu = SwiGLU(d_in=(5 * d) + 4, d_out=d, d_inner=64)
    extractor.norm_final_euclidean = RMSNorm(d)

    h_in = torch.randn(b, k, h, w, d, dtype=torch.float32, requires_grad=True)
    in_mask = torch.ones(b, k, h, w, dtype=torch.bool)
    h_out = torch.randn(b, k, h, w, d, dtype=torch.float32, requires_grad=True)
    out_mask = torch.ones(b, k, h, w, dtype=torch.bool)
    ex_weights = torch.tensor([[0.90, 0.10], [0.20, 0.80]], dtype=torch.float32)

    z_rule, hyp_state, lie_state, xor_state = extractor(
        h_in=h_in,
        in_mask=in_mask,
        h_out=h_out,
        out_mask=out_mask,
        exemplar_weights=ex_weights,
    )

    if z_rule.shape != (b, d) or hyp_state.point_hyperboloid.shape != (b, d + 1):
        return False

    loss = torch.sum(hyp_state.point_hyperboloid.pow(2)) + torch.sum(z_rule.pow(2))
    loss.backward()

    if h_in.grad is None or h_out.grad is None or extractor.probe_query.grad is None:
        return False

    return True
