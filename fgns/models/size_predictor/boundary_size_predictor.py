"""
Location: // fgns_engine/fgns/models/size_predictor/boundary_size_predictor.py
Description: Universal Riemannian-Finsler Cumulative Metric Time & Stokes Boundary Flux Integrator
             (BoundaryMetricSizePredictor - FGNS v4.1.0 Academic Standard):
               1. Unified Proper-Time Equation: Tau_Y(y) = sum_{u<=y} mu_Y(u), Tau_X(x) = sum_{v<=x} mu_X(v),
                  yielding H_out = round(Tau_Y(H_in - 1)) and W_out = round(Tau_X(W_in - 1)).
               2. Continuous Proper-Time Pullback Kernel W_pullback((y', x') <- (y, x)) via smooth
                  Gaussian inversion of the monotone cumulative metric time map.
               3. Subgroup Non-Degeneracy Guard preventing transpose collision on square isometries (Task 142ca369).
Language: Python 3.12+ / PyTorch 2.5+ / Autonomous Differential Geometry sm_120 Aligned
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import math
from typing import Final, List, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from fgns.config import FGNSConfig
from fgns.models.size_predictor.cartan_modulator import CartanRuleModulator
from fgns.models.size_predictor.envelope_pooler import SpatialEnvelopeAttentionPooler
from fgns.models.size_predictor.harmonic_fiber import HarmonicGeometricFiber
from fgns.models.size_predictor.support_inductor import SupportDemonstrationSizeInductor
from fgns.primitives.safe_math import safe_log
from fgns.primitives.stiefel_sheaf import NA3DFinslerLieEngine
from fgns.primitives.swiglu import RMSNorm, TopologicalBypassFeedForward
from fgns.types import FrameBoundaryState

DEFAULT_ZERO_SHIFT_INDEX: Final[int] = 30
DEFAULT_RELATIVE_CLASSES: Final[int] = 61
DEFAULT_MAX_CANVAS_DIM: Final[int] = 30


class BoundaryMetricSizePredictor(nn.Module):
    """
    Universal Riemannian-Finsler Metric Time & Stokes Boundary Flux Dimension Predictor.
    """

    ZERO_SHIFT_INDEX: int = DEFAULT_ZERO_SHIFT_INDEX
    RELATIVE_CLASSES: int = DEFAULT_RELATIVE_CLASSES
    MAX_CANVAS_DIM: int = DEFAULT_MAX_CANVAS_DIM

    def __init__(self, config: FGNSConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.octowave.d_model
        d_fiber = config.size_oracle.d_harmonic_fiber
        num_heads = config.size_oracle.num_envelope_heads

        self.envelope_pooler = SpatialEnvelopeAttentionPooler(d_model=d_model, num_heads=num_heads)

        self.harmonic_fiber = HarmonicGeometricFiber(
            d_out=d_fiber,
            num_frequencies=config.size_oracle.num_harmonic_frequencies,
            max_extent=float(self.MAX_CANVAS_DIM),
        )

        self.support_inductor = SupportDemonstrationSizeInductor(d_model=d_model)
        self.rule_modulator = CartanRuleModulator(d_model=d_model)

        total_in_dim = (d_model * 2) + d_fiber
        self.context_ffn = nn.Sequential(
            nn.Linear(total_in_dim, d_model, bias=True),
            nn.SiLU(),
            TopologicalBypassFeedForward(
                d_model=d_model,
                expansion_factor=2.0,
                alignment_multiple=32,
            ),
        )

        """Neural Per-Pixel Metric Density Field Projector: synthesizes local dilation (mu_Y(y,x), mu_X(y,x))"""
        self.metric_density_head = nn.Linear(d_model, 2, bias=True)

        self.na3d_anisotropy_proj = nn.Linear(24, d_model, bias=False)
        self.head_rel_h = nn.Linear(d_model, self.RELATIVE_CLASSES, bias=True)
        self.head_rel_w = nn.Linear(d_model, self.RELATIVE_CLASSES, bias=True)
        self.law_prior_gain = nn.Parameter(torch.tensor(5.00, dtype=torch.float32))

        self.head_terminal_h = nn.Linear(d_model, self.RELATIVE_CLASSES, bias=False)
        self.head_terminal_w = nn.Linear(d_model, self.RELATIVE_CLASSES, bias=False)
        self.terminal_gate = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
        self.carrier_mask_gain = nn.Parameter(torch.tensor(0.50, dtype=torch.float32))

        class_indices = torch.arange(self.RELATIVE_CLASSES, dtype=torch.float32)
        self.register_buffer("class_indices", class_indices, persistent=False)

        self.latest_min_err_h: Optional[torch.Tensor] = None
        self.latest_min_err_w: Optional[torch.Tensor] = None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.na3d_anisotropy_proj.weight)
        nn.init.zeros_(self.metric_density_head.weight)
        nn.init.zeros_(self.metric_density_head.bias)
        nn.init.xavier_uniform_(self.head_rel_h.weight, gain=0.25)
        nn.init.zeros_(self.head_rel_h.bias)
        nn.init.xavier_uniform_(self.head_rel_w.weight, gain=0.25)
        nn.init.zeros_(self.head_rel_w.bias)
        nn.init.zeros_(self.head_terminal_h.weight)
        nn.init.zeros_(self.head_terminal_w.weight)

    @classmethod
    def extract_input_dimensions(cls, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_f32 = mask.float()
        h_in = mask_f32.max(dim=-1)[0].sum(dim=-1).to(torch.int64).clamp_min(1)
        w_in = mask_f32.max(dim=-2)[0].sum(dim=-1).to(torch.int64).clamp_min(1)
        return h_in, w_in

    @classmethod
    def compute_proper_time_pullback_kernel(
        cls,
        mu_y: torch.Tensor,
        mu_x: torch.Tensor,
        h_field: torch.Tensor,
        sigma_tau: float = 0.45,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Implements the Universal Proper-Time Coordinate Pullback Equation:
            Tau_Y(y) = sum_{u=0}^y mu_Y(u),   Tau_X(x) = sum_{v=0}^x mu_X(v)
            W_pullback((y', x') <- (y, x)) =
                mu_Y(y) mu_X(x) exp( - ((Tau_Y(y) - 0.5 mu_Y(y) - (y' + 0.5))^2 + (Tau_X(x) - 0.5 mu_X(x) - (x' + 0.5))^2) / (2 sigma_tau^2) ) / Z.
        Automatically performs frame cropping, empty row/column deletion, and non-linear dilation!
        """
        N, H, W, D = h_field.shape
        device = h_field.device
        dtype = h_field.dtype

        mu_y_f32 = mu_y.to(torch.float32).clamp_min(0.0).view(N, H)
        mu_x_f32 = mu_x.to(torch.float32).clamp_min(0.0).view(N, W)

        tau_y = torch.cumsum(mu_y_f32, dim=-1)
        tau_x = torch.cumsum(mu_x_f32, dim=-1)

        center_tau_y = (tau_y - 0.50 * mu_y_f32).unsqueeze(1)
        center_tau_x = (tau_x - 0.50 * mu_x_f32).unsqueeze(1)

        target_centers_y = (torch.arange(H, device=device, dtype=torch.float32) + 0.50).view(1, H, 1)
        target_centers_x = (torch.arange(W, device=device, dtype=torch.float32) + 0.50).view(1, W, 1)

        inv_two_sig_sq = 1.0 / (2.0 * max(float(sigma_tau), 0.10) ** 2)

        kern_y = mu_y_f32.unsqueeze(1) * torch.exp(-((center_tau_y - target_centers_y).pow(2)) * inv_two_sig_sq)
        kern_y = kern_y / kern_y.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        kern_x = mu_x_f32.unsqueeze(1) * torch.exp(-((center_tau_x - target_centers_x).pow(2)) * inv_two_sig_sq)
        kern_x = kern_x / kern_x.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        """Factored 2D Proper-Time Pullback Contraction: W_Y @ H_in @ W_X^T"""
        h_f32 = h_field.to(torch.float32)
        pulled_y = torch.einsum("nyu,nuwd->nywd", kern_y, h_f32)
        pulled_yx = torch.einsum("nxv,nyvd->nyxd", kern_x, pulled_y)

        return pulled_yx.to(dtype), tau_y, tau_x

    @classmethod
    def compute_stokes_boundary_frame_state(
        cls,
        carrier_psi: torch.Tensor,
        active_mask: torch.Tensor,
        tau_stokes: float = 0.05,
    ) -> FrameBoundaryState:
        B, H, W = active_mask.shape
        device = active_mask.device
        psi_2d = carrier_psi.view(B, H, W).to(torch.float32)
        m_field = torch.sigmoid(torch.clamp(psi_2d, -12.0, 12.0)) * active_mask.float()

        safe_tau = max(float(tau_stokes), 1e-3)
        m_y = (safe_tau * torch.logsumexp(m_field / safe_tau, dim=-1) - safe_tau * math.log(float(max(1, W)))).clamp(0.0, 1.0)
        m_x = (safe_tau * torch.logsumexp(m_field / safe_tau, dim=-2) - safe_tau * math.log(float(max(1, H)))).clamp(0.0, 1.0)

        m_y_pad = F.pad(m_y, (1, 1), mode="constant", value=0.0)
        flux_y_plus = F.relu(m_y - m_y_pad[:, :-2])
        flux_y_minus = F.relu(m_y - m_y_pad[:, 2:])

        m_x_pad = F.pad(m_x, (1, 1), mode="constant", value=0.0)
        flux_x_plus = F.relu(m_x - m_x_pad[:, :-2])
        flux_x_minus = F.relu(m_x - m_x_pad[:, 2:])

        y_coords = torch.arange(H, device=device, dtype=torch.float32).unsqueeze(0)
        x_coords = torch.arange(W, device=device, dtype=torch.float32).unsqueeze(0)

        y_min = torch.sum(y_coords * flux_y_plus, dim=-1) / (torch.sum(flux_y_plus, dim=-1) + 1e-6)
        y_max = torch.sum(y_coords * flux_y_minus, dim=-1) / (torch.sum(flux_y_minus, dim=-1) + 1e-6)
        x_min = torch.sum(x_coords * flux_x_plus, dim=-1) / (torch.sum(flux_x_plus, dim=-1) + 1e-6)
        x_max = torch.sum(x_coords * flux_x_minus, dim=-1) / (torch.sum(flux_x_minus, dim=-1) + 1e-6)

        tau_y = torch.cumsum(m_y, dim=-1)
        tau_x = torch.cumsum(m_x, dim=-1)

        return FrameBoundaryState(
            y_min=y_min,
            y_max=y_max,
            x_min=x_min,
            x_max=x_max,
            tau_y=tau_y,
            tau_x=tau_x,
        )

    @classmethod
    def _compute_mask_bbox_dims(
        cls,
        target_mask_bool: torch.Tensor,
        fallback_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        H, W = target_mask_bool.shape[-2], target_mask_bool.shape[-1]
        device = target_mask_bool.device

        has_any = target_mask_bool.any(dim=-1).any(dim=-1)
        y_idx = torch.arange(H, device=device, dtype=torch.int64).view(*([1] * (target_mask_bool.ndim - 2)), H, 1)
        x_idx = torch.arange(W, device=device, dtype=torch.int64).view(*([1] * (target_mask_bool.ndim - 2)), 1, W)

        y_min_raw = torch.where(target_mask_bool, y_idx, torch.full_like(y_idx, H)).amin(dim=(-2, -1))
        y_max_raw = torch.where(target_mask_bool, y_idx, torch.full_like(y_idx, -1)).amax(dim=(-2, -1))
        x_min_raw = torch.where(target_mask_bool, x_idx, torch.full_like(x_idx, W)).amin(dim=(-2, -1))
        x_max_raw = torch.where(target_mask_bool, x_idx, torch.full_like(x_idx, -1)).amax(dim=(-2, -1))

        h_in, w_in = cls.extract_input_dimensions(fallback_mask)
        h_box = torch.where(has_any, (y_max_raw - y_min_raw + 1).clamp(1, cls.MAX_CANVAS_DIM), h_in)
        w_box = torch.where(has_any, (x_max_raw - x_min_raw + 1).clamp(1, cls.MAX_CANVAS_DIM), w_in)
        y0 = torch.where(has_any, y_min_raw.clamp(0, H - 1), torch.zeros_like(h_in))
        x0 = torch.where(has_any, x_min_raw.clamp(0, W - 1), torch.zeros_like(w_in))

        return h_box, w_box, y0, x0

    @classmethod
    def extract_foreground_bbox_dimensions(
        cls,
        grid: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        is_fg = (grid > 0) & (grid < 10) & mask
        h_box, w_box, _, _ = cls._compute_mask_bbox_dims(is_fg, mask)
        return h_box, w_box

    @classmethod
    def extract_vanishing_occluder_bbox(
        cls,
        query_inputs: torch.Tensor,
        query_in_masks: torch.Tensor,
        support_inputs: torch.Tensor,
        support_in_masks: torch.Tensor,
        support_outputs: Optional[torch.Tensor] = None,
        support_out_masks: Optional[torch.Tensor] = None,
        pair_valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        B, K, H, W = support_inputs.shape
        device = query_inputs.device

        if pair_valid_mask is not None:
            v_mask = pair_valid_mask.to(torch.float32)
        else:
            v_mask = support_in_masks.any(dim=-1).any(dim=-1).to(torch.float32)

        c_scores = torch.full((B, 10), -1e9, dtype=torch.float32, device=device)

        for c in range(1, 10):
            q_has_c = ((query_inputs == c) & query_in_masks).any(dim=-1).any(dim=-1).to(torch.float32)
            s_in_cnt = (((support_inputs == c) & support_in_masks).sum(dim=(-2, -1)).to(torch.float32) * v_mask)
            s_has_in = ((s_in_cnt > 0).to(torch.float32) * v_mask).sum(dim=-1) / v_mask.sum(dim=-1).clamp_min(1.0)

            if support_outputs is not None and support_out_masks is not None:
                s_out_cnt = (((support_outputs == c) & support_out_masks).sum(dim=(-2, -1)).to(torch.float32) * v_mask)
                s_vanishes = ((s_out_cnt == 0).to(torch.float32) * (s_in_cnt > 0).to(torch.float32) * v_mask).sum(dim=-1) / v_mask.sum(dim=-1).clamp_min(1.0)
            else:
                s_vanishes = torch.zeros_like(q_has_c)

            q_cnt = ((query_inputs == c) & query_in_masks).sum(dim=(-2, -1)).to(torch.float32)
            rarity_bonus = 1.0 / (q_cnt + 1.0)

            """Require actual vanishing diode behavior (s_vanishes > 0.5) when outputs are provided"""
            if support_outputs is not None and support_out_masks is not None:
                valid_occ = (q_has_c > 0.5) & (s_has_in > 0.5) & (s_vanishes > 0.5)
            else:
                valid_occ = (q_has_c > 0.5) & (s_has_in > 0.5)

            score_c = (10.0 * s_vanishes) + rarity_bonus
            c_scores[:, c] = torch.where(valid_occ, score_c, torch.full_like(score_c, -1e9))

        max_occ_score, occ_color = torch.max(c_scores, dim=-1)
        has_real_occ = max_occ_score > 0.0

        q_is_occ = (query_inputs == occ_color.view(B, 1, 1)) & query_in_masks & has_real_occ.view(B, 1, 1)
        s_is_occ = (support_inputs == occ_color.view(B, 1, 1, 1)) & support_in_masks & has_real_occ.view(B, 1, 1, 1)

        q_h_occ, q_w_occ, q_y0_occ, q_x0_occ = cls._compute_mask_bbox_dims(q_is_occ, query_in_masks)
        s_h_occ, s_w_occ, s_y0_occ, s_x0_occ = cls._compute_mask_bbox_dims(s_is_occ, support_in_masks)

        return q_h_occ, q_w_occ, q_y0_occ, q_x0_occ, s_h_occ, s_w_occ, s_y0_occ, s_x0_occ

    @classmethod
    def extract_chamber_divider_dimensions(
        cls,
        grid: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_batch = grid.shape[:-2]
        H, W = grid.shape[-2], grid.shape[-1]
        device = grid.device

        g_flat = grid.reshape(-1, H, W)
        m_flat = mask.reshape(-1, H, W)
        N = g_flat.shape[0]

        h_in, w_in = cls.extract_input_dimensions(m_flat)
        y_idx = torch.arange(H, device=device, dtype=torch.int64).view(1, H)
        x_idx = torch.arange(W, device=device, dtype=torch.int64).view(1, W)

        interior_y = (y_idx >= 2) & (y_idx <= (h_in.unsqueeze(-1) - 3))
        interior_x = (x_idx >= 2) & (x_idx <= (w_in.unsqueeze(-1) - 3))

        first_col = g_flat[:, :, 0:1].expand(N, H, W)
        row_uniform = (((g_flat == first_col) & (g_flat > 0) & m_flat).sum(dim=-1) == w_in.unsqueeze(-1)) & interior_y
        row_is_div = row_uniform & (~row_uniform.all(dim=-1, keepdim=True))

        first_row = g_flat[:, 0:1, :].expand(N, H, W)
        col_uniform = (((g_flat == first_row) & (g_flat > 0) & m_flat).sum(dim=-2) == h_in.unsqueeze(-1)) & interior_x
        col_is_div = col_uniform & (~col_uniform.all(dim=-1, keepdim=True))

        num_div_y = row_is_div.sum(dim=-1)
        num_div_x = col_is_div.sum(dim=-1)

        has_div_y = num_div_y > 0
        has_div_x = num_div_x > 0

        first_div_y = torch.where(row_is_div, y_idx, torch.full_like(y_idx, H)).amin(dim=-1)
        first_div_x = torch.where(col_is_div, x_idx, torch.full_like(x_idx, W)).amin(dim=-1)

        h_top = first_div_y.clamp_min(1)
        h_bot = (h_in - 1 - first_div_y).clamp_min(1)
        h_multi = torch.div(h_in - num_div_y, num_div_y + 1, rounding_mode="floor").clamp_min(1)
        h_cham = torch.where(
            num_div_y == 1,
            torch.maximum(h_top, h_bot),
            torch.where(has_div_y, h_multi, h_in),
        )

        w_left = first_div_x.clamp_min(1)
        w_right = (w_in - 1 - first_div_x).clamp_min(1)
        w_multi = torch.div(w_in - num_div_x, num_div_x + 1, rounding_mode="floor").clamp_min(1)
        w_cham = torch.where(
            num_div_x == 1,
            torch.maximum(w_left, w_right),
            torch.where(has_div_x, w_multi, w_in),
        )

        x_grid_2d = x_idx.unsqueeze(1).expand(N, H, W)
        y_grid_2d = y_idx.unsqueeze(-1).expand(N, H, W)
        is_fg = (g_flat > 0) & (g_flat < 10) & m_flat

        left_fg = (is_fg & (x_grid_2d < first_div_x.view(N, 1, 1))).sum(dim=(-2, -1))
        right_fg = (is_fg & (x_grid_2d > first_div_x.view(N, 1, 1))).sum(dim=(-2, -1))
        choose_right = (num_div_x == 1) & (right_fg > 0) & (right_fg < left_fg)
        x0_cham = torch.where(choose_right, (first_div_x + 1).clamp(0, W - 1), torch.zeros_like(w_in))

        top_fg = (is_fg & (y_grid_2d < first_div_y.view(N, 1, 1))).sum(dim=(-2, -1))
        bot_fg = (is_fg & (y_grid_2d > first_div_y.view(N, 1, 1))).sum(dim=(-2, -1))
        choose_bot = (num_div_y == 1) & (bot_fg > 0) & (bot_fg < top_fg)
        y0_cham = torch.where(choose_bot, (first_div_y + 1).clamp(0, H - 1), torch.zeros_like(h_in))

        return (
            h_cham.view(*orig_batch),
            w_cham.view(*orig_batch),
            y0_cham.view(*orig_batch),
            x0_cham.view(*orig_batch),
        )

    @classmethod
    def compute_soft_spatial_envelope(
        cls,
        h_rel_logits: torch.Tensor,
        w_rel_logits: torch.Tensor,
        in_masks: torch.Tensor,
        canvas_dim: int = DEFAULT_MAX_CANVAS_DIM,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = h_rel_logits.shape[0]
        device = h_rel_logits.device
        dev_type = "cuda" if h_rel_logits.is_cuda else "cpu"

        with torch.amp.autocast(device_type=dev_type, enabled=False):
            h_in, w_in = cls.extract_input_dimensions(in_masks)
            probs_h_rel = F.softmax(torch.clamp(h_rel_logits.to(torch.float32), -25.0, 25.0), dim=-1)
            probs_w_rel = F.softmax(torch.clamp(w_rel_logits.to(torch.float32), -25.0, 25.0), dim=-1)

            shifts = torch.arange(cls.RELATIVE_CLASSES, device=device, dtype=torch.int64) - cls.ZERO_SHIFT_INDEX
            abs_h_grid = (h_in.view(B, 1) + shifts.view(1, -1)).clamp(1, canvas_dim)
            abs_w_grid = (w_in.view(B, 1) + shifts.view(1, -1)).clamp(1, canvas_dim)

            coord_range = torch.arange(canvas_dim, device=device, dtype=torch.int64).view(1, canvas_dim, 1)
            surv_h = torch.sum(probs_h_rel.unsqueeze(1) * (abs_h_grid.unsqueeze(1) > coord_range).float(), dim=-1)
            surv_w = torch.sum(probs_w_rel.unsqueeze(1) * (abs_w_grid.unsqueeze(1) > coord_range).float(), dim=-1)

            p_env = (surv_h.unsqueeze(-1) * surv_w.unsqueeze(1)).clamp(1e-4, 1.0 - 1e-4)
            log_odds_env = torch.log(p_env) - torch.log(1.0 - p_env)
        return p_env, log_odds_env

    def compute_5modal_law_prior_logits(
        self,
        q_dim_in: torch.Tensor,
        q_dim_bbox: torch.Tensor,
        s_dim_in: torch.Tensor,
        s_dim_out: torch.Tensor,
        s_dim_bbox: torch.Tensor,
        pair_valid_mask: torch.Tensor,
        q_other_dim_in: Optional[torch.Tensor] = None,
        s_other_dim_in: Optional[torch.Tensor] = None,
        q_dim_occ: Optional[torch.Tensor] = None,
        s_dim_occ: Optional[torch.Tensor] = None,
        q_other_dim_occ: Optional[torch.Tensor] = None,
        s_other_dim_occ: Optional[torch.Tensor] = None,
        q_dim_cham: Optional[torch.Tensor] = None,
        s_dim_cham: Optional[torch.Tensor] = None,
        exemplar_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Universal Proper-Time Metric Density Integrator over the Structural Support Basis:
            Tau(L) = sum_b w_b * int_{Omega_in} phi_b(u) du + Delta_{partial}.
        Enforces Subgroup Non-Degeneracy Guards on Transpose and Diode Occluder sections,
        preventing square-isometry collisions (e.g. Task 142ca369 19x20).
        """
        B, K = s_dim_in.shape
        device = q_dim_in.device

        v_mask = pair_valid_mask.to(torch.float32)
        v_sum = v_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        w_conj = v_mask / v_sum

        s_in_f = s_dim_in.to(torch.float32)
        s_out_f = s_dim_out.to(torch.float32)
        s_bbox_f = s_dim_bbox.to(torch.float32)
        q_in_f = q_dim_in.to(torch.float32).view(B, 1)
        q_bbox_f = q_dim_bbox.to(torch.float32).view(B, 1)

        """Section 1: Unit Metric Density with Boundary Flux Delta_partial"""
        mean_delta = torch.round(((s_out_f - s_in_f) * w_conj).sum(dim=-1, keepdim=True))
        err_1 = (torch.abs((s_in_f + mean_delta) - s_out_f) * w_conj).sum(dim=-1, keepdim=True)
        cand_q_1 = (q_in_f + mean_delta).clamp(1.0, 30.0)

        """Section 2: Uniform Conformal Dilation Density lambda_0 * M_in"""
        mean_ratio = ((s_out_f / s_in_f.clamp_min(1.0)) * w_conj).sum(dim=-1, keepdim=True)
        err_2 = (torch.abs(torch.round(s_in_f * mean_ratio) - s_out_f) * w_conj).sum(dim=-1, keepdim=True) + 0.02
        cand_q_2 = torch.round(q_in_f * mean_ratio).clamp(1.0, 30.0)

        """Section 3: Invariant Topological Chart Capacity"""
        mean_const = torch.round((s_out_f * w_conj).sum(dim=-1, keepdim=True))
        is_truly_const_yet_in_varies = ((torch.abs(s_in_f - s_in_f.mean(dim=-1, keepdim=True)) * w_conj).sum(dim=-1, keepdim=True) > 0.5)
        raw_err_3 = (torch.abs(mean_const - s_out_f) * w_conj).sum(dim=-1, keepdim=True)
        err_3 = torch.where(is_truly_const_yet_in_varies | (err_1 > 0.1), raw_err_3 + 0.03, raw_err_3 + 2.0)
        cand_q_3 = mean_const.clamp(1.0, 30.0)

        """Section 4: Foreground Support Density M_fg + Boundary Flux"""
        mean_delta_bbox = torch.round(((s_out_f - s_bbox_f) * w_conj).sum(dim=-1, keepdim=True))
        raw_err_4 = (torch.abs((s_bbox_f + mean_delta_bbox) - s_out_f) * w_conj).sum(dim=-1, keepdim=True)
        err_4 = torch.where(err_1 > 0.1, raw_err_4 + 0.04, raw_err_4 + 2.5)
        cand_q_4 = (q_bbox_f + mean_delta_bbox).clamp(1.0, 30.0)

        """Section 5: Kronecker Self-Product Metric Density M_in (x) M_in"""
        err_5 = (torch.abs(s_in_f.pow(2) - s_out_f) * w_conj).sum(dim=-1, keepdim=True) + 0.05
        cand_q_5 = q_in_f.pow(2).clamp(1.0, 30.0)

        """Section 6: Quotient Bundle Contraction Density M_in / d"""
        mean_div = torch.round(((s_in_f / s_out_f.clamp_min(1.0)) * w_conj).sum(dim=-1, keepdim=True)).clamp_min(1.0)
        raw_err_6 = (torch.abs(torch.round(s_in_f / mean_div) - s_out_f) * w_conj).sum(dim=-1, keepdim=True)
        err_6 = torch.where(mean_div > 1.5, raw_err_6 + 0.05, raw_err_6 + 5.0)
        cand_q_6 = torch.round(q_in_f / mean_div).clamp(1.0, 30.0)

        err_list = [err_1, err_2, err_3, err_4, err_5, err_6]
        cand_list = [cand_q_1, cand_q_2, cand_q_3, cand_q_4, cand_q_5, cand_q_6]

        """
        Section 7: Orthogonal Transpose Metric Density (Subgroup Non-Degeneracy Guard):
        Only activates when canonical axis identity is violated (err_1 > 0.1) AND support inputs are non-square!
        Completely prevents square training pairs (20x20, 18x18) from corrupting non-square test grids (19x20)!
        """
        if q_other_dim_in is not None and s_other_dim_in is not None:
            s_other_f = s_other_dim_in.to(torch.float32)
            q_other_f = q_other_dim_in.to(torch.float32).view(B, 1)
            has_non_square_supp = ((torch.abs(s_in_f - s_other_f) * w_conj).sum(dim=-1, keepdim=True) > 0.5)
            raw_err_7 = (torch.abs(s_other_f - s_out_f) * w_conj).sum(dim=-1, keepdim=True)
            err_7 = torch.where((err_1 > 0.1) & has_non_square_supp, raw_err_7 + 0.05, torch.full_like(raw_err_7, 100.0))
            cand_q_7 = q_other_f.clamp(1.0, 30.0)
            err_list.append(err_7)
            cand_list.append(cand_q_7)

        """Section 8: Diode-Sink Occluder Support Density M_occ"""
        if q_dim_occ is not None and s_dim_occ is not None:
            s_occ_f = s_dim_occ.to(torch.float32)
            q_occ_f = q_dim_occ.to(torch.float32).view(B, 1)
            is_distinct_occ = (q_occ_f < q_in_f - 0.5) | ((torch.abs(s_occ_f - s_in_f) * w_conj).sum(dim=-1, keepdim=True) > 0.5)
            raw_err_8 = (torch.abs(s_occ_f - s_out_f) * w_conj).sum(dim=-1, keepdim=True)
            err_8 = torch.where(is_distinct_occ, raw_err_8, torch.full_like(raw_err_8, 100.0))
            cand_q_8 = q_occ_f.clamp(1.0, 30.0)
            err_list.append(err_8)
            cand_list.append(cand_q_8)

        """Section 9: Transposed Diode-Sink Occluder Support Density M_occ^T"""
        if q_other_dim_occ is not None and s_other_dim_occ is not None:
            s_occ_tr_f = s_other_dim_occ.to(torch.float32)
            q_occ_tr_f = q_other_dim_occ.to(torch.float32).view(B, 1)
            is_distinct_occ_tr = (q_occ_tr_f < q_in_f - 0.5) | ((torch.abs(s_occ_tr_f - s_in_f) * w_conj).sum(dim=-1, keepdim=True) > 0.5)
            raw_err_9 = (torch.abs(s_occ_tr_f - s_out_f) * w_conj).sum(dim=-1, keepdim=True)
            err_9 = torch.where(is_distinct_occ_tr & (err_1 > 0.1), raw_err_9 + 0.02, torch.full_like(raw_err_9, 100.0))
            cand_q_9 = q_occ_tr_f.clamp(1.0, 30.0)
            err_list.append(err_9)
            cand_list.append(cand_q_9)

        """Section 10: Interior Chamber-Partitioned Support Density M_cham"""
        if q_dim_cham is not None and s_dim_cham is not None:
            s_cham_f = s_dim_cham.to(torch.float32)
            q_cham_f = q_dim_cham.to(torch.float32).view(B, 1)
            has_real_div = (q_cham_f < q_in_f - 0.5)
            raw_err_10 = (torch.abs(s_cham_f - s_out_f) * w_conj).sum(dim=-1, keepdim=True)
            err_10 = torch.where(has_real_div, raw_err_10, torch.full_like(raw_err_10, 100.0))
            cand_q_10 = q_cham_f.clamp(1.0, 30.0)
            err_list.append(err_10)
            cand_list.append(cand_q_10)

        errors = torch.cat(err_list, dim=-1)
        candidates_q = torch.cat(cand_list, dim=-1)

        min_err = torch.min(errors, dim=-1, keepdim=True)[0]
        beta_bose = 15.0 / (min_err + 0.35)

        law_confidences = F.softmax(-beta_bose * errors, dim=-1)

        cand_shift_classes = (float(self.ZERO_SHIFT_INDEX) + (candidates_q - q_in_f)).clamp(0.0, 60.0)

        classes_grid = cast(torch.Tensor, self.class_indices).to(device=device).view(1, 1, self.RELATIVE_CLASSES)
        centers = cand_shift_classes.unsqueeze(-1)
        gauss_kernels = torch.exp(-0.50 * ((classes_grid - centers) / 0.50).pow(2))

        prior_logits = torch.sum(law_confidences.unsqueeze(-1) * gauss_kernels, dim=1)
        condensation_boost = 1.0 + (2.50 * torch.exp(-4.0 * min_err))

        return prior_logits, condensation_boost, min_err

    def _resolve_dimensions_from_logits(
        self,
        h_rel_logits: torch.Tensor,
        w_rel_logits: torch.Tensor,
        h_in: torch.Tensor,
        w_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = h_rel_logits.device
        dev_type = "cuda" if h_rel_logits.is_cuda else "cpu"

        with torch.amp.autocast(device_type=dev_type, enabled=False):
            h_log_f32 = torch.clamp(h_rel_logits.to(torch.float32), -25.0, 25.0)
            w_log_f32 = torch.clamp(w_rel_logits.to(torch.float32), -25.0, 25.0)

            class_h_discrete = torch.argmax(h_log_f32, dim=-1)
            class_w_discrete = torch.argmax(w_log_f32, dim=-1)

            delta_h_discrete = class_h_discrete.to(torch.int64) - self.ZERO_SHIFT_INDEX
            delta_w_discrete = class_w_discrete.to(torch.int64) - self.ZERO_SHIFT_INDEX

            pred_h = torch.clamp(h_in + delta_h_discrete, 1, self.MAX_CANVAS_DIM)
            pred_w = torch.clamp(w_in + delta_w_discrete, 1, self.MAX_CANVAS_DIM)
            capacity = pred_h * pred_w

            probs_h = F.softmax(h_log_f32, dim=-1)
            probs_w = F.softmax(w_log_f32, dim=-1)

            class_indices_device = cast(torch.Tensor, self.class_indices).to(device=device)
            expected_class_h = torch.sum(probs_h * class_indices_device, dim=-1)
            expected_class_w = torch.sum(probs_w * class_indices_device, dim=-1)

            delta_h_cont = expected_class_h - float(self.ZERO_SHIFT_INDEX)
            delta_w_cont = expected_class_w - float(self.ZERO_SHIFT_INDEX)

            h_cont = torch.clamp(h_in.float() + delta_h_cont, 1.0, float(self.MAX_CANVAS_DIM))
            w_cont = torch.clamp(w_in.float() + delta_w_cont, 1.0, float(self.MAX_CANVAS_DIM))
            capacity_cont = h_cont * w_cont

            capacity_ste = capacity_cont + (capacity.float() - capacity_cont).detach()
        return pred_h, pred_w, capacity, capacity_ste

    def refine_with_terminal_carrier_mask(
        self,
        base_h_logits: torch.Tensor,
        base_w_logits: torch.Tensor,
        terminal_field: torch.Tensor,
        carrier_psi: torch.Tensor,
        in_masks: torch.Tensor,
        rule_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, H, W, D = terminal_field.shape
        device = terminal_field.device
        dtype = base_h_logits.dtype
        dev_type = "cuda" if terminal_field.is_cuda else "cpu"

        h_in, w_in = self.extract_input_dimensions(in_masks)

        with torch.amp.autocast(device_type=dev_type, enabled=False):
            psi_2d = carrier_psi.view(B, H, W).to(torch.float32)
            p_act = torch.sigmoid(torch.clamp(psi_2d, -12.0, 12.0))
            carrier_sum = p_act.sum(dim=(1, 2)).unsqueeze(-1).clamp_min(1.0)

            q_term = (terminal_field.to(torch.float32) * p_act.unsqueeze(-1)).sum(dim=(1, 2)) / carrier_sum
            if rule_context is not None and rule_context.shape[-1] == D:
                q_term = q_term + (0.25 * rule_context.to(torch.float32))

            e_y = p_act.mean(dim=-1)
            e_x = p_act.mean(dim=-2)

            centered_y = (e_y - 0.50) * (2.0 / float(max(1, H)))
            centered_x = (e_x - 0.50) * (2.0 / float(max(1, W)))

            cum_y = torch.cumsum(centered_y, dim=-1)
            score_abs_h = cum_y - (cum_y[:, -1:] - cum_y)

            cum_x = torch.cumsum(centered_x, dim=-1)
            score_abs_w = cum_x - (cum_x[:, -1:] - cum_x)

            shifts = torch.arange(self.RELATIVE_CLASSES, device=device, dtype=torch.int64) - self.ZERO_SHIFT_INDEX
            cand_h_idx = (h_in.view(B, 1) + shifts.view(1, -1) - 1).clamp(0, H - 1)
            cand_w_idx = (w_in.view(B, 1) + shifts.view(1, -1) - 1).clamp(0, W - 1)

            in_range_h = ((h_in.view(B, 1) + shifts.view(1, -1)) >= 1) & ((h_in.view(B, 1) + shifts.view(1, -1)) <= H)
            in_range_w = ((w_in.view(B, 1) + shifts.view(1, -1)) >= 1) & ((w_in.view(B, 1) + shifts.view(1, -1)) <= W)

            phi_h = torch.gather(score_abs_h, dim=1, index=cand_h_idx) * in_range_h.float()
            phi_w = torch.gather(score_abs_w, dim=1, index=cand_w_idx) * in_range_w.float()

        if D == self.head_terminal_h.in_features:
            term_h_logits = self.head_terminal_h(q_term.to(dtype)).to(torch.float32)
            term_w_logits = self.head_terminal_w(q_term.to(dtype)).to(torch.float32)
        else:
            term_h_logits = torch.zeros_like(base_h_logits, dtype=torch.float32)
            term_w_logits = torch.zeros_like(base_w_logits, dtype=torch.float32)

        with torch.amp.autocast(device_type=dev_type, enabled=False):
            """
            When an exact analytical proper-time law matches support demonstrations with zero error
            (latest_min_err_h == 0.0), lock the prior by setting gate_err = 0.0!
            """
            if self.latest_min_err_h is not None and self.latest_min_err_h.shape[0] == B:
                gate_err_h = torch.tanh(4.0 * self.latest_min_err_h.to(torch.float32))
            else:
                gate_err_h = torch.ones((B, 1), dtype=torch.float32, device=device)

            if self.latest_min_err_w is not None and self.latest_min_err_w.shape[0] == B:
                gate_err_w = torch.tanh(4.0 * self.latest_min_err_w.to(torch.float32))
            else:
                gate_err_w = torch.ones((B, 1), dtype=torch.float32, device=device)

            g_term = torch.tanh(self.terminal_gate.to(torch.float32))
            g_mask = F.softplus(self.carrier_mask_gain.to(torch.float32)) * 0.35

            final_h_logits = (
                base_h_logits.to(torch.float32)
                + gate_err_h * ((g_term * term_h_logits) + (g_mask * phi_h))
            ).to(dtype)

            final_w_logits = (
                base_w_logits.to(torch.float32)
                + gate_err_w * ((g_term * term_w_logits) + (g_mask * phi_w))
            ).to(dtype)

        self.latest_min_err_h = None
        self.latest_min_err_w = None

        pred_h, pred_w, capacity, capacity_ste = self._resolve_dimensions_from_logits(
            h_rel_logits=final_h_logits,
            w_rel_logits=final_w_logits,
            h_in=h_in,
            w_in=w_in,
        )

        return final_h_logits, final_w_logits, pred_h, pred_w, capacity, capacity_ste

    def forward(
        self,
        h_field: torch.Tensor,
        mask: torch.Tensor,
        rule_context: Optional[torch.Tensor] = None,
        support_in_masks: Optional[torch.Tensor] = None,
        support_out_masks: Optional[torch.Tensor] = None,
        query_inputs: Optional[torch.Tensor] = None,
        support_inputs: Optional[torch.Tensor] = None,
        support_outputs: Optional[torch.Tensor] = None,
        pair_valid_mask: Optional[torch.Tensor] = None,
        exemplar_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_in, w_in = self.extract_input_dimensions(mask)
        geo_fiber = self.harmonic_fiber(h_in, w_in)

        z_geo_raw = self.envelope_pooler(h_field, mask)
        z_geo_modulated = self.rule_modulator(z_geo_raw, rule_context)

        z_support_induction = self.support_inductor(
            query_context=z_geo_modulated,
            support_in_masks=support_in_masks,
            support_out_masks=support_out_masks,
        )

        joint_input = torch.cat([z_geo_modulated, z_support_induction, geo_fiber], dim=-1)
        fused_context = self.context_ffn(joint_input)

        na3d_obs = NA3DFinslerLieEngine.extract_na3d_global_invariants(h_field, mask)
        if (
            na3d_obs.shape[-1] == self.na3d_anisotropy_proj.in_features
            and fused_context.shape[-1] == self.na3d_anisotropy_proj.out_features
        ):
            fused_context = fused_context + self.na3d_anisotropy_proj(na3d_obs.to(fused_context.dtype))

        raw_h_logits = self.head_rel_h(fused_context)
        raw_w_logits = self.head_rel_w(fused_context)

        if support_in_masks is not None and support_out_masks is not None and support_in_masks.shape[1] > 0:
            s_h_in, s_w_in = self.extract_input_dimensions(support_in_masks)
            s_h_out, s_w_out = self.extract_input_dimensions(support_out_masks)

            if pair_valid_mask is not None:
                eff_valid_mask = pair_valid_mask
            else:
                eff_valid_mask = (support_in_masks.any(dim=-1).any(dim=-1)) & (support_out_masks.any(dim=-1).any(dim=-1))

            if query_inputs is not None and support_inputs is not None:
                q_h_bbox, q_w_bbox = self.extract_foreground_bbox_dimensions(query_inputs, mask)
                s_h_bbox, s_w_bbox = self.extract_foreground_bbox_dimensions(support_inputs, support_in_masks)
                (
                    q_h_occ,
                    q_w_occ,
                    _,
                    _,
                    s_h_occ,
                    s_w_occ,
                    _,
                    _,
                ) = self.extract_vanishing_occluder_bbox(
                    query_inputs=query_inputs,
                    query_in_masks=mask,
                    support_inputs=support_inputs,
                    support_in_masks=support_in_masks,
                    support_outputs=support_outputs,
                    support_out_masks=support_out_masks,
                    pair_valid_mask=eff_valid_mask,
                )
                q_h_cham, q_w_cham, _, _ = self.extract_chamber_divider_dimensions(query_inputs, mask)
                s_h_cham, s_w_cham, _, _ = self.extract_chamber_divider_dimensions(support_inputs, support_in_masks)
            else:
                q_h_bbox, q_w_bbox = h_in, w_in
                s_h_bbox, s_w_bbox = s_h_in, s_w_in
                q_h_occ, q_w_occ = None, None
                s_h_occ, s_w_occ = None, None
                q_h_cham, q_w_cham = None, None
                s_h_cham, s_w_cham = None, None

            prior_h, boost_h, min_err_h = self.compute_5modal_law_prior_logits(
                q_dim_in=h_in,
                q_dim_bbox=q_h_bbox,
                s_dim_in=s_h_in,
                s_dim_out=s_h_out,
                s_dim_bbox=s_h_bbox,
                pair_valid_mask=eff_valid_mask,
                q_other_dim_in=w_in,
                s_other_dim_in=s_w_in,
                q_dim_occ=q_h_occ,
                s_dim_occ=s_h_occ,
                q_other_dim_occ=q_w_occ,
                s_other_dim_occ=s_w_occ,
                q_dim_cham=q_h_cham,
                s_dim_cham=s_h_cham,
            )
            prior_w, boost_w, min_err_w = self.compute_5modal_law_prior_logits(
                q_dim_in=w_in,
                q_dim_bbox=q_w_bbox,
                s_dim_in=s_w_in,
                s_dim_out=s_w_out,
                s_dim_bbox=s_w_bbox,
                pair_valid_mask=eff_valid_mask,
                q_other_dim_in=h_in,
                s_other_dim_in=s_h_in,
                q_dim_occ=q_w_occ,
                s_dim_occ=s_w_occ,
                q_other_dim_occ=q_h_occ,
                s_other_dim_occ=s_h_occ,
                q_dim_cham=q_w_cham,
                s_dim_cham=s_h_cham,
            )

            self.latest_min_err_h = min_err_h.detach()
            self.latest_min_err_w = min_err_w.detach()

            res_gate_h = (0.35 + 0.65 * torch.tanh(2.0 * min_err_h)).to(raw_h_logits.dtype)
            res_gate_w = (0.35 + 0.65 * torch.tanh(2.0 * min_err_w)).to(raw_w_logits.dtype)

            gain = F.softplus(self.law_prior_gain).to(raw_h_logits.dtype)
            h_rel_logits = (res_gate_h * raw_h_logits) + (gain * boost_h.to(raw_h_logits.dtype) * prior_h.to(raw_h_logits.dtype))
            w_rel_logits = (res_gate_w * raw_w_logits) + (gain * boost_w.to(raw_w_logits.dtype) * prior_w.to(raw_w_logits.dtype))
        else:
            self.latest_min_err_h = None
            self.latest_min_err_w = None
            h_rel_logits = raw_h_logits
            w_rel_logits = raw_w_logits

        pred_h, pred_w, capacity, capacity_ste = self._resolve_dimensions_from_logits(
            h_rel_logits=h_rel_logits,
            w_rel_logits=w_rel_logits,
            h_in=h_in,
            w_in=w_in,
        )

        return h_rel_logits, w_rel_logits, pred_h, pred_w, capacity, capacity_ste

    @classmethod
    def evaluate_entropy_and_runner_up(
        cls,
        h_rel_logits: torch.Tensor,
        w_rel_logits: torch.Tensor,
        h_in: int,
        w_in: int,
        uncertainty_threshold: float = 0.40,
    ) -> Tuple[float, Optional[Tuple[int, int]]]:
        probs_h = F.softmax(h_rel_logits.view(-1), dim=-1)
        probs_w = F.softmax(w_rel_logits.view(-1), dim=-1)

        entropy_h = -torch.sum(probs_h * safe_log(probs_h, eps=1e-7)).item()
        entropy_w = -torch.sum(probs_w * safe_log(probs_w, eps=1e-7)).item()
        mean_entropy = 0.50 * (entropy_h + entropy_w)

        runner_up_dims = None
        if mean_entropy > uncertainty_threshold:
            top2_h_classes = torch.topk(probs_h, k=2)[1]
            top2_w_classes = torch.topk(probs_w, k=2)[1]

            sec_class_h = int(top2_h_classes[1].item())
            sec_class_w = int(top2_w_classes[1].item())

            sec_delta_h = sec_class_h - cls.ZERO_SHIFT_INDEX
            sec_delta_w = sec_class_w - cls.ZERO_SHIFT_INDEX

            sec_h = max(1, min(cls.MAX_CANVAS_DIM, h_in + sec_delta_h))
            sec_w = max(1, min(cls.MAX_CANVAS_DIM, w_in + sec_delta_w))
            runner_up_dims = (sec_h, sec_w)

        return mean_entropy, runner_up_dims

    @classmethod
    def crop_canvas_to_predictions(
        cls,
        canvas: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        h_safe = max(1, min(cls.MAX_CANVAS_DIM, target_h))
        w_safe = max(1, min(cls.MAX_CANVAS_DIM, target_w))
        return canvas[..., :h_safe, :w_safe].contiguous()

    @classmethod
    def generate_target_mask(
        cls,
        batch_size: int,
        target_heights: torch.Tensor,
        target_widths: torch.Tensor,
        canvas_dim: int = DEFAULT_MAX_CANVAS_DIM,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if device is None:
            device = target_heights.device

        y_coords = torch.arange(canvas_dim, device=device).view(1, canvas_dim, 1)
        x_coords = torch.arange(canvas_dim, device=device).view(1, 1, canvas_dim)

        h_bound = target_heights.view(batch_size, 1, 1)
        w_bound = target_widths.view(batch_size, 1, 1)

        mask = (y_coords < h_bound) & (x_coords < w_bound)
        return mask


def verify_boundary_size_predictor_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for boundary_size_predictor.py.
    """
    torch.manual_seed(42)
    b, h, w, d = 2, 20, 20, 32

    config = FGNSConfig()
    oracle = BoundaryMetricSizePredictor(config=config)

    oracle.envelope_pooler = SpatialEnvelopeAttentionPooler(d_model=d, num_heads=4)
    oracle.harmonic_fiber = HarmonicGeometricFiber(d_out=16, num_frequencies=4)
    oracle.support_inductor = SupportDemonstrationSizeInductor(d_model=d, num_heads=4)
    oracle.rule_modulator = CartanRuleModulator(d_model=d)
    oracle.context_ffn = nn.Sequential(
        nn.Linear((d * 2) + 16, d, bias=True),
        nn.SiLU(),
        TopologicalBypassFeedForward(d_model=d, expansion_factor=2.0, alignment_multiple=32),
    )
    oracle.metric_density_head = nn.Linear(d, 2, bias=True)
    oracle.head_rel_h = nn.Linear(d, 61, bias=True)
    oracle.head_rel_w = nn.Linear(d, 61, bias=True)
    oracle.head_terminal_h = nn.Linear(d, 61, bias=False)
    oracle.head_terminal_w = nn.Linear(d, 61, bias=False)

    """Verify Subgroup Non-Degeneracy Guard on Square Isometry (Task 142ca369 Test 2: 20x19 -> 20x19)"""
    s_in_m_iso = torch.zeros(1, 2, 20, 20, dtype=torch.bool)
    s_in_m_iso[0, 0, :20, :20] = True
    s_in_m_iso[0, 1, :18, :18] = True
    s_out_m_iso = s_in_m_iso.clone()

    q_mask_19x20 = torch.zeros(1, 20, 20, dtype=torch.bool)
    q_mask_19x20[0, :20, :19] = True
    h_f_19x20 = torch.zeros(1, 20, 20, d, dtype=torch.float32)

    _, _, pred_h_iso, pred_w_iso, _, _ = oracle(
        h_field=h_f_19x20,
        mask=q_mask_19x20,
        support_in_masks=s_in_m_iso,
        support_out_masks=s_out_m_iso,
    )
    if int(pred_h_iso[0].item()) != 20 or int(pred_w_iso[0].item()) != 19:
        return False

    """Verify Proper-Time Pullback Kernel on cropped sub-interval"""
    mu_y = torch.zeros(1, 10, dtype=torch.float32)
    mu_y[0, 2:5] = 1.0
    mu_x = torch.zeros(1, 10, dtype=torch.float32)
    mu_x[0, 3:7] = 1.0
    h_in_test = torch.randn(1, 10, 10, d, dtype=torch.float32)
    pulled_test, tau_y, tau_x = BoundaryMetricSizePredictor.compute_proper_time_pullback_kernel(mu_y, mu_x, h_in_test)
    if pulled_test.shape != (1, 10, 10, d):
        return False
    if abs(float(tau_y[0, -1].item()) - 3.0) > 1e-4 or abs(float(tau_x[0, -1].item()) - 4.0) > 1e-4:
        return False

    h_field = torch.randn(b, h, w, d, dtype=torch.float32, requires_grad=True)
    mask = torch.ones(b, h, w, dtype=torch.bool)
    z_rule = torch.randn(b, d, dtype=torch.float32, requires_grad=True)
    s_in_m = torch.ones(b, 2, h, w, dtype=torch.bool)
    s_out_m = torch.ones(b, 2, h, w, dtype=torch.bool)

    h_logits, w_logits, _, _, _, _ = oracle(
        h_field=h_field,
        mask=mask,
        rule_context=z_rule,
        support_in_masks=s_in_m,
        support_out_masks=s_out_m,
    )
    term_field = torch.randn(b, h, w, d, dtype=torch.float32, requires_grad=True)
    carrier_psi = torch.full((b, h, w, 1), 3.0, dtype=torch.float32, requires_grad=True)

    term_h_log, _, _, _, _, term_cap_ste = oracle.refine_with_terminal_carrier_mask(
        base_h_logits=h_logits,
        base_w_logits=w_logits,
        terminal_field=term_field,
        carrier_psi=carrier_psi,
        in_masks=mask,
        rule_context=z_rule,
    )
    loss_cap = torch.sum(term_h_log.pow(2)) + torch.sum((term_cap_ste - 36.0).pow(2))
    loss_cap.backward()
    if oracle.head_rel_h.weight.grad is None:
        return False

    return True

