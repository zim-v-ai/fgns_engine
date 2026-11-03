"""
Location: // fgns_engine/fgns/models/size_predictor/boundary_size_predictor.py
Description: Universal 1-Jet Stokes & Coupled Boltzmann-Woodbury Dimension Predictor
             (BoundaryMetricSizePredictor - FGNS v4.4.0 Academic Standard):
               1. Vectorized Per-Color Extremal Sub-Mask Extractor (extract_extremal_color_bbox_dimensions):
                  extracts both Largest-Object BBox (L_max_col) and Smallest-Pattern BBox (L_min_col)
                  in a single 4D one-hot pass with zero loops!
               2. Coupled Boltzmann-Gibbs & Woodbury-Tikhonov 1-Jet Operator: each geometric law votes
                  at its exact integer displacement peak in Z_61 (preventing multi-law averaging blur),
                  while the continuous Woodbury-Sherman-Morrison integral handles composite warps!
               3. Strict Occam Prior Hierarchy: Transpose (W_in) is suppressed when inputs are square
                  or isometric, and Occluder crops require verified color annihilation.
Language: Python 3.12+ / PyTorch 2.5+ / Autonomous Differential Geometry sm_120 Aligned
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import math
from typing import Final, Optional, Tuple, cast

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
from fgns.primitives.swiglu import TopologicalBypassFeedForward
from fgns.types import FrameBoundaryState

DEFAULT_ZERO_SHIFT_INDEX: Final[int] = 30
DEFAULT_RELATIVE_CLASSES: Final[int] = 61
DEFAULT_MAX_CANVAS_DIM: Final[int] = 30
WOODBURY_TIKHONOV_REG: Final[float] = 1e-3


class BoundaryMetricSizePredictor(nn.Module):
    """
    Universal 1-Jet Stokes & Coupled Boltzmann-Woodbury Integrator for Canvas Dimension Prediction.
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
            TopologicalBypassFeedForward(d_model=d_model, expansion_factor=2.0, alignment_multiple=32),
        )

        self.metric_density_head = nn.Linear(d_model, 2, bias=True)
        self.na3d_anisotropy_proj = nn.Linear(24, d_model, bias=False)
        self.head_rel_h = nn.Linear(d_model, self.RELATIVE_CLASSES, bias=True)
        self.head_rel_w = nn.Linear(d_model, self.RELATIVE_CLASSES, bias=True)
        self.law_prior_gain = nn.Parameter(torch.tensor(5.00, dtype=torch.float32))

        self.head_terminal_h = nn.Linear(d_model, self.RELATIVE_CLASSES, bias=False)
        self.head_terminal_w = nn.Linear(d_model, self.RELATIVE_CLASSES, bias=False)
        self.terminal_gate = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.carrier_mask_gain = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))

        self.register_buffer("class_indices", torch.arange(self.RELATIVE_CLASSES, dtype=torch.float32), persistent=False)
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
        m_f = mask.float()
        return (
            m_f.max(dim=-1)[0].sum(dim=-1).to(torch.int64).clamp_min(1),
            m_f.max(dim=-2)[0].sum(dim=-1).to(torch.int64).clamp_min(1),
        )

    @classmethod
    def compute_proper_time_pullback_kernel(
        cls,
        mu_y: torch.Tensor,
        mu_x: torch.Tensor,
        h_field: torch.Tensor,
        sigma_tau: float = 0.45,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N, H, W, _ = h_field.shape
        device, dtype = h_field.device, h_field.dtype

        mu_y_f = mu_y.to(torch.float32).clamp_min(0.0).view(N, H)
        mu_x_f = mu_x.to(torch.float32).clamp_min(0.0).view(N, W)
        tau_y = torch.cumsum(mu_y_f, dim=-1)
        tau_x = torch.cumsum(mu_x_f, dim=-1)

        cy = (tau_y - 0.50 * mu_y_f).unsqueeze(1)
        cx = (tau_x - 0.50 * mu_x_f).unsqueeze(1)
        ty = (torch.arange(H, device=device, dtype=torch.float32) + 0.50).view(1, H, 1)
        tx = (torch.arange(W, device=device, dtype=torch.float32) + 0.50).view(1, W, 1)

        inv_2s2 = 1.0 / (2.0 * max(float(sigma_tau), 0.10) ** 2)
        ky = mu_y_f.unsqueeze(1) * torch.exp(-((cy - ty).pow(2)) * inv_2s2)
        kx = mu_x_f.unsqueeze(1) * torch.exp(-((cx - tx).pow(2)) * inv_2s2)
        ky = ky / ky.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        kx = kx / kx.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        pulled_y = torch.einsum("nyu,nuwd->nywd", ky, h_field.to(torch.float32))
        pulled_yx = torch.einsum("nxv,nyvd->nyxd", kx, pulled_y)
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
        m_field = torch.sigmoid(torch.clamp(carrier_psi.view(B, H, W).to(torch.float32), -12.0, 12.0)) * active_mask.float()

        st = max(float(tau_stokes), 1e-3)
        m_y = (st * torch.logsumexp(m_field / st, dim=-1) - st * math.log(float(max(1, W)))).clamp(0.0, 1.0)
        m_x = (st * torch.logsumexp(m_field / st, dim=-2) - st * math.log(float(max(1, H)))).clamp(0.0, 1.0)

        my_p = F.pad(m_y, (1, 1), mode="constant", value=0.0)
        mx_p = F.pad(m_x, (1, 1), mode="constant", value=0.0)
        fy_pos, fy_neg = F.relu(m_y - my_p[:, :-2]), F.relu(m_y - my_p[:, 2:])
        fx_pos, fx_neg = F.relu(m_x - mx_p[:, :-2]), F.relu(m_x - mx_p[:, 2:])

        yc = torch.arange(H, device=device, dtype=torch.float32).unsqueeze(0)
        xc = torch.arange(W, device=device, dtype=torch.float32).unsqueeze(0)

        return FrameBoundaryState(
            y_min=torch.sum(yc * fy_pos, dim=-1) / (torch.sum(fy_pos, dim=-1) + 1e-6),
            y_max=torch.sum(yc * fy_neg, dim=-1) / (torch.sum(fy_neg, dim=-1) + 1e-6),
            x_min=torch.sum(xc * fx_pos, dim=-1) / (torch.sum(fx_pos, dim=-1) + 1e-6),
            x_max=torch.sum(xc * fx_neg, dim=-1) / (torch.sum(fx_neg, dim=-1) + 1e-6),
            tau_y=torch.cumsum(m_y, dim=-1),
            tau_x=torch.cumsum(m_x, dim=-1),
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

        yi = torch.arange(H, device=device, dtype=torch.int64).view(*([1] * (target_mask_bool.ndim - 2)), H, 1)
        xi = torch.arange(W, device=device, dtype=torch.int64).view(*([1] * (target_mask_bool.ndim - 2)), 1, W)

        ymin = torch.where(target_mask_bool, yi, torch.full_like(yi, H)).amin(dim=(-2, -1))
        ymax = torch.where(target_mask_bool, yi, torch.full_like(yi, -1)).amax(dim=(-2, -1))
        xmin = torch.where(target_mask_bool, xi, torch.full_like(xi, W)).amin(dim=(-2, -1))
        xmax = torch.where(target_mask_bool, xi, torch.full_like(xi, -1)).amax(dim=(-2, -1))

        h_in, w_in = cls.extract_input_dimensions(fallback_mask)
        h_box = torch.where(has_any, (ymax - ymin + 1).clamp(1, cls.MAX_CANVAS_DIM), h_in)
        w_box = torch.where(has_any, (xmax - xmin + 1).clamp(1, cls.MAX_CANVAS_DIM), w_in)
        return (
            h_box,
            w_box,
            torch.where(has_any, ymin.clamp(0, H - 1), torch.zeros_like(h_in)),
            torch.where(has_any, xmin.clamp(0, W - 1), torch.zeros_like(w_in)),
        )

    @classmethod
    def extract_foreground_bbox_dimensions(
        cls,
        grid: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_box, w_box, _, _ = cls._compute_mask_bbox_dims((grid > 0) & (grid < 10) & mask, mask)
        return h_box, w_box

    @classmethod
    def extract_extremal_color_bbox_dimensions(
        cls,
        grid: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vectorized Per-Color Sub-Mask Bounding Box Extractor:
        Extracts the bounding box dimensions (H_max_col, W_max_col) of the largest non-background
        sub-canvas color object and (H_min_col, W_min_col) of the smallest non-trivial color object!
        """
        orig_batch = grid.shape[:-2]
        H, W = grid.shape[-2], grid.shape[-1]
        device = grid.device

        g_f = grid.reshape(-1, H, W)
        m_f = mask.reshape(-1, H, W)
        N = g_f.shape[0]
        h_in, w_in = cls.extract_input_dimensions(m_f)

        oh = (F.one_hot(g_f.clamp(0, 9), 10).bool()) & m_f.unsqueeze(-1)
        oh[:, :, :, 0] = False

        yi = torch.arange(H, device=device, dtype=torch.int64).view(1, H, 1, 1)
        xi = torch.arange(W, device=device, dtype=torch.int64).view(1, 1, W, 1)

        ymin = torch.where(oh, yi, torch.full_like(yi, H)).amin(dim=(1, 2))
        ymax = torch.where(oh, yi, torch.full_like(yi, -1)).amax(dim=(1, 2))
        xmin = torch.where(oh, xi, torch.full_like(xi, W)).amin(dim=(1, 2))
        xmax = torch.where(oh, xi, torch.full_like(xi, -1)).amax(dim=(1, 2))

        hc = (ymax - ymin + 1).clamp(1, cls.MAX_CANVAS_DIM)
        wc = (xmax - xmin + 1).clamp(1, cls.MAX_CANVAS_DIM)
        mass_c = oh.float().sum(dim=(1, 2))

        is_sub_canvas = (mass_c >= 2.0) & ((hc < h_in.unsqueeze(-1)) | (wc < w_in.unsqueeze(-1)))
        valid_c = torch.where(is_sub_canvas, mass_c, torch.zeros_like(mass_c))
        has_sub = is_sub_canvas.any(dim=-1)

        max_idx = torch.argmax(valid_c, dim=-1, keepdim=True)
        h_max_c = torch.where(has_sub, torch.gather(hc, 1, max_idx).squeeze(-1), h_in)
        w_max_c = torch.where(has_sub, torch.gather(wc, 1, max_idx).squeeze(-1), w_in)

        min_cand = torch.where(is_sub_canvas, mass_c, torch.full_like(mass_c, 1e9))
        min_idx = torch.argmin(min_cand, dim=-1, keepdim=True)
        h_min_c = torch.where(has_sub, torch.gather(hc, 1, min_idx).squeeze(-1), h_in)
        w_min_c = torch.where(has_sub, torch.gather(wc, 1, min_idx).squeeze(-1), w_in)

        return (
            h_max_c.view(*orig_batch),
            w_max_c.view(*orig_batch),
            h_min_c.view(*orig_batch),
            w_min_c.view(*orig_batch),
        )

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
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        B, K, _, _ = support_inputs.shape
        v_m = (
            pair_valid_mask.to(torch.float32).unsqueeze(-1)
            if pair_valid_mask is not None
            else support_in_masks.any(dim=-1).any(dim=-1).to(torch.float32).unsqueeze(-1)
        )
        v_denom = v_m.sum(dim=1).clamp_min(1.0)

        q_oh = F.one_hot(query_inputs.clamp(0, 9), 10).float() * query_in_masks.float().unsqueeze(-1)
        s_in_oh = F.one_hot(support_inputs.clamp(0, 9), 10).float() * support_in_masks.float().unsqueeze(-1)

        q_cnt = q_oh.sum(dim=(1, 2))
        s_in_cnt = s_in_oh.sum(dim=(2, 3)) * v_m
        q_has = q_cnt > 0.5
        s_has_rate = ((s_in_cnt > 0.5).float() * v_m).sum(dim=1) / v_denom

        if support_outputs is not None and support_out_masks is not None:
            s_out_oh = F.one_hot(support_outputs.clamp(0, 9), 10).float() * support_out_masks.float().unsqueeze(-1)
            s_out_cnt = s_out_oh.sum(dim=(2, 3)) * v_m
            s_van_rate = (((s_out_cnt == 0.0) & (s_in_cnt > 0.5)).float() * v_m).sum(dim=1) / v_denom
            valid_c = q_has & (s_has_rate > 0.5) & (s_van_rate > 0.5)
        else:
            s_van_rate = torch.zeros_like(q_cnt)
            valid_c = q_has & (s_has_rate > 0.5)

        valid_c[:, 0] = False
        scores = torch.where(valid_c, (10.0 * s_van_rate) + (1.0 / (q_cnt + 1.0)), torch.full_like(q_cnt, -1e9))
        max_sc, occ_col = torch.max(scores, dim=-1)
        has_occ = max_sc > 0.0

        q_occ = (query_inputs == occ_col.view(B, 1, 1)) & query_in_masks & has_occ.view(B, 1, 1)
        s_occ = (support_inputs == occ_col.view(B, 1, 1, 1)) & support_in_masks & has_occ.view(B, 1, 1, 1)

        qh, qw, qy0, qx0 = cls._compute_mask_bbox_dims(q_occ, query_in_masks)
        sh, sw, sy0, sx0 = cls._compute_mask_bbox_dims(s_occ, support_in_masks)
        return qh, qw, qy0, qx0, sh, sw, sy0, sx0

    @classmethod
    def extract_chamber_divider_dimensions(
        cls,
        grid: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_batch = grid.shape[:-2]
        H, W = grid.shape[-2], grid.shape[-1]
        device = grid.device

        g_f = grid.reshape(-1, H, W)
        m_f = mask.reshape(-1, H, W)
        N = g_f.shape[0]
        h_in, w_in = cls.extract_input_dimensions(m_f)

        yi = torch.arange(H, device=device, dtype=torch.int64).view(1, H)
        xi = torch.arange(W, device=device, dtype=torch.int64).view(1, W)

        row_u = (((g_f == g_f[:, :, 0:1]) & (g_f > 0) & m_f).sum(dim=-1) == w_in.unsqueeze(-1)) & (yi >= 2) & (yi <= h_in.unsqueeze(-1) - 3)
        col_u = (((g_f == g_f[:, 0:1, :]) & (g_f > 0) & m_f).sum(dim=-2) == h_in.unsqueeze(-1)) & (xi >= 2) & (xi <= w_in.unsqueeze(-1) - 3)

        row_div = row_u & (~row_u.all(dim=-1, keepdim=True))
        col_div = col_u & (~col_u.all(dim=-1, keepdim=True))

        nd_y, nd_x = row_div.sum(dim=-1), col_div.sum(dim=-1)
        f_dy = torch.where(row_div, yi, torch.full_like(yi, H)).amin(dim=-1)
        f_dx = torch.where(col_div, xi, torch.full_like(xi, W)).amin(dim=-1)

        h_cham = torch.where(
            nd_y == 1,
            torch.maximum(f_dy.clamp_min(1), (h_in - 1 - f_dy).clamp_min(1)),
            torch.where(nd_y > 0, torch.div(h_in - nd_y, nd_y + 1, rounding_mode="floor").clamp_min(1), h_in),
        )
        w_cham = torch.where(
            nd_x == 1,
            torch.maximum(f_dx.clamp_min(1), (w_in - 1 - f_dx).clamp_min(1)),
            torch.where(nd_x > 0, torch.div(w_in - nd_x, nd_x + 1, rounding_mode="floor").clamp_min(1), w_in),
        )

        is_fg = (g_f > 0) & (g_f < 10) & m_f
        xg2, yg2 = xi.unsqueeze(1).expand(N, H, W), yi.unsqueeze(-1).expand(N, H, W)
        l_fg = (is_fg & (xg2 < f_dx.view(N, 1, 1))).sum(dim=(-2, -1))
        r_fg = (is_fg & (xg2 > f_dx.view(N, 1, 1))).sum(dim=(-2, -1))
        t_fg = (is_fg & (yg2 < f_dy.view(N, 1, 1))).sum(dim=(-2, -1))
        b_fg = (is_fg & (yg2 > f_dy.view(N, 1, 1))).sum(dim=(-2, -1))

        x0_c = torch.where((nd_x == 1) & (r_fg > 0) & (r_fg < l_fg), (f_dx + 1).clamp(0, W - 1), torch.zeros_like(w_in))
        y0_c = torch.where((nd_y == 1) & (b_fg > 0) & (b_fg < t_fg), (f_dy + 1).clamp(0, H - 1), torch.zeros_like(h_in))
        return h_cham.view(*orig_batch), w_cham.view(*orig_batch), y0_c.view(*orig_batch), x0_c.view(*orig_batch)

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
            ph = F.softmax(torch.clamp(h_rel_logits.to(torch.float32), -25.0, 25.0), dim=-1)
            pw = F.softmax(torch.clamp(w_rel_logits.to(torch.float32), -25.0, 25.0), dim=-1)

            shifts = torch.arange(cls.RELATIVE_CLASSES, device=device, dtype=torch.int64) - cls.ZERO_SHIFT_INDEX
            ah = (h_in.view(B, 1) + shifts.view(1, -1)).clamp(1, canvas_dim)
            aw = (w_in.view(B, 1) + shifts.view(1, -1)).clamp(1, canvas_dim)

            cr = torch.arange(canvas_dim, device=device, dtype=torch.int64).view(1, canvas_dim, 1)
            sh = torch.sum(ph.unsqueeze(1) * (ah.unsqueeze(1) > cr).float(), dim=-1)
            sw = torch.sum(pw.unsqueeze(1) * (aw.unsqueeze(1) > cr).float(), dim=-1)

            p_env = (sh.unsqueeze(-1) * sw.unsqueeze(1)).clamp(1e-4, 1.0 - 1e-4)
            return p_env, torch.log(p_env) - torch.log(1.0 - p_env)

    def compute_universal_proper_time_prior(
        self,
        q_phi: torch.Tensor,
        s_phi: torch.Tensor,
        s_dim_out: torch.Tensor,
        pair_valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Coupled Boltzmann-Gibbs & Woodbury-Tikhonov 1-Jet Stokes Selector:
        Evaluates discrete geometric hypotheses at their exact integer peaks in Z_61 (preventing
        multi-law averaging blur between correlated laws) and blends with the Woodbury-Tikhonov
        proper-time integral for composite warps!
        """
        B, K, M_raw = s_phi.shape
        device = q_phi.device
        dev_type = "cuda" if q_phi.is_cuda else "cpu"

        with torch.amp.autocast(device_type=dev_type, enabled=False):
            v_mask = pair_valid_mask.to(torch.float32)
            v_3d = v_mask.unsqueeze(-1)
            w_conj = v_mask / v_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            w_3d = w_conj.unsqueeze(-1)

            s_f = s_phi.to(torch.float32).clamp_min(1.0)
            q_f = q_phi.to(torch.float32).clamp_min(1.0)
            s_out_f = s_dim_out.to(torch.float32).clamp_min(1.0)

            s_in = s_f[:, :, 0]
            q_in = q_f[:, 0]

            s_hb = s_f[:, :, 1]
            q_hb = q_f[:, 1]
            s_hc = s_f[:, :, 2]
            q_hc = q_f[:, 2]
            s_ho = s_f[:, :, 3]
            q_ho = q_f[:, 3]
            s_other = s_f[:, :, 4]
            q_other = q_f[:, 4]
            s_wo_tr = s_f[:, :, 5]
            q_wo_tr = q_f[:, 5]

            if M_raw >= 8:
                s_hmax_c = s_f[:, :, 6]
                q_hmax_c = q_f[:, 6]
                s_hmin_c = s_f[:, :, 7]
                q_hmin_c = q_f[:, 7]
            else:
                s_hmax_c = s_hb
                q_hmax_c = q_hb
                s_hmin_c = s_hb
                q_hmin_c = q_hb

            """1. Additive / Isometric shift: L_out = L_in + delta"""
            mean_d_in = torch.round((w_conj * (s_out_f - s_in)).sum(dim=1, keepdim=True))
            err_1 = (w_conj * torch.abs(s_out_f - (s_in + mean_d_in))).sum(dim=1)
            cand_q_1 = (q_in + mean_d_in.squeeze(-1)).clamp(1.0, 30.0)

            """2. Multiplicative homothety: L_out = round(r * L_in)"""
            mean_ratio = (w_conj * (s_out_f / s_in)).sum(dim=1, keepdim=True)
            err_2 = (w_conj * torch.abs(s_out_f - torch.round(s_in * mean_ratio))).sum(dim=1) + 0.01
            cand_q_2 = torch.round(q_in * mean_ratio.squeeze(-1)).clamp(1.0, 30.0)

            """3. Constant target dimension: L_out = C_out"""
            mean_const = torch.round((w_conj * s_out_f).sum(dim=1, keepdim=True))
            err_3 = (w_conj * torch.abs(s_out_f - mean_const)).sum(dim=1) + 0.02
            cand_q_3 = mean_const.squeeze(-1).clamp(1.0, 30.0)

            """4. Global Foreground BBox + offset: L_out = L_bbox + delta_bbox"""
            mean_d_hb = torch.round((w_conj * (s_out_f - s_hb)).sum(dim=1, keepdim=True))
            active_hb = ((torch.abs(s_hb - s_in) * w_conj).sum(dim=1) > 0.1) | (torch.abs(q_hb - q_in) > 0.1)
            err_4 = torch.where(
                active_hb,
                (w_conj * torch.abs(s_out_f - (s_hb + mean_d_hb))).sum(dim=1) + 0.03,
                torch.full_like(err_1, 50.0),
            )
            cand_q_4 = (q_hb + mean_d_hb.squeeze(-1)).clamp(1.0, 30.0)

            """5. Chamber Divider Dimension: L_out = L_cham"""
            active_hc = ((torch.abs(s_hc - s_in) * w_conj).sum(dim=1) > 0.1) | (torch.abs(q_hc - q_in) > 0.1)
            err_5 = torch.where(
                active_hc,
                (w_conj * torch.abs(s_out_f - s_hc)).sum(dim=1) + 0.025,
                torch.full_like(err_1, 50.0),
            )
            cand_q_5 = q_hc.clamp(1.0, 30.0)

            """6. Vanishing Occluder BBox: L_out = L_occ"""
            active_ho = ((torch.abs(s_ho - s_in) * w_conj).sum(dim=1) > 0.1) & (torch.abs(q_ho - q_in) > 0.1)
            err_6 = torch.where(
                active_ho,
                (w_conj * torch.abs(s_out_f - s_ho)).sum(dim=1) + 0.03,
                torch.full_like(err_1, 50.0),
            )
            cand_q_6 = q_ho.clamp(1.0, 30.0)

            """7. Transposed Vanishing Occluder BBox: L_out = W_occ"""
            active_wo_tr = ((torch.abs(s_wo_tr - s_other) * w_conj).sum(dim=1) > 0.1) & (torch.abs(q_wo_tr - q_other) > 0.1)
            err_7 = torch.where(
                active_wo_tr,
                (w_conj * torch.abs(s_out_f - s_wo_tr)).sum(dim=1) + 0.035,
                torch.full_like(err_1, 50.0),
            )
            cand_q_7 = q_wo_tr.clamp(1.0, 30.0)

            """8. Largest Single-Color Sub-Mask BBox: L_out = L_max_col"""
            mean_d_maxc = torch.round((w_conj * (s_out_f - s_hmax_c)).sum(dim=1, keepdim=True))
            active_maxc = ((torch.abs(s_hmax_c - s_in) * w_conj).sum(dim=1) > 0.1) | (torch.abs(q_hmax_c - q_in) > 0.1)
            err_8 = torch.where(
                active_maxc,
                (w_conj * torch.abs(s_out_f - (s_hmax_c + mean_d_maxc))).sum(dim=1) + 0.04,
                torch.full_like(err_1, 50.0),
            )
            cand_q_8 = (q_hmax_c + mean_d_maxc.squeeze(-1)).clamp(1.0, 30.0)

            """9. Smallest Non-Trivial Single-Color Sub-Mask BBox: L_out = L_min_col"""
            active_minc = ((torch.abs(s_hmin_c - s_in) * w_conj).sum(dim=1) > 0.1) | (torch.abs(q_hmin_c - q_in) > 0.1)
            err_9 = torch.where(
                active_minc,
                (w_conj * torch.abs(s_out_f - s_hmin_c)).sum(dim=1) + 0.045,
                torch.full_like(err_1, 50.0),
            )
            cand_q_9 = q_hmin_c.clamp(1.0, 30.0)

            """10. Transposed Input Dimension: L_out = W_in (strictly when non-isometric & non-square!)"""
            non_square_s = ((torch.abs(s_in - s_other) * w_conj).sum(dim=1) > 0.1) & (err_1 > 0.1)
            err_10 = torch.where(
                non_square_s,
                (w_conj * torch.abs(s_out_f - s_other)).sum(dim=1) + 0.05,
                torch.full_like(err_1, 50.0),
            )
            cand_q_10 = q_other.clamp(1.0, 30.0)

            """11. Continuous Woodbury-Tikhonov 1-Jet Stokes Integral"""
            s_l0, q_l0 = s_f[:, :, 0:1], q_f[:, 0:1]
            delta_s = (s_out_f.unsqueeze(-1) - s_l0) * v_3d
            mean_out_s = (s_out_f.unsqueeze(-1) * w_3d).sum(dim=1, keepdim=True)

            psi_s = torch.cat(
                [
                    torch.ones_like(s_l0),
                    s_f[:, :, 1:6] - s_l0,
                    s_l0 * (torch.log(s_l0) / 3.40),
                    mean_out_s.expand(B, K, 1) - s_l0,
                ],
                dim=-1,
            ) * v_3d

            psi_q = torch.cat(
                [
                    torch.ones_like(q_l0),
                    q_f[:, 1:6] - q_l0,
                    q_l0 * (torch.log(q_l0) / 3.40),
                    mean_out_s.squeeze(1) - q_l0,
                ],
                dim=-1,
            )

            col_norm_sq = (psi_s.pow(2) * w_3d).sum(dim=1).clamp_min(1e-4)
            beta_opt = ((psi_s * delta_s * w_3d).sum(dim=1) / col_norm_sq).clamp(-2.0, 2.0)
            res_col = (torch.abs(delta_s - (psi_s * beta_opt.unsqueeze(1))) * w_3d).sum(dim=1)
            signal_mag = 0.50 * ((torch.abs(psi_s) * w_3d).sum(dim=1) + torch.abs(psi_q))
            fisher_active = torch.tanh(signal_mag * 2.0)
            fisher_active[:, 0] = 1.0

            g_fisher = (F.softmax(-18.0 * (res_col + 4.0 * (1.0 - fisher_active)), dim=-1) * fisher_active) / col_norm_sq.clamp_min(1.0)
            psi_s_g = psi_s * g_fisher.unsqueeze(1)
            gram_k = 0.50 * (torch.bmm(psi_s_g, psi_s.transpose(1, 2)) + torch.bmm(psi_s, psi_s_g.transpose(1, 2)))
            eye_k = torch.eye(K, dtype=torch.float32, device=device).unsqueeze(0).expand(B, K, K)
            gram_spd = gram_k + (WOODBURY_TIKHONOV_REG * eye_k)
            trace_spd = torch.diagonal(gram_spd, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1e-4)
            gram_inv = eye_k / trace_spd
            two_eye = 2.0 * eye_k
            for _ in range(6):
                gram_inv = torch.bmm(gram_inv, two_eye - torch.bmm(gram_spd, gram_inv))

            w_woodbury = torch.bmm(psi_s_g.transpose(1, 2), torch.bmm(gram_inv, delta_s)).squeeze(-1)
            pred_s_wood = s_in + torch.round(torch.sum(psi_s * w_woodbury.unsqueeze(1), dim=-1))
            err_11 = (w_conj * torch.abs(s_out_f - pred_s_wood)).sum(dim=1) + 0.06
            cand_q_11 = (q_in + torch.round(torch.sum(psi_q * w_woodbury, dim=-1))).clamp(1.0, 30.0)

            all_errs = torch.stack(
                [err_1, err_2, err_3, err_4, err_5, err_6, err_7, err_8, err_9, err_10, err_11],
                dim=-1,
            )
            all_cands_q = torch.stack(
                [cand_q_1, cand_q_2, cand_q_3, cand_q_4, cand_q_5, cand_q_6, cand_q_7, cand_q_8, cand_q_9, cand_q_10, cand_q_11],
                dim=-1,
            )

            min_err = torch.min(all_errs, dim=-1, keepdim=True)[0]
            law_weights = F.softmax(-18.0 * (all_errs - min_err), dim=-1)

            cand_shifts = (float(self.ZERO_SHIFT_INDEX) + (all_cands_q - q_in.unsqueeze(-1))).clamp(0.0, 60.0)
            classes_grid = cast(torch.Tensor, self.class_indices).to(device=device).view(1, 1, self.RELATIVE_CLASSES)
            gauss_peaks = torch.exp(-0.50 * ((classes_grid - cand_shifts.unsqueeze(-1)) / 0.50).pow(2))

            prior_logits = torch.sum(law_weights.unsqueeze(-1) * gauss_peaks, dim=1)
            condensation_boost = 1.0 + (2.00 * torch.exp(-4.0 * min_err))

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
            hl = torch.clamp(h_rel_logits.to(torch.float32), -25.0, 25.0)
            wl = torch.clamp(w_rel_logits.to(torch.float32), -25.0, 25.0)

            dh_d = torch.argmax(hl, dim=-1).to(torch.int64) - self.ZERO_SHIFT_INDEX
            dw_d = torch.argmax(wl, dim=-1).to(torch.int64) - self.ZERO_SHIFT_INDEX

            pred_h = torch.clamp(h_in + dh_d, 1, self.MAX_CANVAS_DIM)
            pred_w = torch.clamp(w_in + dw_d, 1, self.MAX_CANVAS_DIM)
            capacity = pred_h * pred_w

            ci = cast(torch.Tensor, self.class_indices).to(device=device)
            dh_c = torch.sum(F.softmax(hl, dim=-1) * ci, dim=-1) - float(self.ZERO_SHIFT_INDEX)
            dw_c = torch.sum(F.softmax(wl, dim=-1) * ci, dim=-1) - float(self.ZERO_SHIFT_INDEX)

            hc = torch.clamp(h_in.float() + dh_c, 1.0, float(self.MAX_CANVAS_DIM))
            wc = torch.clamp(w_in.float() + dw_c, 1.0, float(self.MAX_CANVAS_DIM))
            cap_c = hc * wc
            return pred_h, pred_w, capacity, cap_c + (capacity.float() - cap_c).detach()

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
            p_act = torch.sigmoid(torch.clamp(carrier_psi.view(B, H, W).to(torch.float32), -12.0, 12.0))
            q_term = (terminal_field.to(torch.float32) * p_act.unsqueeze(-1)).sum(dim=(1, 2)) / p_act.sum(dim=(1, 2)).unsqueeze(-1).clamp_min(1.0)
            if rule_context is not None and rule_context.shape[-1] == D:
                q_term = q_term + (0.25 * rule_context.to(torch.float32))

            h_psi = p_act.max(dim=-1)[0].sum(dim=-1, keepdim=True).clamp(1.0, float(H))
            w_psi = p_act.max(dim=-2)[0].sum(dim=-1, keepdim=True).clamp(1.0, float(W))

            sh_c = (float(self.ZERO_SHIFT_INDEX) + (h_psi - h_in.float().unsqueeze(-1))).clamp(0.0, 60.0)
            sw_c = (float(self.ZERO_SHIFT_INDEX) + (w_psi - w_in.float().unsqueeze(-1))).clamp(0.0, 60.0)

            ci = cast(torch.Tensor, self.class_indices).to(device=device).view(1, self.RELATIVE_CLASSES)
            phi_h = torch.exp(-0.50 * ((ci - sh_c) / 1.50).pow(2))
            phi_w = torch.exp(-0.50 * ((ci - sw_c) / 1.50).pow(2))

        if D == self.head_terminal_h.in_features:
            th_log = self.head_terminal_h(q_term.to(dtype)).to(torch.float32)
            tw_log = self.head_terminal_w(q_term.to(dtype)).to(torch.float32)
        else:
            th_log = torch.zeros_like(base_h_logits, dtype=torch.float32)
            tw_log = torch.zeros_like(base_w_logits, dtype=torch.float32)

        with torch.amp.autocast(device_type=dev_type, enabled=False):
            ge_h = torch.tanh(2.5 * self.latest_min_err_h.to(torch.float32)) if (self.latest_min_err_h is not None and self.latest_min_err_h.shape[0] == B) else torch.ones((B, 1), dtype=torch.float32, device=device)
            ge_w = torch.tanh(2.5 * self.latest_min_err_w.to(torch.float32)) if (self.latest_min_err_w is not None and self.latest_min_err_w.shape[0] == B) else torch.ones((B, 1), dtype=torch.float32, device=device)

            gt = torch.tanh(self.terminal_gate.to(torch.float32))
            gm = F.softplus(self.carrier_mask_gain.to(torch.float32)) * 0.35

            final_h = (base_h_logits.to(torch.float32) + (0.10 + 0.90 * ge_h) * (gt * th_log) + ge_h * (gm * phi_h)).to(dtype)
            final_w = (base_w_logits.to(torch.float32) + (0.10 + 0.90 * ge_w) * (gt * tw_log) + ge_w * (gm * phi_w)).to(dtype)

        self.latest_min_err_h = None
        self.latest_min_err_w = None

        pred_h, pred_w, cap, cap_ste = self._resolve_dimensions_from_logits(final_h, final_w, h_in, w_in)
        return final_h, final_w, pred_h, pred_w, cap, cap_ste

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
        z_geo = self.rule_modulator(self.envelope_pooler(h_field, mask), rule_context)
        z_ind = self.support_inductor(query_context=z_geo, support_in_masks=support_in_masks, support_out_masks=support_out_masks)

        fused_ctx = self.context_ffn(torch.cat([z_geo, z_ind, geo_fiber], dim=-1))
        na3d_obs = NA3DFinslerLieEngine.extract_na3d_global_invariants(h_field, mask)
        if na3d_obs.shape[-1] == self.na3d_anisotropy_proj.in_features and fused_ctx.shape[-1] == self.na3d_anisotropy_proj.out_features:
            fused_ctx = fused_ctx + self.na3d_anisotropy_proj(na3d_obs.to(fused_ctx.dtype))

        raw_h, raw_w = self.head_rel_h(fused_ctx), self.head_rel_w(fused_ctx)

        if support_in_masks is not None and support_out_masks is not None and support_in_masks.shape[1] > 0:
            s_hi, s_wi = self.extract_input_dimensions(support_in_masks)
            s_ho, s_wo = self.extract_input_dimensions(support_out_masks)
            eff_v = pair_valid_mask if pair_valid_mask is not None else (support_in_masks.any(dim=-1).any(dim=-1) & support_out_masks.any(dim=-1).any(dim=-1))

            if query_inputs is not None and support_inputs is not None:
                q_hb, q_wb = self.extract_foreground_bbox_dimensions(query_inputs, mask)
                s_hb, s_wb = self.extract_foreground_bbox_dimensions(support_inputs, support_in_masks)
                q_ho, q_wo, _, _, s_ho_occ, s_wo_occ, _, _ = self.extract_vanishing_occluder_bbox(
                    query_inputs, mask, support_inputs, support_in_masks, support_outputs, support_out_masks, eff_v
                )
                q_hc, q_wc, _, _ = self.extract_chamber_divider_dimensions(query_inputs, mask)
                s_hc, s_wc, _, _ = self.extract_chamber_divider_dimensions(support_inputs, support_in_masks)
                q_hmax, q_wmax, q_hmin, q_wmin = self.extract_extremal_color_bbox_dimensions(query_inputs, mask)
                s_hmax, s_wmax, s_hmin, s_wmin = self.extract_extremal_color_bbox_dimensions(support_inputs, support_in_masks)
            else:
                q_hb = q_ho = q_hc = q_hmax = q_hmin = h_in
                q_wb = q_wo = q_wc = q_wmax = q_wmin = w_in
                s_hb = s_ho_occ = s_hc = s_hmax = s_hmin = s_hi
                s_wb = s_wo_occ = s_wc = s_wmax = s_wmin = s_wi

            q_phi_h = torch.stack([h_in, q_hb, q_hc, q_ho, w_in, q_wo, q_hmax, q_hmin], dim=-1)
            s_phi_h = torch.stack([s_hi, s_hb, s_hc, s_ho_occ, s_wi, s_wo_occ, s_hmax, s_hmin], dim=-1)
            q_phi_w = torch.stack([w_in, q_wb, q_wc, q_wo, h_in, q_ho, q_wmax, q_wmin], dim=-1)
            s_phi_w = torch.stack([s_wi, s_wb, s_wc, s_wo_occ, s_hi, s_ho_occ, s_wmax, s_wmin], dim=-1)

            pr_h, bst_h, err_h = self.compute_universal_proper_time_prior(q_phi_h, s_phi_h, s_ho, eff_v)
            pr_w, bst_w, err_w = self.compute_universal_proper_time_prior(q_phi_w, s_phi_w, s_wo, eff_v)

            self.latest_min_err_h, self.latest_min_err_w = err_h.detach(), err_w.detach()
            gain = torch.clamp(F.softplus(self.law_prior_gain), max=6.0).to(raw_h.dtype)
            h_rel = ((0.35 + 0.65 * torch.tanh(2.5 * err_h)).to(raw_h.dtype) * raw_h) + (gain * bst_h.to(raw_h.dtype) * pr_h.to(raw_h.dtype))
            w_rel = ((0.35 + 0.65 * torch.tanh(2.5 * err_w)).to(raw_w.dtype) * raw_w) + (gain * bst_w.to(raw_w.dtype) * pr_w.to(raw_w.dtype))
        else:
            self.latest_min_err_h = self.latest_min_err_w = None
            h_rel, w_rel = raw_h, raw_w

        pred_h, pred_w, cap, cap_ste = self._resolve_dimensions_from_logits(h_rel, w_rel, h_in, w_in)
        return h_rel, w_rel, pred_h, pred_w, cap, cap_ste

    @classmethod
    def evaluate_entropy_and_runner_up(
        cls,
        h_rel_logits: torch.Tensor,
        w_rel_logits: torch.Tensor,
        h_in: int,
        w_in: int,
        uncertainty_threshold: float = 0.40,
    ) -> Tuple[float, Optional[Tuple[int, int]]]:
        ph, pw = F.softmax(h_rel_logits.view(-1), dim=-1), F.softmax(w_rel_logits.view(-1), dim=-1)
        mean_ent = 0.50 * float((-torch.sum(ph * safe_log(ph)) - torch.sum(pw * safe_log(pw))).item())
        if mean_ent <= uncertainty_threshold:
            return mean_ent, None
        sh = int(torch.topk(ph, k=2)[1][1].item()) - cls.ZERO_SHIFT_INDEX
        sw = int(torch.topk(pw, k=2)[1][1].item()) - cls.ZERO_SHIFT_INDEX
        return mean_ent, (max(1, min(cls.MAX_CANVAS_DIM, h_in + sh)), max(1, min(cls.MAX_CANVAS_DIM, w_in + sw)))

    @classmethod
    def crop_canvas_to_predictions(cls, canvas: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        return canvas[..., : max(1, min(cls.MAX_CANVAS_DIM, target_h)), : max(1, min(cls.MAX_CANVAS_DIM, target_w))].contiguous()

    @classmethod
    def generate_target_mask(
        cls,
        batch_size: int,
        target_heights: torch.Tensor,
        target_widths: torch.Tensor,
        canvas_dim: int = DEFAULT_MAX_CANVAS_DIM,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        dev = device if device is not None else target_heights.device
        yc = torch.arange(canvas_dim, device=dev).view(1, canvas_dim, 1)
        xc = torch.arange(canvas_dim, device=dev).view(1, 1, canvas_dim)
        return (yc < target_heights.view(batch_size, 1, 1)) & (xc < target_widths.view(batch_size, 1, 1))


def verify_boundary_size_predictor_diagnostics() -> bool:
    """Diagnostic Verification Gate for boundary_size_predictor.py."""
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

    s_in_m_iso = torch.zeros(1, 2, 20, 20, dtype=torch.bool)
    s_in_m_iso[0, 0, :20, :20] = True
    s_in_m_iso[0, 1, :18, :18] = True
    q_mask_19x20 = torch.zeros(1, 20, 20, dtype=torch.bool)
    q_mask_19x20[0, :20, :19] = True

    _, _, pred_h_iso, pred_w_iso, _, _ = oracle(
        h_field=torch.zeros(1, 20, 20, d),
        mask=q_mask_19x20,
        support_in_masks=s_in_m_iso,
        support_out_masks=s_in_m_iso.clone(),
    )
    if int(pred_h_iso[0].item()) != 20 or int(pred_w_iso[0].item()) != 19:
        return False

    h_field = torch.randn(b, h, w, d, requires_grad=True)
    mask = torch.ones(b, h, w, dtype=torch.bool)
    z_rule = torch.randn(b, d, requires_grad=True)
    s_m = torch.ones(b, 2, h, w, dtype=torch.bool)

    h_logits, w_logits, _, _, _, _ = oracle(h_field, mask, rule_context=z_rule, support_in_masks=s_m, support_out_masks=s_m)
    term_f = torch.randn(b, h, w, d, requires_grad=True)
    c_psi = torch.full((b, h, w, 1), -3.0, requires_grad=True)

    th_log, _, ph_neg, pw_neg, _, t_cap = oracle.refine_with_terminal_carrier_mask(h_logits, w_logits, term_f, c_psi, mask, z_rule)
    if int(ph_neg[0].item()) != 20 or int(pw_neg[0].item()) != 20:
        return False

    (torch.sum(th_log.pow(2)) + torch.sum((t_cap - 36.0).pow(2))).backward()
    return oracle.head_rel_h.weight.grad is not None
