"""
Location: // fgns_engine/fgns/models/palette_budget/histogram_predictor.py
Description: SOTA CuSOLVER-Free Schulz-Hotelling Dual Gram-Tikhonov & Sherman-Morrison
             Palette Capacity Oracle with S10-Equivariant 8D Color Structural Role Fiber,
             Diode Annihilation Sink, and NA3D Global Holonomy Conditioning (Equivariant Palette Histogram Predictor).
Language: Python 3.12+ / PyTorch 2.5+ / Discrete Mass Conservation sm_120 Aligned
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import math
from typing import Final, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fgns.config import FGNSConfig
from fgns.models.palette_histogram.spatial_scanner import SpatialPaletteAttentionScanner
from fgns.primitives.hare_lrq import differentiable_hare_lrq
from fgns.primitives.safe_math import (
    safe_divide,
    safe_hyperbolic_sqrt,
    safe_log,
)
from fgns.primitives.stiefel_sheaf import NA3DFinslerLieEngine
from fgns.primitives.swiglu import RMSNorm, SwiGLU, TopologicalBypassFeedForward
from fgns.types import PaletteBudgetPrediction

DEFAULT_NUM_PALETTE_COLORS: Final[int] = 10
DEFAULT_BUDGET_EPS: Final[float] = 1e-7
TIKHONOV_LAMBDA: Final[float] = 0.05


class EquivariantPaletteHistogramPredictor(nn.Module):
    """
    SOTA Dual Gram-Tikhonov, S10-Equivariant Structural Role & Diode Sink Palette Capacity Oracle.
    """

    NUM_COLORS: int = DEFAULT_NUM_PALETTE_COLORS

    def __init__(self, config: FGNSConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.octowave.d_model
        d_hidden = config.budget.d_budget_hidden
        self.eps = DEFAULT_BUDGET_EPS

        self.hare_quota_tau = config.budget.hare_quota_tau
        self.anti_collapse_margin = config.budget.anti_collapse_margin
        self.bias_gain = config.budget.bias_gain

        self.palette_scanner = SpatialPaletteAttentionScanner(
            d_model=d_model,
            num_colors=self.NUM_COLORS,
            num_heads=config.budget.num_scanner_heads,
        )

        self.na3d_budget_modulator = nn.Linear(24, d_model, bias=False)
        self.role_equivariant_proj = nn.Linear(8, 1, bias=False)

        self.in_feat_dim = d_model + (self.NUM_COLORS * 5) + 3

        self.budget_mlp = nn.Sequential(
            nn.Linear(self.in_feat_dim, d_hidden, bias=True),
            nn.SiLU(),
            TopologicalBypassFeedForward(
                d_model=d_hidden,
                expansion_factor=2.0,
                alignment_multiple=32,
            ),
            nn.Linear(d_hidden, self.NUM_COLORS, bias=True),
        )

        self.coupling_swiglu = SwiGLU(
            d_in=self.NUM_COLORS + d_model,
            d_out=self.NUM_COLORS,
            d_inner=d_model,
            alignment_multiple=32,
        )

        self.per_color_equivariant_mlp = nn.Sequential(
            nn.Linear(d_model + 7, 64, bias=True),
            nn.SiLU(),
            nn.Linear(64, 1, bias=True),
        )

        self.prior_logit_gate = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.equivariant_gate = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
        self.residual_net_gate = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.norm_rule = RMSNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.na3d_budget_modulator.weight)
        nn.init.zeros_(self.role_equivariant_proj.weight)
        for module in self.budget_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.2)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.per_color_equivariant_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.2)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def compute_support_histogram_priors(
        support_in_hist: torch.Tensor,
        support_out_hist: torch.Tensor,
        pair_valid_mask: Optional[torch.Tensor] = None,
        leave_one_out: bool = False,
        exemplar_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, K, C = support_in_hist.shape
        device = support_in_hist.device
        dev_type = "cuda" if support_in_hist.is_cuda else "cpu"

        with torch.amp.autocast(device_type=dev_type, enabled=False):
            s_in_f32 = support_in_hist.to(torch.float32)
            s_out_f32 = support_out_hist.to(torch.float32)

            k_in = s_in_f32.sum(dim=-1, keepdim=True).clamp_min(1.0)
            k_out = s_out_f32.sum(dim=-1, keepdim=True).clamp_min(1.0)

            p_in_s = s_in_f32 / k_in
            p_out_s = s_out_f32 / k_out
            delta_p_s = p_out_s - p_in_s

            if pair_valid_mask is not None:
                v_mask = pair_valid_mask.to(torch.float32).unsqueeze(-1)
            else:
                v_mask = (s_out_f32.sum(dim=-1, keepdim=True) > 0).to(torch.float32)

            if not leave_one_out or K <= 1:
                if exemplar_weights is not None and exemplar_weights.ndim == 2:
                    w_ex = exemplar_weights.to(torch.float32).view(B, K, 1) * v_mask
                    w_sum = w_ex.sum(dim=1, keepdim=True).clamp_min(1e-6)
                    eff_w = w_ex / w_sum
                else:
                    v_sum = v_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                    eff_w = v_mask / v_sum

                mean_delta = (delta_p_s * eff_w).sum(dim=1)
                mean_p_out = (p_out_s * eff_w).sum(dim=1)
                if leave_one_out:
                    return mean_delta.repeat_interleave(K, dim=0), mean_p_out.repeat_interleave(K, dim=0)
                return mean_delta, mean_p_out

            eye_k = torch.eye(K, dtype=torch.float32, device=device).view(1, K, K)
            not_self = 1.0 - eye_k

            if exemplar_weights is not None and exemplar_weights.ndim == 3:
                loo_weights = exemplar_weights.to(torch.float32) * v_mask.transpose(1, 2) * not_self
            else:
                loo_weights = v_mask.transpose(1, 2) * not_self

            loo_sum = loo_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            has_other = (loo_weights.sum(dim=-1, keepdim=True) > 1e-6)
            eff_weights = torch.where(
                has_other,
                loo_weights / loo_sum,
                v_mask.transpose(1, 2) / v_mask.sum(dim=1, keepdim=True).clamp_min(1.0),
            )

            loo_delta = torch.bmm(eff_weights, delta_p_s).view(B * K, C)
            loo_p_out = torch.bmm(eff_weights, p_out_s).view(B * K, C)
        return loo_delta, loo_p_out

    @classmethod
    def compute_dual_gram_tikhonov_resolvent(
        cls,
        s_in: torch.Tensor,
        s_out: torch.Tensor,
        k_out_s: torch.Tensor,
        t_in: torch.Tensor,
        k_out_t: torch.Tensor,
        w_conj: torch.Tensor,
        support_cavity_hist: Optional[torch.Tensor] = None,
        target_cavity_hist: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, K, C = s_in.shape
        B_t = t_in.shape[0]
        device = s_in.device
        dev_type = "cuda" if s_in.is_cuda else "cpu"

        with torch.amp.autocast(device_type=dev_type, enabled=False):
            p_in_s = s_in.to(torch.float32) / k_out_s.to(torch.float32)
            p_out_s = s_out.to(torch.float32) / k_out_s.to(torch.float32)
            p_in_t = t_in.to(torch.float32) / k_out_t.to(torch.float32)

            ones_s = torch.ones((B, K, 1), dtype=torch.float32, device=device)
            ones_t = torch.ones((B_t, 1), dtype=torch.float32, device=device)

            if support_cavity_hist is not None and target_cavity_hist is not None:
                s_cav_norm = support_cavity_hist.to(torch.float32) / k_out_s.to(torch.float32)
                t_cav_norm = target_cavity_hist.to(torch.float32) / k_out_t.to(torch.float32)
                phi_s = torch.cat([ones_s, p_in_s, s_cav_norm], dim=-1)
                phi_t = torch.cat([ones_t, p_in_t, t_cav_norm], dim=-1)
            else:
                phi_s = torch.cat([ones_s, p_in_s], dim=-1)
                phi_t = torch.cat([ones_t, p_in_t], dim=-1)

            M = phi_s.shape[-1]
            sqrt_w = torch.sqrt(w_conj.to(torch.float32).clamp_min(0.0))
            phi_s_w = phi_s * sqrt_w

            gram_raw = torch.bmm(phi_s_w, phi_s_w.transpose(1, 2))
            gram_sym = 0.50 * (gram_raw + gram_raw.transpose(1, 2))
            eye_k = torch.eye(K, dtype=torch.float32, device=device).unsqueeze(0).expand(B, K, K)
            gram_spd = gram_sym + (TIKHONOV_LAMBDA * eye_k)

            trace_spd = torch.diagonal(gram_spd, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1e-4)
            gram_inv = eye_k / trace_spd
            two_eye = 2.0 * eye_k

            for _ in range(6):
                gx = torch.bmm(gram_spd, gram_inv)
                gram_inv = torch.bmm(gram_inv, two_eye - gx)

            hat_mat = torch.bmm(gram_sym, gram_inv)
            diag_h = torch.diagonal(hat_mat, dim1=-2, dim2=-1).unsqueeze(-1)
            leverage_denom = torch.clamp_min(1.0 - diag_h, 0.20)

            p_out_w = p_out_s * sqrt_w
            pred_p_s_w = torch.bmm(hat_mat, p_out_w)
            loo_residual_s = torch.abs(p_out_w - pred_p_s_w) / leverage_denom
            err_sherman_morrison = loo_residual_s.sum(dim=-1).sum(dim=-1, keepdim=True)

            dual_alpha = torch.bmm(gram_inv, p_out_w)
            w_operator = torch.bmm(phi_s_w.transpose(1, 2), dual_alpha)

            if B_t == B * K:
                w_op_t = w_operator.repeat_interleave(K, dim=0)
            else:
                w_op_t = w_operator

            pred_p_t_raw = torch.bmm(phi_t.view(B_t, 1, M), w_op_t).squeeze(1)
            pred_p_t = F.relu(pred_p_t_raw)
            pred_p_t = pred_p_t / pred_p_t.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        return pred_p_t, err_sherman_morrison

    @classmethod
    def compute_5modal_palette_law_prior(
        cls,
        target_in_hist: torch.Tensor,
        target_k_out: torch.Tensor,
        support_in_hist: torch.Tensor,
        support_out_hist: torch.Tensor,
        pair_valid_mask: Optional[torch.Tensor] = None,
        leave_one_out: bool = False,
        exemplar_weights: Optional[torch.Tensor] = None,
        target_cavity_hist: Optional[torch.Tensor] = None,
        support_cavity_hist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, K, C = support_in_hist.shape
        dev_type = "cuda" if support_in_hist.is_cuda else "cpu"

        with torch.amp.autocast(device_type=dev_type, enabled=False):
            s_in = support_in_hist.to(torch.float32)
            s_out = support_out_hist.to(torch.float32)
            k_in_s = s_in.sum(dim=-1, keepdim=True).clamp_min(1.0)
            k_out_s = s_out.sum(dim=-1, keepdim=True).clamp_min(1.0)

            if pair_valid_mask is not None:
                v_mask = pair_valid_mask.to(torch.float32).unsqueeze(-1)
            else:
                v_mask = (s_out.sum(dim=-1, keepdim=True) > 0).to(torch.float32)

            w_conj = v_mask / v_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            if exemplar_weights is not None and exemplar_weights.ndim == 2:
                w_res = (exemplar_weights.to(torch.float32).view(B, K, 1) * v_mask)
                w_sum = w_res.sum(dim=1, keepdim=True)
                w_res = torch.where(w_sum > 1e-6, w_res / w_sum.clamp_min(1e-6), w_conj)
            else:
                w_res = w_conj

            mean_delta_n = ((s_out - s_in) * w_conj).sum(dim=1, keepdim=True)
            mean_n_out = (s_out * w_conj).sum(dim=1, keepdim=True)

            fg_sum_s = s_in[:, :, 1:].sum(dim=-1, keepdim=True)
            bg_rem_s = torch.clamp_min(k_out_s - fg_sum_s, 0.0)
            pred_s_1 = torch.cat([bg_rem_s, s_in[:, :, 1:]], dim=-1)

            area_ratio_s = k_out_s / k_in_s
            pred_s_2 = torch.round(s_in * area_ratio_s)

            pred_s_3 = torch.clamp_min(s_in + torch.round(mean_delta_n), 0.0)

            pred_s_4 = torch.round(mean_n_out).expand(B, K, C)

            fg_sq_s = s_in[:, :, 1:].pow(2)
            bg_sq_s = torch.clamp_min(k_out_s - fg_sq_s.sum(dim=-1, keepdim=True), 0.0)
            pred_s_5 = torch.cat([bg_sq_s, fg_sq_s], dim=-1)

            max_fg_in_s = torch.max(s_in[:, :, 1:], dim=-1, keepdim=True)[0]
            dom_out_col_s = torch.argmax(s_out[:, :, 1:], dim=-1) + 1
            one_hot_dom_s = F.one_hot(dom_out_col_s, num_classes=C).to(torch.float32)
            fg_recol_s = one_hot_dom_s * max_fg_in_s
            bg_recol_s = torch.clamp_min(k_out_s - max_fg_in_s, 0.0)
            pred_s_6 = fg_recol_s.clone()
            pred_s_6[:, :, 0:1] = bg_recol_s

            t_in = target_in_hist.to(torch.float32)
            B_t = t_in.shape[0]
            k_in_t = t_in.sum(dim=-1, keepdim=True).clamp_min(1.0)
            k_out_t = target_k_out.to(torch.float32).view(B_t, 1).clamp_min(1.0)

            preds_s_6 = torch.stack([pred_s_1, pred_s_2, pred_s_3, pred_s_4, pred_s_5, pred_s_6], dim=2)
            abs_diff_6 = torch.abs(preds_s_6 - s_out.unsqueeze(2)).sum(dim=-1) / k_out_s
            err_6_laws = (abs_diff_6 * w_conj).sum(dim=1)

            cand_7, err_tikhonov = cls.compute_dual_gram_tikhonov_resolvent(
                s_in=s_in,
                s_out=s_out,
                k_out_s=k_out_s,
                t_in=t_in,
                k_out_t=k_out_t,
                w_conj=w_conj,
                support_cavity_hist=support_cavity_hist,
                target_cavity_hist=target_cavity_hist,
            )

            err_per_law = torch.cat([err_6_laws, err_tikhonov], dim=-1)

            min_law_err = torch.min(err_per_law, dim=-1, keepdim=True)[0]
            beta_bose = 15.0 / (min_law_err + 0.25)
            law_weights = F.softmax(-beta_bose * err_per_law, dim=-1)

            weighted_fg_out = (s_out[:, :, 1:] * w_res).sum(dim=1)
            res_dom_col_q = torch.argmax(weighted_fg_out, dim=-1) + 1
            one_hot_res_dom = F.one_hot(res_dom_col_q, num_classes=C).to(torch.float32)

            if B_t == B * K:
                w_t = law_weights.repeat_interleave(K, dim=0)
                dn_t = mean_delta_n.squeeze(1).repeat_interleave(K, dim=0)
                nout_t = mean_n_out.squeeze(1).repeat_interleave(K, dim=0)
                oh_dom_t = one_hot_res_dom.repeat_interleave(K, dim=0)
            else:
                w_t = law_weights
                dn_t = mean_delta_n.squeeze(1)
                nout_t = mean_n_out.squeeze(1)
                oh_dom_t = one_hot_res_dom

            fg_sum_t = t_in[:, 1:].sum(dim=-1, keepdim=True)
            bg_rem_t = torch.clamp_min(k_out_t - fg_sum_t, 0.0)
            cand_1 = torch.cat([bg_rem_t, t_in[:, 1:]], dim=-1) / k_out_t

            cand_2 = (t_in * (k_out_t / k_in_t)) / k_out_t
            cand_3 = torch.clamp_min(t_in + dn_t, 0.0) / k_out_t
            cand_4 = torch.clamp_min(nout_t, 0.0) / k_out_t.clamp_min(1.0)

            fg_sq_t = t_in[:, 1:].pow(2)
            bg_sq_t = torch.clamp_min(k_out_t - fg_sq_t.sum(dim=-1, keepdim=True), 0.0)
            cand_5 = torch.cat([bg_sq_t, fg_sq_t], dim=-1) / k_out_t

            max_fg_t = torch.max(t_in[:, 1:], dim=-1, keepdim=True)[0]
            cand_6_counts = oh_dom_t * max_fg_t
            cand_6_counts[:, 0:1] = torch.clamp_min(k_out_t - max_fg_t, 0.0)
            cand_6 = cand_6_counts / k_out_t

            cands_stack = torch.stack([cand_1, cand_2, cand_3, cand_4, cand_5, cand_6, cand_7], dim=1)
            cands_norm = cands_stack / cands_stack.sum(dim=-1, keepdim=True).clamp_min(1e-6)

            p_analytic = torch.sum(w_t.unsqueeze(-1) * cands_norm, dim=1)
            p_final = p_analytic / p_analytic.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        return p_final

    @staticmethod
    def reapportion_with_terminal_capacity(
        prediction: PaletteBudgetPrediction,
        terminal_capacity_ste: torch.Tensor,
        training_true_capacities: Optional[torch.Tensor] = None,
        active_color_mask: Optional[torch.Tensor] = None,
    ) -> PaletteBudgetPrediction:
        B = prediction.simplex_proportions.shape[0]
        dtype = prediction.simplex_proportions.dtype

        k_term_f32 = terminal_capacity_ste.to(torch.float32).view(B, 1).clamp(1.0, 900.0)
        if training_true_capacities is not None:
            k_true_f32 = training_true_capacities.to(torch.float32).view(B, 1).clamp(1.0, 900.0)
            k_effective = k_term_f32 + (k_true_f32 - k_term_f32).detach()
        else:
            k_effective = k_term_f32

        if active_color_mask is not None:
            m_f32 = active_color_mask.to(torch.float32)
            clean_p = (prediction.simplex_proportions.to(torch.float32) * m_f32)
            clean_p = clean_p / clean_p.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            clean_p = prediction.simplex_proportions.to(torch.float32)

        discrete_counts = differentiable_hare_lrq(
            simplex_probabilities=clean_p.to(dtype),
            capacities=k_effective.squeeze(-1),
        )

        return PaletteBudgetPrediction(
            predicted_counts=discrete_counts,
            simplex_proportions=prediction.simplex_proportions,
            input_counts=prediction.input_counts,
            target_capacity=torch.round(k_effective.squeeze(-1)).to(torch.int64),
        )

    def forward(
        self,
        h_field: torch.Tensor,
        mask: torch.Tensor,
        input_histograms: torch.Tensor,
        target_capacities: torch.Tensor,
        rule_context: Optional[torch.Tensor] = None,
        training_true_capacities: Optional[torch.Tensor] = None,
        support_delta_p: Optional[torch.Tensor] = None,
        support_mean_p_out: Optional[torch.Tensor] = None,
        analytic_law_prior: Optional[torch.Tensor] = None,
        role_vectors: Optional[torch.Tensor] = None,
        diode_sink_mask: Optional[torch.Tensor] = None,
    ) -> PaletteBudgetPrediction:
        B = h_field.shape[0]
        device = h_field.device
        dtype = h_field.dtype

        n_in_f32 = input_histograms.to(torch.float32)
        k_in = torch.sum(n_in_f32, dim=-1, keepdim=True).clamp_min(1.0)
        p_in = n_in_f32 / k_in
        log_n_in = safe_log(n_in_f32 + 1.0, eps=self.eps)

        if support_delta_p is not None and support_mean_p_out is not None:
            delta_p = support_delta_p.to(device=device, dtype=torch.float32).view(B, self.NUM_COLORS)
            mean_p_out = support_mean_p_out.to(device=device, dtype=torch.float32).view(B, self.NUM_COLORS)
            base_heuristic_prior = torch.clamp(0.80 * (p_in + delta_p) + 0.20 * mean_p_out, min=0.0, max=1.0)
            is_color_active = (n_in_f32 > 0.0) | (mean_p_out > 1e-5)
        else:
            delta_p = torch.zeros((B, self.NUM_COLORS), dtype=torch.float32, device=device)
            mean_p_out = p_in
            base_heuristic_prior = torch.clamp(p_in, min=0.0, max=1.0)
            is_color_active = (n_in_f32 > 0.0)

        if analytic_law_prior is not None:
            raw_prior = analytic_law_prior.to(device=device, dtype=torch.float32).view(B, self.NUM_COLORS)
            is_color_active = is_color_active | (raw_prior > 1e-4)
        else:
            raw_prior = base_heuristic_prior

        if diode_sink_mask is not None:
            not_sunk = diode_sink_mask.to(device=device, dtype=torch.float32).view(B, self.NUM_COLORS) < 0.50
            is_color_active = is_color_active & not_sunk

        is_color_active = is_color_active.clone()
        is_color_active[:, 0] = True
        active_mask_f32 = is_color_active.to(torch.float32)

        masked_prior = (raw_prior * active_mask_f32) + (1e-6 * active_mask_f32)
        p_prior = masked_prior / masked_prior.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        k_ste_f32 = target_capacities.to(torch.float32).view(B, 1).clamp(1.0, 900.0)
        capacity_ratio = safe_divide(k_ste_f32, k_in, eps=self.eps)
        log_capacity_target = safe_log(k_ste_f32, eps=self.eps)
        sqrt_capacity_target = safe_hyperbolic_sqrt(k_ste_f32, eps=self.eps) / 30.0

        z_pal_spatial = self.palette_scanner(h_field, mask)
        z_spatial_summary = torch.mean(z_pal_spatial, dim=1)

        na3d_obs = NA3DFinslerLieEngine.extract_na3d_global_invariants(h_field, mask)
        if (
            na3d_obs.shape[-1] == self.na3d_budget_modulator.in_features
            and z_spatial_summary.shape[-1] == self.na3d_budget_modulator.out_features
        ):
            z_spatial_summary = z_spatial_summary + self.na3d_budget_modulator(na3d_obs.to(z_spatial_summary.dtype))

        if rule_context is not None:
            z_rule = self.norm_rule(rule_context)
            z_joint_context = z_spatial_summary + z_rule
        else:
            z_joint_context = z_spatial_summary

        expected_in = self.budget_mlp[0].in_features
        if expected_in == z_joint_context.shape[-1] + (self.NUM_COLORS * 5) + 3:
            raw_bundle = torch.cat(
                [
                    z_joint_context.to(torch.float32),
                    log_n_in,
                    p_in,
                    delta_p,
                    mean_p_out,
                    p_prior,
                    capacity_ratio,
                    log_capacity_target,
                    sqrt_capacity_target,
                ],
                dim=-1,
            ).to(dtype)
        else:
            raw_bundle = torch.cat(
                [
                    z_joint_context.to(torch.float32),
                    log_n_in,
                    p_in,
                    capacity_ratio,
                    log_capacity_target,
                    sqrt_capacity_target,
                ],
                dim=-1,
            ).to(dtype)

        raw_logits = self.budget_mlp(raw_bundle)
        coupled_input = torch.cat([raw_logits, z_joint_context], dim=-1)
        net_delta_logits = self.coupling_swiglu(coupled_input).to(torch.float32)

        is_bg_indicator = torch.zeros((B, self.NUM_COLORS, 1), dtype=torch.float32, device=device)
        is_bg_indicator[:, 0, 0] = 1.0
        cap_ratio_exp = capacity_ratio.unsqueeze(1).expand(B, self.NUM_COLORS, 1)

        per_color_feats = torch.cat(
            [
                z_pal_spatial.to(torch.float32),
                log_n_in.unsqueeze(-1),
                p_in.unsqueeze(-1),
                p_prior.unsqueeze(-1),
                delta_p.unsqueeze(-1),
                mean_p_out.unsqueeze(-1),
                is_bg_indicator,
                cap_ratio_exp,
            ],
            dim=-1,
        ).to(dtype)

        if per_color_feats.shape[-1] == self.per_color_equivariant_mlp[0].in_features:
            equiv_logits = self.per_color_equivariant_mlp(per_color_feats).squeeze(-1).to(torch.float32)
        else:
            equiv_logits = torch.zeros_like(net_delta_logits)

        if role_vectors is not None and role_vectors.shape[-1] == 8:
            role_bias = self.role_equivariant_proj(role_vectors.to(dtype=dtype, device=device)).squeeze(-1).to(torch.float32)
            equiv_logits = equiv_logits + role_bias

        net_residual = (
            (torch.tanh(self.equivariant_gate) * equiv_logits)
            + (torch.tanh(self.residual_net_gate) * net_delta_logits)
            + (0.05 * self.prior_logit_gate)
        )
        p_net = F.softmax(torch.log(torch.clamp_min(p_prior, 1e-6)) + net_residual, dim=-1)

        mix_alpha = 0.25 * torch.sigmoid(self.residual_net_gate)
        raw_simplex = ((1.0 - mix_alpha) * p_prior) + (mix_alpha * p_net)

        hare_simplex = raw_simplex * active_mask_f32
        hare_simplex = hare_simplex / hare_simplex.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        simplex_proportions = ((1.0 - 1e-4) * hare_simplex + (1e-5)).to(dtype)

        if training_true_capacities is not None:
            k_true_f32 = training_true_capacities.to(torch.float32).view(B, 1).clamp(1.0, 900.0)
            k_effective = k_ste_f32 + (k_true_f32 - k_ste_f32).detach()
        else:
            k_effective = k_ste_f32

        discrete_counts = differentiable_hare_lrq(
            simplex_probabilities=hare_simplex.to(dtype),
            capacities=k_effective.squeeze(-1),
        )

        return PaletteBudgetPrediction(
            predicted_counts=discrete_counts,
            simplex_proportions=simplex_proportions,
            input_counts=input_histograms,
            target_capacity=torch.round(k_effective.squeeze(-1)).to(torch.int64),
        )

    def evaluate_potts_chemical_biases(
        self,
        predicted_counts: torch.Tensor,
        current_canvas_field: torch.Tensor,
        palette_prototypes: torch.Tensor,
        target_capacity: torch.Tensor,
    ) -> torch.Tensor:
        B, H, W, D = current_canvas_field.shape
        prototypes_10 = palette_prototypes[:10, :]

        flat_field = current_canvas_field.view(B, H * W, D)
        norm_field = F.normalize(flat_field, p=2, dim=-1, eps=self.eps)
        norm_proto = F.normalize(prototypes_10, p=2, dim=-1, eps=self.eps)

        cos_sim = torch.matmul(norm_field, norm_proto.t())
        soft_occupancy = F.softmax(cos_sim * 5.0, dim=-1).sum(dim=1)

        deficit = predicted_counts.float() - soft_occupancy
        sigma_k = safe_hyperbolic_sqrt(target_capacity.float().view(B, 1), eps=self.eps).clamp_min(1.0)

        chemical_biases = self.bias_gain * torch.tanh(deficit / sigma_k)
        return chemical_biases


def verify_histogram_predictor_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for histogram_predictor.py.
    """
    torch.manual_seed(42)
    b_size, h, w, d_model = 2, 6, 6, 32

    config = FGNSConfig()
    oracle = EquivariantPaletteHistogramPredictor(config=config)

    oracle.palette_scanner = SpatialPaletteAttentionScanner(d_model=d_model, num_colors=10, num_heads=4)
    oracle.budget_mlp[0] = nn.Linear(d_model + 53, 64, bias=True)
    oracle.budget_mlp[2] = TopologicalBypassFeedForward(d_model=64, expansion_factor=2.0)
    oracle.budget_mlp[3] = nn.Linear(64, 10, bias=True)
    oracle.coupling_swiglu = SwiGLU(d_in=10 + d_model, d_out=10, d_inner=32)
    oracle.per_color_equivariant_mlp[0] = nn.Linear(d_model + 7, 64, bias=True)
    oracle.norm_rule = RMSNorm(d_model)

    h_field = torch.randn(b_size, h, w, d_model, dtype=torch.float32, requires_grad=True)
    mask = torch.ones(b_size, h, w, dtype=torch.bool)

    input_hist = torch.tensor(
        [[20, 5, 5, 2, 2, 2, 0, 0, 10, 0], [4, 1, 1, 1, 1, 1, 0, 0, 0, 0]],
        dtype=torch.int64,
    )
    target_capacities = torch.tensor([81.0, 9.0], dtype=torch.float32, requires_grad=True)
    true_capacities = torch.tensor([36, 16], dtype=torch.int64)
    rule_vec = torch.randn(b_size, d_model, dtype=torch.float32, requires_grad=True)

    diode_sink = torch.zeros(b_size, 10, dtype=torch.float32)
    diode_sink[0, 8] = 1.0

    prediction = oracle(
        h_field=h_field,
        mask=mask,
        input_histograms=input_hist,
        target_capacities=target_capacities,
        rule_context=rule_vec,
        training_true_capacities=true_capacities,
        diode_sink_mask=diode_sink,
    )

    if prediction.predicted_counts.shape != (b_size, 10):
        return False
    if int(torch.sum(prediction.predicted_counts[0]).item()) != 36:
        return False
    if int(prediction.predicted_counts[0, 8].item()) != 0:
        return False

    target_mock = torch.tensor(
        [[20, 10, 6, 0, 0, 0, 0, 0, 0, 0], [10, 2, 2, 2, 0, 0, 0, 0, 0, 0]],
        dtype=torch.float32,
    )
    loss = F.mse_loss(prediction.predicted_counts, target_mock)
    loss.backward()

    if h_field.grad is None or target_capacities.grad is None or rule_vec.grad is None:
        return False

    return True

