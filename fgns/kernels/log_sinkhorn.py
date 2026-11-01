"""
Location: // fgns_engine/fgns/kernels/log_sinkhorn.py
Description: SOTA Log-Domain Kantorovich Dual Potential Sinkhorn-Knopp & Asymmetric Finsler-Markov
             Diode Projector on Birkhoff Polytope B_11.
             Core Architecture for FGNS v4.0.0 (Finsler-Gauge Sheaf):
               1. Strict log-space Kantorovich dual potential iterations via LogSumExp.
               2. S10-Equivariant Structural Role Randers Metric Coupling: evaluates asymmetric role
                  affinities F_Randers(rho_i, rho_j) over 8D topological role descriptors (Theorem 3).
               3. Diode Annihilation Sink Support: allows vanished occluder (8) or divider (4) colors
                  to be absorbed without polluting active output columns.
               4. Numerically immune to FP16/BF16 underflow/overflow even at extreme temperatures tau -> 0.001.
               5. Straight-Through Estimator (STE) greedy Hungarian projection for discrete inference.
Language: Python 3.12+ / PyTorch 2.5+ / Differentiable Optimal Transport
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fgns.primitives.safe_math import safe_log


def log_sinkhorn_iterations(
    log_cost: torch.Tensor,
    log_r: torch.Tensor,
    log_c: torch.Tensor,
    tau: Union[float, torch.Tensor] = 0.05,
    num_iters: int = 20,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluates Kantorovich dual potentials strictly in log-space:
        f^{t+1} = log(r) - LogSumExp_j((log_cost / tau) + g^t)
        g^{t+1} = log(c) - LogSumExp_i((log_cost / tau) + f^{t+1})
    """
    device = log_cost.device
    dtype = log_cost.dtype

    if isinstance(tau, torch.Tensor):
        tau_clamped = torch.clamp(tau.to(device=device, dtype=dtype), min=eps)
    else:
        tau_clamped = max(float(tau), eps)

    scaled_cost = log_cost / tau_clamped

    f_pot = torch.zeros_like(log_r, dtype=dtype, device=device)
    g_pot = torch.zeros_like(log_c, dtype=dtype, device=device)

    for _ in range(num_iters):
        f_kernel = scaled_cost + g_pot.unsqueeze(-2)
        f_pot = log_r - torch.logsumexp(f_kernel, dim=-1)

        g_kernel = scaled_cost + f_pot.unsqueeze(-1)
        g_pot = log_c - torch.logsumexp(g_kernel, dim=-2)

    log_p = scaled_cost + f_pot.unsqueeze(-1) + g_pot.unsqueeze(-2)
    p_matrix = torch.exp(torch.clamp(log_p, min=-35.0, max=0.0))

    return p_matrix, log_p


class LogDomainSinkhornBirkhoffProjector(nn.Module):
    """
    SOTA Log-Domain Sinkhorn-Knopp & Asymmetric Finsler-Markov Projector on B_V (V=11 for ARC-AGI-2).
    """

    def __init__(
        self,
        d_model: int = 128,
        vocab_size: int = 11,
        num_iterations: int = 25,
        tau_default: float = 0.05,
        min_tau: float = 0.01,
        max_tau: float = 0.50,
        marginal_penalty_weight: float = 1.00,
        sharpness_weight: float = 0.20,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_iterations = num_iterations
        self.min_tau = float(min_tau)
        self.max_tau = float(max_tau)
        self.marginal_penalty_weight = float(marginal_penalty_weight)
        self.sharpness_weight = float(sharpness_weight)
        self.eps = eps

        """Linear projection mapping invariant task rule vector into V x V affinity matrix"""
        self.affinity_projection = nn.Linear(d_model, vocab_size * vocab_size, bias=True)

        """Asymmetric Randers Wind Covector on 8D Color Structural Role Space"""
        self.randers_role_wind = nn.Parameter(torch.zeros(8, dtype=torch.float32))
        self.role_coupling_gain = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))

        """Continuous bounded temperature parameterization: tau in [min_tau, max_tau]"""
        raw_tau_init = math.log((tau_default - min_tau) / (max_tau - tau_default + 1e-6))
        self.tau_param = nn.Parameter(torch.tensor(raw_tau_init, dtype=torch.float32))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.affinity_projection.weight, gain=0.1)
        with torch.no_grad():
            identity_bias = torch.eye(self.vocab_size).view(-1) * 3.0
            self.affinity_projection.bias.copy_(identity_bias)

    @property
    def temperature(self) -> torch.Tensor:
        sig = torch.sigmoid(self.tau_param)
        return self.min_tau + (self.max_tau - self.min_tau) * sig

    def get_temperature_scalar(self) -> float:
        return float(self.temperature.detach().item())

    def compute_finsler_role_affinity(
        self,
        in_roles: torch.Tensor,
        out_roles: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Evaluates the Asymmetric Finsler-Randers Role Affinity Matrix over 8D structural roles:
            A_role(i -> j) = -||rho_j - rho_i||_2 - <beta_wind, rho_j - rho_i>.
        """
        r_in = in_roles.to(torch.float32)
        r_out = out_roles.to(torch.float32) if out_roles is not None else r_in

        delta_rho = r_out.unsqueeze(-3) - r_in.unsqueeze(-2)
        riemann_dist = torch.sqrt(torch.sum(delta_rho * delta_rho, dim=-1) + self.eps)

        w_norm = torch.sqrt(torch.sum(self.randers_role_wind.pow(2)) + self.eps)
        wind_bounded = (torch.tanh(w_norm) / w_norm) * self.randers_role_wind
        randers_drift = torch.sum(delta_rho * wind_bounded, dim=-1)

        finsler_dist = riemann_dist + randers_drift
        affinity_10 = -finsler_dist
        return F.pad(affinity_10, (0, 1, 0, 1), mode="constant", value=0.0)

    def compute_birkhoff_matrix(
        self,
        rule_vector: torch.Tensor,
        tau_override: Optional[Union[float, torch.Tensor]] = None,
        target_column_proportions: Optional[torch.Tensor] = None,
        role_vectors: Optional[torch.Tensor] = None,
        diode_sink_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        orig_shape = rule_vector.shape
        batch_dims = orig_shape[:-1]
        v = self.vocab_size
        device = rule_vector.device
        dtype = rule_vector.dtype

        raw_logits = self.affinity_projection(rule_vector).view(*batch_dims, v, v)

        if role_vectors is not None:
            role_aff = self.compute_finsler_role_affinity(role_vectors).to(dtype=dtype, device=device)
            g_role = torch.tanh(self.role_coupling_gain.to(dtype=dtype, device=device))
            raw_logits = raw_logits + (g_role * role_aff)

        if diode_sink_mask is not None:
            sink_pad = F.pad(diode_sink_mask.to(dtype=dtype, device=device), (0, 1), mode="constant", value=0.0)
            raw_logits = raw_logits - (8.0 * sink_pad.unsqueeze(-2))

        if tau_override is not None:
            tau = tau_override
        else:
            tau = self.temperature

        if target_column_proportions is not None:
            pad_weight = 1.0 / float(v)
            c_target = F.pad(target_column_proportions.float(), (0, 1), mode="constant", value=pad_weight)
            c_target = c_target / c_target.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            log_c = safe_log(c_target.to(dtype=dtype, device=device), eps=self.eps)
        else:
            uniform_c = torch.full((*batch_dims, v), 1.0 / float(v), dtype=dtype, device=device)
            log_c = torch.log(uniform_c)

        uniform_r = torch.full((*batch_dims, v), 1.0 / float(v), dtype=dtype, device=device)
        log_r = torch.log(uniform_r)

        p_matrix, _ = log_sinkhorn_iterations(
            log_cost=raw_logits,
            log_r=log_r,
            log_c=log_c,
            tau=tau,
            num_iters=self.num_iterations,
            eps=self.eps,
        )

        row_sums = torch.sum(p_matrix, dim=-1, keepdim=True)
        p_normalized = p_matrix / torch.clamp(row_sums, min=self.eps)
        return p_normalized

    def compute_marginal_divergence_loss(
        self,
        p_matrix: torch.Tensor,
        target_column_proportions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float, float]:
        v = p_matrix.shape[-1]
        device = p_matrix.device
        dtype = p_matrix.dtype

        row_sums = torch.sum(p_matrix, dim=-1)
        col_sums = torch.sum(p_matrix, dim=-2)

        target_row = 1.0
        if target_column_proportions is not None:
            pad_weight = 1.0 / float(v)
            c_target = F.pad(target_column_proportions.float(), (0, 1), mode="constant", value=pad_weight)
            c_target = (c_target / c_target.sum(dim=-1, keepdim=True).clamp_min(self.eps)) * float(v)
            c_target = c_target.to(dtype=dtype, device=device)
        else:
            c_target = torch.full_like(col_sums, 1.0)

        row_err = torch.mean((row_sums - target_row).pow(2))
        col_err = torch.mean((col_sums - c_target).pow(2))

        total_marginal_loss = 0.50 * (row_err + col_err)
        return total_marginal_loss, float(row_err.detach().item()), float(col_err.detach().item())

    def compute_sharpness_potential(self, p_matrix: torch.Tensor) -> Tuple[torch.Tensor, float]:
        v = p_matrix.shape[-1]
        p_clamped = torch.clamp(p_matrix, min=0.0, max=1.0)
        row_purity = torch.sum(p_clamped.pow(2), dim=-1)
        mean_purity = torch.mean(row_purity)

        sharpness_loss = torch.clamp(1.0 - (mean_purity / float(v)), min=0.0, max=1.0)
        return sharpness_loss, float(mean_purity.detach().item())

    def sample_hard_permutation(
        self,
        p_matrix: torch.Tensor,
        straight_through: bool = True,
    ) -> torch.Tensor:
        orig_shape = p_matrix.shape
        v = orig_shape[-1]
        flat_p = p_matrix.reshape(-1, v, v)
        b_count = flat_p.shape[0]

        hard_p = torch.zeros_like(flat_p)

        for b in range(b_count):
            cost_grid = flat_p[b].detach().clone()
            for _ in range(v):
                max_pos = torch.argmax(cost_grid)
                row_idx = max_pos // v
                col_idx = max_pos % v

                hard_p[b, row_idx, col_idx] = 1.0
                cost_grid[row_idx, :] = -1e9
                cost_grid[:, col_idx] = -1e9

        hard_p = hard_p.view(*orig_shape)

        if straight_through:
            return hard_p - p_matrix.detach() + p_matrix
        return hard_p

    def forward(
        self,
        rule_vector: torch.Tensor,
        tau_override: Optional[Union[float, torch.Tensor]] = None,
        target_column_proportions: Optional[torch.Tensor] = None,
        role_vectors: Optional[torch.Tensor] = None,
        diode_sink_mask: Optional[torch.Tensor] = None,
        crystallize: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        p_soft = self.compute_birkhoff_matrix(
            rule_vector=rule_vector,
            tau_override=tau_override,
            target_column_proportions=target_column_proportions,
            role_vectors=role_vectors,
            diode_sink_mask=diode_sink_mask,
        )

        marginal_loss, r_err, c_err = self.compute_marginal_divergence_loss(
            p_matrix=p_soft,
            target_column_proportions=target_column_proportions,
        )

        sharp_loss, purity = self.compute_sharpness_potential(p_soft)

        total_loss = (self.marginal_penalty_weight * marginal_loss) + (self.sharpness_weight * sharp_loss)

        if crystallize:
            p_final = self.sample_hard_permutation(p_soft, straight_through=True)
        else:
            p_final = p_soft

        telemetry = {
            "birkhoff_loss_total": float(total_loss.detach().item()),
            "birkhoff_row_err": r_err,
            "birkhoff_col_err": c_err,
            "birkhoff_purity": purity,
            "birkhoff_tau": self.get_temperature_scalar(),
        }

        return p_final, total_loss, telemetry


def verify_log_sinkhorn_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for log_sinkhorn.py.
    """
    torch.manual_seed(42)
    b, d, v = 2, 32, 11

    projector = LogDomainSinkhornBirkhoffProjector(
        d_model=d,
        vocab_size=v,
        num_iterations=20,
        tau_default=0.05,
    )

    rule_vec = torch.randn(b, d, dtype=torch.float32, requires_grad=True)
    roles = torch.rand(b, 10, 8, dtype=torch.float32, requires_grad=True)
    sink = torch.zeros(b, 10, dtype=torch.float32)
    sink[:, 8] = 1.0

    p_mat, loss_birk, _ = projector(rule_vec, role_vectors=roles, diode_sink_mask=sink)

    if p_mat.shape != (b, v, v):
        return False

    row_sums = torch.sum(p_mat, dim=-1)
    if float(torch.max(torch.abs(row_sums - 1.0)).item()) > 1e-4:
        return False

    p_cool, _, _ = projector(rule_vec, tau_override=0.001)
    if torch.isnan(p_cool).any() or torch.isinf(p_cool).any():
        return False

    p_ste, _, _ = projector(rule_vec, crystallize=True)
    test_loss = torch.sum(p_ste.pow(2)) + loss_birk + torch.sum(p_mat.pow(2))
    test_loss.backward()

    if rule_vec.grad is None or projector.tau_param.grad is None or roles.grad is None:
        return False
    if torch.isnan(rule_vec.grad).any() or torch.isnan(projector.tau_param.grad).any():
        return False

    return True