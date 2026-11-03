"""
Location: // fgns_engine/fgns/primitives/potts_latching.py
Description: Annealed Potts-Ginzburg-Landau Multi-Well Energy Landscape, Dual-Bridge STE Crystallizer,
             and Parseval Fiber-Normalized Zero-Gap Max-Envelope Sphere Readout Engine (FGNS v5.0.0):
               1. Orthonormal Grassmannian Palette Manifold on the unit hypersphere S^(D-1).
               2. Theorem I Parseval Fiber-Isolated L2 Normalization: normalizes the 10D color
                  projection c_fiber in R^10 onto the unit sphere S^9 via exact L2 norm
                  ||c_fiber||_2 = sqrt(sum_{c=0}^9 c_c^2), completely eliminating orthogonal
                  117D geometric norm dilution without empirical constants!
               3. Closed-Loop Dynamic Chemical Potential Negative Feedback: beta_c(tau) evaluated from h_tau.
               4. Zero-Gap Max-Envelope Pad Coupling: Z_10(y, x) = max_{c=0..9} stop_grad(Z_c) - psi(y, x),
                  establishing exact sign isomorphism argmax_{0..10} Z_k in {0..9} <=> psi(y, x) > 0.
Language: Python 3.12+ / PyTorch 2.5+ / Statistical Mechanics sm_120 Aligned
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import math
from typing import Final, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fgns.primitives.safe_math import (
    hyperbolic_saturating_norm,
    safe_divide,
    safe_hyperbolic_sqrt,
    safe_reciprocal,
)
from fgns.primitives.swiglu import RMSNorm

DEFAULT_ENERGY_SIGMA_SQ: Final[float] = 0.25
DEFAULT_RESTORATIVE_GAIN: Final[float] = 0.25
DEFAULT_POTTS_EPS: Final[float] = 1e-7
DEFAULT_CHEMICAL_GAIN: Final[float] = 1.00
DEFAULT_COSINE_LOGIT_SCALE: Final[float] = 10.00


class OrthonormalPaletteFrame(nn.Module):
    """
    Maintains an orthonormal Grassmannian frame of discrete color prototypes on unit hypersphere S^(D-1).
    Guarantees maximal angular separation between color attractors: <e_i, e_j> = delta_ij.
    """

    def __init__(self, vocab_size: int = 11, d_model: int = 128) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.raw_prototypes = nn.Parameter(torch.empty(vocab_size, d_model))
        self._initialize_orthonormal_manifold()

    def _initialize_orthonormal_manifold(self) -> None:
        gaussian_matrix = torch.randn(self.d_model, self.vocab_size)
        q_mat, _ = torch.linalg.qr(gaussian_matrix)
        ortho_basis = q_mat.t()[: self.vocab_size, :]
        with torch.no_grad():
            self.raw_prototypes.copy_(ortho_basis)

    def get_normalized_prototypes(self) -> torch.Tensor:
        norm_sq = torch.sum(self.raw_prototypes * self.raw_prototypes, dim=-1, keepdim=True)
        norm = safe_hyperbolic_sqrt(norm_sq, eps=1e-12)
        return self.raw_prototypes / (norm + 1e-12)

    def compute_orthogonality_loss(self) -> torch.Tensor:
        e = self.get_normalized_prototypes()
        gram = torch.matmul(e, e.t())
        identity = torch.eye(self.vocab_size, device=e.device, dtype=e.dtype)
        diff = gram - identity
        return torch.sum(diff * diff)


class StraightThroughPottsCrystallizer(nn.Module):
    """
    Dual-Bridge Differentiable Straight-Through Estimator (STE) Operator.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        continuous_field: torch.Tensor,
        prototypes: torch.Tensor,
        affinity_probabilities: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        discrete_tokens = torch.argmax(affinity_probabilities, dim=-1)
        winner_prototypes = F.embedding(discrete_tokens, prototypes)

        orig_shape = continuous_field.shape
        flat_affinity = affinity_probabilities.reshape(-1, prototypes.shape[0])
        soft_expectation = torch.matmul(flat_affinity, prototypes).view(*orig_shape)

        bridge_surrogate = 0.50 * (continuous_field + soft_expectation)
        crystallized_field = (
            bridge_surrogate
            - bridge_surrogate.detach()
            + winner_prototypes.detach()
        )

        return crystallized_field, discrete_tokens


class PottsLatchingLayer(nn.Module):
    """
    Multi-Well Potts-Ginzburg-Landau Energy Landscape & Parseval Fiber-Normalized Sphere Readout Module.
    """

    def __init__(
        self,
        d_model: int = 128,
        vocab_size: int = 11,
        energy_sigma_sq: float = DEFAULT_ENERGY_SIGMA_SQ,
        restorative_force_gain: float = DEFAULT_RESTORATIVE_GAIN,
        chemical_gain: float = DEFAULT_CHEMICAL_GAIN,
        cosine_logit_scale_init: float = DEFAULT_COSINE_LOGIT_SCALE,
        eps: float = DEFAULT_POTTS_EPS,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.energy_sigma_sq = energy_sigma_sq
        self.restorative_force_gain = restorative_force_gain
        self.chemical_gain = chemical_gain
        self.eps = eps

        self.palette = OrthonormalPaletteFrame(vocab_size=vocab_size, d_model=d_model)
        self.crystallizer = StraightThroughPottsCrystallizer()

        self.logit_scale = nn.Parameter(torch.tensor(float(cosine_logit_scale_init), dtype=torch.float32))
        self.color_logit_bias = nn.Parameter(torch.zeros(10, dtype=torch.float32))
        self.chem_logit_gate = nn.Parameter(torch.tensor(0.20, dtype=torch.float32))

        """Non-Linear Asymmetric Casimir-Chiral Observable Readout Projection"""
        num_cells = max(1, d_model // 16)
        self.casimir_dim = 4 * num_cells
        self.casimir_color_proj = nn.Linear(self.casimir_dim, 10, bias=False)
        self.casimir_logit_gate = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))

        self.reaction_gate = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )
        self.norm = RMSNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.orthogonal_(self.casimir_color_proj.weight, gain=0.25)
        for module in self.reaction_gate.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.25)
                nn.init.zeros_(module.bias)

    def evaluate_dynamic_chemical_biases(
        self,
        h_current: torch.Tensor,
        target_counts: torch.Tensor,
        target_capacity: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, H, W, D = h_current.shape
        prototypes_10 = self.palette.get_normalized_prototypes()[:10, :]

        mask_f32 = target_mask.float().unsqueeze(-1)
        flat_field = (h_current * mask_f32).view(B, H * W, D)

        norm_field = F.normalize(flat_field, p=2, dim=-1, eps=self.eps)
        norm_proto = F.normalize(prototypes_10, p=2, dim=-1, eps=self.eps)

        cos_sim = torch.matmul(norm_field, norm_proto.t())
        soft_probs = F.softmax(cos_sim * 8.0, dim=-1)

        mask_flat = target_mask.view(B, H * W, 1).float()
        soft_occupancy = torch.sum(soft_probs * mask_flat, dim=1)

        deficit = target_counts.float() - soft_occupancy
        sigma_k = safe_hyperbolic_sqrt(target_capacity.float().view(B, 1), eps=self.eps).clamp_min(1.0)

        dynamic_biases = self.chemical_gain * torch.tanh(deficit / sigma_k)
        return dynamic_biases

    def compute_gibbs_affinity(
        self,
        h: torch.Tensor,
        tau: float,
        budget_chemical_potentials: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        orig_shape = h.shape
        batch_dims = orig_shape[:-1]
        prototypes = self.palette.get_normalized_prototypes()

        h_flat = F.normalize(h.reshape(-1, self.d_model), p=2, dim=-1, eps=self.eps)
        p_norm = F.normalize(prototypes, p=2, dim=-1, eps=self.eps)
        dot_product = torch.matmul(h_flat, p_norm.t())

        dist_sq_flat = torch.clamp_min(2.0 - 2.0 * dot_product, 0.0)
        dist_sq = dist_sq_flat.view(*batch_dims, self.vocab_size)

        safe_tau = max(float(tau), 0.25)
        scaled_energy = -dist_sq / (2.0 * safe_tau * self.energy_sigma_sq)

        if budget_chemical_potentials is not None:
            chem_gate = torch.tanh(self.chem_logit_gate.to(scaled_energy.dtype)) * 0.50
            b_pad = F.pad(budget_chemical_potentials * chem_gate, (0, 1), mode="constant", value=0.0)

            if len(batch_dims) == 3:
                b_size = batch_dims[0]
                potential_bias = b_pad.view(b_size, 1, 1, self.vocab_size)
            elif len(batch_dims) == 1:
                potential_bias = b_pad.view(batch_dims[0], self.vocab_size)
            else:
                potential_bias = b_pad.view(batch_dims[0], *([1] * (len(batch_dims) - 1)), self.vocab_size)

            scaled_energy = scaled_energy + potential_bias

        affinity = F.softmax(scaled_energy, dim=-1)
        return affinity, dist_sq

    def compute_unified_sphere_logits(
        self,
        h: torch.Tensor,
        carrier_psi: torch.Tensor,
        budget_chemical_potentials: Optional[torch.Tensor] = None,
        casimir_invariants: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Theorem I Parseval Fiber-Normalized Sphere Readout & Zero-Gap Max-Envelope Coupling:
            c_raw(y, x)   = <h(y, x), e_{0:9}> in R^10
            c_fiber(y, x) = c_raw(y, x) + tanh(g_Casimir) * (I_Casimir(y, x) W_Casimir^T)
            c_unit(y, x)  = c_fiber(y, x) / sqrt(||c_fiber(y, x)||_2^2 + eps) in S^9
            Z_c(y, x)     = gamma_cos * c_unit_c(y, x) + b_class_c + 0.5 * tanh(g_chem) * beta_c
            Z_10(y, x)    = max_{c=0..9}(stop_grad(Z_c(y, x))) - psi(y, x).
        By Parseval's identity ||P_color h||_R^D == ||c_raw||_R^10, normalizing on S^9 completely
        eliminates orthogonal H_geom^(117) norm dilution with zero heuristic constants!
        """
        target_dtype = h.dtype
        h_f32 = h.to(torch.float32)
        psi_f32 = carrier_psi.to(torch.float32)
        if psi_f32.ndim == h_f32.ndim - 1:
            psi_f32 = psi_f32.unsqueeze(-1)

        prototypes_10 = self.palette.get_normalized_prototypes()[:10, :].to(torch.float32)

        raw_color_coords = torch.matmul(h_f32, prototypes_10.t())

        if casimir_invariants is not None and casimir_invariants.shape[-1] == self.casimir_dim:
            g_cas = torch.tanh(self.casimir_logit_gate.to(torch.float32)) * 0.35
            cas_coords = F.linear(
                casimir_invariants.to(torch.float32),
                self.casimir_color_proj.weight.to(torch.float32),
            )
            fiber_coords = raw_color_coords + (g_cas * cas_coords)
        else:
            fiber_coords = raw_color_coords

        fiber_norm_sq = torch.sum(fiber_coords * fiber_coords, dim=-1, keepdim=True)
        fiber_norm = safe_hyperbolic_sqrt(fiber_norm_sq, eps=1e-12)
        c_unit = fiber_coords / (fiber_norm + 1e-7)

        scale_clamped = torch.clamp(self.logit_scale.to(torch.float32), min=2.0, max=25.0)
        color_logits = (c_unit * scale_clamped) + self.color_logit_bias

        if budget_chemical_potentials is not None:
            b_size = h.shape[0]
            chem_gate = torch.tanh(self.chem_logit_gate.to(torch.float32)) * 0.50
            chem_bias = (budget_chemical_potentials.to(torch.float32) * chem_gate).view(b_size, 1, 1, 10)
            color_logits = color_logits + chem_bias

        max_colors_detached = torch.amax(color_logits.detach(), dim=-1, keepdim=True)
        pad_logit = max_colors_detached - torch.clamp(psi_f32, min=-25.0, max=25.0)

        unified_logits = torch.cat([color_logits, pad_logit], dim=-1)
        return unified_logits.to(target_dtype)

    def embed_draft_probabilities(
        self,
        color_probs_10: torch.Tensor,
        active_mask_prob: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prototypes_10 = self.palette.get_normalized_prototypes()[:10, :].to(color_probs_10.dtype)
        draft_vec = torch.matmul(color_probs_10, prototypes_10) * math.sqrt(float(self.d_model))
        if active_mask_prob is not None:
            prob_exp = (
                active_mask_prob.unsqueeze(-1)
                if active_mask_prob.ndim == draft_vec.ndim - 1
                else active_mask_prob
            )
            draft_vec = draft_vec * prob_exp.to(draft_vec.dtype)
        return draft_vec

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor,
        tau: float,
        crystallize: bool = False,
        budget_potentials: Optional[torch.Tensor] = None,
        target_counts: Optional[torch.Tensor] = None,
        target_capacity: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        orig_shape = h.shape
        mask_f32 = mask.float().unsqueeze(-1)
        prototypes = self.palette.get_normalized_prototypes()

        if target_counts is not None and target_capacity is not None and h.dim() == 4:
            active_biases = self.evaluate_dynamic_chemical_biases(
                h_current=h,
                target_counts=target_counts,
                target_capacity=target_capacity,
                target_mask=mask,
            )
        else:
            active_biases = budget_potentials

        affinity, _ = self.compute_gibbs_affinity(
            h=h,
            tau=tau,
            budget_chemical_potentials=active_biases,
        )

        flat_affinity = affinity.reshape(-1, self.vocab_size)
        latched_state_flat = torch.matmul(flat_affinity, prototypes) * math.sqrt(float(self.d_model))
        latched_state = latched_state_flat.view(*orig_shape)

        restoring_force = latched_state - h
        gate = torch.sigmoid(self.reaction_gate(h))
        h_relaxed = h + (restoring_force * gate * self.restorative_force_gain)

        if crystallize:
            crystallized_field, tokens = self.crystallizer(
                continuous_field=h_relaxed,
                prototypes=prototypes * math.sqrt(float(self.d_model)),
                affinity_probabilities=affinity,
            )
            h_out = self.norm(crystallized_field) * mask_f32
        else:
            tokens = torch.argmax(affinity, dim=-1)
            h_out = self.norm(h_relaxed) * mask_f32

        return h_out, tokens, affinity, active_biases


def verify_potts_latching_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for potts_latching.py.
    """
    torch.manual_seed(42)
    b, h, w, d = 2, 4, 4, 32
    vocab = 11

    layer = PottsLatchingLayer(d_model=d, vocab_size=vocab, energy_sigma_sq=0.25)

    ortho_loss = layer.palette.compute_orthogonality_loss()
    if float(ortho_loss.item()) > 1e-4:
        return False

    h_test = torch.randn(b, h, w, d, dtype=torch.float32)
    target_mask = torch.ones(b, h, w, dtype=torch.bool)

    target_counts = torch.zeros(b, 10, dtype=torch.float32)
    target_counts[:, 1] = 16.0
    target_counts[:, 2] = 0.0
    target_capacity = torch.tensor([16.0, 16.0], dtype=torch.float32)

    biases = layer.evaluate_dynamic_chemical_biases(
        h_current=h_test,
        target_counts=target_counts,
        target_capacity=target_capacity,
        target_mask=target_mask,
    )

    if biases.shape != (b, 10):
        return False
    if float(biases[0, 1].item()) <= 0.10 or float(biases[0, 2].item()) > 0.05:
        return False

    carrier_psi = torch.randn(b, h, w, dtype=torch.float32, requires_grad=True)
    casimir_obs = torch.randn(b, h, w, 4 * (d // 16), dtype=torch.float32, requires_grad=True)
    unified_logits = layer.compute_unified_sphere_logits(
        h=h_test,
        carrier_psi=carrier_psi,
        budget_chemical_potentials=biases,
        casimir_invariants=casimir_obs,
    )
    if unified_logits.shape != (b, h, w, 11):
        return False

    max_0_9 = torch.amax(unified_logits[..., :10], dim=-1)
    z_10 = unified_logits[..., 10]
    recovered_psi = max_0_9 - z_10
    if float(torch.max(torch.abs(recovered_psi - carrier_psi)).item()) > 1e-4:
        return False

    """Verify Theorem I Exact Sign Equivalence: argmax in {0..9} <=> carrier_psi > 0"""
    top_class = torch.argmax(unified_logits, dim=-1)
    active_by_argmax = top_class < 10
    active_by_psi = carrier_psi > 0.0
    if not torch.equal(active_by_argmax, active_by_psi):
        return False

    h_field = torch.randn(b, h, w, d, dtype=torch.float32, requires_grad=True)
    mask = torch.ones(b, h, w, dtype=torch.bool)
    mask[:, :, 2:] = False

    h_out, tokens, affinity, _ = layer(
        h=h_field,
        mask=mask,
        tau=0.02,
        crystallize=True,
        target_counts=target_counts,
        target_capacity=target_capacity,
    )

    if torch.isnan(h_out).any() or torch.isinf(h_out).any():
        return False
    if not torch.all(h_out[:, :, 2:, :] == 0.0):
        return False

    loss = torch.sum(h_out.pow(2)) + torch.sum(unified_logits.pow(2)) + ortho_loss
    loss.backward()

    if h_field.grad is None or layer.palette.raw_prototypes.grad is None or carrier_psi.grad is None or casimir_obs.grad is None:
        return False
    if torch.isnan(h_field.grad).any() or torch.isnan(layer.palette.raw_prototypes.grad).any():
        return False
    if float(torch.norm(h_field.grad).item()) <= 0.0:
        return False

    return True