"""
Location: // fgns_engine/fgns/primitives/potts_latching.py
Description: SOTA Annealed Potts-Ginzburg-Landau Multi-Well Energy Landscape, Dual-Bridge STE Crystallizer,
             and Calibrated Unified Isometric Sphere Readout Engine on S^(D-1).
             Core Mathematical Foundation for FGNS v1.0.0 (Sovereign):
               1. Orthonormal Grassmannian Palette Manifold on the unit hypersphere S^(D-1).
               2. Closed-Loop Dynamic Chemical Potential Negative Feedback: beta_c(tau) evaluated from h_tau.
               3. Scale-Invariant Grand Canonical Occupancy Deficit normalized by sqrt(K_target).
               4. Dual-Bridge Straight-Through Estimator (STE): snaps forward states to discrete e_{c*}
                  while routing backward gradients simultaneously through h_relaxed and soft Gibbs affinity.
               5. Calibrated Unified Sphere Readout (Theorem 1): Evaluates Z_{0..9} from continuous fields
                  with gated chemical bias and learnable class priors, preventing global color flooding.
               6. Strict Dirichlet zero-boundary enforcement on all masked padding regions.
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
DEFAULT_RESTORATIVE_GAIN: Final[float] = 0.35
DEFAULT_POTTS_EPS: Final[float] = 1e-7
DEFAULT_CHEMICAL_GAIN: Final[float] = 1.00
DEFAULT_COSINE_LOGIT_SCALE: Final[float] = 12.00


class OrthonormalPaletteFrame(nn.Module):
    """
    Maintains an orthonormal Grassmannian frame of discrete color prototypes on unit hypersphere S^(D-1).
    Guarantees maximal angular separation between color attractors: <e_i, e_j> = delta_ij.
    Supports uninterrupted gradient propagation flowing back into prototype coordinates during TTT.
    """

    def __init__(self, vocab_size: int = 11, d_model: int = 128) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        """Unconstrained learnable prototype embeddings"""
        self.raw_prototypes = nn.Parameter(torch.empty(vocab_size, d_model))
        self._initialize_orthonormal_manifold()

    def _initialize_orthonormal_manifold(self) -> None:
        """Initializes prototypes via QR factorization to form an exact orthonormal basis"""
        gaussian_matrix = torch.randn(self.d_model, self.vocab_size)
        q_mat, _ = torch.linalg.qr(gaussian_matrix)
        ortho_basis = q_mat.t()[: self.vocab_size, :]
        with torch.no_grad():
            self.raw_prototypes.copy_(ortho_basis)

    def get_normalized_prototypes(self) -> torch.Tensor:
        """
        Projects raw embeddings onto unit hypersphere S^(D-1) via safe hyperbolic norm:
            e_c = raw_e_c / ||raw_e_c||_2.
        Returns: [VocabSize, D_model]
        """
        norm_sq = torch.sum(self.raw_prototypes * self.raw_prototypes, dim=-1, keepdim=True)
        norm = safe_hyperbolic_sqrt(norm_sq, eps=1e-12)
        return self.raw_prototypes / (norm + 1e-12)

    def compute_orthogonality_loss(self) -> torch.Tensor:
        """
        Penalizes deviation from Gram matrix identity: ||E E^T - I||_F^2.
        Prevents prototype collapse and maintains maximum entropy separation.
        """
        e = self.get_normalized_prototypes()
        gram = torch.matmul(e, e.t())
        identity = torch.eye(self.vocab_size, device=e.device, dtype=e.dtype)
        diff = gram - identity
        return torch.sum(diff * diff)


class StraightThroughPottsCrystallizer(nn.Module):
    """
    Dual-Bridge Differentiable Straight-Through Estimator (STE) Operator.
    Snaps continuous latent features to exact discrete color prototypes on the forward pass
    while routing analytical gradients through both the continuous field and the soft Gibbs expectation:
        h_crystal = 0.5 * (h + h_soft) - detach(0.5 * (h + h_soft)) + e_{c*}.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        continuous_field: torch.Tensor,
        prototypes: torch.Tensor,
        affinity_probabilities: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            continuous_field: Continuous relaxed state field [..., D_model].
            prototypes: Normalized color prototypes [VocabSize, D_model].
            affinity_probabilities: Gibbs probability distribution over colors [..., VocabSize].

        Returns:
            crystallized_field: Discrete-snapped tensor [..., D_model].
            discrete_tokens: Argmax integer class indices [...].
        """
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
    SOTA Multi-Well Potts-Ginzburg-Landau Energy Landscape & Calibrated Sphere Readout Module.
    Simulates non-equilibrium physical relaxation of continuous feature fields into discrete palette crystals.
    Features gated chemical potential coupling and learnable class priors so that global budget quotas
    guide crystallization without overpowering local spatial features.
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

        """Learnable inverse temperature scale gamma_cos and class prior bias for Sphere Readout"""
        self.logit_scale = nn.Parameter(torch.tensor(float(cosine_logit_scale_init), dtype=torch.float32))
        self.color_logit_bias = nn.Parameter(torch.zeros(10, dtype=torch.float32))
        self.chem_logit_gate = nn.Parameter(torch.tensor(0.20, dtype=torch.float32))

        """Spatial reaction gating controlling local continuous fluidity"""
        self.reaction_gate = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )
        self.norm = RMSNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initializes reaction gates to allow fluid exploration before crystallization"""
        for module in self.reaction_gate.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def evaluate_dynamic_chemical_biases(
        self,
        h_current: torch.Tensor,
        target_counts: torch.Tensor,
        target_capacity: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Closed-Loop Dynamic Chemical Potential Negative Feedback Controller.
        Evaluates the instant color deficit by inspecting the soft occupancy of the CURRENT field h_current
        strictly within the active boundaries of target_mask.
        """
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
        """
        Evaluates Boltzmann-Gibbs thermodynamic affinity distribution over color prototypes.
        """
        orig_shape = h.shape
        batch_dims = orig_shape[:-1]
        prototypes = self.palette.get_normalized_prototypes()

        h_flat = h.reshape(-1, self.d_model)

        h_norm_sq = torch.sum(h_flat * h_flat, dim=-1, keepdim=True)
        p_norm_sq = torch.sum(prototypes * prototypes, dim=-1).view(1, self.vocab_size)
        dot_product = torch.matmul(h_flat, prototypes.t())

        dist_sq_flat = torch.clamp_min(h_norm_sq + p_norm_sq - 2.0 * dot_product, 0.0)
        dist_sq = dist_sq_flat.view(*batch_dims, self.vocab_size)

        safe_tau = max(float(tau), 0.10)
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
    ) -> torch.Tensor:
        """
        SOTA Theorem 1: Calibrated Unified Isometric Sphere Readout & Algebraic Carrier Field Decoupling.
        Evaluates domain color logits Z_{0..9} from continuous field h as scaled cosine similarities
        to orthonormal palette prototypes E_{0..9} on S^(D-1) plus gated chemical bias:
            Z_c(y, x) = gamma_cos * <h_unit(y, x), e_c> + b_class_c + tanh(g_chem) * beta_c.
        Evaluates exterior PAD token logit Z_10 algebraically via carrier log-odds field psi(y, x):
            Z_10(y, x) = LogSumExp_{c=0..9}(Z_c(y, x)) - psi(y, x).
        """
        target_dtype = h.dtype
        h_f32 = h.to(torch.float32)
        psi_f32 = carrier_psi.to(torch.float32)
        if psi_f32.ndim == h_f32.ndim - 1:
            psi_f32 = psi_f32.unsqueeze(-1)

        prototypes_10 = self.palette.get_normalized_prototypes()[:10, :].to(torch.float32)

        """1. Normalize field onto unit hypersphere S^(D-1) with safe hyperbolic norm"""
        h_norm_sq = torch.sum(h_f32 * h_f32, dim=-1, keepdim=True)
        h_unit = h_f32 / (safe_hyperbolic_sqrt(h_norm_sq, eps=1e-12) + 1e-7)

        """2. Isometric Cosine Projection onto Colors 0..9 + Learnable Class Prior"""
        scale_clamped = torch.clamp(self.logit_scale.to(torch.float32), min=2.0, max=30.0)
        color_logits = (torch.matmul(h_unit, prototypes_10.t()) * scale_clamped) + self.color_logit_bias

        if budget_chemical_potentials is not None:
            b_size = h.shape[0]
            chem_gate = torch.tanh(self.chem_logit_gate.to(torch.float32)) * 0.50
            chem_bias = (budget_chemical_potentials.to(torch.float32) * chem_gate).view(b_size, 1, 1, 10)
            color_logits = color_logits + chem_bias

        """3. Algebraic Carrier Decoupling for PAD Token 10: Z_10 = LSE(Z_{0..9}) - psi"""
        lse_colors = torch.logsumexp(color_logits, dim=-1, keepdim=True)
        pad_logit = lse_colors - torch.clamp(psi_f32, min=-25.0, max=25.0)

        unified_logits = torch.cat([color_logits, pad_logit], dim=-1)
        return unified_logits.to(target_dtype)

    def embed_draft_probabilities(
        self,
        color_probs_10: torch.Tensor,
        active_mask_prob: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Projects 10-color probability simplex distribution P in Delta^9 back onto S^(D-1):
            E_draft(y, x) = (sum_{c=0}^9 P_c(y, x) * e_c) * active_prob(y, x).
        """
        prototypes_10 = self.palette.get_normalized_prototypes()[:10, :].to(color_probs_10.dtype)
        draft_vec = torch.matmul(color_probs_10, prototypes_10)
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
        """
        Executes thermodynamic Ginzburg-Landau relaxation and straight-through crystallization.
        """
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
        latched_state_flat = torch.matmul(flat_affinity, prototypes)
        latched_state = latched_state_flat.view(*orig_shape)

        restoring_force = latched_state - h
        gate = torch.sigmoid(self.reaction_gate(h))
        h_relaxed = h + (restoring_force * gate * self.restorative_force_gain)

        if crystallize or tau <= 0.05:
            crystallized_field, tokens = self.crystallizer(
                continuous_field=h_relaxed,
                prototypes=prototypes,
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
    Validates:
      1. Orthogonality of palette prototypes: ||E E^T - I||_F < 1e-4 upon initialization.
      2. Dynamic Chemical Negative Feedback: Quota overshoot produces negative potential, deficit positive.
      3. Theorem 1 Unified Sphere Readout: LogSumExp(Z_0..9) - Z_10 identically equals carrier_psi.
      4. Dual-Bridge STE Crystallization: Gradients flow cleanly into both h_field and raw_prototypes.
      5. Dirichlet Boundary Invariance: Masked padding positions remain identically zero.
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
    unified_logits = layer.compute_unified_sphere_logits(
        h=h_test,
        carrier_psi=carrier_psi,
        budget_chemical_potentials=biases,
    )
    if unified_logits.shape != (b, h, w, 11):
        return False

    lse_0_9 = torch.logsumexp(unified_logits[..., :10], dim=-1)
    z_10 = unified_logits[..., 10]
    recovered_psi = lse_0_9 - z_10
    if float(torch.max(torch.abs(recovered_psi - carrier_psi)).item()) > 1e-4:
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

    if h_field.grad is None or layer.palette.raw_prototypes.grad is None or carrier_psi.grad is None:
        return False
    if torch.isnan(h_field.grad).any() or torch.isnan(layer.palette.raw_prototypes.grad).any():
        return False
    if float(torch.norm(h_field.grad).item()) <= 0.0:
        return False

    return True