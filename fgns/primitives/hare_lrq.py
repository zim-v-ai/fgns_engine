"""
Location: // fgns_engine/fgns/primitives/hare_lrq.py
Description: SOTA Zero-Sync Vectorized Differentiable Hamilton-Hare Largest Remainder Quota (LRQ) Projector.
             Core Mathematical Atom for FGNS v1.0.0 (Sovereign):
               1. Fundamental bridge of Equivariant Palette Histogram Predictor (NPCO).
               2. Snaps continuous simplex proportions p in Delta^(C-1) and capacity K into N^C.
               3. Enforces exact physical mass conservation: sum_{c=0}^(C-1) N_c* == round(K) <= 900.
               4. 100% GPU-vectorized remainder ranking via double argsort, completely eliminating
                  CPU-GPU synchronization (.item()) and Python batch loops.
               5. Dual-gradient Autograd Function propagating analytical derivatives to p and K.
Invariants: Strict integer output N_c* in N, exact sum identity sum(N*) == round(K),
            non-vanishing gradients dL/dp and dL/dK, zero hash comment compliance.
Language: Python 3.12+ / PyTorch 2.5+ / Discrete Optimization sm_120 Aligned
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


class StraightThroughHareQuota(torch.autograd.Function):
    """
    Zero-Sync Vectorized Differentiable Hamilton-Hare Largest Remainder Quota Autograd Operator.
    Forward pass executes discrete apportionment on GPU registers yielding exact integer counts.
    Backward pass evaluates analytical chain rule derivatives w.r.t both probability simplex
    and continuous capacity via a smooth bilinear surrogate envelope.
    """

    @staticmethod
    def forward(
        ctx,
        simplex_probabilities: torch.Tensor,
        capacities: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apportions integer multiset counts from continuous proportions and total capacity
        using pure vectorized tensor operations with zero CPU-GPU synchronization.

        Args:
            ctx: PyTorch autograd context.
            simplex_probabilities: Class proportions of shape [Batch, NumClasses] in Delta^(C-1).
            capacities: Continuous or integer target area capacity of shape [Batch] or [Batch, 1].

        Returns:
            discrete_counts: Integer multiset counts tensor of shape [Batch, NumClasses] summing to K.
        """
        dtype = simplex_probabilities.dtype
        batch_size, num_classes = simplex_probabilities.shape

        cap_f32 = capacities.to(dtype=torch.float32).view(batch_size, 1)
        p_f32 = simplex_probabilities.to(dtype=torch.float32)

        """1. Ensure probability simplex normalization: sum_c p_c == 1.0"""
        sum_p = torch.sum(p_f32, dim=-1, keepdim=True).clamp_min(1e-12)
        p_normalized = p_f32 / sum_p

        """2. Evaluate ideal continuous quotas: q_c = p_c * K"""
        continuous_quotas = p_normalized * cap_f32

        """3. Evaluate guaranteed integer baseline allocations: N_base = floor(q_c)"""
        integer_baseline = torch.floor(continuous_quotas)
        allocated_so_far = torch.sum(integer_baseline, dim=-1, keepdim=True)

        """4. Evaluate residual integer seat deficit: R = clamp(round(K) - sum(N_base), 0, C)"""
        target_k_int = torch.round(cap_f32).to(torch.int64)
        deficit = torch.clamp(target_k_int - allocated_so_far.to(torch.int64), min=0, max=num_classes)

        """5. Evaluate fractional remainders: r_c = q_c - N_base in [0, 1)"""
        remainders = continuous_quotas - integer_baseline

        """
        6. Zero-Sync Vectorized Remainder Ranking on GPU:
        Double argsort computes exact 0-indexed descending rank per class without .item() or loops!
        """
        sorted_order = torch.argsort(remainders, dim=-1, descending=True, stable=True)
        rank_positions = torch.argsort(sorted_order, dim=-1)
        bonus_allocations = (rank_positions < deficit).to(dtype=torch.float32)

        discrete_counts = integer_baseline + bonus_allocations

        ctx.save_for_backward(p_normalized, cap_f32)
        return discrete_counts.to(dtype)

    @staticmethod
    def backward(
        ctx,
        grad_counts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Analytical dual gradient backpropagation.
        Propagates sensitivity vectors through the continuous surrogate envelope:
            q_c = p_c * K.
            dL/dp_c = K * grad_c
            dL/dK   = sum_c (p_c * grad_c).
        """
        p_normalized, cap_f32 = ctx.saved_tensors
        g_f32 = grad_counts.to(torch.float32)

        grad_probabilities = g_f32 * cap_f32
        grad_capacities = torch.sum(g_f32 * p_normalized, dim=-1)

        return grad_probabilities.to(grad_counts.dtype), grad_capacities.to(grad_counts.dtype)


def differentiable_hare_lrq(
    simplex_probabilities: torch.Tensor,
    capacities: torch.Tensor,
) -> torch.Tensor:
    """
    Public functional gateway for Differentiable Hamilton-Hare Largest Remainder Quota projection.
    Enforces the fundamental physical discrete conservation identity:
        sum_c N_c* == round(K) <= 900.
    Preserves continuous gradient pathways flowing into both simplex logits and capacity estimators.
    """
    orig_shape = simplex_probabilities.shape
    num_classes = orig_shape[-1]
    batch_dims = orig_shape[:-1]

    flat_batch = 1
    for b_dim in batch_dims:
        flat_batch *= b_dim

    flat_probs = simplex_probabilities.reshape(flat_batch, num_classes)
    flat_caps = capacities.reshape(flat_batch)

    discrete_counts_flat = StraightThroughHareQuota.apply(flat_probs, flat_caps)
    return discrete_counts_flat.view(*orig_shape)


def continuous_quota_surplus(
    simplex_probabilities: torch.Tensor,
    capacities: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluates continuous fractional remainders and integer baselines without discrete snapping.
    """
    p_norm = F.normalize(simplex_probabilities.to(torch.float32), p=1, dim=-1)
    cap_exp = capacities.to(torch.float32).unsqueeze(-1)

    continuous_q = p_norm * cap_exp
    baseline = torch.floor(continuous_q)
    remainders = continuous_q - baseline

    return baseline.to(simplex_probabilities.dtype), remainders.to(simplex_probabilities.dtype)


def verify_hare_lrq_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for hare_lrq.py.
    Validates:
      1. Discrete Conservation Identity: sum_c N_c* == round(K) across arbitrary batch instances.
      2. Non-Negativity Invariant: N_c* >= 0 for all classes.
      3. Largest Remainder Property: Surplus is allocated strictly to largest fractional remainders.
      4. Dual-Gradient Flow: Gradients propagate simultaneously to probabilities and capacity.
      5. Rank Polymorphism: Handles 2D matrices and multi-dimensional spatial batches identically.
    """
    torch.manual_seed(42)

    batch_size = 4
    num_classes = 10
    raw_logits = torch.randn(batch_size, num_classes, dtype=torch.float32, requires_grad=True)
    probs = F.softmax(raw_logits, dim=-1)

    target_caps = torch.tensor([9.0, 81.0, 100.0, 400.0], dtype=torch.float32, requires_grad=True)

    discrete_counts = differentiable_hare_lrq(probs, target_caps)

    if discrete_counts.shape != (batch_size, num_classes):
        return False
    if torch.any(discrete_counts < 0.0):
        return False
    if torch.any(discrete_counts != torch.floor(discrete_counts)):
        return False

    allocated_sums = torch.sum(discrete_counts, dim=-1)
    target_rounds = torch.round(target_caps)
    sum_errors = torch.abs(allocated_sums - target_rounds)
    if float(torch.max(sum_errors).item()) > 1e-5:
        return False

    test_p = torch.tensor([[0.44, 0.34, 0.22]], dtype=torch.float32)
    test_k = torch.tensor([10.0], dtype=torch.float32)
    test_n = differentiable_hare_lrq(test_p, test_k)
    if abs(float(torch.sum(test_n).item()) - 10.0) > 1e-5:
        return False
    if int(test_n[0, 0].item()) != 5 or int(test_n[0, 1].item()) != 3 or int(test_n[0, 2].item()) != 2:
        return False

    target_profile = torch.tensor([
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1.0, 0.0],
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        [40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0],
    ], dtype=torch.float32)

    loss = F.mse_loss(discrete_counts, target_profile)
    loss.backward()

    if raw_logits.grad is None or target_caps.grad is None:
        return False
    if torch.isnan(raw_logits.grad).any() or torch.isnan(target_caps.grad).any():
        return False
    if float(torch.norm(raw_logits.grad).item()) <= 0.0 or float(torch.norm(target_caps.grad).item()) <= 0.0:
        return False

    p_3d = torch.ones(2, 3, 10, dtype=torch.float32) / 10.0
    k_3d = torch.full((2, 3), 30.0, dtype=torch.float32)
    n_3d = differentiable_hare_lrq(p_3d, k_3d)

    if n_3d.shape != (2, 3, 10):
        return False
    if float(torch.max(torch.abs(torch.sum(n_3d, dim=-1) - 30.0)).item()) > 1e-5:
        return False

    return True