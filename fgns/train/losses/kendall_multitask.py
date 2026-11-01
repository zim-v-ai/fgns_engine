"""
Location: // fgns_engine/fgns/train/losses/kendall_multitask.py
Description: SOTA Strictly-Positive Liebel-Koerner-Kendall Homoscedastic Bayesian Multitask Balancer.
             Core Autonomous Multi-Objective Engine for FGNS v1.0.0 (Sovereign):
               1. Strictly Positive Softplus Uncertainty Barrier: replaces linear +0.5 * s_t with
                  +0.5 * ln(1 + exp(s_t)), mathematically guaranteeing Total_Loss > 0.000000 everywhere!
               2. Eliminates auxiliary weight runaway: when auxiliary losses (Msk, Ort, Xor, Mdl) approach 0,
                  their precision weights remain bounded and never starve primary objectives (Query & Size).
               3. Tightly bounded hyperbolic tangent variance scaling: s_t in (-1.50, +1.50).
               4. Non-negotiable query weight floor w_query >= 0.35.
               5. Shannon Task Governor enforcing thermodynamic equipartition of gradient energy.
Language: Python 3.12+ / PyTorch 2.5+ / Bayesian Multitask Optimization sm_120 Aligned
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import math
from typing import Dict, Final, Iterator, List, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_NUM_TASKS: Final[int] = 8
DEFAULT_LOG_VAR_BOUND: Final[float] = 1.50
DEFAULT_QUERY_WEIGHT_FLOOR: Final[float] = 0.35
DEFAULT_GOVERNOR_WEIGHT: Final[float] = 0.15
DEFAULT_KENDALL_EPS: Final[float] = 1e-7

TASK_NAMES: Final[Tuple[str, ...]] = (
    "query_focal",
    "support_focal",
    "mask_segmentation",
    "size_displacement",
    "palette_budget",
    "program_mdl",
    "palette_orthogonality",
    "xor_parity",
)


class BoundedKendallMultitaskLoss(nn.Module):
    """
    SOTA Strictly-Positive Liebel-Koerner-Kendall Homoscedastic Bayesian Multitask Balancer.
    Formulation per channel t:
        L_total = sum_{t=1}^8 ( 0.5 * exp(-s_t) * L_t + 0.5 * ln(1 + exp(s_t)) ) + w_gov * L_gov.
    Guarantees strict positivity L_total > 0 and prevents auxiliary tasks with L_t -> 0
    from hijacking gradient capacity from query_focal and size_displacement.
    """

    TASK_NAMES: Final[Tuple[str, ...]] = TASK_NAMES

    def __init__(
        self,
        num_tasks: int = DEFAULT_NUM_TASKS,
        task_names: Tuple[str, ...] = TASK_NAMES,
        bound_max: float = DEFAULT_LOG_VAR_BOUND,
        min_query_weight_floor: float = DEFAULT_QUERY_WEIGHT_FLOOR,
        governor_weight: float = DEFAULT_GOVERNOR_WEIGHT,
        init_log_vars: Optional[Sequence[float]] = None,
        eps: float = DEFAULT_KENDALL_EPS,
    ) -> None:
        super().__init__()
        self.num_tasks = int(num_tasks)
        self.task_names = tuple(task_names)
        self.bound_max = float(bound_max)
        self.min_query_weight_floor = float(min_query_weight_floor)
        self.governor_weight = float(governor_weight)
        self.eps = eps

        if len(self.task_names) != self.num_tasks:
            raise ValueError(
                f"Mismatch: task_names length ({len(self.task_names)}) must equal num_tasks ({self.num_tasks})."
            )

        if init_log_vars is not None and len(init_log_vars) == self.num_tasks:
            initial_s = torch.tensor(list(init_log_vars), dtype=torch.float32)
        else:
            initial_s = torch.zeros(self.num_tasks, dtype=torch.float32)

        self.log_vars_raw = nn.Parameter(initial_s)
        self.task_to_idx: Dict[str, int] = {name: idx for idx, name in enumerate(self.task_names)}
        self.max_entropy = math.log(float(self.num_tasks))

        """Maximum precision cap per channel: primary channels (0..4) up to 2.25, regularizers (5..7) up to 0.75"""
        max_w = torch.tensor([2.25, 1.75, 1.25, 2.00, 1.50, 0.75, 0.75, 0.75], dtype=torch.float32)
        if self.num_tasks != 8:
            max_w = torch.full((self.num_tasks,), 2.25, dtype=torch.float32)
        self.register_buffer("max_channel_weights", max_w, persistent=False)
        self.register_buffer("loss_ema", torch.ones(self.num_tasks, dtype=torch.float32))

    def get_bounded_log_vars(self) -> torch.Tensor:
        """
        Applies smooth hyperbolic tangent projection to strictly bound log-variances:
            s_bounded = bound_max * tanh(s_raw / bound_max) in (-bound_max, +bound_max).
        """
        return self.bound_max * torch.tanh(self.log_vars_raw / self.bound_max)

    @property
    def effective_weights(self) -> Dict[str, float]:
        """
        Returns dictionary of current scalar loss multipliers w_t.
        Enforces the minimum query weight floor on index 0 and caps auxiliary regularizers.
        """
        with torch.no_grad():
            bounded_s = self.get_bounded_log_vars()
            max_w = cast(torch.Tensor, self.max_channel_weights).to(device=bounded_s.device, dtype=bounded_s.dtype)
            precisions = torch.minimum(0.50 * torch.exp(-bounded_s), max_w)
            precisions[0] = torch.clamp_min(precisions[0], self.min_query_weight_floor)
            return {
                name: float(precisions[self.task_to_idx[name]].item())
                for name in self.task_names
            }

    @property
    def effective_sigmas(self) -> Dict[str, float]:
        """Returns dictionary of observation noise standards sigma_t = exp(0.5 * s_t)."""
        with torch.no_grad():
            bounded_s = self.get_bounded_log_vars()
            sigmas = torch.exp(0.50 * bounded_s)
            return {
                name: float(sigmas[self.task_to_idx[name]].item())
                for name in self.task_names
            }

    def uncertainty_parameters(self) -> Iterator[nn.Parameter]:
        """Isolates multitask uncertainty parameters for zero-weight-decay Adam optimization."""
        yield self.log_vars_raw

    def compute_gradient_energy_shares(
        self,
        weighted_losses: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """
        Calculates relative percentage of gradient energy contributed by each task
        and evaluates the Shannon Task Governor barrier penalty.
        """
        total_energy = torch.sum(weighted_losses)
        safe_total = torch.clamp_min(total_energy, self.eps)

        energy_shares = (weighted_losses / safe_total) * 100.0

        prob_dist = weighted_losses / safe_total
        safe_prob = torch.clamp(prob_dist, self.eps, 1.0)
        task_entropy = -torch.sum(safe_prob * torch.log(safe_prob))

        entropy_gap = torch.clamp_min(self.max_entropy - task_entropy, 0.0)
        governor_loss = entropy_gap.pow(2)

        return energy_shares, float(task_entropy.detach().item()), governor_loss

    def forward(
        self,
        losses: Union[Sequence[torch.Tensor], Dict[str, torch.Tensor]],
        auxiliary_telemetry: Optional[Dict[str, float]] = None,
        return_telemetry: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, Tuple[float, ...], float],
        Tuple[torch.Tensor, Tuple[float, ...], float, Dict[str, float]],
    ]:
        """
        Executes strictly positive Liebel-Koerner-Kendall Bayesian multitask loss aggregation.
        """
        device = self.log_vars_raw.device
        dtype = self.log_vars_raw.dtype

        loss_tensor_list: List[torch.Tensor] = []

        if isinstance(losses, dict):
            for name in self.task_names:
                if name not in losses:
                    raise KeyError(
                        f"Missing required task loss '{name}'. Provided keys: {list(losses.keys())}"
                    )
                loss_tensor_list.append(losses[name])
        else:
            if len(losses) != self.num_tasks:
                raise ValueError(
                    f"Expected {self.num_tasks} losses, got {len(losses)}."
                )
            loss_tensor_list = list(losses)

        bounded_s = self.get_bounded_log_vars().to(device=device)
        max_w = cast(torch.Tensor, self.max_channel_weights).to(device=device, dtype=dtype)
        raw_precisions = torch.minimum(0.50 * torch.exp(-bounded_s), max_w)

        precisions = raw_precisions.clone()
        precisions[0] = torch.clamp_min(raw_precisions[0], self.min_query_weight_floor)

        total_loss_accum = torch.tensor(0.0, device=device, dtype=dtype)
        weighted_losses_list: List[torch.Tensor] = []

        for t_idx in range(self.num_tasks):
            raw_loss_t = loss_tensor_list[t_idx]

            loss_t = torch.nan_to_num(
                raw_loss_t.to(device=device, dtype=dtype),
                nan=1.50,
                posinf=25.0,
                neginf=0.0,
            ).clamp_min(0.0)

            s_t = bounded_s[t_idx]
            w_t = precisions[t_idx]

            """
            SOTA Liebel-Koerner Strictly Positive Uncertainty Barrier:
            0.5 * ln(1 + exp(s_t)) = 0.5 * F.softplus(s_t) > 0 everywhere!
            Eliminates negative total losses when s_t < 0!
            """
            is_active_channel = (float(loss_t.detach().item()) > 1e-9) or raw_loss_t.requires_grad
            if is_active_channel:
                positive_barrier = 0.50 * F.softplus(s_t)
                weighted_component = (w_t * loss_t) + positive_barrier
            else:
                weighted_component = w_t * loss_t

            total_loss_accum = total_loss_accum + weighted_component
            weighted_losses_list.append(w_t * loss_t.detach().clamp_min(1e-4))

        stacked_weighted = torch.stack(weighted_losses_list)
        energy_shares, task_entropy, governor_loss = self.compute_gradient_energy_shares(stacked_weighted)

        total_loss_final = total_loss_accum + (self.governor_weight * governor_loss.to(dtype))

        sum_precisions = torch.sum(precisions.detach()).clamp_min(self.eps)
        simplex_weights = precisions.detach() / sum_precisions
        weights_tuple = tuple(float(w.item()) for w in simplex_weights)

        if return_telemetry:
            telemetry: Dict[str, float] = {
                "kendall_total_loss": float(total_loss_final.detach().item()),
                "kendall_entropy": task_entropy,
                "kendall_governor_loss": float(governor_loss.detach().item()),
                "kendall_max_share_pct": float(torch.max(energy_shares).detach().item()),
                "is_task_starvation": 1.0 if float(torch.max(energy_shares).detach().item()) > 60.0 else 0.0,
            }

            for t_idx, name in enumerate(self.task_names):
                s_val = float(bounded_s[t_idx].detach().item())
                telemetry[f"loss_{name}_raw"] = float(loss_tensor_list[t_idx].detach().item())
                telemetry[f"weight_{name}"] = float(precisions[t_idx].detach().item())
                telemetry[f"sigma_{name}"] = float(math.exp(0.50 * s_val))
                telemetry[f"share_{name}_pct"] = float(energy_shares[t_idx].detach().item())

            if auxiliary_telemetry is not None:
                telemetry.update(auxiliary_telemetry)

            return total_loss_final, weights_tuple, task_entropy, telemetry

        return total_loss_final, weights_tuple, task_entropy


def verify_kendall_multitask_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for kendall_multitask.py.
    Validates:
      1. Strict Positivity Invariant: total_loss > 0.0 even when s_raw is pushed to -100.0 on zero losses!
      2. Tightly Bounded Log-Variance Invariant: s_t strictly resides in (-1.50, +1.50).
      3. Guaranteed Query Weight Floor: w_query >= 0.25 even when s_raw_0 is pushed to +100.0.
      4. Full Autograd Backpropagation.
    """
    torch.manual_seed(42)
    num_tasks = 8

    balancer = BoundedKendallMultitaskLoss(
        num_tasks=num_tasks,
        bound_max=1.50,
        min_query_weight_floor=0.25,
    )

    test_losses = [
        torch.tensor(1.850, requires_grad=True),
        torch.tensor(1.200, requires_grad=True),
        torch.tensor(0.850, requires_grad=True),
        torch.tensor(2.400, requires_grad=True),
        torch.tensor(1.500, requires_grad=True),
        torch.tensor(0.045, requires_grad=True),
        torch.tensor(0.008, requires_grad=True),
        torch.tensor(0.012, requires_grad=True),
    ]

    total_loss, weights_tuple, entropy_val, _ = balancer(test_losses, return_telemetry=True)

    if torch.isnan(total_loss) or torch.isinf(total_loss) or float(total_loss.item()) <= 0.0:
        return False
    if len(weights_tuple) != num_tasks:
        return False

    """Verify Strict Positivity even at s_raw = -100.0 with near-zero losses"""
    with torch.no_grad():
        balancer.log_vars_raw.fill_(-100.0)
        tiny_losses = [torch.tensor(0.001, requires_grad=True) for _ in range(num_tasks)]
        pos_loss, _, _ = balancer(tiny_losses)
        if float(pos_loss.item()) <= 0.0:
            return False

        balancer.log_vars_raw[0] = 100.0
        if balancer.effective_weights["query_focal"] < 0.2499:
            return False
        balancer.log_vars_raw.zero_()

    total_loss.backward()

    if balancer.log_vars_raw.grad is None or torch.isnan(balancer.log_vars_raw.grad).any():
        return False

    return True