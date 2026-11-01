"""
Location: // fgns_engine/fgns/train/losses/true_spectral_mdl.py
Description: Master True Spectral Minimum Description Length (MDL) Multitask Loss Engine.
             Apex Loss Coordinator for FGNS v2.1.0 (Finsler-Gauge Sheaf):
               1. Complete multi-objective orchestration integrating all 8 specialized task surfaces.
               2. Deep Trajectory Supervision across 4-step Dual-State cellular TRM-ACCRO reasoning passes.
               3. Stride-agnostic .reshape(B * K, ...) handling across multi-fold TTT batches (B > 1, K > 1).
               4. Universal ordinal size displacement supervision combining cross-entropy, Smooth-L1, and EMD.
               5. Joint Query + Support palette capacity oracle (NPCO) Laplace-smoothed multiset supervision.
               6. Non-Degenerate Sheaf Dirichlet MDL + CuSOLVER-Free Newton-Schulz Schatten-1 & Entropy.
Language: Python 3.12+ / PyTorch 2.5+ / Master Loss Orchestration sm_120 Aligned
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import math
from typing import Dict, Final, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fgns.config import FGNSConfig
from fgns.models.palette_histogram.budget_loss import evaluate_budget_loss
from fgns.primitives.schatten_norm import von_neumann_spectral_entropy
from fgns.primitives.symplectic_involution import symplectic_bilinear_form
from fgns.train.losses.focal_canvas import ActiveDomainFocalLoss
from fgns.train.losses.kendall_multitask import BoundedKendallMultitaskLoss
from fgns.train.losses.mask_segmentation import DifferentiableActiveMaskLoss
from fgns.train.losses.mdl_complexity import SpectralProgramComplexity
from fgns.train.losses.size_loss import UniversalTargetSizeLoss
from fgns.train.losses.xor_regularization import XORParityRegularizationLoss
from fgns.types import BatchData, LossTelemetry, ModelOutput

DEFAULT_TRIANGULAR_PROGRESSION: Final[bool] = True


class SchattenNuclearMDLObjective(nn.Module):
    """
    Master True Spectral Minimum Description Length (MDL) Multitask Loss Engine.
    Unifies all 8 objective channels under Bounded Kendall Bayesian Uncertainty Balancing.
    """

    def __init__(self, config: FGNSConfig) -> None:
        super().__init__()
        self.config = config
        vocab_size = config.geometry.vocab_size

        self.focal_engine = ActiveDomainFocalLoss(
            vocab_size=vocab_size,
            fg_weight=config.mdl.fg_class_weight,
            bg_weight=config.mdl.bg_class_weight,
            gamma=config.mdl.focal_gamma,
            kappa=0.25,
        )

        self.mask_engine = DifferentiableActiveMaskLoss(
            pad_token_id=config.geometry.pad_token_id,
            bce_weight=1.00,
            dice_weight=1.00,
            contour_weight=0.50,
        )

        self.size_engine = UniversalTargetSizeLoss(
            relative_classes=config.geometry.relative_dim_classes,
            zero_shift_index=config.geometry.zero_shift_index,
            label_smoothing=0.05,
            metric_l1_weight=0.50,
            wasserstein_weight=0.10,
        )

        self.mdl_engine = SpectralProgramComplexity(
            lambda_nuclear=config.mdl.lambda_nuclear_rank,
            lambda_entropy=config.mdl.lambda_spectral_entropy,
            schatten_p=config.mdl.schatten_p_order,
            eps=config.mdl.schatten_epsilon,
        )

        self.xor_loss_engine = XORParityRegularizationLoss(
            weight=0.25,
            self_weight=1.00,
            asym_weight=0.50,
            anti_collapse_weight=0.35,
            min_response_margin=0.50,
        )

        self.kendall_balancer = BoundedKendallMultitaskLoss(
            num_tasks=8,
            bound_max=config.kendall.log_var_bound_max,
            min_query_weight_floor=config.kendall.min_query_weight_floor,
            governor_weight=config.kendall.governor_weight,
            init_log_vars=config.kendall.init_log_vars,
        )

    def compute_triangular_weights(self, num_steps: int, device: torch.device) -> torch.Tensor:
        steps = max(1, num_steps)
        step_indices = torch.arange(1, steps + 1, dtype=torch.float32, device=device)
        triangular_sum = float((steps * (steps + 1)) // 2)
        return step_indices / triangular_sum

    def forward(
        self,
        output: ModelOutput,
        batch: BatchData,
        adaptation_deltas: Optional[Sequence[torch.Tensor]] = None,
        prototype_ortho_loss: Optional[torch.Tensor] = None,
        codebook_loss: Optional[torch.Tensor] = None,
    ) -> LossTelemetry:
        device = output.query_logits.device
        dtype = output.query_logits.dtype

        if output.trajectory_logits and len(output.trajectory_logits) > 0:
            query_trajectory = output.trajectory_logits
        else:
            query_trajectory = (output.query_logits,)

        num_query_steps = len(query_trajectory)
        triangular_weights_q = self.compute_triangular_weights(num_query_steps, device)

        """1. Active-Grid Focal & Boundary Mask Loss over Query Trajectory"""
        query_traj_loss = torch.tensor(0.0, device=device, dtype=dtype)
        mask_traj_loss = torch.tensor(0.0, device=device, dtype=dtype)
        mean_mask_iou = 0.0

        has_query_targets = (
            batch.query.targets is not None and batch.query.target_masks is not None
        )

        if has_query_targets and batch.query.targets is not None and batch.query.target_masks is not None:
            for step_idx, q_logits_step in enumerate(query_trajectory):
                w_tau = triangular_weights_q[step_idx]

                loss_step = self.focal_engine(
                    logits=q_logits_step,
                    targets=batch.query.targets,
                    target_masks=batch.query.target_masks,
                )
                query_traj_loss = query_traj_loss + (w_tau * loss_step)

                step_mask_loss, step_iou = self.mask_engine(
                    logits=q_logits_step,
                    target_masks=batch.query.target_masks,
                )
                mask_traj_loss = mask_traj_loss + (w_tau * step_mask_loss)

                if step_idx == num_query_steps - 1:
                    mean_mask_iou = step_iou

        """2. Deep Trajectory Supervision over Valid Support Demonstrations (Stride-Safe .reshape)"""
        B, K, H, W = batch.support.inputs.shape
        supp_traj_loss = torch.tensor(0.0, device=device, dtype=dtype)
        supp_mask_traj_loss = torch.tensor(0.0, device=device, dtype=dtype)

        if output.all_support_logits and len(output.all_support_logits) > 0:
            supp_trajectory = output.all_support_logits
        else:
            supp_trajectory = [output.support_logits]

        num_supp_steps = len(supp_trajectory)
        triangular_weights_s = self.compute_triangular_weights(num_supp_steps, device)

        if batch.support.pair_valid_mask is not None:
            valid_pair_exp = batch.support.pair_valid_mask.view(B, K, 1, 1).expand(B, K, H, W)
            effective_supp_out_masks = batch.support.out_masks & valid_pair_exp
        else:
            effective_supp_out_masks = batch.support.out_masks

        supp_targets_flat = batch.support.outputs.reshape(B * K, H, W)
        supp_masks_flat = effective_supp_out_masks.reshape(B * K, H, W)

        for step_idx, s_logits_step in enumerate(supp_trajectory):
            w_tau = triangular_weights_s[step_idx]
            s_logits_flat = s_logits_step.reshape(B * K, H, W, -1)

            loss_supp_flat = self.focal_engine(
                logits=s_logits_flat,
                targets=supp_targets_flat,
                target_masks=supp_masks_flat,
            )
            supp_traj_loss = supp_traj_loss + (w_tau * loss_supp_flat)

            if step_idx == num_supp_steps - 1:
                s_m_loss, _ = self.mask_engine(
                    logits=s_logits_flat,
                    target_masks=supp_masks_flat,
                )
                supp_mask_traj_loss = s_m_loss

        mask_traj_loss = 0.70 * mask_traj_loss + 0.30 * supp_mask_traj_loss

        """3. Universal 61-Class Size Displacement Loss (The First Lock)"""
        has_size_targets = (
            batch.query.target_heights is not None and batch.query.target_widths is not None
        )

        if has_size_targets and batch.query.target_heights is not None and batch.query.target_widths is not None:
            size_loss = self.size_engine(
                h_pred=output.query_h_pred,
                w_pred=output.query_w_pred,
                h_target=batch.query.target_heights,
                w_target=batch.query.target_widths,
                in_masks=batch.query.in_masks,
            )
            true_cap_f32 = (batch.query.target_heights.float() * batch.query.target_widths.float()).clamp_min(1.0)
            cap_bridge_loss = F.smooth_l1_loss(
                torch.log(output.pred_capacity_ste.float().clamp_min(1.0)),
                torch.log(true_cap_f32),
                beta=0.25,
            )
            size_loss = size_loss + (0.20 * cap_bridge_loss)
        else:
            size_loss = torch.tensor(0.0, device=device, dtype=dtype)

        """4. Joint Query + Support Palette Budget Loss (The Second Lock)"""
        has_budget_targets = (
            batch.query.target_histograms is not None
            and batch.query.target_capacities is not None
            and output.query_budget is not None
        )

        if (
            has_budget_targets
            and output.query_budget is not None
            and batch.query.target_histograms is not None
            and batch.query.target_capacities is not None
        ):
            budget_loss, _, _, loss_anti_col, _ = evaluate_budget_loss(
                prediction=output.query_budget,
                target_histograms=batch.query.target_histograms,
                target_capacities=batch.query.target_capacities,
                anti_collapse_margin=self.config.budget.anti_collapse_margin,
                anti_collapse_weight=self.config.budget.anti_collapse_weight,
                smooth_l1_beta=self.config.budget.smooth_l1_beta,
                support_predictions=output.support_budgets,
                support_target_histograms=batch.support.target_histograms,
                support_target_capacities=batch.support.target_capacities,
                support_pair_valid_mask=batch.support.pair_valid_mask,
            )
            anti_collapse_scalar = float(loss_anti_col.detach().item())
        else:
            budget_loss = torch.tensor(0.0, device=device, dtype=dtype)
            anti_collapse_scalar = 0.0

        """5. True Kolmogorov-Solomonoff Spectral & Sheaf Dirichlet MDL Loss"""
        sheaf_mdl_term = (
            0.10 * torch.mean(output.sheaf_dirichlet_energy)
            if output.sheaf_dirichlet_energy is not None
            else torch.tensor(0.0, device=device, dtype=dtype)
        )

        if output.hyperbolic_rule is not None:
            lie_op = output.hyperbolic_rule.symmetric_invariant + output.hyperbolic_rule.commutator
            d_op = max(2, lie_op.shape[-1])
            vn_entropy, eff_rk_tensor = von_neumann_spectral_entropy(lie_op)
            lie_spectral_mdl = 0.05 * torch.mean(vn_entropy / math.log(float(d_op))).to(dtype)
            base_s_entropy = float(torch.mean(vn_entropy).detach().item())
            base_eff_rank = float(torch.mean(eff_rk_tensor).detach().item())
        else:
            lie_spectral_mdl = torch.tensor(0.0, device=device, dtype=dtype)
            base_s_entropy = float(sheaf_mdl_term.detach().item())
            base_eff_rank = 1.0

        if adaptation_deltas is not None and len(adaptation_deltas) > 0:
            lora_mdl_loss, s_entropy, eff_rank = self.mdl_engine(adaptation_deltas)
            mdl_loss = lora_mdl_loss + sheaf_mdl_term + lie_spectral_mdl
        else:
            mdl_loss = sheaf_mdl_term + lie_spectral_mdl
            s_entropy = base_s_entropy
            eff_rank = base_eff_rank

        """6. Grassmannian Prototype Orthogonality + Birkhoff Sharpness Potential"""
        v_dim = output.p_color_matrix.shape[-1]
        birkhoff_purity = torch.sum(output.p_color_matrix.to(torch.float32).pow(2), dim=(-2, -1)) / float(max(1, v_dim))
        birkhoff_sharp_loss = torch.clamp(1.0 - torch.mean(birkhoff_purity), min=0.0, max=1.0).to(dtype)

        if prototype_ortho_loss is not None:
            ortho_loss = prototype_ortho_loss.to(device=device, dtype=dtype) + (0.15 * birkhoff_sharp_loss)
        else:
            ortho_loss = 0.15 * birkhoff_sharp_loss

        """7. Symplectic True XOR Parity Involution Regularization Loss"""
        if output.xor_state is not None:
            base_xor = self.xor_loss_engine(
                xor_state=output.xor_state,
                target_masks=batch.query.target_masks if has_query_targets else None,
            )
            flux_vec = output.xor_state.vorticity_flux.to(torch.float32)
            sel_vec = output.xor_state.parity_selector.to(torch.float32)
            if flux_vec.shape == sel_vec.shape and flux_vec.shape[-1] % 2 == 0:
                symp_area = torch.abs(symplectic_bilinear_form(flux_vec, sel_vec))
                symp_barrier = torch.mean(F.relu(0.75 - symp_area).pow(2)).to(dtype)
            else:
                symp_barrier = torch.tensor(0.02, device=device, dtype=dtype)
            xor_loss = base_xor + (0.08 * symp_barrier)
        else:
            xor_loss = torch.tensor(0.0, device=device, dtype=dtype)

        """8. Hyperbolic Rule State Chirality & Codebook Clustering Metric"""
        if codebook_loss is not None:
            lorentz_rule_metric = float(codebook_loss.detach().item())
        elif output.hyperbolic_rule is not None:
            k_commutator = output.hyperbolic_rule.commutator
            lorentz_rule_metric = float(torch.norm(k_commutator, p="fro").detach().item())
        else:
            lorentz_rule_metric = 0.0

        eight_channel_losses = [
            query_traj_loss,
            supp_traj_loss,
            mask_traj_loss,
            size_loss,
            budget_loss,
            mdl_loss,
            ortho_loss,
            xor_loss,
        ]

        total_loss, kendall_weights, multitask_entropy = self.kendall_balancer(eight_channel_losses)

        return LossTelemetry(
            total_loss=total_loss,
            loss_query=float(query_traj_loss.detach().item()),
            loss_support=float(supp_traj_loss.detach().item()),
            loss_mask=float(mask_traj_loss.detach().item()),
            loss_size=float(size_loss.detach().item()),
            loss_budget=float(budget_loss.detach().item()),
            loss_anti_collapse=anti_collapse_scalar,
            loss_program_mdl=float(mdl_loss.detach().item()),
            loss_orthogonality=float(ortho_loss.detach().item()),
            loss_xor_parity=float(xor_loss.detach().item()),
            loss_lorentz_rule=lorentz_rule_metric,
            mask_iou=mean_mask_iou,
            spectral_entropy=s_entropy,
            effective_rank=eff_rank,
            kendall_weights=kendall_weights,
            kendall_entropy=multitask_entropy,
        )


def verify_true_spectral_mdl_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for true_spectral_mdl.py.
    Explicitly tests non-contiguous [:, 1:, ...] support slices with B=2, K=6!
    """
    torch.manual_seed(42)
    b_size, k_supp, h, w, vocab_size = 2, 6, 4, 4, 11
    d_model = 32

    config = FGNSConfig()
    loss_engine = SchattenNuclearMDLObjective(config=config)

    combined_traj = [
        torch.randn(b_size, 1 + k_supp, h, w, vocab_size, dtype=torch.float32, requires_grad=True)
        for _ in range(4)
    ]
    traj_list = [step[:, 0] for step in combined_traj]
    supp_traj_list = [step[:, 1:] for step in combined_traj]

    q_h_pred = torch.randn(b_size, 61, dtype=torch.float32, requires_grad=True)
    q_w_pred = torch.randn(b_size, 61, dtype=torch.float32, requires_grad=True)

    from fgns.types import (
        EntitySlots,
        PaletteBudgetPrediction,
        QueryBatch,
        SupportBatch,
        XORParityState,
    )

    dummy_slots = EntitySlots(
        geometry=torch.zeros(b_size, 1, 16),
        color=torch.zeros(b_size, 1, 8),
        position=torch.zeros(b_size, 1, 4),
        euler=torch.zeros(b_size, 1, 1),
        mask=torch.ones(b_size, 1, 1),
    )

    dummy_xor = XORParityState(
        vorticity_flux=torch.randn(b_size, h, w, d_model, requires_grad=True),
        parity_selector=torch.randn(b_size, h, w, d_model),
        involution_norm=torch.ones(b_size),
    )

    dummy_budget = PaletteBudgetPrediction(
        predicted_counts=torch.tensor([[10, 2, 2, 2, 0, 0, 0, 0, 0, 0], [10, 2, 2, 2, 0, 0, 0, 0, 0, 0]], dtype=torch.float32),
        simplex_proportions=torch.ones(b_size, 10) / 10.0,
        input_counts=torch.zeros(b_size, 10, dtype=torch.int64),
        target_capacity=torch.tensor([16, 16], dtype=torch.int64),
    )

    output = ModelOutput(
        query_logits=traj_list[-1],
        trajectory_logits=tuple(traj_list),
        support_logits=supp_traj_list[-1],
        all_query_logits=traj_list,
        all_support_logits=supp_traj_list,
        query_h_pred=q_h_pred,
        query_w_pred=q_w_pred,
        pred_capacity_ste=torch.tensor([16.0, 16.0]),
        crystallized_tokens=torch.zeros(b_size, h, w, dtype=torch.int64),
        p_color_matrix=torch.eye(vocab_size).unsqueeze(0).expand(b_size, -1, -1),
        slots=dummy_slots,
        xor_state=dummy_xor,
        query_budget=dummy_budget,
        sheaf_dirichlet_energy=torch.tensor([0.15, 0.20], requires_grad=True),
    )

    pair_valid = torch.ones(b_size, k_supp, dtype=torch.bool)
    pair_valid[:, 3:] = False

    batch = BatchData(
        support=SupportBatch(
            inputs=torch.zeros(b_size, k_supp, h, w, dtype=torch.int64),
            in_masks=torch.ones(b_size, k_supp, h, w, dtype=torch.bool),
            outputs=torch.zeros(b_size, k_supp, h, w, dtype=torch.int64),
            out_masks=torch.ones(b_size, k_supp, h, w, dtype=torch.bool),
            pair_valid_mask=pair_valid,
        ),
        query=QueryBatch(
            inputs=torch.zeros(b_size, h, w, dtype=torch.int64),
            in_masks=torch.ones(b_size, h, w, dtype=torch.bool),
            targets=torch.zeros(b_size, h, w, dtype=torch.int64),
            target_masks=torch.ones(b_size, h, w, dtype=torch.bool),
            target_heights=torch.tensor([h, h], dtype=torch.int64),
            target_widths=torch.tensor([w, w], dtype=torch.int64),
            target_histograms=torch.tensor([[16, 0, 0, 0, 0, 0, 0, 0, 0, 0], [16, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int64),
            target_capacities=torch.tensor([16, 16], dtype=torch.int64),
        ),
    )

    telemetry = loss_engine(output, batch)

    if torch.isnan(telemetry.total_loss) or torch.isinf(telemetry.total_loss):
        return False
    if len(telemetry.kendall_weights) != 8 or telemetry.kendall_entropy <= 0.0:
        return False
    if telemetry.loss_program_mdl <= 0.0 or telemetry.loss_xor_parity <= 0.0:
        return False

    telemetry.total_loss.backward()

    for t_step in combined_traj:
        if t_step.grad is None or torch.isnan(t_step.grad).any():
            return False
        if float(torch.norm(t_step.grad).item()) <= 0.0:
            return False

    return True

