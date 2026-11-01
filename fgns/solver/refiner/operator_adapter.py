"""
Location: // fgns_engine/fgns/solver/refiner/operator_adapter.py
Description: Full-Network Geodesic SO(3)^C Test-Time Adaptation (TransductiveOperatorAdapter)
             with D4 x S9 Color-Permutation Automorphism Regularization (FGNS v4.1.0):
               1. Zero LoRA bottleneck: full expressive adaptation of the neural manifold.
               2. Stochastic D4 x S9 Color-Permutation Augmentation on TTT steps, preventing
                  shortcut color memorization and compelling relational support induction!
               3. Zero-Shot Step-0 Analytical Consensus Lock for exact D4/Bravais/Eikonal tasks.
Language: Python 3.12+ / PyTorch 2.5+ / Test-Time Optimization sm_120 Aligned
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import copy
import math
import random
import time
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fgns.config import FGNSConfig
from fgns.env.augment import (
    AutomorphismEngine,
    ColorPermutation,
    D4Group,
    D4Symmetry,
)
from fgns.env.batcher import FastVectorizedBatcher
from fgns.env.task import ARCPair, ARCTask
from fgns.models.fgns_network import FinslerGaugeSheafNetwork
from fgns.models.size_predictor.boundary_size_predictor import BoundaryMetricSizePredictor
from fgns.train.losses.true_spectral_mdl import SchattenNuclearMDLObjective
from fgns.types import BatchData, LossTelemetry, ModelOutput

DEFAULT_LOOCV_TAU: Final[float] = 0.05
DEFAULT_MAX_TTT_STEPS: Final[int] = 30000
MAX_PTTSA_NORM: Final[float] = 2.50
MAX_BATCHED_FOLDS: Final[int] = 3


class TransductiveOperatorAdapter:
    """
    Full-Network Geodesic Test-Time Adaptation Engine under D4 x S9 Group Automorphisms.
    """

    @staticmethod
    def _clone_state_dict_in_vram(
        model: nn.Module,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        state_snap = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if k not in ("pttsa_rule_delta", "pttsa_so3_omega")
        }
        rule_delta = getattr(model, "pttsa_rule_delta", None)
        pttsa_snap = (
            rule_delta.detach().clone()
            if isinstance(rule_delta, torch.Tensor)
            else torch.zeros(1)
        )
        so3_omega = getattr(model, "pttsa_so3_omega", None)
        so3_snap = (
            so3_omega.detach().clone()
            if isinstance(so3_omega, torch.Tensor)
            else torch.zeros(1)
        )
        return state_snap, pttsa_snap, so3_snap

    @staticmethod
    def _extract_core_operator_matrices(model: FinslerGaugeSheafNetwork) -> List[torch.Tensor]:
        return [
            model.lie_extractor.synthesizer_swiglu.proj_out.weight,
            model.size_oracle.head_rel_h.weight,
            model.size_oracle.head_terminal_h.weight,
            model.budget_oracle.coupling_swiglu.proj_out.weight,
            model.accro_decoder.force_synthesizer.force_swiglu.proj_out.weight,
            model.accro_decoder.rewrite_gate_proj.weight,
            model.slot_binder.kinematics_z_proj.weight,
            model.na3d_rotor_proj.weight,
            model.pullback_router.weight,
        ]

    @staticmethod
    def _verify_size_and_pixel_match(
        output: ModelOutput,
        batch: BatchData,
    ) -> Tuple[bool, float]:
        q_target = batch.query.targets
        q_mask = batch.query.target_masks
        q_pred = output.crystallized_tokens

        if q_target is None or q_mask is None:
            return False, 0.0

        h_in, w_in = BoundaryMetricSizePredictor.extract_input_dimensions(batch.query.in_masks)
        pred_dh = torch.argmax(output.query_h_pred, dim=-1).to(torch.int64) - 30
        pred_dw = torch.argmax(output.query_w_pred, dim=-1).to(torch.int64) - 30
        pred_h = int((h_in[0] + pred_dh[0]).clamp(1, 30).item())
        pred_w = int((w_in[0] + pred_dw[0]).clamp(1, 30).item())

        true_h = int(batch.query.target_heights[0].item()) if batch.query.target_heights is not None else pred_h
        true_w = int(batch.query.target_widths[0].item()) if batch.query.target_widths is not None else pred_w

        size_ok = (pred_h == true_h) and (pred_w == true_w)

        total_active = q_mask.sum().item()
        correct_active = ((q_pred == q_target) & q_mask).sum().item()
        match_pct = (float(correct_active) / float(max(1, total_active))) * 100.0

        is_perfect = size_ok and (match_pct >= 99.99)
        return is_perfect, match_pct

    @classmethod
    def evaluate_honest_augmented_loocv(
        cls,
        model: FinslerGaugeSheafNetwork,
        train_pairs: Sequence[ARCPair],
        loss_engine: SchattenNuclearMDLObjective,
        device: torch.device,
        use_augmentation: bool = True,
    ) -> Tuple[int, float, float, float, float, List[str]]:
        n_pairs = len(train_pairs)
        if n_pairs <= 1:
            return 0, 999.0, 0.0, 0.0, 0.0, ["N/A (Single Pair)"]

        was_training = model.training
        model.eval()

        solved_count = 0
        loss_accum = 0.0
        fold_details: List[str] = []
        min_pct = 100.0
        max_pct = 0.0
        last_pct = 0.0

        pttsa_delta = getattr(model, "pttsa_rule_delta", None)
        amp_device = "cuda" if device.type == "cuda" else "cpu"
        amp_enabled = device.type == "cuda"

        with torch.no_grad():
            for q_idx in range(n_pairs):
                support_canonical = [p for i, p in enumerate(train_pairs) if i != q_idx]
                query_canonical = [train_pairs[q_idx]]

                batch_canon = FastVectorizedBatcher.assemble_batch([support_canonical], [query_canonical], device)

                with torch.amp.autocast(device_type=amp_device, dtype=torch.bfloat16, enabled=amp_enabled):
                    output_canon = model(
                        batch_canon,
                        tau=DEFAULT_LOOCV_TAU,
                        current_step_ratio=1.0,
                        rule_perturbation=pttsa_delta,
                    )
                    telemetry_canon = loss_engine(output_canon, batch_canon)

                loss_val = telemetry_canon.loss_query + (0.25 * telemetry_canon.loss_size)
                loss_accum += loss_val

                is_canonical_perfect, match_pct = cls._verify_size_and_pixel_match(output_canon, batch_canon)
                min_pct = min(min_pct, match_pct)
                max_pct = max(max_pct, match_pct)
                last_pct = match_pct

                if is_canonical_perfect and use_augmentation:
                    sym_challenge = D4Group.ALL[(q_idx + 1) % 8]
                    cmap_challenge = ColorPermutation.identity()

                    aug_support = [
                        AutomorphismEngine.transform_pair(p, sym_challenge, cmap_challenge)
                        for p in support_canonical
                    ]
                    aug_query = [
                        AutomorphismEngine.transform_pair(query_canonical[0], sym_challenge, cmap_challenge)
                    ]

                    batch_aug = FastVectorizedBatcher.assemble_batch([aug_support], [aug_query], device)

                    with torch.amp.autocast(device_type=amp_device, dtype=torch.bfloat16, enabled=amp_enabled):
                        output_aug = model(
                            batch_aug,
                            tau=DEFAULT_LOOCV_TAU,
                            current_step_ratio=1.0,
                            rule_perturbation=pttsa_delta,
                        )

                    aug_perfect, aug_pct = cls._verify_size_and_pixel_match(output_aug, batch_aug)

                    if aug_perfect:
                        solved_count += 1
                        fold_details.append(f"F{q_idx + 1}: 100% [Inv OK]")
                    else:
                        fold_details.append(f"F{q_idx + 1}: 100% [Inv Fail {aug_pct:.0f}%]")
                elif is_canonical_perfect:
                    solved_count += 1
                    fold_details.append(f"F{q_idx + 1}: 100%")
                else:
                    fold_details.append(f"F{q_idx + 1}: {match_pct:.0f}%")

        if was_training:
            model.train()

        mean_loss = loss_accum / float(n_pairs)
        return solved_count, mean_loss, min_pct, max_pct, last_pct, fold_details

    @classmethod
    def refine_task_weights(
        cls,
        base_model: FinslerGaugeSheafNetwork,
        task: ARCTask,
        config: FGNSConfig,
        time_budget_secs: int = 333,
        device: Optional[torch.device] = None,
    ) -> Tuple[FinslerGaugeSheafNetwork, FinslerGaugeSheafNetwork]:
        if device is None:
            device = torch.device(config.hardware.device_name if torch.cuda.is_available() else "cpu")

        start_time = time.time()
        train_pairs = list(task.train)
        total_train_pairs = len(train_pairs)

        model = copy.deepcopy(base_model).to(device)

        frozen_ttt_prefixes = (
            "projector.lie_resonator",
            "projector.spectral_topology",
            "chordal_coord_proj",
            "query_canvas_initializer",
        )
        for name, param in model.named_parameters():
            if any(name.startswith(pref) for pref in frozen_ttt_prefixes):
                param.requires_grad = False
            else:
                param.requires_grad = True

        model.pttsa_rule_delta = nn.Parameter(
            torch.zeros((1, config.d_model), dtype=torch.float32, device=device)
        )
        model.pttsa_so3_omega = nn.Parameter(
            torch.zeros((1, model.num_na3d_cells, 3), dtype=torch.float32, device=device)
        )

        network_params = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and "pttsa_rule_delta" not in n and "pttsa_so3_omega" not in n
        ]
        pttsa_params = [model.pttsa_rule_delta, model.pttsa_so3_omega]

        loss_engine = SchattenNuclearMDLObjective(config).to(device)

        use_fused = torch.cuda.is_available() and device.type == "cuda"
        optimizer = torch.optim.AdamW(
            [
                {"params": pttsa_params, "lr": config.ttt.base_learning_rate * 3.0, "weight_decay": 0.0},
                {"params": network_params, "lr": config.ttt.base_learning_rate, "weight_decay": 1e-4},
            ],
            fused=use_fused,
        )

        initial_snap = cls._clone_state_dict_in_vram(model)
        first_consensus_snap: Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]] = None
        polished_mdl_snap = initial_snap
        fallback_snap = initial_snap

        min_mdl_score = float("inf")
        best_loo_solved = 0
        consensus_engaged = False
        post_consensus_remaining = config.ttt.post_consensus_polish_steps
        momentum_reset_done = False

        sep_line = "=" * 60
        banner = (
            f"{sep_line}\n"
            f"   [FGNS v4.1.0 TRANSDUCTIVE ADAPTER] Task: {task.task_id} | Pairs: {total_train_pairs} | Budget: {time_budget_secs}s\n"
            f"{sep_line}\n"
        )
        print(banner, end="")

        step = 0
        max_steps = min(DEFAULT_MAX_TTT_STEPS, config.ttt.max_gradient_steps)
        pttsa_warmup_steps = min(15, config.ttt.pttsa_steps)

        amp_device = "cuda" if device.type == "cuda" else "cpu"
        amp_enabled = config.hardware.enable_bf16_amp and (device.type == "cuda")

        (
            init_solved,
            init_loss,
            init_min_pct,
            init_max_pct,
            _,
            init_fold_strs,
        ) = cls.evaluate_honest_augmented_loocv(
            model=model,
            train_pairs=train_pairs,
            loss_engine=loss_engine,
            device=device,
            use_augmentation=config.ttt.use_augmented_loocv,
        )
        if init_solved == total_train_pairs and total_train_pairs > 1:
            consensus_engaged = True
            best_loo_solved = init_solved
            min_mdl_score = init_loss
            first_consensus_snap = initial_snap
            polished_mdl_snap = initial_snap
            post_consensus_remaining = min(20, config.ttt.post_consensus_polish_steps)
            print(
                f"   🏆 [ZERO-SHOT ANALYTICAL CONSENSUS LOCKED] All {total_train_pairs} folds solved at step 0 "
                f"(Loss: {init_loss:.4f})! Executing rapid 20-step MDL polish...\n"
            )

        canonical_color_map = ColorPermutation.identity()
        use_s9_colors = config.ttt.enable_s9_color_augmentation
        s9_prob = config.ttt.s9_augmentation_probability

        while True:
            step += 1
            elapsed = time.time() - start_time

            if elapsed >= float(time_budget_secs):
                print(f"   [BUDGET EXPIRED] TTT reached step {step:,} in {elapsed:.1f}s. Solved: {best_loo_solved}/{total_train_pairs}\n")
                break

            progress = min(1.0, float(step) / float(max_steps))
            current_lr = config.ttt.base_learning_rate * (
                config.ttt.cosine_lr_floor_ratio
                + (1.0 - config.ttt.cosine_lr_floor_ratio) * 0.50 * (1.0 + math.cos(math.pi * progress))
            )

            if step <= pttsa_warmup_steps:
                optimizer.param_groups[0]["lr"] = config.ttt.base_learning_rate * 4.5
                optimizer.param_groups[1]["lr"] = current_lr * 0.25
            else:
                optimizer.param_groups[0]["lr"] = current_lr * 1.5
                optimizer.param_groups[1]["lr"] = current_lr * 0.75

            tau = config.compute_cooling_temperature(step=step, total_steps=max_steps)

            sym_step = D4Group.ALL[step % 8]

            """
            D4 x S9 Color-Permutation Automorphism Regularization:
            Samples a stochastic non-background color permutation pi_t in S9 during exploration steps
            so the network cannot memorize static color IDs, while anchoring on identity every 3rd step
            and during post-consensus polish!
            """
            if use_s9_colors and (not consensus_engaged) and (step % 3 != 0) and (random.random() < s9_prob):
                step_color_map = ColorPermutation.random(preserve_background=True)
            else:
                step_color_map = canonical_color_map

            aug_pairs = [AutomorphismEngine.transform_pair(p, sym_step, step_color_map) for p in train_pairs]

            support_batch_list: List[List[ARCPair]] = []
            query_batch_list: List[List[ARCPair]] = []

            if total_train_pairs > 1:
                num_folds_in_batch = min(total_train_pairs, MAX_BATCHED_FOLDS)
                for b_i in range(num_folds_in_batch):
                    q_idx = (step + b_i) % total_train_pairs
                    supp_b = [p for i, p in enumerate(aug_pairs) if i != q_idx]
                    if len(supp_b) > 1:
                        random.shuffle(supp_b)
                    quer_b = [aug_pairs[q_idx]]
                    support_batch_list.append(supp_b)
                    query_batch_list.append(quer_b)
            else:
                support_batch_list.append(aug_pairs)
                query_batch_list.append(aug_pairs)

            batch = FastVectorizedBatcher.assemble_batch(support_batch_list, query_batch_list, device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=amp_device, dtype=torch.bfloat16, enabled=amp_enabled):
                output = model(
                    batch,
                    tau=tau,
                    current_step_ratio=progress,
                    rule_perturbation=model.pttsa_rule_delta,
                )

                operator_matrices = cls._extract_core_operator_matrices(model)
                proto_loss = model.projector.palette.compute_orthogonality_loss()

                telemetry = loss_engine(
                    output=output,
                    batch=batch,
                    adaptation_deltas=operator_matrices,
                    prototype_ortho_loss=proto_loss,
                )
                loss = telemetry.total_loss

            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                for p in pttsa_params + network_params:
                    if p.grad is not None:
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=1.0, neginf=-1.0)
                torch.nn.utils.clip_grad_norm_(pttsa_params + network_params, max_norm=1.0)
                optimizer.step()

            with torch.no_grad():
                u_norm = torch.norm(model.pttsa_rule_delta, p=2, dim=-1, keepdim=True)
                if float(u_norm.max().item()) > MAX_PTTSA_NORM:
                    model.pttsa_rule_delta.mul_(MAX_PTTSA_NORM / u_norm.clamp_min(1e-7))
                if model.pttsa_so3_omega is not None:
                    w_norm = torch.norm(model.pttsa_so3_omega, p=2, dim=-1, keepdim=True)
                    if float(w_norm.max().item()) > math.pi:
                        model.pttsa_so3_omega.mul_(math.pi / w_norm.clamp_min(1e-7))

            if consensus_engaged:
                post_consensus_remaining -= 1
                if post_consensus_remaining <= 0:
                    print(f"[SPECTRAL MDL POLISH COMPLETE] Finished polish at step {step:,} ({elapsed:.1f}s). Min MDL: {min_mdl_score:.4f}\n")
                    break

            if step % config.ttt.validation_interval_steps == 0 or step == 1:
                (
                    loo_solved,
                    loo_loss,
                    min_pct,
                    max_pct,
                    _,
                    fold_strs,
                ) = cls.evaluate_honest_augmented_loocv(
                    model=model,
                    train_pairs=train_pairs,
                    loss_engine=loss_engine,
                    device=device,
                    use_augmentation=config.ttt.use_augmented_loocv,
                )

                true_mdl = loo_loss + telemetry.loss_program_mdl

                if min_pct >= config.ttt.momentum_reset_threshold_pct and not momentum_reset_done:
                    optimizer.state.clear()
                    momentum_reset_done = True
                    print(f"   >>> [MOMENTUM FLUSH] Cleared optimizer kinetic buffers at step {step:,} (Spread: {min_pct:.1f}%)\n")

                if loo_solved > best_loo_solved or (loo_solved == best_loo_solved and true_mdl < min_mdl_score):
                    best_loo_solved = loo_solved
                    min_mdl_score = true_mdl
                    current_snap = cls._clone_state_dict_in_vram(model)
                    fallback_snap = current_snap

                    if not consensus_engaged:
                        first_consensus_snap = current_snap
                    polished_mdl_snap = current_snap

                if step % (config.ttt.validation_interval_steps * 2) == 0 or step == 1:
                    log_str = (
                        f"   -> [TTT Step {step:>5} | {elapsed:>3.0f}s] "
                        f"LR: {current_lr:.2e} | "
                        f"LOOCV: {loo_solved}/{total_train_pairs} | "
                        f"Loss: {loo_loss:.4f} | "
                        f"True_MDL: {true_mdl:.4f} | "
                        f"H_k: {telemetry.kendall_entropy:.2f} | "
                        f"Spread: [{min_pct:.0f}% - {max_pct:.0f}%] | "
                        f"Folds: [{', '.join(fold_strs)}]\n"
                    )
                    print(log_str, end="")

                if loo_solved == total_train_pairs and total_train_pairs > 1 and not consensus_engaged:
                    consensus_engaged = True
                    current_snap = cls._clone_state_dict_in_vram(model)
                    first_consensus_snap = current_snap
                    polished_mdl_snap = current_snap
                    print(
                        f"   🏆 [FIRST CONSENSUS LOCKED FOR ATTEMPT 1] Solved all {total_train_pairs} folds under D4 invariance at step {step:,}! "
                        f"Commencing Spectral MDL polish for Attempt 2...\n"
                    )

        if first_consensus_snap is None:
            first_consensus_snap = fallback_snap

        model_first = copy.deepcopy(base_model).to(device)
        model_first.load_state_dict(first_consensus_snap[0], strict=False)
        model_first.pttsa_rule_delta = nn.Parameter(first_consensus_snap[1].clone())
        model_first.pttsa_so3_omega = nn.Parameter(first_consensus_snap[2].clone())

        model_polished = model
        model_polished.load_state_dict(polished_mdl_snap[0], strict=False)
        model_polished.pttsa_rule_delta = nn.Parameter(polished_mdl_snap[1].clone())
        model_polished.pttsa_so3_omega = nn.Parameter(polished_mdl_snap[2].clone())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return model_first, model_polished


def verify_operator_adapter_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for operator_adapter.py.
    """
    torch.manual_seed(42)

    config = FGNSConfig()
    device = torch.device("cpu")

    base_model = FinslerGaugeSheafNetwork(config=config).to(device)

    from fgns.env.task import Grid

    g1 = Grid([[1, 2], [3, 4]])
    g2 = Grid([[2, 1], [4, 3]])
    pair1 = ARCPair(input_grid=g1, output_grid=g2)
    pair2 = ARCPair(input_grid=g2, output_grid=g1)
    test_pair = ARCPair(input_grid=g1, output_grid=None)

    mock_task = ARCTask(
        task_id="mock_ttt_001",
        train=(pair1, pair2),
        test=(test_pair,),
    )

    m_first, m_polished = TransductiveOperatorAdapter.refine_task_weights(
        base_model=base_model,
        task=mock_task,
        config=config,
        time_budget_secs=2,
        device=device,
    )

    if not isinstance(m_first, FinslerGaugeSheafNetwork) or not isinstance(m_polished, FinslerGaugeSheafNetwork):
        return False

    return True

