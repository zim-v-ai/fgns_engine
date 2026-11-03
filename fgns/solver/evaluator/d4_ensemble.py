"""
Location: // fgns_engine/fgns/solver/evaluator/d4_ensemble.py
Description: SOTA Pure Neural D4 Dihedral Reynolds Ensemble & Sub-Canvas Cropped Exact Orbit Inverter.
             Core Decision-Theoretic Engine for FGNS v4.0.0 (Finsler-Gauge Sheaf):
               1. Exact Sub-Canvas Cropped Probability Inversion: applies D4Group.invert_tensor
                  strictly on the active cropped sub-canvas [:calc_h, :calc_w, :] BEFORE accumulating
                  into the 30x30 buffer, completely eradicating the (30 - H, 30 - W) padding shift bug!
               2. Holistic Orbit Consensus across all 8 isometric orientations.
               3. Zero DSL overrides: 100% pure neural manifold evaluation.
Language: Python 3.12+ / PyTorch 2.5+ / Group Representation Theory sm_120 Aligned
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import collections
import math
from typing import Dict, Final, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from fgns.env.augment import (
    AutomorphismEngine,
    ColorPermutation,
    D4Group,
    D4Symmetry,
)
from fgns.env.batcher import FastVectorizedBatcher
from fgns.env.task import ARCPair, ARCTask, Grid
from fgns.models.fgns_network import FinslerGaugeSheafNetwork
from fgns.types import GridData

DEFAULT_REYNOLDS_TAU: Final[float] = 0.05
DEFAULT_ENTROPY_EPS: Final[float] = 1e-7


class D4ReynoldsEnsembleEngine:
    """
    SOTA Pure Neural D4 Dihedral Group Reynolds & Exact Cropped Orbit Consensus Projector Engine.
    """

    latest_holistic_modes: Tuple[Optional[GridData], Optional[GridData]] = (None, None)

    @classmethod
    def invert_cropped_probability_field(
        cls,
        probs_sym: torch.Tensor,
        calc_h: int,
        calc_w: int,
        sym: int,
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Crops probs_sym [30, 30, V] to active bounds [:calc_h, :calc_w, :] BEFORE applying
        D4Group.invert_tensor, guaranteeing zero (30 - H, 30 - W) padding shift!
        """
        cropped = probs_sym[:calc_h, :calc_w, :]
        permuted = cropped.permute(2, 0, 1)
        inverted = D4Group.invert_tensor(permuted, sym).contiguous()
        restored_sub = inverted.permute(1, 2, 0)
        restored_h, restored_w, _ = restored_sub.shape
        return restored_sub, restored_h, restored_w

    @classmethod
    def evaluate_d4_marginalized_field(
        cls,
        model: FinslerGaugeSheafNetwork,
        task: ARCTask,
        test_idx: int,
        device: torch.device,
        tau_override: float = DEFAULT_REYNOLDS_TAU,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int], Optional[Tuple[int, int]], Tuple[int, int]]:
        was_training = model.training
        model.eval()

        canvas_dim = FastVectorizedBatcher.CANVAS_DIM
        vocab_size = model.vocab_size
        d4_symmetries = D4Group.ALL
        num_symmetries = len(d4_symmetries)

        restored_prob_accum = torch.zeros((canvas_dim, canvas_dim, vocab_size), dtype=torch.float32, device=device)
        restored_weight_accum = torch.zeros((canvas_dim, canvas_dim, 1), dtype=torch.float32, device=device)
        p_color_accum = torch.zeros((vocab_size, vocab_size), dtype=torch.float32, device=device)

        dim_votes: Dict[Tuple[int, int], int] = collections.defaultdict(int)
        holistic_scores: Dict[GridData, float] = collections.defaultdict(float)
        holistic_counts: Dict[GridData, int] = collections.defaultdict(int)

        test_in_grid = task.test[test_idx].input_grid
        canonical_cmap = ColorPermutation.identity()
        pttsa_delta = getattr(model, "pttsa_rule_delta", None)

        amp_device = "cuda" if device.type == "cuda" else "cpu"
        amp_enabled = (
            (device.type == "cuda")
            and torch.cuda.is_available()
            and (torch.cuda.get_device_capability(device)[0] >= 8)
        )

        with torch.no_grad():
            for sym in d4_symmetries:
                aug_train = [AutomorphismEngine.transform_pair(p, sym, canonical_cmap) for p in task.train]
                aug_test = [AutomorphismEngine.transform_pair(p, sym, canonical_cmap) for p in task.test]

                support = aug_train
                query = [aug_test[test_idx]]

                batch = FastVectorizedBatcher.assemble_batch([support], [query], device)

                with torch.amp.autocast(device_type=amp_device, dtype=torch.bfloat16, enabled=amp_enabled):
                    output = model(
                        batch,
                        tau=tau_override,
                        current_step_ratio=1.0,
                        rule_perturbation=pttsa_delta,
                    )

                pred_sym_h = int(output.query_h_pred[0].argmax().item()) - 30
                pred_sym_w = int(output.query_w_pred[0].argmax().item()) - 30

                aug_test_in = D4Group.apply_grid(test_in_grid, sym)
                aug_h_in, aug_w_in = aug_test_in.height, aug_test_in.width

                calc_h = max(1, min(canvas_dim, aug_h_in + pred_sym_h))
                calc_w = max(1, min(canvas_dim, aug_w_in + pred_sym_w))

                step_logits = output.query_logits[0].float()
                probs_sym = F.softmax(step_logits, dim=-1)

                """Exact Sub-Canvas Cropped Probability Inversion (Zero Padding Shift!)"""
                restored_sub, restored_h, restored_w = cls.invert_cropped_probability_field(
                    probs_sym=probs_sym,
                    calc_h=calc_h,
                    calc_w=calc_w,
                    sym=sym,
                )

                dim_votes[(restored_h, restored_w)] += 1

                max_p_sub, tok_sub = torch.max(restored_sub[:, :, :10], dim=-1)
                mean_conf = float(max_p_sub.mean().item())

                cand_grid = GridData.from_tensor(tok_sub)
                holistic_scores[cand_grid] += (1.0 + mean_conf)
                holistic_counts[cand_grid] += 1

                restored_prob_accum[:restored_h, :restored_w, :] += restored_sub
                restored_weight_accum[:restored_h, :restored_w, :] += 1.0
                p_color_accum += output.p_color_matrix[0].float()

        if was_training:
            model.train()

        uniform_fallback = torch.full_like(restored_prob_accum, 1.0 / float(vocab_size))
        mean_probs = torch.where(
            restored_weight_accum > 0.5,
            restored_prob_accum / restored_weight_accum.clamp_min(1.0),
            uniform_fallback,
        )
        mean_probs = mean_probs / mean_probs.sum(dim=-1, keepdim=True).clamp_min(DEFAULT_ENTROPY_EPS)
        mean_p_color = p_color_accum / float(num_symmetries)

        log_probs = torch.log(mean_probs.clamp_min(DEFAULT_ENTROPY_EPS))
        entropy_field = -torch.sum(mean_probs * log_probs, dim=-1)

        sorted_dim_votes = sorted(dim_votes.items(), key=lambda kv: kv[1], reverse=True)
        primary_dimensions = sorted_dim_votes[0][0]

        secondary_dimensions = (
            sorted_dim_votes[1][0]
            if len(sorted_dim_votes) > 1 and sorted_dim_votes[1][1] >= 2
            else None
        )

        ranked_holistic = sorted(
            holistic_scores.items(),
            key=lambda kv: (holistic_counts[kv[0]], kv[1]),
            reverse=True,
        )
        mode_1_holistic: Optional[GridData] = None
        mode_2_holistic: Optional[GridData] = None

        for g_cand, _ in ranked_holistic:
            if (g_cand.height, g_cand.width) == primary_dimensions and holistic_counts[g_cand] >= 2:
                if mode_1_holistic is None:
                    mode_1_holistic = g_cand
                elif mode_2_holistic is None and g_cand.data != mode_1_holistic.data:
                    mode_2_holistic = g_cand
            elif mode_2_holistic is None and holistic_counts[g_cand] >= 2:
                if mode_1_holistic is None or g_cand.data != mode_1_holistic.data:
                    mode_2_holistic = g_cand

        cls.latest_holistic_modes = (mode_1_holistic, mode_2_holistic)

        min_v = min(dim_votes.values()) if dim_votes else 0
        max_v = max(dim_votes.values()) if dim_votes else 0

        return mean_probs, entropy_field, mean_p_color, primary_dimensions, secondary_dimensions, (min_v, max_v)


def verify_d4_ensemble_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for d4_ensemble.py.
    """
    torch.manual_seed(42)
    canvas_dim = 30
    vocab_size = 11

    h_sub, w_sub = 29, 29
    true_grid_29 = torch.zeros((h_sub, w_sub), dtype=torch.int64)
    true_grid_29[0, :] = 8
    true_grid_29[-1, :] = 8
    true_grid_29[:, 0] = 8
    true_grid_29[:, -1] = 8
    true_grid_29[1, 1] = 3
    true_grid_29[5, 10] = 4

    accum_29 = torch.zeros((canvas_dim, canvas_dim, vocab_size), dtype=torch.float32)
    for sym in D4Group.ALL:
        rot_g = D4Group.apply_tensor(true_grid_29, sym)
        ch, cw = rot_g.shape
        padded_probs = torch.zeros((canvas_dim, canvas_dim, vocab_size), dtype=torch.float32)
        padded_probs[:ch, :cw, :] = F.one_hot(rot_g, num_classes=vocab_size).float()
        padded_probs[ch:, :, 10] = 1.0
        padded_probs[:, cw:, 10] = 1.0

        rest_sub, rh, rw = D4ReynoldsEnsembleEngine.invert_cropped_probability_field(
            probs_sym=padded_probs,
            calc_h=ch,
            calc_w=cw,
            sym=sym,
        )
        if (rh, rw) != (h_sub, w_sub):
            return False
        accum_29[:rh, :rw, :] += rest_sub

    pred_29 = torch.argmax(accum_29[:h_sub, :w_sub, :10], dim=-1)
    if not torch.equal(pred_29, true_grid_29):
        return False

    return True