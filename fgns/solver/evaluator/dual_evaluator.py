"""
Location: // fgns_engine/fgns/solver/evaluator/dual_evaluator.py
Description: SOTA Master Pure Neural Decision-Theoretic Dual Attempt Evaluator Engine
             with True-MDL Priority Ordering & Holistic D4 Orbit Mode Selection.
             Apex Evaluation Orchestrator for FGNS v3.3.0 (Finsler-Gauge Sheaf):
               1. Theorem 2 True-MDL Priority: assigns the Schatten-1 polished snapshot
                  (model_polished) to Attempt 1 and the first-consensus snapshot (model_first)
                  or Plackett-Luce Mode-2 bifurcation to Attempt 2.
               2. Holistic D4 Orbit Priority: selects whole-grid Condorcet modes C_1* and C_2*.
               3. Absolute Non-Duplication Invariant: Attempt 2 != Attempt 1.
Language: Python 3.12+ / PyTorch 2.5+ / Decision-Theoretic Optimization sm_120 Aligned
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import math
from typing import Final, List, Optional, Sequence, Tuple, Union

import torch

from fgns.config import FGNSConfig
from fgns.env.task import ARCPair, ARCTask, Grid
from fgns.models.fgns_network import FinslerGaugeSheafNetwork
from fgns.solver.evaluator.d4_ensemble import D4ReynoldsEnsembleEngine
from fgns.solver.evaluator.plackett_luce import PlackettLuceModalSynthesizer
from fgns.types import AttemptPrediction, GridData

DEFAULT_SNAPSHOT1_TAU: Final[float] = 0.02
DEFAULT_SNAPSHOT2_TAU: Final[float] = 0.05


class DualAttemptEvaluator:
    """
    SOTA Master Pure Neural Decision-Theoretic Dual Attempt Evaluation Orchestrator.
    """

    @classmethod
    def evaluate_task(
        cls,
        model_first: FinslerGaugeSheafNetwork,
        model_polished: FinslerGaugeSheafNetwork,
        task: ARCTask,
        config: FGNSConfig,
        device: Optional[torch.device] = None,
    ) -> List[AttemptPrediction]:
        if device is None:
            device = torch.device(config.hardware.device_name if torch.cuda.is_available() else "cpu")

        predictions: List[AttemptPrediction] = []

        for test_idx in range(len(task.test)):
            test_input_grid = task.test[test_idx].input_grid

            """
            1. Evaluate Attempt 1 via Polished True-MDL Snapshot (Theorem 2 Priority):
            Extracts the noise-purged, minimum Kolmogorov-Schatten complexity hypothesis first.
            """
            (
                probs_1,
                entropy_1,
                p_col_1,
                (target_h_1, target_w_1),
                sec_dim_1,
                spread_1,
            ) = D4ReynoldsEnsembleEngine.evaluate_d4_marginalized_field(
                model=model_polished,
                task=task,
                test_idx=test_idx,
                device=device,
                tau_override=DEFAULT_SNAPSHOT1_TAU,
            )
            holistic_1_a, holistic_1_b = D4ReynoldsEnsembleEngine.latest_holistic_modes

            if holistic_1_a is not None:
                attempt_1 = holistic_1_a
            else:
                attempt_1 = PlackettLuceModalSynthesizer.synthesize_candidate_grid(
                    probs=probs_1,
                    target_h=target_h_1,
                    target_w=target_w_1,
                )

            """2. Evaluate Candidate Attempt 2 via First Consensus Snapshot"""
            (
                probs_2,
                entropy_2,
                p_col_2,
                (target_h_2, target_w_2),
                sec_dim_2,
                spread_2,
            ) = D4ReynoldsEnsembleEngine.evaluate_d4_marginalized_field(
                model=model_first,
                task=task,
                test_idx=test_idx,
                device=device,
                tau_override=DEFAULT_SNAPSHOT2_TAU,
            )
            holistic_2_a, holistic_2_b = D4ReynoldsEnsembleEngine.latest_holistic_modes

            if holistic_2_a is not None:
                attempt_2_primary = holistic_2_a
            else:
                attempt_2_primary = PlackettLuceModalSynthesizer.synthesize_candidate_grid(
                    probs=probs_2,
                    target_h=target_h_2,
                    target_w=target_w_2,
                )

            """3. Strict Non-Duplication Invariant Enforcement with Holistic Mode-2 Priority"""
            is_duplicate = (
                attempt_2_primary.data == attempt_1.data
                and attempt_2_primary.height == attempt_1.height
                and attempt_2_primary.width == attempt_1.width
            )

            if is_duplicate:
                alt_holistic = holistic_1_b if holistic_1_b is not None else holistic_2_b
                if (
                    alt_holistic is not None
                    and (
                        alt_holistic.data != attempt_1.data
                        or alt_holistic.height != attempt_1.height
                        or alt_holistic.width != attempt_1.width
                    )
                ):
                    attempt_2 = alt_holistic
                else:
                    candidate_dim_2 = sec_dim_1 if sec_dim_1 is not None else sec_dim_2
                    attempt_2 = PlackettLuceModalSynthesizer.synthesize_orthogonal_mode2(
                        primary_grid=attempt_1,
                        mean_probs=probs_1,
                        entropy_field=entropy_1,
                        p_color_matrix=p_col_1,
                        test_input=test_input_grid,
                        secondary_dimensions=candidate_dim_2,
                    )
            else:
                attempt_2 = attempt_2_primary

            """4. Shannon Spatial Uncertainty Entropy Confidence Scoring"""
            active_entropy_patch = entropy_1[:target_h_1, :target_w_1]
            mean_spatial_entropy = float(active_entropy_patch.mean().item())

            log_10 = math.log(10.0)
            confidence_scalar = max(0.0, min(1.0, 1.0 - (mean_spatial_entropy / log_10)))

            pred_monad = AttemptPrediction(
                attempt_1=attempt_1,
                attempt_2=attempt_2,
                confidence=confidence_scalar,
                d4_vote_spread=spread_1,
            )
            predictions.append(pred_monad)

        return predictions


def verify_dual_evaluator_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for dual_evaluator.py.
    """
    torch.manual_seed(42)

    g1 = Grid([[1, 2], [3, 4]])
    g2 = Grid([[5, 6], [7, 8]])
    pair_train = ARCPair(input_grid=g1, output_grid=g2)
    pair_test1 = ARCPair(input_grid=g1, output_grid=None)
    pair_test2 = ARCPair(input_grid=g2, output_grid=None)

    mock_task = ARCTask(
        task_id="mock_eval_001",
        train=(pair_train,),
        test=(pair_test1, pair_test2),
    )

    config = FGNSConfig()
    device = torch.device("cpu")

    model_first = FinslerGaugeSheafNetwork(config=config).to(device)
    model_polished = FinslerGaugeSheafNetwork(config=config).to(device)

    preds = DualAttemptEvaluator.evaluate_task(
        model_first=model_first,
        model_polished=model_polished,
        task=mock_task,
        config=config,
        device=device,
    )

    if len(preds) != 2:
        return False

    for p in preds:
        is_same = (
            p.attempt_1.data == p.attempt_2.data
            and p.attempt_1.height == p.attempt_2.height
            and p.attempt_1.width == p.attempt_2.width
        )
        if is_same:
            return False

        if p.confidence < 0.0 or p.confidence > 1.0001:
            return False

        if p.attempt_1.height < 1 or p.attempt_1.height > 30:
            return False
        if p.attempt_1.width < 1 or p.attempt_1.width > 30:
            return False

    return True

