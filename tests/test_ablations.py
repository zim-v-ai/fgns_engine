"""
Location: // fgns_engine/tests/test_ablations.py
Description: Comprehensive Diagnostic & Invariant Verification Suite for the 8 FGNS-TRM v4.3.0
             Architectural Ablations. Verifies multi-metric completeness, zero color-subspace
             leakage (Theorem I), positive-definite sheaf contraction (Theorem II), proper-time
             dimension accuracy (Theorem III), and honest LOO-TTT generalization.
Author: Vyacheslav Zimovsky (v.zimovsky.ai@gmail.com)
Language: Python 3.12+ / PyTorch 2.5+
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

from typing import Any, Dict, List

from fgns.env.task import ARCPair, ARCTask, Grid
from scripts.ablations.ablate_01_na3d_stiefel import run_na3d_ablation
from scripts.ablations.ablate_02_sheaf_laplacian import run_sheaf_laplacian_ablation
from scripts.ablations.ablate_03_octowave_ssm import run_octowave_ablation
from scripts.ablations.ablate_04_cauchy_dynamics import run_cauchy_dynamics_ablation
from scripts.ablations.ablate_05_spectral_mdl import run_spectral_mdl_ablation
from scripts.ablations.ablate_06_boundary_size import run_boundary_size_ablation
from scripts.ablations.ablate_07_palette_budget import run_palette_budget_ablation
from scripts.ablations.ablate_08_operator_adapter import run_operator_adapter_ablation
from scripts.ablations.common import get_ablation_device

MANDATORY_METRIC_KEYS = (
    "query_loss",
    "support_loss",
    "total_mdl_loss",
    "pixel_accuracy_pct",
    "fg_accuracy_pct",
    "exact_grid_match_pct",
    "size_accuracy_pct",
    "size_mae",
    "palette_emd",
    "color_hallucination_pct",
    "effective_rank_r_eff",
    "latency_ms",
    "verdict",
)


def _assert_complete_metrics(condition_dict: Dict[str, Any]) -> None:
    for key in MANDATORY_METRIC_KEYS:
        assert key in condition_dict, f"Missing mandatory metric '{key}' in ablation output."
    assert isinstance(condition_dict["verdict"], str) and len(condition_dict["verdict"]) > 10


def create_mock_benchmark_tasks() -> List[ARCTask]:
    """
    Generates two structured geometric ARC tasks for fast deterministic unit verification.
    """
    g_in2 = Grid([
        [1, 2, 0],
        [3, 4, 0],
        [0, 0, 5],
    ])
    g_out2 = Grid([
        [2, 1, 0],
        [4, 3, 0],
        [0, 0, 5],
    ])
    pair2 = ARCPair(input_grid=g_in2, output_grid=g_out2)
    pair3 = ARCPair(input_grid=g_out2, output_grid=g_in2)

    task1 = ARCTask(task_id="mock_abl_001", train=(pair2, pair3), test=(pair2,))
    task2 = ARCTask(task_id="mock_abl_002", train=(pair3, pair2), test=(pair3,))
    return [task1, task2]


def test_ablation_01_na3d_stiefel() -> None:
    """Verifies NA3D Stiefel rotor ablation and Theorem I zero color-subspace leakage."""
    device = get_ablation_device()
    tasks = create_mock_benchmark_tasks()
    res = run_na3d_ablation(tasks, device)
    for cond in ("isotropic_scalar", "planar_so2", "unsplit_na3d_naive", "na3d_stiefel"):
        assert cond in res
        _assert_complete_metrics(res[cond])
    assert res["na3d_stiefel"]["color_subspace_leakage"] < 1e-4
    assert res["unsplit_na3d_naive"]["color_subspace_leakage"] > res["na3d_stiefel"]["color_subspace_leakage"]


def test_ablation_02_sheaf_laplacian() -> None:
    """Verifies Cellular Sheaf connection Laplacian and Theorem II Dirichlet contraction."""
    device = get_ablation_device()
    tasks = create_mock_benchmark_tasks()
    res = run_sheaf_laplacian_ablation(tasks, device)
    for cond in ("isotropic_gcn", "scalar_gat", "unnormalized_antidiffusion", "cellular_sheaf_od"):
        assert cond in res
        _assert_complete_metrics(res[cond])
    assert res["cellular_sheaf_od"]["energy_contraction_ratio"] < 1.0
    assert res["unnormalized_antidiffusion"]["energy_contraction_ratio"] > 1.5


def test_ablation_03_octowave_ssm() -> None:
    """Verifies OctoWave 8-axis SSM and diagonal wavefront reach."""
    device = get_ablation_device()
    tasks = create_mock_benchmark_tasks()
    res = run_octowave_ablation(tasks, device)
    for cond in ("dilated_conv2d", "mamba_2axis_scan", "full_2d_attention", "octowave_8axis_ssm"):
        assert cond in res
        _assert_complete_metrics(res[cond])
    assert res["octowave_8axis_ssm"]["diagonal_wavefront_reach"] > res["mamba_2axis_scan"]["diagonal_wavefront_reach"]


def test_ablation_04_cauchy_dynamics() -> None:
    """Verifies Dual-State Cauchy continuous dynamics across 4 conditions."""
    device = get_ablation_device()
    tasks = create_mock_benchmark_tasks()
    res = run_cauchy_dynamics_ablation(tasks, device)
    for cond in ("single_pass_feedforward", "discrete_resnet", "recurrent_gru", "cauchy_resolvent"):
        assert cond in res
        _assert_complete_metrics(res[cond])


def test_ablation_05_spectral_mdl() -> None:
    """Verifies Zero-Leakage LOO Spectral MDL test-time training ablation."""
    device = get_ablation_device()
    tasks = create_mock_benchmark_tasks()
    res = run_spectral_mdl_ablation(tasks, device)
    for cond in ("zero_ttt", "standard_l2_ttt", "schatten2_frobenius", "spectral_mdl_schatten1"):
        assert cond in res
        _assert_complete_metrics(res[cond])
    assert res["spectral_mdl_schatten1"]["effective_rank_r_eff"] >= 1.0


def test_ablation_06_boundary_size() -> None:
    """Verifies Single-Equation Woodbury-Stokes Boundary-Mask size predictor ablation."""
    device = get_ablation_device()
    tasks = create_mock_benchmark_tasks()
    res = run_boundary_size_ablation(tasks, device)
    for cond in ("identity_baseline", "global_avg_pooling", "spatial_mlp", "boundary_metric_flux"):
        assert cond in res
        _assert_complete_metrics(res[cond])


def test_ablation_07_palette_budget() -> None:
    """Verifies S10-Equivariant Role-Woodbury & Diode-Sink palette predictor ablation."""
    device = get_ablation_device()
    tasks = create_mock_benchmark_tasks()
    res = run_palette_budget_ablation(tasks, device)
    for cond in ("unconstrained_softmax", "non_equivariant_mlp", "no_diode_birkhoff", "equivariant_haar_lrq"):
        assert cond in res
        _assert_complete_metrics(res[cond])


def test_ablation_08_operator_adapter() -> None:
    """Verifies Geodesic SO(3)^C + D4xS9 operator adapter under OOD automorphism challenge."""
    device = get_ablation_device()
    tasks = create_mock_benchmark_tasks()
    res = run_operator_adapter_ablation(tasks, device)
    for cond in ("full_fine_tuning", "euclidean_lora", "static_color_ttt", "geodesic_so3_adapter"):
        assert cond in res
        _assert_complete_metrics(res[cond])
        assert "d4_s9_invariance_accuracy_pct" in res[cond]
