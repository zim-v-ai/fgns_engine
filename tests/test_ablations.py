"""
Location: // fgns_engine/tests/test_ablations.py
Description: Diagnostic Verification Suite for the 8 FGNS Architectural Ablations.
              Verifies that each ablation module runs end-to-end on real tensor batches.
Author: Vyacheslav Zimovsky (v.zimovsky.ai@gmail.com)
Language: Python 3.12+ / PyTorch 2.5+
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import torch

from fgns.env.task import ARCPair, ARCTask, Grid
from scripts.ablations.common import get_ablation_device
from scripts.ablations.ablate_01_na3d_stiefel import run_na3d_ablation
from scripts.ablations.ablate_02_sheaf_laplacian import run_sheaf_laplacian_ablation
from scripts.ablations.ablate_03_octowave_ssm import run_octowave_ablation
from scripts.ablations.ablate_04_cauchy_dynamics import run_cauchy_dynamics_ablation
from scripts.ablations.ablate_05_spectral_mdl import run_spectral_mdl_ablation
from scripts.ablations.ablate_06_boundary_size import run_boundary_size_ablation
from scripts.ablations.ablate_07_palette_budget import run_palette_budget_ablation
from scripts.ablations.ablate_08_operator_adapter import run_operator_adapter_ablation


def create_mock_benchmark_tasks() -> list[ARCTask]:
    """Generates two lightweight mock ARC tasks for isolated dry-run testing."""
    g_in = Grid([[1, 2, 0], [3, 4, 0], [0, 0, 5]])
    g_out = Grid([[2, 1, 0], [4, 3, 0], [0, 0, 5]])
    pair1 = ARCPair(input_grid=g_in, output_grid=g_out)
    pair2 = ARCPair(input_grid=g_out, output_grid=g_in)

    task1 = ARCTask(task_id="mock_abl_001", train=(pair1, pair2), test=(pair1,))
    task2 = ARCTask(task_id="mock_abl_002", train=(pair2, pair1), test=(pair2,))
    return [task1, task2]


def test_ablation_01_na3d_stiefel() -> None:
    """Verifies NA3D Stiefel rotor ablation harness."""
    device = get_ablation_device()
    tasks = create_mock_benchmark_tasks()
    res = run_na3d_ablation(tasks, device)
    assert "na3d_stiefel" in res
    assert "isotropic_scalar" in res
    assert res["na3d_stiefel"]["steps"] > 0


def test_ablation_02_sheaf_laplacian() -> None:
    """Verifies Cellular Sheaf connection Laplacian ablation harness."""
    device = get_ablation_device()
    tasks = create_mock_benchmark_tasks()
    res = run_sheaf_laplacian_ablation(tasks, device)
    assert "cellular_sheaf_od" in res
    assert "isotropic_gcn" in res
    assert res["cellular_sheaf_od"]["dirichlet_energy"] >= 0.0


def test_ablation_03_octowave_ssm() -> None:
    """Verifies OctoWave 8-axis SSM ablation harness."""
    device = get_ablation_device()
    tasks = create_mock_benchmark_tasks()
    res = run_octowave_ablation(tasks, device)
    assert "octowave_8axis_ssm" in res
    assert "dilated_conv2d" in res
    assert res["octowave_8axis_ssm"]["steps"] > 0


def test_ablation_04_cauchy_dynamics() -> None:
    """Verifies Cauchy continuous dynamics ablation harness."""
    device = get_ablation_device()
    tasks = create_mock_benchmark_tasks()
    res = run_cauchy_dynamics_ablation(tasks, device)
    assert "cauchy_resolvent" in res
    assert "discrete_resnet" in res
    assert res["cauchy_resolvent"]["monotonic_descent_pct"] >= 0.0


def test_ablation_05_spectral_mdl() -> None:
    """Verifies Spectral MDL test-time training ablation harness."""
    device = get_ablation_device()
    tasks = create_mock_benchmark_tasks()
    res = run_spectral_mdl_ablation(tasks, device)
    assert "spectral_mdl_schatten1" in res
    assert "standard_l2_ttt" in res
    assert res["spectral_mdl_schatten1"]["effective_rank_r_eff"] >= 1.0


def test_ablation_06_boundary_size() -> None:
    """Verifies boundary metric size predictor ablation harness."""
    device = get_ablation_device()
    tasks = create_mock_benchmark_tasks()
    res = run_boundary_size_ablation(tasks, device)
    assert "boundary_metric_flux" in res
    assert "identity_baseline" in res
    assert res["boundary_metric_flux"]["total"] > 0


def test_ablation_07_palette_budget() -> None:
    """Verifies equivariant palette budget predictor ablation harness."""
    device = get_ablation_device()
    tasks = create_mock_benchmark_tasks()
    res = run_palette_budget_ablation(tasks, device)
    assert "equivariant_haar_lrq" in res
    assert "unconstrained_softmax" in res
    assert res["equivariant_haar_lrq"]["earth_movers_distance"] >= 0.0


def test_ablation_08_operator_adapter() -> None:
    """Verifies transductive operator adapter ablation harness."""
    device = get_ablation_device()
    tasks = create_mock_benchmark_tasks()
    res = run_operator_adapter_ablation(tasks, device)
    assert "geodesic_so3_adapter" in res
    assert "euclidean_lora" in res
    assert res["geodesic_so3_adapter"]["mean_steps_to_convergence"] > 0
