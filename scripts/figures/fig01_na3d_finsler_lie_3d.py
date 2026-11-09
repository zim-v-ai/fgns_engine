"""
Location: // fgns_engine/scripts/figures/fig01_na3d_finsler_lie_3d.py
Description: Nature Physics / PRX Publication Figure 1 — Native Rust 3D Ray-Traced Non-Abelian
             3D Lie Simplex (NA3D) inside Cutaway Finsler-Randers Indicatrix Shell, 3D
             Orthogonal Fiber Bundle R^128 = H_color^(11) (+) H_geom^(117), and Empirical
             Verification of Lemma 1 (Zero Color-Subspace Leakage on Public Evaluation Task 1818057f).
Author: Vyacheslav Zimovsky (v.zimovsky.ai@gmail.com)
License: MIT-0
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Optional, Tuple

import matplotlib
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fgns.env.batcher import FastVectorizedBatcher
from fgns.primitives.stiefel_sheaf import NA3DFinslerLieEngine
from scripts.figures.style_utils import (
    add_figure_header_banner,
    apply_nature_prx_rcparams,
    create_fresh_fgns_model_from_checkpoint,
    draw_panel_badge,
    format_float_matrix_compact,
    format_grid_2d_compact,
    format_sci_matrix_compact,
    get_arc2_eval_task,
    load_master_ablation_json,
    record_figure_telemetry,
    render_arc_grid,
)

matplotlib.use("Agg")


def _compute_real_wilson_plaquette_field(
    h_field_full: torch.Tensor,
    y_slice: slice,
    x_slice: slice,
) -> np.ndarray:
    """
    Computes the authentic 2x2 Yang-Mills-Wilson plaquette curvature field
    K_plaq(y, x) = 1 - cos(pi * tanh(||F_plaquette(y, x)||_2 / scale)) on the full 22x22
    lattice of Task 1818057f before slicing the interior window [8:19, 10:21].
    """
    B, H, W, D = h_field_full.shape
    num_cells = D // 16
    cells = h_field_full.to(torch.float32).view(B, H, W, num_cells, 16)
    l_face = cells[..., 0:3]
    r_face = cells[..., 6:9]
    v_beta = cells[..., 9:12]
    v_norm = torch.sqrt(torch.sum(v_beta * v_beta, dim=-1, keepdim=True) + 1e-6)
    beta_wind = (torch.tanh(v_norm) / v_norm) * v_beta
    spin_local = torch.cross(l_face, r_face, dim=-1)

    pack_9 = torch.cat([l_face, beta_wind, spin_local], dim=-1).view(B, H, W, num_cells * 9)
    pack_pad = F.pad(pack_9.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="replicate").permute(0, 2, 3, 1)

    nb_n = pack_pad[:, :-2, 1:-1, :]
    nb_s = pack_pad[:, 2:, 1:-1, :]
    nb_w = pack_pad[:, 1:-1, :-2, :]
    nb_e = pack_pad[:, 1:-1, 2:, :]
    nb_stack = torch.stack([nb_n, nb_s, nb_w, nb_e], dim=1).view(B, 4, H, W, num_cells, 9)

    l_nb = nb_stack[..., 0:3]
    spin_nb = nb_stack[..., 6:9]
    r_exp = r_face.unsqueeze(1)
    spin_exp = spin_local.unsqueeze(1)

    omega_gap = torch.cross(r_exp.expand_as(l_nb), l_nb, dim=-1)
    comm_1 = torch.cross(spin_exp.expand_as(spin_nb), spin_nb, dim=-1)
    comm_2 = torch.cross(spin_exp.expand_as(comm_1), comm_1, dim=-1) - torch.cross(spin_nb, comm_1, dim=-1)
    omega_bch = omega_gap + (0.50 * comm_1) + ((1.0 / 12.0) * comm_2)

    omega_s = omega_bch[:, 1]
    omega_e = omega_bch[:, 3]
    omega_e_south = torch.cat([omega_e[:, 1:, :, :, :], omega_e[:, -1:, :, :, :]], dim=1)
    omega_s_east = torch.cat([omega_s[:, :, 1:, :, :], omega_s[:, :, -1:, :, :]], dim=2)

    f_plaquette = (
        omega_e
        + omega_s_east
        - omega_e_south
        - omega_s
        + 0.50 * (torch.cross(omega_e, omega_s_east, dim=-1) - torch.cross(omega_s, omega_e_south, dim=-1))
    )
    f_norm_full = torch.sqrt(torch.sum(f_plaquette * f_plaquette, dim=-1) + 1e-6).mean(dim=-1)[0]
    f_patch = f_norm_full[y_slice, x_slice]

    f_min = f_patch.min()
    f_span = (f_patch.max() - f_min).clamp_min(1e-6)
    f_rel = (f_patch - f_min) / f_span
    wilson_curv = 0.50 * (1.0 - torch.cos(torch.tanh(1.35 * f_rel) * math.pi))
    wilson_norm = wilson_curv / wilson_curv.max().clamp_min(1e-6)
    return wilson_norm.detach().cpu().numpy()


def extract_real_na3d_tensors() -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
]:
    """
    Executes authentic 0-TTT forward passes and Lemma 1 (11 (+) 117 MGS2 orthogonal subspace
    split vs. unsplit SO(3)^8 Rodrigues holonomy) directly on FinslerGaugeSheafNetwork
    for ARC-AGI-2 Public Evaluation Task `1818057f`.
    """
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False

    try:
        report = load_master_ablation_json()
        res_01 = report["results"]["01_na3d_stiefel"]
        bench_leak_unsplit = float(res_01["unsplit_na3d_naive"]["color_subspace_leakage"])
        bench_leak_split = float(res_01["na3d_stiefel"]["color_subspace_leakage"])

        model, device = create_fresh_fgns_model_from_checkpoint(seed=42)
        task = get_arc2_eval_task("1818057f")
        support = list(task.train)
        query = [task.test[0]]
        static_k = max(1, min(6, len(support)))
        batch = FastVectorizedBatcher.assemble_batch([support], [query], device=device, static_k=static_k)

        inp_grid = task.test[0].input_grid.to_tensor().numpy()
        assert task.test[0].output_grid is not None
        out_grid = task.test[0].output_grid.to_tensor().numpy()
        y_sl = slice(8, 19)
        x_sl = slice(10, 21)
        patch_in = inp_grid[y_sl, x_sl].copy()
        patch_out = out_grid[y_sl, x_sl].copy()

        with torch.no_grad():
            output_split = model(batch, tau=0.05, current_step_ratio=1.0)
            pred_full_split = torch.argmax(
                output_split.query_logits[0, :22, :22, :10].to(torch.float32), dim=-1
            ).cpu().numpy()
            pred_split_patch_base = pred_full_split[y_sl, x_sl].copy()

            full_grid_t = torch.tensor(inp_grid, dtype=torch.long, device=device).unsqueeze(0)
            full_mask_t = torch.ones_like(full_grid_t, dtype=torch.bool)
            z_emb_full = model.projector(full_grid_t, full_mask_t)
            h_lie_full = model.octowave_ssm(z_emb_full, full_mask_t)
            h_lie_patch = h_lie_full[:, y_sl, x_sl, :]

            plaq_np = _compute_real_wilson_plaquette_field(h_lie_full, y_sl, x_sl)

            e_proto_all = model.projector.palette.get_normalized_prototypes().to(torch.float32)
            e_proto_10 = e_proto_all[:10]
            d_model = e_proto_all.shape[-1]
            c_cells = d_model // 16

            z_rule_base = (
                output_split.hyperbolic_rule.tangent_velocity.to(torch.float32)
                if output_split.hyperbolic_rule is not None
                else torch.ones((1, d_model), dtype=torch.float32, device=device)
            )
            probe_gen = torch.Generator(device=device).manual_seed(42)
            z_probe = z_rule_base + 2.0 * torch.randn(
                (1, d_model), dtype=torch.float32, device=device, generator=probe_gen
            )

            def _unsplit_na3d_holonomy(
                h_field: torch.Tensor,
                z_rule: torch.Tensor,
                gain_override: Optional[float] = None,
            ) -> torch.Tensor:
                orig_shape = h_field.shape
                b_r = z_rule.shape[0]
                h_f32 = h_field.to(torch.float32).view(b_r, -1, c_cells, 16)
                omega_r = model.na3d_rotor_proj(z_rule).to(torch.float32).view(b_r, 1, c_cells, 3)
                l_r = NA3DFinslerLieEngine.apply_rodrigues_rotation(h_f32[..., 0:3], omega_r)
                c_r = NA3DFinslerLieEngine.apply_rodrigues_rotation(h_f32[..., 3:6], omega_r)
                r_r = NA3DFinslerLieEngine.apply_rodrigues_rotation(h_f32[..., 6:9], omega_r)
                rot_all = torch.cat([l_r, c_r, r_r, h_f32[..., 9:16]], dim=-1).view(b_r, -1, d_model)
                h_flat_f32 = h_f32.view(b_r, -1, d_model)
                g_na3d = (
                    float(gain_override)
                    if gain_override is not None
                    else float(torch.tanh(model.na3d_action_gate.to(torch.float32)).item())
                )
                return (h_flat_f32 + g_na3d * (rot_all - h_flat_f32)).view(*orig_shape).to(h_field.dtype)

            geom_context = h_lie_patch[0, :10, 0, :].to(torch.float32)
            q_basis = model._get_exact_orthonormal_palette_basis()
            gram_q = torch.matmul(q_basis, q_basis.t())
            mgs2_ortho_err = float(torch.max(torch.abs(gram_q - torch.eye(11, device=device))).item())

            geom_perp = geom_context - torch.matmul(torch.matmul(geom_context, q_basis.t()), q_basis)
            h_color_probes = (e_proto_10 + 0.35 * geom_perp).view(1, 10, 1, d_model)

            h_after_split = model._apply_na3d_cell_holonomy_action(h_color_probes, z_probe)
            h_after_unsplit = _unsplit_na3d_holonomy(h_color_probes, z_probe)

            diff_split = (h_after_split[0, :, 0, :] - h_color_probes[0, :, 0, :]).to(torch.float32)
            diff_unsplit = (h_after_unsplit[0, :, 0, :] - h_color_probes[0, :, 0, :]).to(torch.float32)

            leak_matrix_split_t = torch.abs(torch.matmul(diff_split, e_proto_10.t()))
            leak_matrix_unsplit_t = torch.abs(torch.matmul(diff_unsplit, e_proto_10.t()))

            leak_split_val = float(torch.max(leak_matrix_split_t).item())
            leak_unsplit_val = float(torch.max(leak_matrix_unsplit_t).item())

            leak_matrix_split = leak_matrix_split_t.detach().cpu().numpy()
            leak_matrix_unsplit = leak_matrix_unsplit_t.detach().cpu().numpy()

            # Construct latent patch state with natural 117D geometric energy from h_lie_patch
            pred_patch_t = torch.tensor(pred_split_patch_base, dtype=torch.long, device=device).unsqueeze(0)
            h_lie_flat = h_lie_patch.to(torch.float32).reshape(1, -1, d_model)
            geom_patch_perp = (
                h_lie_flat - torch.matmul(torch.matmul(h_lie_flat, q_basis.t()), q_basis)
            ).view(1, 11, 11, d_model)
            h_patch_clean = F.embedding(pred_patch_t, e_proto_all) + (0.85 * geom_patch_perp)

            # 1. Apply Lemma 1 MGS2-Split NA3D Holonomy (3 steps) and decode on S^9
            h_patch_split_rot = h_patch_clean
            for step_rot in range(3):
                h_patch_split_rot = model._apply_na3d_cell_holonomy_action(
                    h_patch_split_rot, z_probe * (2.5 + 0.8 * step_rot)
                )
            psi_ones = torch.full((1, 11, 11, 1), 5.0, dtype=torch.float32, device=device)
            logits_split = model.accro_decoder.potts_latching.compute_unified_sphere_logits(
                h=h_patch_split_rot,
                carrier_psi=psi_ones,
            )
            pred_split_patch = torch.argmax(logits_split[0, :, :, :10], dim=-1).detach().cpu().numpy()

            # 2. Apply Unsplit SO(3)^8 Holonomy (3 steps) to the exact same state and decode on S^9
            h_patch_unsplit_rot = h_patch_clean
            for step_rot in range(3):
                h_patch_unsplit_rot = _unsplit_na3d_holonomy(
                    h_patch_unsplit_rot, z_probe * (2.5 + 0.8 * step_rot), gain_override=1.0
                )
            logits_unsplit = model.accro_decoder.potts_latching.compute_unified_sphere_logits(
                h=h_patch_unsplit_rot,
                carrier_psi=psi_ones,
            )
            corrupted_patch = torch.argmax(logits_unsplit[0, :, :, :10], dim=-1).detach().cpu().numpy()

            split_patch_acc = 100.0 * float((pred_split_patch == patch_out).mean())
            unsplit_patch_acc = 100.0 * float((corrupted_patch == patch_out).mean())

            record_figure_telemetry(
                "fig01_na3d_lemma1",
                {
                    "evaluation_mode": "0_ttt_single_pass_transductive",
                    "task_id": "1818057f",
                    "patch_window": "y=8..18, x=10..20 (11x11)",
                    "mgs2_gram_orthogonality_error": mgs2_ortho_err,
                    "unit_prototype_10x10_unsplit_max_leakage": leak_unsplit_val,
                    "unit_prototype_10x10_split_max_leakage": leak_split_val,
                    "benchmark_25task_unsplit_max_leakage": bench_leak_unsplit,
                    "benchmark_25task_split_max_leakage": bench_leak_split,
                    "leakage_reduction_ratio_patch": leak_unsplit_val / max(leak_split_val, 1e-12),
                    "leakage_reduction_ratio_benchmark": bench_leak_unsplit / max(bench_leak_split, 1e-12),
                    "split_patch_accuracy_pct": round(split_patch_acc, 2),
                    "unsplit_rotated_patch_accuracy_pct": round(unsplit_patch_acc, 2),
                    "leak_matrix_unsplit_mean": float(leak_matrix_unsplit.mean()),
                    "leak_matrix_split_mean": float(leak_matrix_split.mean()),
                    "input_patch_11x11": format_grid_2d_compact(patch_in),
                    "target_patch_11x11": format_grid_2d_compact(patch_out),
                    "unsplit_corrupted_patch_11x11": format_grid_2d_compact(corrupted_patch),
                    "lemma1_split_predicted_patch_11x11": format_grid_2d_compact(pred_split_patch),
                    "wilson_plaquette_curvature_11x11": format_float_matrix_compact(plaq_np, decimals=3),
                    "leak_matrix_unsplit_10x10": format_float_matrix_compact(leak_matrix_unsplit, decimals=4),
                    "leak_matrix_split_10x10_sci": format_sci_matrix_compact(leak_matrix_split, precision=2),
                },
            )

        return (
            patch_in,
            patch_out,
            pred_split_patch,
            corrupted_patch,
            plaq_np,
            leak_matrix_unsplit,
            leak_matrix_split,
            leak_unsplit_val,
            leak_split_val,
            bench_leak_unsplit,
            bench_leak_split,
        )
    finally:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32


def _load_rust_render(output_dir: Path, filename: str) -> Optional[np.ndarray]:
    """
    Loads a native Rust 3D ray-traced image (.ppm or .png) from `artifacts/rust_renders`,
    automatically syncing the PNG copy if the PPM is newer.
    """
    primary_dir = PROJECT_ROOT / "artifacts" / "rust_renders"
    fallback_dir = output_dir / "rust_renders"
    for rust_dir in (primary_dir, fallback_dir):
        ppm_file = rust_dir / f"{filename}.ppm"
        png_file = rust_dir / f"{filename}.png"
        if ppm_file.exists():
            img = Image.open(ppm_file).convert("RGB")
            img.save(png_file)
            return np.asarray(img)
        if png_file.exists():
            img = Image.open(png_file).convert("RGB")
            return np.asarray(img)
    return None


def generate_fig01(output_dir: Optional[Path] = None) -> Path:
    """
    Renders Figure 1: Native 3D Ray-Traced NA3D Oriented Simplex inside Finsler-Randers
    Indicatrix Shell, 3D Orthogonal Fiber Bundle H_color^(11) (+) H_geom^(117), and
    Empirical Verification of Lemma 1 on Public Evaluation Task 1818057f.
    """
    apply_nature_prx_rcparams()
    if output_dir is None:
        output_dir = PROJECT_ROOT / "artifacts" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        patch_in,
        patch_out,
        pred_split_patch,
        corrupted_patch,
        plaq_np,
        leak_matrix_unsplit,
        leak_matrix_split,
        leak_unsplit_val,
        leak_split_val,
        bench_leak_unsplit,
        bench_leak_split,
    ) = extract_real_na3d_tensors()

    scene1_img = _load_rust_render(output_dir, "scene1_na3d_finsler")
    scene2_img = _load_rust_render(output_dir, "scene2_orthogonal_bundle")

    fig = plt.figure(figsize=(19.2, 10.8), dpi=350)
    add_figure_header_banner(
        fig,
        "Figure 1 | Non-Abelian 3D Lie Simplex (NA3D), Asymmetric Finsler-Randers Indicatrix & Lemma 1 Orthogonal Bundle",
        "Native Rust GGX 3D ray-traced manifold geometry paired with real NA3D tensors from ARC-AGI-2 Task 1818057f verifying machine-zero color-subspace leakage",
        y_top=0.958,
    )

    outer_gs = GridSpec(
        2,
        2,
        figure=fig,
        left=0.048,
        right=0.950,
        top=0.842,
        bottom=0.068,
        hspace=0.38,
        wspace=0.18,
        width_ratios=[1.12, 1.0],
        height_ratios=[1.08, 0.96],
    )

    gs_a = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[0, 0], wspace=0.16, width_ratios=[1.18, 0.95])
    ax_a1 = fig.add_subplot(gs_a[0, 0])
    ax_a2 = fig.add_subplot(gs_a[0, 1])

    draw_panel_badge(
        ax_a1,
        "a",
        "3D NA3D Oriented Simplex & Asymmetric Finsler-Randers Indicatrix Shell",
        r"Ray-traced 3D simplex $(\vec{L},\vec{C},\vec{R})\in(\mathbb{R}^3)^3$ inside $F(x,\mathbf{v}) = \sqrt{\mathbf{v}^\top A(x)\mathbf{v}} + \boldsymbol{\beta}_x^\top\mathbf{v} = 1$",
        badge_bg="#0F172A",
        x_pos=-0.04,
        y_pos=1.13,
    )

    if scene1_img is not None:
        ax_a1.imshow(scene1_img, interpolation="lanczos")
        h_s1, w_s1 = scene1_img.shape[:2]
        ax_a1.text(
            0.03 * w_s1,
            0.07 * h_s1,
            r"Finsler-Randers Shell $F(x,\mathbf{v})=1$",
            fontsize=7.5,
            fontweight="bold",
            color="#E2E8F0",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#0F172A", edgecolor="#3B82F6", lw=1.1, alpha=0.92),
        )
        ax_a1.text(
            0.56 * w_s1,
            0.07 * h_s1,
            r"Wilson Plaquette $[R_y, R_x] \neq 0$",
            fontsize=7.3,
            fontweight="bold",
            color="#F5D0FE",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#0F172A", edgecolor="#D946EF", lw=1.1, alpha=0.92),
        )
        ax_a1.text(
            0.03 * w_s1,
            0.93 * h_s1,
            r"$\vec{L}$ (Sapphire), $\vec{C}$ (Emerald), $\vec{R}$ (Ruby), Drift $\vec{\beta}$ (Gold)",
            fontsize=7.1,
            fontweight="bold",
            color="#FFFFFF",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#0F172A", edgecolor="#38BDF8", lw=1.0, alpha=0.92),
        )
    ax_a1.set_xticks([])
    ax_a1.set_yticks([])
    ax_a1.set_title(
        "3D Ray-Traced NA3D Simplex & Indicatrix\nAsymmetric directional attention modulator",
        fontsize=8.8,
        fontweight="bold",
        color="#0F172A",
        pad=5.5,
    )

    theta = np.linspace(0, 2 * np.pi, 360)
    rx_riem = np.cos(theta)
    ry_riem = np.sin(theta)
    g_xx, g_yy = 1.15, 0.88
    omega_x, omega_y = -0.34, -0.18
    norm_g = np.sqrt(g_xx * (np.cos(theta) ** 2) + g_yy * (np.sin(theta) ** 2))
    dot_omega = omega_x * np.cos(theta) + omega_y * np.sin(theta)
    r_finsler = 1.0 / (norm_g + dot_omega)
    fx_ind = r_finsler * np.cos(theta)
    fy_ind = r_finsler * np.sin(theta)

    ax_a2.axhline(0.0, color="#CBD5E1", lw=0.8, zorder=1)
    ax_a2.axvline(0.0, color="#CBD5E1", lw=0.8, zorder=1)
    for rad in [0.5, 1.0, 1.5]:
        ax_a2.add_patch(Circle((0, 0), rad, fill=False, edgecolor="#E2E8F0", lw=0.7, ls=":", zorder=1))

    ax_a2.plot(rx_riem, ry_riem, linestyle="--", linewidth=1.6, color="#64748B", label=r"Riemannian ($\boldsymbol{\beta}=\mathbf{0}$)", zorder=3)
    ax_a2.fill(fx_ind, fy_ind, color="#3B82F6", alpha=0.16, zorder=4)
    ax_a2.plot(fx_ind, fy_ind, linewidth=2.2, color="#1D4ED8", label=r"Randers ($\|\boldsymbol{\beta}\|=0.38$)", zorder=5)

    angle_fwd = np.deg2rad(28.0)
    idx_fwd = int(np.argmin(np.abs(theta - angle_fwd)))
    idx_bwd = int(np.argmin(np.abs(theta - (angle_fwd + np.pi))))
    ax_a2.annotate(
        "",
        xy=(fx_ind[idx_fwd], fy_ind[idx_fwd]),
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="-|>,head_width=0.25,head_length=0.3", color="#059669", lw=2.0),
        zorder=8,
    )
    ax_a2.annotate(
        "",
        xy=(fx_ind[idx_bwd], fy_ind[idx_bwd]),
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="-|>,head_width=0.25,head_length=0.3", color="#E11D48", lw=2.0),
        zorder=8,
    )
    ax_a2.text(
        0.15,
        -1.26,
        r"$F(+\mathbf{v})=0.62 \neq F(-\mathbf{v})=1.38$",
        fontsize=7.5,
        fontweight="bold",
        color="#0F172A",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#F8FAFC", edgecolor="#94A3B8", lw=0.9),
    )
    ax_a2.set_xlim(-1.85, 2.25)
    ax_a2.set_ylim(-1.48, 1.55)
    ax_a2.set_aspect("equal", adjustable="box")
    ax_a2.set_xlabel(r"Tangent Velocity $v_x$", fontsize=8.2)
    ax_a2.set_ylabel(r"Tangent Velocity $v_y$", fontsize=8.2)
    ax_a2.set_title(
        "Equatorial Indicatrix Cross-Section\n" + r"Breaks $d(p,q) = d(q,p)$ symmetry",
        fontsize=8.8,
        fontweight="bold",
        color="#0F172A",
        pad=5.5,
    )
    ax_a2.legend(loc="upper left", fontsize=7.3, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    gs_b = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[0, 1], wspace=0.20, width_ratios=[1.15, 0.95])
    ax_b1 = fig.add_subplot(gs_b[0, 0])
    ax_b2 = fig.add_subplot(gs_b[0, 1])

    draw_panel_badge(
        ax_b1,
        "b",
        "3D Orthogonal Fiber Bundle & Plaquette Holonomy Field",
        r"$\mathbb{R}^{128} = \mathcal{H}_{\mathrm{color}}^{(11)} \oplus \mathcal{H}_{\mathrm{geom}}^{(117)}$ prevents $\mathrm{SO}(3)^8$ gauge transport from rotating discrete color IDs",
        badge_bg="#1E40AF",
        x_pos=-0.04,
        y_pos=1.13,
    )

    if scene2_img is not None:
        ax_b1.imshow(scene2_img, interpolation="lanczos")
        h_s2, w_s2 = scene2_img.shape[:2]
        ax_b1.text(
            0.03 * w_s2,
            0.07 * h_s2,
            r"Split $\mathrm{SO}(3)$ Tori in $\mathcal{H}_{\mathrm{geom}}^{(117)}$",
            fontsize=7.3,
            fontweight="bold",
            color="#A7F3D0",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#0F172A", edgecolor="#10B981", lw=1.1, alpha=0.92),
        )
        ax_b1.text(
            0.55 * w_s2,
            0.07 * h_s2,
            "Unsplit Leak (Red Helix)",
            fontsize=7.1,
            fontweight="bold",
            color="#FECDD3",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#0F172A", edgecolor="#F43F5E", lw=1.1, alpha=0.92),
        )
        ax_b1.text(
            0.03 * w_s2,
            0.93 * h_s2,
            r"Base Platform: 11 Orthonormal Color Fibers $\mathcal{H}_{\mathrm{color}}^{(11)}$",
            fontsize=7.1,
            fontweight="bold",
            color="#FFFFFF",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#0F172A", edgecolor="#38BDF8", lw=1.0, alpha=0.92),
        )
    ax_b1.set_xticks([])
    ax_b1.set_yticks([])
    ax_b1.set_title(
        "3D Ray-Traced Orthogonal Fiber Bundle\nHorizontal orbits conserve discrete palette",
        fontsize=8.8,
        fontweight="bold",
        color="#0F172A",
        pad=5.5,
    )

    im_plaq = ax_b2.imshow(plaq_np, cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal")
    for r in range(12):
        ax_b2.axhline(r - 0.5, color="#FFFFFF", linewidth=0.35, alpha=0.25)
        ax_b2.axvline(r - 0.5, color="#FFFFFF", linewidth=0.35, alpha=0.25)
    ax_b2.set_xticks([])
    ax_b2.set_yticks([])
    ax_b2.set_title(
        "Wilson Plaquette Curvature (Task 1818057f)\n"
        + r"$\mathcal{K}_{\mathrm{plaq}}(y,x) = 1 - \cos(\pi\tanh\|\mathbf{F}_{\mathrm{plaq}}\|_2)$",
        fontsize=8.8,
        fontweight="bold",
        color="#0F172A",
        pad=5.5,
    )
    cbar_b = fig.colorbar(im_plaq, ax=ax_b2, fraction=0.046, pad=0.04)
    cbar_b.ax.tick_params(labelsize=7.2)
    cbar_b.set_label("Normalized Plaquette Curvature", fontsize=7.6, color="#334155")

    gs_c = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[1, 0], wspace=0.18)
    ax_c1 = fig.add_subplot(gs_c[0, 0])
    ax_c2 = fig.add_subplot(gs_c[0, 1])
    ax_c3 = fig.add_subplot(gs_c[0, 2])

    draw_panel_badge(
        ax_c1,
        "c",
        "Visual Verification of Lemma 1: Orthogonal Subspace Split Prevents Color Corruption",
        r"Task 1818057f patch [8:19, 10:21]: unsplit $\mathrm{SO}(3)^8$ holonomy vs. $\mathcal{H}_{\mathrm{color}}^{(11)} \oplus \mathcal{H}_{\mathrm{geom}}^{(117)}$ MGS2 projection",
        badge_bg="#0F172A",
        x_pos=-0.06,
        y_pos=1.18,
    )

    render_arc_grid(
        ax_c1,
        patch_out,
        title="Target Symbolic Patch",
        subtitle="Exact palette identities",
        show_values=True,
        title_fontsize=8.6,
    )

    bad_cells = [
        (r, c, "#FFFFFF")
        for r in range(11)
        for c in range(11)
        if corrupted_patch[r, c] != patch_out[r, c]
    ]
    unsplit_acc_pct = 100.0 * float((corrupted_patch == patch_out).mean())
    render_arc_grid(
        ax_c2,
        corrupted_patch,
        title=r"Unsplit $\mathrm{SO}(3)^8$ Holonomy",
        subtitle=f"Color Drift ({unsplit_acc_pct:.1f}% Acc | Leak {leak_unsplit_val:.3f})",
        show_values=True,
        highlight_cells=bad_cells if len(bad_cells) <= 60 else None,
        title_color="#B91C1C",
        title_fontsize=8.6,
    )

    split_match_pct = 100.0 * float((pred_split_patch == patch_out).mean())
    render_arc_grid(
        ax_c3,
        pred_split_patch,
        title="Lemma 1 Split NA3D (Ours)",
        subtitle=f"Leak = {leak_split_val:.2e} ({split_match_pct:.0f}% Exact)",
        show_values=True,
        title_color="#047857",
        title_fontsize=8.6,
    )

    gs_d = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[1, 1], wspace=0.25)
    ax_d1 = fig.add_subplot(gs_d[0, 0])
    ax_d2 = fig.add_subplot(gs_d[0, 1])

    draw_panel_badge(
        ax_d1,
        "d",
        r"Measured $10\times 10$ Prototype Cross-Leakage Matrix (Patch vs. 25-Task Benchmark)",
        r"Coupling $|\langle \mathbf{h}_{\mathrm{after}}(\mathbf{e}_c) - \mathbf{h}_{\mathrm{before}}(\mathbf{e}_c), \mathbf{e}_{c'}\rangle|$ under $\mathrm{SO}(3)^8$ holonomy",
        badge_bg="#047857",
        x_pos=-0.04,
        y_pos=1.18,
    )

    vmax_leak = max(0.32, float(leak_matrix_unsplit.max()))
    im_d1 = ax_d1.imshow(leak_matrix_unsplit, cmap="Reds", vmin=0.0, vmax=vmax_leak, aspect="equal")
    ax_d1.set_xticks(range(10))
    ax_d1.set_yticks(range(10))
    ax_d1.set_xticklabels([str(i) for i in range(10)], fontsize=7.5)
    ax_d1.set_yticklabels([str(i) for i in range(10)], fontsize=7.5)
    ax_d1.set_xlabel("Target Color Symbol $c'$", fontsize=8.2)
    ax_d1.set_ylabel("Source Color Symbol $c$", fontsize=8.2)
    ax_d1.set_title(
        r"Unsplit $\mathrm{SO}(3)^8$ Baseline"
        + f"\nPatch Max = {leak_unsplit_val:.3f} (25-Task = {bench_leak_unsplit:.3f})",
        fontsize=8.4,
        fontweight="bold",
        color="#B91C1C",
        pad=5.5,
    )

    ax_d2.imshow(leak_matrix_split, cmap="Blues", vmin=0.0, vmax=vmax_leak, aspect="equal")
    ax_d2.set_xticks(range(10))
    ax_d2.set_yticks(range(10))
    ax_d2.set_xticklabels([str(i) for i in range(10)], fontsize=7.5)
    ax_d2.set_yticklabels([str(i) for i in range(10)], fontsize=7.5)
    ax_d2.set_xlabel("Target Color Symbol $c'$", fontsize=8.2)
    ax_d2.set_title(
        "Orthogonal-Split NA3D (Ours)\n"
        + f"Patch = {leak_split_val:.2e} (25-Task = {bench_leak_split:.2e})",
        fontsize=8.4,
        fontweight="bold",
        color="#047857",
        pad=5.5,
    )
    ratio_patch = leak_unsplit_val / max(leak_split_val, 1e-12)
    ratio_bench = bench_leak_unsplit / max(bench_leak_split, 1e-12)
    ax_d2.text(
        4.5,
        4.5,
        "IEEE-754 FP32\nOrthogonal Projection\n"
        + r"$\mathbf{Q}\mathbf{P}_{\mathrm{geom}} \equiv \mathbf{0}$"
        + f"\nPatch: {ratio_patch:.1e}x Lower\n25-Task: {ratio_bench:.1e}x Lower",
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color="#065F46",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#ECFDF5", edgecolor="#10B981", lw=1.2, alpha=0.95),
    )

    cbar_d = fig.colorbar(im_d1, ax=[ax_d1, ax_d2], fraction=0.032, pad=0.04)
    cbar_d.ax.tick_params(labelsize=7.5)
    cbar_d.set_label("Color Subspace Coupling Magnitude", fontsize=8.0)

    png_path = output_dir / "fig01_na3d_finsler_lie_3d.png"
    pdf_path = output_dir / "fig01_na3d_finsler_lie_3d.pdf"
    fig.savefig(png_path, dpi=350)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


if __name__ == "__main__":
    out = generate_fig01()
    print(f"Saved: {out}")
