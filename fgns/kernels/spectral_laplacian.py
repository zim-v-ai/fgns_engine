"""
Location: // fgns_engine/fgns/kernels/spectral_laplacian.py
Description: Matrix-Free Chebyshev Graph Laplacian, Reference-Retaining Zero-ABA Cached CUDA
             Finsler-Chordal Cavity Solver, Two-Shell Punctured-Gibbs Bravais Lattice Cohomology,
             Morphological Cross Detector, and Full-Diameter Geodesic Eikonal Propagator (FGNS v5.0.0):
               1. Zero-ABA Lifecycle-Safe Bundle Cache: retains tensor references alongside version
                  counters and explicit clear_cache() boundaries, completely preventing
                  CUDACachingAllocator pointer-recycling collisions across tasks.
               2. Universal Two-Shell Punctured-Gibbs Bravais Resonator: evaluates 1-shell and 2-shell
                  punctured neighborhood inner products A_phase((y, x), (y + m*T_y, x + n*T_x))
                  via smooth Boltzmann-Gibbs weights without any wall-length or background heuristics.
               3. Pre-Eroded 3x3 Moore Chebyshev Eikonal Distance and 48-step 4-Connected
                  Von Neumann Geodesic Flood around arbitrary L/U-shaped rooms.
               4. Exact 5-Cell Von Neumann Plus-Cross Morphological Erosion Operator B_+.
Language: Python 3.12+ / PyTorch 2.5+ / Native CUDA C++17 sm_120 & sm_89 Aligned
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import math
from typing import Dict, Final, List, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from fgns.kernels.cuda_fgns_ops import cuda_finsler_chordal_cavity_op
from fgns.primitives.swiglu import RMSNorm, SwiGLU
from fgns.types import BravaisPeriodicityState, ColorRoleState, EikonalLevelSetState

DEFAULT_LAPLACIAN_EPS: Final[float] = 1e-7


class StencilGraphLaplacianOps:
    """
    Fast F.pad slice-vectorized 2D stencil implementation of the normalized graph Laplacian.
    """

    DIRECTIONS: Tuple[Tuple[int, int, float], ...] = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, 0.70710678),
        (-1, 1, 0.70710678),
        (1, -1, 0.70710678),
        (1, 1, 0.70710678),
    )

    @classmethod
    def shift_tensor_2d(cls, x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        B, H, W, C = x.shape
        x_pad = F.pad(x.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="constant", value=0.0).permute(0, 2, 3, 1)
        y0 = 1 - dy
        x0 = 1 - dx
        return x_pad[:, y0 : y0 + H, x0 : x0 + W, :]

    @classmethod
    def compute_edge_weights_and_degree(
        cls, grid: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, W = grid.shape
        device = grid.device
        mask_f32 = mask.float().unsqueeze(-1)
        g_f32 = grid.float().unsqueeze(-1)

        g_pad = F.pad(g_f32.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="constant", value=-1.0).permute(0, 2, 3, 1)
        m_pad = F.pad(mask_f32.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="constant", value=0.0).permute(0, 2, 3, 1)

        nb_g = torch.cat(
            [g_pad[:, (1 - dy) : (1 - dy + H), (1 - dx) : (1 - dx + W), :] for dy, dx, _ in cls.DIRECTIONS],
            dim=-1,
        )
        nb_m = torch.cat(
            [m_pad[:, (1 - dy) : (1 - dy + H), (1 - dx) : (1 - dx + W), :] for dy, dx, _ in cls.DIRECTIONS],
            dim=-1,
        )
        metric_w = torch.tensor(
            [m for _, _, m in cls.DIRECTIONS], dtype=torch.float32, device=device
        ).view(1, 1, 1, 8)

        edge_weights = (g_f32 == nb_g).float() * mask_f32 * nb_m * metric_w
        degree_accum = torch.sum(edge_weights, dim=-1, keepdim=True)

        is_positive_deg = degree_accum > 0.0
        inv_sqrt_deg = torch.where(
            is_positive_deg,
            torch.rsqrt(torch.clamp_min(degree_accum, DEFAULT_LAPLACIAN_EPS)),
            torch.zeros_like(degree_accum),
        ) * mask_f32

        return edge_weights, inv_sqrt_deg

    @classmethod
    def apply_normalized_laplacian(
        cls,
        v: torch.Tensor,
        edge_weights: torch.Tensor,
        inv_sqrt_deg: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        B, H, W, C = v.shape
        mask_f32 = mask.float().unsqueeze(-1)
        v_scaled = v * inv_sqrt_deg

        v_pad = F.pad(v_scaled.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="constant", value=0.0).permute(0, 2, 3, 1)
        nb_stack = torch.stack(
            [v_pad[:, (1 - dy) : (1 - dy + H), (1 - dx) : (1 - dx + W), :] for dy, dx, _ in cls.DIRECTIONS],
            dim=3,
        )

        adjacency_v = torch.sum(edge_weights.unsqueeze(-1) * nb_stack, dim=3)
        normalized_adjacency_v = adjacency_v * inv_sqrt_deg
        return (-normalized_adjacency_v) * mask_f32


class PoissonDirichletCavitySolver:
    """
    Reference-Retaining Zero-ABA Cached Native CUDA C++ Finsler-Chordal,
    Vectorized Color Structural Role, Two-Shell Punctured-Gibbs Bravais Lattice,
    Morphological Plus-Cross, and Full-Diameter Geodesic Eikonal Level-Set Engine.
    """

    _BUNDLE_CACHE_DICT: Dict[
        Tuple[int, int, Tuple[int, ...], str, int, int],
        Tuple[
            torch.Tensor,
            torch.Tensor,
            Tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ],
    ] = {}

    SHELL_1_DIRS: Tuple[Tuple[int, int], ...] = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    SHELL_2_DIRS: Tuple[Tuple[int, int], ...] = (
        (-2, 0), (2, 0), (0, -2), (0, 2),
        (-2, -1), (-2, 1), (2, -1), (2, 1),
        (-1, -2), (1, -2), (-1, 2), (1, 2),
    )

    @classmethod
    def clear_cache(cls) -> None:
        cls._BUNDLE_CACHE_DICT.clear()

    @classmethod
    def _compute_full_cached(
        cls,
        grid: torch.Tensor,
        mask: torch.Tensor,
        num_iterations: int = 12,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        with torch.no_grad():
            key = (
                int(grid.data_ptr()),
                int(mask.data_ptr()),
                tuple(grid.shape),
                str(grid.device),
                int(grid._version),
                int(mask._version),
            )
            if key in cls._BUNDLE_CACHE_DICT:
                cached_grid, cached_mask, cached_res = cls._BUNDLE_CACHE_DICT[key]
                if cached_grid is grid and cached_mask is mask:
                    return cached_res

            res = cuda_finsler_chordal_cavity_op(grid, mask, num_iterations=num_iterations)
            if len(cls._BUNDLE_CACHE_DICT) >= 6:
                cls._BUNDLE_CACHE_DICT.clear()
            cls._BUNDLE_CACHE_DICT[key] = (grid, mask, res)
            return res

    @classmethod
    def compute_morphological_plus_cross_mask(
        cls,
        grid: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Morphological 5-Cell Von Neumann Plus-Cross Erosion Operator B_+:
        A pixel (y, x) of color c > 0 is a cross center iff all 4 cardinal neighbors (N, S, W, E)
        equal c and at least 3 of its 4 diagonal neighbors (NW, NE, SW, SE) differ from c.
        """
        orig_batch = grid.shape[:-2]
        H, W = grid.shape[-2], grid.shape[-1]

        g_flat = grid.reshape(-1, H, W)
        m_flat = mask.reshape(-1, H, W)

        g_pad = F.pad(g_flat.unsqueeze(1).float(), (1, 1, 1, 1), mode="constant", value=-1.0).squeeze(1).to(torch.int64)
        m_pad = F.pad(m_flat.unsqueeze(1).float(), (1, 1, 1, 1), mode="constant", value=0.0).squeeze(1) > 0.5

        g_n = g_pad[:, :-2, 1:-1]
        g_s = g_pad[:, 2:, 1:-1]
        g_w = g_pad[:, 1:-1, :-2]
        g_e = g_pad[:, 1:-1, 2:]

        m_n = m_pad[:, :-2, 1:-1]
        m_s = m_pad[:, 2:, 1:-1]
        m_w = m_pad[:, 1:-1, :-2]
        m_e = m_pad[:, 1:-1, 2:]

        g_nw = g_pad[:, :-2, :-2]
        g_ne = g_pad[:, :-2, 2:]
        g_sw = g_pad[:, 2:, :-2]
        g_se = g_pad[:, 2:, 2:]

        m_nw = m_pad[:, :-2, :-2]
        m_ne = m_pad[:, :-2, 2:]
        m_sw = m_pad[:, 2:, :-2]
        m_se = m_pad[:, 2:, 2:]

        is_fg = (g_flat > 0) & (g_flat < 10) & m_flat
        card_all_same = (
            (g_n == g_flat) & m_n
            & (g_s == g_flat) & m_s
            & (g_w == g_flat) & m_w
            & (g_e == g_flat) & m_e
        )
        diag_diff_count = (
            ((g_nw != g_flat) & m_nw).to(torch.int64)
            + ((g_ne != g_flat) & m_ne).to(torch.int64)
            + ((g_sw != g_flat) & m_sw).to(torch.int64)
            + ((g_se != g_flat) & m_se).to(torch.int64)
        )

        is_center = is_fg & card_all_same & (diag_diff_count >= 3)
        c_pad = F.pad(is_center.unsqueeze(1).float(), (1, 1, 1, 1), mode="constant", value=0.0).squeeze(1) > 0.5

        arm_from_n = c_pad[:, :-2, 1:-1] & (g_n == g_flat)
        arm_from_s = c_pad[:, 2:, 1:-1] & (g_s == g_flat)
        arm_from_w = c_pad[:, 1:-1, :-2] & (g_w == g_flat)
        arm_from_e = c_pad[:, 1:-1, 2:] & (g_e == g_flat)

        is_plus_full = (is_center | arm_from_n | arm_from_s | arm_from_w | arm_from_e) & is_fg
        return is_plus_full.to(torch.float32).view(*orig_batch, H, W)

    @classmethod
    def _recover_cardinal_rays_and_eikonal(
        cls,
        grid: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            _,
            h_span,
            w_span,
            _,
            ey,
            ex,
            _,
            is_wall,
            is_seed,
        ) = cls._compute_full_cached(grid, mask, num_iterations=10)

        ell_n = torch.round(0.50 * (h_span * (1.0 + ey) + 1.0)).clamp_min(1.0)
        ell_s = torch.round(0.50 * (h_span * (1.0 - ey) + 1.0)).clamp_min(1.0)
        ell_w = torch.round(0.50 * (w_span * (1.0 + ex) + 1.0)).clamp_min(1.0)
        ell_e = torch.round(0.50 * (w_span * (1.0 - ex) + 1.0)).clamp_min(1.0)

        min_ray = torch.minimum(torch.minimum(ell_n, ell_s), torch.minimum(ell_w, ell_e))
        d_eikonal = F.relu(min_ray - 1.0) * (1.0 - is_wall) * mask.float().unsqueeze(-1)

        return (
            ell_n.squeeze(-1),
            ell_s.squeeze(-1),
            ell_w.squeeze(-1),
            ell_e.squeeze(-1),
            d_eikonal.squeeze(-1),
            is_wall.squeeze(-1),
            is_seed.squeeze(-1),
        )

    @classmethod
    def _extract_wall_and_seed_masks(
        cls,
        grid: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        res = cls._compute_full_cached(grid, mask, num_iterations=10)
        is_wall = res[7]
        is_seed = res[8]
        mask_f32 = mask.float().unsqueeze(-1)
        is_transparent = (1.0 - is_wall).clamp(0.0, 1.0) * mask_f32
        return is_transparent, is_wall, is_seed

    @classmethod
    def compute_finsler_chordal_bundle(
        cls,
        grid: torch.Tensor,
        mask: torch.Tensor,
        num_iterations: int = 12,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            m_cav_bg,
            h_span,
            w_span,
            r_harm,
            ey,
            ex,
            seed_f,
            _,
            _,
        ) = cls._compute_full_cached(grid, mask, num_iterations=num_iterations)
        dipole_yx = torch.cat([ey, ex], dim=-1)
        return m_cav_bg, h_span, w_span, r_harm, dipole_yx, seed_f

    @classmethod
    def solve_cavity_potential(
        cls,
        grid: torch.Tensor,
        mask: torch.Tensor,
        num_iterations: int = 12,
    ) -> torch.Tensor:
        mask_f32 = mask.float().unsqueeze(-1)
        m_cavity_bg, h_span, w_span, r_harmonic, _, _ = cls.compute_finsler_chordal_bundle(
            grid, mask, num_iterations=num_iterations
        )
        min_span = torch.minimum(h_span, w_span)
        span_depth = torch.tanh((min_span - 1.0) / 6.0) * m_cavity_bg
        harm_depth = torch.tanh(r_harmonic / 4.0) * m_cavity_bg
        return (0.50 * m_cavity_bg + 0.25 * span_depth + 0.25 * harm_depth) * mask_f32

    @classmethod
    def extract_color_structural_roles(
        cls,
        grid: torch.Tensor,
        mask: torch.Tensor,
        support_inputs: Optional[torch.Tensor] = None,
        support_in_masks: Optional[torch.Tensor] = None,
        support_outputs: Optional[torch.Tensor] = None,
        support_out_masks: Optional[torch.Tensor] = None,
    ) -> ColorRoleState:
        orig_batch = grid.shape[:-2]
        H, W = grid.shape[-2], grid.shape[-1]
        device = grid.device

        g_flat = grid.reshape(-1, H, W)
        m_flat = mask.reshape(-1, H, W)
        N = g_flat.shape[0]

        m_f32 = m_flat.float()
        active_h = m_f32.max(dim=-1)[0].sum(dim=-1).clamp_min(1.0)
        active_w = m_f32.max(dim=-2)[0].sum(dim=-1).clamp_min(1.0)
        active_area = m_f32.sum(dim=(-2, -1)).clamp_min(1.0)

        (
            m_cav_bg,
            _,
            _,
            _,
            _,
            _,
            _,
            is_wall,
            is_seed,
        ) = cls._compute_full_cached(g_flat, m_flat, num_iterations=10)

        y_idx = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1, 1)
        x_idx = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W, 1)

        is_border_px = (
            (y_idx.squeeze(-1) == 0)
            | (y_idx.squeeze(-1) == (active_h.view(N, 1, 1) - 1.0))
            | (x_idx.squeeze(-1) == 0)
            | (x_idx.squeeze(-1) == (active_w.view(N, 1, 1) - 1.0))
        ) & m_flat
        border_total = is_border_px.float().sum(dim=(-2, -1)).clamp_min(1.0).unsqueeze(-1)

        is_c = (F.one_hot(g_flat.clamp(0, 9), num_classes=10).bool()) & m_flat.unsqueeze(-1)
        c_f32 = is_c.float()
        c_mass = c_f32.sum(dim=(1, 2))
        has_c = (c_mass > 0.5).float()

        r1_density = (c_mass / active_area.unsqueeze(-1)).clamp(0.0, 1.0)
        r2_border = ((c_f32 * is_border_px.float().unsqueeze(-1)).sum(dim=(1, 2)) / border_total).clamp(0.0, 1.0)

        row_fill = (c_f32.sum(dim=2) / active_w.view(N, 1, 1)).max(dim=1)[0]
        col_fill = (c_f32.sum(dim=1) / active_h.view(N, 1, 1)).max(dim=1)[0]
        r3_divider = torch.maximum(row_fill, col_fill).clamp(0.0, 1.0) * has_c

        ymin_c = torch.where(is_c, y_idx, torch.full_like(y_idx, float(H))).amin(dim=(1, 2))
        ymax_c = torch.where(is_c, y_idx, torch.full_like(y_idx, -1.0)).amax(dim=(1, 2))
        xmin_c = torch.where(is_c, x_idx, torch.full_like(x_idx, float(W))).amin(dim=(1, 2))
        xmax_c = torch.where(is_c, x_idx, torch.full_like(x_idx, -1.0)).amax(dim=(1, 2))

        h_box = (ymax_c - ymin_c + 1.0).clamp_min(1.0)
        w_box = (xmax_c - xmin_c + 1.0).clamp_min(1.0)
        box_area = h_box * w_box
        r4_rect_fill = ((c_mass / box_area) * has_c * (c_mass >= 4.0).float()).clamp(0.0, 1.0)

        seed_c = (c_f32 * is_seed).sum(dim=(1, 2)) / c_mass.clamp_min(1.0)
        r5_seed = (seed_c * has_c).clamp(0.0, 1.0)

        wall_c = (c_f32 * is_wall).sum(dim=(1, 2)) / c_mass.clamp_min(1.0)
        r6_wall = (wall_c * has_c).clamp(0.0, 1.0)

        cav_adj = (c_f32 * m_cav_bg).sum(dim=(1, 2)) / m_cav_bg.sum(dim=(1, 2)).clamp_min(1.0)
        r7_cavity = (cav_adj * has_c).clamp(0.0, 1.0)

        r8_aspect = (torch.minimum(h_box, w_box) / torch.maximum(h_box, w_box).clamp_min(1.0)) * has_c

        role_vectors = torch.stack(
            [r1_density, r2_border, r3_divider, r4_rect_fill, r5_seed, r6_wall, r7_cavity, r8_aspect],
            dim=-1,
        )

        act_h_exp = active_h.unsqueeze(-1)
        act_w_exp = active_w.unsqueeze(-1)
        is_interior_y = (ymin_c >= 1.0) & (ymax_c <= (act_h_exp - 2.0))
        is_interior_x = (xmin_c >= 1.0) & (xmax_c <= (act_w_exp - 2.0))
        is_thin_line = ((h_box == act_h_exp) & (w_box <= 2.0) & is_interior_x) | (
            (w_box == act_w_exp) & (h_box <= 2.0) & is_interior_y
        )

        div_mat = torch.where(
            (r3_divider > 0.95) & is_thin_line,
            r3_divider + (1.0 - r1_density),
            torch.full_like(r3_divider, -1e9),
        )
        occ_mat = torch.where(
            (r4_rect_fill > 0.98) & (h_box >= 2.0) & (w_box >= 2.0) & (r1_density < 0.50),
            r4_rect_fill + (1.0 / (c_mass + 1.0)),
            torch.full_like(r4_rect_fill, -1e9),
        )
        seed_mat = torch.where(
            (c_mass == 1.0) & (r5_seed > 0.95),
            r5_seed + 1.0,
            torch.full_like(r5_seed, -1e9),
        )

        div_mat[:, 0] = -1e9
        occ_mat[:, 0] = -1e9
        seed_mat[:, 0] = -1e9

        diode_sink = torch.zeros((N, 10), dtype=torch.float32, device=device)
        if (
            support_inputs is not None
            and support_in_masks is not None
            and support_outputs is not None
            and support_out_masks is not None
        ):
            B_s, K_s = support_inputs.shape[:2]
            s_in_oh = F.one_hot(support_inputs.clamp(0, 9), 10).bool() & support_in_masks.unsqueeze(-1)
            s_out_oh = F.one_hot(support_outputs.clamp(0, 9), 10).bool() & support_out_masks.unsqueeze(-1)
            s_in_has = s_in_oh.any(dim=(2, 3)).float()
            s_out_has = s_out_oh.any(dim=(2, 3)).float()
            vanished = ((s_in_has > 0.5) & (s_out_has < 0.5)).float().sum(dim=1) / s_in_has.sum(dim=1).clamp_min(1.0)
            if N == B_s * K_s:
                vanished = vanished.repeat_interleave(K_s, dim=0)
            diode_sink = (vanished > 0.80).float()
            diode_sink[:, 0] = 0.0

            occ_mat = occ_mat + (5.0 * diode_sink)
            div_mat = div_mat + (5.0 * diode_sink)

        max_div_val, max_div_col = torch.max(div_mat, dim=-1)
        divider_color = torch.where(max_div_val > 0.0, max_div_col, torch.full_like(max_div_col, -1))

        max_occ_val, max_occ_col = torch.max(occ_mat, dim=-1)
        occluder_color = torch.where(max_occ_val > 0.0, max_occ_col, torch.full_like(max_occ_col, -1))

        max_seed_val, max_seed_col = torch.max(seed_mat, dim=-1)
        seed_color = torch.where(max_seed_val > 0.0, max_seed_col, torch.full_like(max_seed_col, -1))

        return ColorRoleState(
            role_vectors=role_vectors.view(*orig_batch, 10, 8),
            diode_sink_mask=diode_sink.view(*orig_batch, 10),
            divider_color=divider_color.view(*orig_batch),
            occluder_color=occluder_color.view(*orig_batch),
            seed_color=seed_color.view(*orig_batch),
        )

    @classmethod
    def compute_bravais_lattice_periodicity(
        cls,
        grid: torch.Tensor,
        mask: torch.Tensor,
        max_period: int = 8,
    ) -> BravaisPeriodicityState:
        """
        Universal Two-Shell Punctured-Gibbs Bravais Lattice Resonator:
        Evaluates 1-shell and 2-shell punctured neighborhood inner products
        A_phase((y, x), (y + m*T_y, x + n*T_x)) via smooth Boltzmann-Gibbs weights
        without any wall-length or background-color heuristics.
        """
        orig_batch = grid.shape[:-2]
        H, W = grid.shape[-2], grid.shape[-1]
        device = grid.device

        g_flat = grid.reshape(-1, H, W)
        m_flat = mask.reshape(-1, H, W)
        N = g_flat.shape[0]

        g_pad1 = F.pad(g_flat.unsqueeze(1).float(), (1, 1, 1, 1), mode="constant", value=-1.0).squeeze(1).to(torch.int64)
        m_pad1 = F.pad(m_flat.unsqueeze(1).float(), (1, 1, 1, 1), mode="constant", value=0.0).squeeze(1) > 0.5

        nb1_tokens = torch.stack(
            [g_pad1[:, (1 + dy) : (1 + dy + H), (1 + dx) : (1 + dx + W)] for dy, dx in cls.SHELL_1_DIRS],
            dim=-1,
        )
        nb1_valid = torch.stack(
            [m_pad1[:, (1 + dy) : (1 + dy + H), (1 + dx) : (1 + dx + W)] for dy, dx in cls.SHELL_1_DIRS],
            dim=-1,
        )

        g_pad2 = F.pad(g_flat.unsqueeze(1).float(), (2, 2, 2, 2), mode="constant", value=-1.0).squeeze(1).to(torch.int64)
        m_pad2 = F.pad(m_flat.unsqueeze(1).float(), (2, 2, 2, 2), mode="constant", value=0.0).squeeze(1) > 0.5

        nb2_tokens = torch.stack(
            [g_pad2[:, (2 + dy) : (2 + dy + H), (2 + dx) : (2 + dx + W)] for dy, dx in cls.SHELL_2_DIRS],
            dim=-1,
        )
        nb2_valid = torch.stack(
            [m_pad2[:, (2 + dy) : (2 + dy + H), (2 + dx) : (2 + dx + W)] for dy, dx in cls.SHELL_2_DIRS],
            dim=-1,
        )

        y_int = torch.arange(H, device=device, dtype=torch.int64).view(1, H, 1).expand(N, H, W)
        x_int = torch.arange(W, device=device, dtype=torch.int64).view(1, 1, W).expand(N, H, W)
        b_id = torch.arange(N, device=device, dtype=torch.int64).view(N, 1, 1).expand(N, H, W)

        active_h = m_flat.float().max(dim=-1)[0].sum(dim=-1).to(torch.int64).clamp_min(1).view(N, 1, 1)
        active_w = m_flat.float().max(dim=-2)[0].sum(dim=-1).to(torch.int64).clamp_min(1).view(N, 1, 1)
        is_interior = (
            (y_int >= 1)
            & (y_int < active_h - 1)
            & (x_int >= 1)
            & (x_int < active_w - 1)
            & m_flat
        )

        one_hot = F.one_hot(g_flat.clamp(0, 9), num_classes=10).to(torch.float32) * m_flat.float().unsqueeze(-1)

        mult_vec = torch.tensor([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6], device=device, dtype=torch.int64).view(
            1, 1, 1, 12
        )
        b_4d = b_id.unsqueeze(-1).expand(N, H, W, 12)
        y_4d = y_int.unsqueeze(-1).expand(N, H, W, 12)
        x_4d = x_int.unsqueeze(-1).expand(N, H, W, 12)

        best_score_x = torch.zeros((N, H, W), dtype=torch.float32, device=device)
        best_tx = torch.ones((N, H, W), dtype=torch.int64, device=device)
        best_coset_x = one_hot.clone()

        for tx in range(2, min(max_period + 1, W)):
            shifts_x = x_int.unsqueeze(-1) + (mult_vec * tx)
            in_bounds_x = (shifts_x >= 1) & (shifts_x < (active_w - 1).unsqueeze(-1)) & is_interior.unsqueeze(-1)
            xs_c = shifts_x.clamp(0, W - 1)

            s1_tok = nb1_tokens[b_4d, y_4d, xs_c]
            s1_val = nb1_valid[b_4d, y_4d, xs_c]
            j1_val = (nb1_valid.unsqueeze(-2) & s1_val & in_bounds_x.unsqueeze(-1)).float()
            a1_x = ((nb1_tokens.unsqueeze(-2) == s1_tok).float() * j1_val).sum(dim=-1) / j1_val.sum(dim=-1).clamp_min(1.0)

            s2_tok = nb2_tokens[b_4d, y_4d, xs_c]
            s2_val = nb2_valid[b_4d, y_4d, xs_c]
            j2_val = (nb2_valid.unsqueeze(-2) & s2_val & in_bounds_x.unsqueeze(-1)).float()
            a2_x = ((nb2_tokens.unsqueeze(-2) == s2_tok).float() * j2_val).sum(dim=-1) / j2_val.sum(dim=-1).clamp_min(1.0)

            a_phase_x = (0.65 * a1_x + 0.35 * torch.maximum(a1_x, a2_x)) * in_bounds_x.float()
            w_gibbs_x = a_phase_x.pow(10) * torch.exp(-12.0 * (1.0 - a_phase_x)) * in_bounds_x.float()

            w_sum_x = w_gibbs_x.sum(dim=-1)
            mean_a_x = (a_phase_x * w_gibbs_x).sum(dim=-1) / w_sum_x.clamp_min(1e-5)
            score_x = (mean_a_x * torch.tanh(w_sum_x * 2.5)) - (0.004 * float(tx))

            gathered_oh_x = one_hot[b_4d, y_4d, xs_c] * w_gibbs_x.unsqueeze(-1)
            vote_x = (one_hot * 0.12) + gathered_oh_x.sum(dim=-2)
            vote_x_norm = vote_x / vote_x.sum(dim=-1, keepdim=True).clamp_min(1e-5)

            better_x = score_x > (best_score_x + 1e-4)
            best_score_x = torch.where(better_x, score_x, best_score_x)
            best_tx = torch.where(better_x, torch.full_like(best_tx, tx), best_tx)
            best_coset_x = torch.where(better_x.unsqueeze(-1), vote_x_norm, best_coset_x)

        best_score_y = torch.zeros((N, H, W), dtype=torch.float32, device=device)
        best_ty = torch.ones((N, H, W), dtype=torch.int64, device=device)
        best_coset_y = one_hot.clone()

        for ty in range(2, min(max_period + 1, H)):
            shifts_y = y_int.unsqueeze(-1) + (mult_vec * ty)
            in_bounds_y = (shifts_y >= 1) & (shifts_y < (active_h - 1).unsqueeze(-1)) & is_interior.unsqueeze(-1)
            ys_c = shifts_y.clamp(0, H - 1)

            s1_tok_y = nb1_tokens[b_4d, ys_c, x_4d]
            s1_val_y = nb1_valid[b_4d, ys_c, x_4d]
            j1_val_y = (nb1_valid.unsqueeze(-2) & s1_val_y & in_bounds_y.unsqueeze(-1)).float()
            a1_y = ((nb1_tokens.unsqueeze(-2) == s1_tok_y).float() * j1_val_y).sum(dim=-1) / j1_val_y.sum(dim=-1).clamp_min(1.0)

            s2_tok_y = nb2_tokens[b_4d, ys_c, x_4d]
            s2_val_y = nb2_valid[b_4d, ys_c, x_4d]
            j2_val_y = (nb2_valid.unsqueeze(-2) & s2_val_y & in_bounds_y.unsqueeze(-1)).float()
            a2_y = ((nb2_tokens.unsqueeze(-2) == s2_tok_y).float() * j2_val_y).sum(dim=-1) / j2_val_y.sum(dim=-1).clamp_min(1.0)

            a_phase_y = (0.65 * a1_y + 0.35 * torch.maximum(a1_y, a2_y)) * in_bounds_y.float()
            w_gibbs_y = a_phase_y.pow(10) * torch.exp(-12.0 * (1.0 - a_phase_y)) * in_bounds_y.float()

            w_sum_y = w_gibbs_y.sum(dim=-1)
            mean_a_y = (a_phase_y * w_gibbs_y).sum(dim=-1) / w_sum_y.clamp_min(1e-5)
            score_y = (mean_a_y * torch.tanh(w_sum_y * 2.5)) - (0.004 * float(ty))

            gathered_oh_y = one_hot[b_4d, ys_c, x_4d] * w_gibbs_y.unsqueeze(-1)
            vote_y = (one_hot * 0.12) + gathered_oh_y.sum(dim=-2)
            vote_y_norm = vote_y / vote_y.sum(dim=-1, keepdim=True).clamp_min(1e-5)

            better_y = score_y > (best_score_y + 1e-4)
            best_score_y = torch.where(better_y, score_y, best_score_y)
            best_ty = torch.where(better_y, torch.full_like(best_ty, ty), best_ty)
            best_coset_y = torch.where(better_y.unsqueeze(-1), vote_y_norm, best_coset_y)

        choose_y = best_score_y >= best_score_x
        conf_max = torch.maximum(best_score_y, best_score_x) * is_interior.float()
        chosen_coset = torch.where(choose_y.unsqueeze(-1), best_coset_y, best_coset_x)

        heal_gate = (conf_max.pow(4)).clamp(0.0, 1.0).unsqueeze(-1)
        healed_probs = ((1.0 - heal_gate) * one_hot) + (heal_gate * chosen_coset)
        healed_probs = healed_probs / healed_probs.sum(dim=-1, keepdim=True).clamp_min(1e-5)

        return BravaisPeriodicityState(
            period_y=best_ty.view(*orig_batch, H, W),
            period_x=best_tx.view(*orig_batch, H, W),
            period_confidence=conf_max.view(*orig_batch, H, W),
            healed_logits_10=healed_probs.view(*orig_batch, H, W, 10),
        )

    @classmethod
    def compute_eikonal_level_set_propagation(
        cls,
        grid: torch.Tensor,
        mask: torch.Tensor,
    ) -> EikonalLevelSetState:
        """
        Pre-Eroded 3x3 Moore Chebyshev Eikonal Distance d_inf(y, x) & 48-Step 4-Connected
        Von Neumann Geodesic Propagator around arbitrary non-convex L/U-shaped rooms (Task 13e47133).
        """
        orig_batch = grid.shape[:-2]
        H, W = grid.shape[-2], grid.shape[-1]
        device = grid.device

        g_flat = grid.reshape(-1, H, W)
        m_flat = mask.reshape(-1, H, W)
        N = g_flat.shape[0]

        (
            ell_n,
            ell_s,
            ell_w,
            ell_e,
            _,
            is_wall,
            is_seed,
        ) = cls._recover_cardinal_rays_and_eikonal(g_flat, m_flat)

        alive_interior = ((is_wall < 0.5) & m_flat).float().unsqueeze(1)
        curr_shell = alive_interior
        d_eikonal_acc = torch.zeros_like(curr_shell)

        for _ in range(15):
            c_pad = F.pad(curr_shell, (1, 1, 1, 1), mode="constant", value=0.0)
            curr_shell = -F.max_pool2d(-c_pad, kernel_size=3, stride=1, padding=0) * alive_interior
            d_eikonal_acc = d_eikonal_acc + curr_shell
            if not bool((curr_shell > 0.5).any().item()):
                break

        d_eikonal = d_eikonal_acc.squeeze(1)

        one_hot = F.one_hot(g_flat.clamp(0, 9), num_classes=10).to(torch.float32) * m_flat.float().unsqueeze(-1)
        seed_mask = (is_seed > 0.5) & m_flat
        d_int = torch.round(d_eikonal).to(torch.int64).clamp(0, 15)

        lvl_indices = torch.arange(4, device=device, dtype=torch.int64).view(1, 1, 1, 4)
        is_lvl_seed = (seed_mask.unsqueeze(-1) & (d_int.unsqueeze(-1) == lvl_indices)).to(torch.float32)

        trans_mask = ((is_wall < 0.5) & m_flat).float().view(N, H, W, 1)
        field_40 = (is_lvl_seed.unsqueeze(-1) * one_hot.unsqueeze(-2)).reshape(N, H, W, 40) * trans_mask

        y_grid = torch.arange(H, device=device, dtype=torch.int64).view(1, H, 1).expand(N, H, W)
        x_grid = torch.arange(W, device=device, dtype=torch.int64).view(1, 1, W).expand(N, H, W)
        b_idx = torch.arange(N, device=device, dtype=torch.int64).view(N, 1, 1).expand(N, H, W)

        y0_cam = (y_grid - ell_n.to(torch.int64) + 1).clamp(0, H - 1)
        y1_cam = (y_grid + ell_s.to(torch.int64)).clamp(1, H)
        x0_cam = (x_grid - ell_w.to(torch.int64) + 1).clamp(0, W - 1)
        x1_cam = (x_grid + ell_e.to(torch.int64)).clamp(1, W)

        cx_pad = F.pad(field_40, (0, 0, 1, 0), mode="constant", value=0.0).cumsum(dim=2)
        field_40 = ((cx_pad[b_idx, y_grid, x1_cam, :] - cx_pad[b_idx, y_grid, x0_cam, :]) > 0.05).to(torch.float32) * trans_mask

        cy_pad = F.pad(field_40, (0, 0, 0, 0, 1, 0), mode="constant", value=0.0).cumsum(dim=1)
        field_40 = ((cy_pad[b_idx, y1_cam, x_grid, :] - cy_pad[b_idx, y0_cam, x_grid, :]) > 0.05).to(torch.float32) * trans_mask

        f_perm = field_40.permute(0, 3, 1, 2)
        t_perm = trans_mask.permute(0, 3, 1, 2)
        for _ in range(48):
            f_pad = F.pad(f_perm, (1, 1, 1, 1), mode="constant", value=0.0)
            f_card = torch.maximum(
                f_perm,
                torch.maximum(
                    torch.maximum(f_pad[:, :, :-2, 1:-1], f_pad[:, :, 2:, 1:-1]),
                    torch.maximum(f_pad[:, :, 1:-1, :-2], f_pad[:, :, 1:-1, 2:]),
                ),
            )
            f_perm = f_card * t_perm
        field_40 = f_perm.permute(0, 2, 3, 1)

        depth_color_table = field_40.view(N, H, W, 4, 10)
        depth_color_norm = depth_color_table / depth_color_table.sum(dim=-1, keepdim=True).clamp_min(1e-5)

        depth_has_seed = (depth_color_table.sum(dim=-1) > 0.1).float()
        num_seeds_in_cam = depth_has_seed.sum(dim=-1).clamp_min(0.0)

        k_safe = torch.clamp_min(num_seeds_in_cam.to(torch.int64), 1)
        mod_idx = torch.remainder(d_int, k_safe).clamp(0, 3)

        propagated_col = depth_color_norm[b_idx, y_grid, x_grid, mod_idx, :]

        has_d0 = depth_has_seed[:, :, :, 0] > 0.5
        has_d1 = depth_has_seed[:, :, :, 1] > 0.5
        single_inner_seed = (num_seeds_in_cam == 1.0) & (~has_d0) & has_d1
        inner_col = depth_color_norm[:, :, :, 1, :]
        propagated_col = torch.where(
            (single_inner_seed & (d_int >= 1)).unsqueeze(-1),
            inner_col,
            torch.where(single_inner_seed.unsqueeze(-1), one_hot, propagated_col),
        )

        has_any_seed = (num_seeds_in_cam >= 1.0) & (is_wall < 0.5) & (propagated_col.sum(dim=-1) > 0.1)
        level_set_probs = torch.where(has_any_seed.unsqueeze(-1), propagated_col, one_hot)
        level_set_probs = level_set_probs / level_set_probs.sum(dim=-1, keepdim=True).clamp_min(1e-5)

        return EikonalLevelSetState(
            wall_distance=d_eikonal.view(*orig_batch, H, W),
            level_set_probs_10=level_set_probs.view(*orig_batch, H, W, 10),
            chamber_seed_count=num_seeds_in_cam.view(*orig_batch, H, W),
        )


class SpectralGraphTopology(nn.Module):
    """
    Comprehensive Differentiable Graph Laplacian, Eikonal Harmonic & Finsler-Chordal Vision Engine.
    """

    def __init__(
        self,
        num_eigenmodes: int = 8,
        chebyshev_order: int = 4,
        d_out: int = 128,
    ):
        super().__init__()
        self.num_eigenmodes = num_eigenmodes
        self.chebyshev_order = chebyshev_order
        self.d_out = d_out

        q_init, _ = torch.linalg.qr(torch.randn(num_eigenmodes, num_eigenmodes))
        self.register_buffer("probe_basis_q", q_init, persistent=True)

        raw_topo_channels = num_eigenmodes + 3

        self.swiglu_proj = SwiGLU(
            d_in=raw_topo_channels,
            d_out=d_out,
            alignment_multiple=32,
        )
        self.chordal_modulator = nn.Linear(6, d_out, bias=False)
        self.eikonal_harmonic_proj = nn.Linear(5, d_out, bias=False)
        self.norm = RMSNorm(d_out)

        self.latest_chordal_bundle: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None

        nn.init.xavier_uniform_(self.chordal_modulator.weight, gain=0.25)
        nn.init.zeros_(self.eikonal_harmonic_proj.weight)

    def _generate_orthonormal_dct_probes(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        freq_pairs = (
            (0, 1), (1, 0), (1, 1), (0, 2),
            (2, 0), (1, 2), (2, 1), (2, 2),
        )

        y_idx = torch.arange(height, dtype=dtype, device=device).view(height, 1)
        x_idx = torch.arange(width, dtype=dtype, device=device).view(1, width)

        h_flt = float(max(1, height))
        w_flt = float(max(1, width))

        modes: List[torch.Tensor] = []
        for u, v in freq_pairs[: self.num_eigenmodes]:
            arg_y = (math.pi * float(u) * (2.0 * y_idx + 1.0)) / (2.0 * h_flt)
            arg_x = (math.pi * float(v) * (2.0 * x_idx + 1.0)) / (2.0 * w_flt)
            mode_2d = torch.cos(arg_y) * torch.cos(arg_x)
            modes.append(mode_2d)

        stacked_dct = torch.stack(modes, dim=-1)
        rotated_basis = torch.matmul(stacked_dct, cast(torch.Tensor, self.probe_basis_q))
        return rotated_basis.unsqueeze(0).expand(batch_size, height, width, self.num_eigenmodes)

    def forward(self, grid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, H, W = grid.shape
        mask_f32 = mask.float().unsqueeze(-1)

        edge_weights, inv_sqrt_deg = StencilGraphLaplacianOps.compute_edge_weights_and_degree(
            grid, mask
        )

        v0 = self._generate_orthonormal_dct_probes(
            batch_size=B,
            height=H,
            width=W,
            device=grid.device,
            dtype=torch.float32,
        ) * mask_f32

        t_prev = v0
        t_curr = StencilGraphLaplacianOps.apply_normalized_laplacian(
            v0, edge_weights, inv_sqrt_deg, mask
        )

        chebyshev_accumulator = t_prev + t_curr

        for _ in range(2, self.chebyshev_order):
            l_t_curr = StencilGraphLaplacianOps.apply_normalized_laplacian(
                t_curr, edge_weights, inv_sqrt_deg, mask
            )
            t_next = (2.0 * l_t_curr - t_prev) * mask_f32
            chebyshev_accumulator = chebyshev_accumulator + t_next
            t_prev = t_curr
            t_curr = t_next

        norm_sq = torch.sum(chebyshev_accumulator * chebyshev_accumulator, dim=-1, keepdim=True)
        is_pos_norm = norm_sq > 0.0
        norm_factor = torch.where(
            is_pos_norm,
            torch.rsqrt(torch.clamp_min(norm_sq, DEFAULT_LAPLACIAN_EPS)),
            torch.zeros_like(norm_sq),
        )
        betti_0_modes = chebyshev_accumulator * norm_factor * mask_f32

        chordal_bundle = PoissonDirichletCavitySolver.compute_finsler_chordal_bundle(
            grid, mask, num_iterations=12
        )
        self.latest_chordal_bundle = chordal_bundle
        (
            m_cavity_bg,
            h_span,
            w_span,
            r_harmonic,
            dipole_yx,
            seed_field,
        ) = chordal_bundle

        _, _, _, _, d_eik, _, _ = PoissonDirichletCavitySolver._recover_cardinal_rays_and_eikonal(grid, mask)
        d_eik_4d = d_eik.unsqueeze(-1)
        eikonal_harmonics = torch.cat(
            [
                d_eik_4d / 15.0,
                torch.cos(math.pi * d_eik_4d),
                torch.sin(math.pi * d_eik_4d),
                torch.cos((2.0 * math.pi / 3.0) * d_eik_4d),
                torch.sin((2.0 * math.pi / 3.0) * d_eik_4d),
            ],
            dim=-1,
        ) * mask_f32

        min_span = torch.minimum(h_span, w_span)
        cavity_potential = (
            0.50 * m_cavity_bg
            + 0.25 * torch.tanh((min_span - 1.0) / 6.0) * m_cavity_bg
            + 0.25 * torch.tanh(r_harmonic / 4.0) * m_cavity_bg
        ) * mask_f32

        degree_field = torch.sum(edge_weights, dim=-1, keepdim=True)
        boundary_contrast = (degree_field < 4.0).float() * mask_f32
        euler_density = (1.0 - 0.25 * degree_field) * mask_f32

        raw_bundle = torch.cat(
            [betti_0_modes, cavity_potential, boundary_contrast, euler_density],
            dim=-1,
        )

        chordal_features = torch.cat(
            [
                m_cavity_bg,
                torch.log(h_span.clamp_min(1.0)) / 3.40,
                torch.log(w_span.clamp_min(1.0)) / 3.40,
                dipole_yx,
                seed_field,
            ],
            dim=-1,
        ) * mask_f32

        projected = self.swiglu_proj(raw_bundle)
        if projected.shape[-1] == self.chordal_modulator.out_features:
            projected = projected + self.chordal_modulator(chordal_features.to(projected.dtype))
        if projected.shape[-1] == self.eikonal_harmonic_proj.out_features:
            projected = projected + self.eikonal_harmonic_proj(eikonal_harmonics.to(projected.dtype))

        topo_embeddings = self.norm(projected) * mask_f32.to(projected.dtype)

        return topo_embeddings


def verify_spectral_laplacian_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for spectral_laplacian.py.
    """
    PoissonDirichletCavitySolver.clear_cache()
    torch.manual_seed(42)
    b, h, w, d = 2, 7, 7, 32

    topo = SpectralGraphTopology(num_eigenmodes=8, chebyshev_order=4, d_out=d)

    grid = torch.zeros((b, h, w), dtype=torch.int64)
    grid[0, :, 3] = 4
    grid[0, 1:4, 0:2] = 8
    grid[0, 0, 5] = 5
    mask = torch.ones((b, h, w), dtype=torch.bool)

    out = topo(grid, mask)
    if out.shape != (b, h, w, d) or torch.isnan(out).any():
        return False

    roles = PoissonDirichletCavitySolver.extract_color_structural_roles(grid[:1], mask[:1])
    if int(roles.divider_color[0].item()) != 4:
        return False
    if int(roles.occluder_color[0].item()) != 8:
        return False
    if int(roles.seed_color[0].item()) != 5:
        return False

    cam_grid = torch.full((1, 15, 6), 3, dtype=torch.int64)
    for r in range(1, 14):
        cam_grid[0, r, 1:5] = 8
        if (r - 1) % 3 != 0:
            cam_grid[0, r, 2:4] = 2
    cam_grid[0, 12, 2] = 8
    cam_mask = torch.ones_like(cam_grid, dtype=torch.bool)
    bravais = PoissonDirichletCavitySolver.compute_bravais_lattice_periodicity(cam_grid, cam_mask, max_period=8)
    if int(torch.argmax(bravais.healed_logits_10[0, 12, 2]).item()) != 2:
        return False

    train1_135a = torch.tensor(
        [[
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
            [3, 2, 1, 3, 1, 3, 1, 3, 3, 3, 1, 2, 3],
            [3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        ]],
        dtype=torch.int64,
    )
    m_135a = torch.ones_like(train1_135a, dtype=torch.bool)
    brav_t1 = PoissonDirichletCavitySolver.compute_bravais_lattice_periodicity(train1_135a, m_135a, max_period=8)
    pred_t1 = torch.argmax(brav_t1.healed_logits_10[0], dim=-1)
    if int(pred_t1[2, 8].item()) != 1 or int(pred_t1[2, 1].item()) != 2 or int(pred_t1[2, 11].item()) != 2:
        return False

    room_grid = torch.ones((1, 6, 9), dtype=torch.int64)
    room_grid[0, :, 4] = 2
    room_grid[0, 0, 0] = 7
    room_grid[0, 1, 1] = 8
    room_mask = torch.ones_like(room_grid, dtype=torch.bool)
    eik_state = PoissonDirichletCavitySolver.compute_eikonal_level_set_propagation(room_grid, room_mask)
    if float(eik_state.wall_distance[0, 2, 3].item()) != 0.0:
        return False
    pred_ring0 = int(torch.argmax(eik_state.level_set_probs_10[0, 2, 3]).item())
    pred_ring1 = int(torch.argmax(eik_state.level_set_probs_10[0, 2, 2]).item())
    if pred_ring0 != 7 or pred_ring1 != 8:
        return False

    plus_grid = torch.full((1, 5, 5), 2, dtype=torch.int64)
    plus_grid[0, 2, 1:4] = 4
    plus_grid[0, 1:4, 2] = 4
    plus_mask = torch.ones_like(plus_grid, dtype=torch.bool)
    plus_ind = PoissonDirichletCavitySolver.compute_morphological_plus_cross_mask(plus_grid, plus_mask)
    if int(plus_ind.sum().item()) != 5 or float(plus_ind[0, 2, 2].item()) != 1.0:
        return False

    loss = torch.sum(out ** 2)
    loss.backward()
    if topo.swiglu_proj.proj_out.weight.grad is None:
        return False

    PoissonDirichletCavitySolver.clear_cache()
    return True