"""
Location: scripts/figures/fig00_kaggle_card_thumbnail.py
Description: Generates SOTA 2:1 (2800x1400 and 560x280 PNG, plus native SVG and PDF vector)
             architectural cover thumbnails for the Kaggle ARC Prize 2026 Paper Track Writeup:
             - Light Theme (fig00_kaggle_card_thumbnail.*): Warm editorial ivory Distill/Nature
               isometric 5-stage architecture pipeline with flat in-plane lattice sockets and
               plane-normal 3D active tiles, split orthogonal fiber bundle (H_color (+) H_geom),
               Stiefel cellular sheaf lattice, spectral harmonic field, open-channel inter-stage
               streamlines, and large editorial typography.
             - Dark Theme (fig00_cover_variant_dark_manifold.*): Modern Kaggle carbon-charcoal
               dark UI palette with signature warm yellow-gold (#FACC15 / #F59E0B) and cyan
               accents.
Author: Vyacheslav Zimovsky (v.zimovsky.ai@gmail.com)
License: MIT-0
"""

from pathlib import Path
from typing import Optional
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch, Polygon
from matplotlib.path import Path as MplPath
import numpy as np
from PIL import Image


def _pt_tuple(p: np.ndarray) -> tuple[float, float]:
    """Helper to ensure point coordinates match Matplotlib tuple[float, float] typing."""
    return (float(p[0]), float(p[1]))


Y_CENTER = -0.16
EU = (0.88, -0.36)
EV = (0.0, 1.36)
SLOPE_Y = 0.016


def iso_pt(x_stage: float, u: float, v: float) -> np.ndarray:
    """Projects 3D stage coordinates (x_stage, u, v) into 2D isometric canvas coordinates."""
    cx = x_stage + u * EU[0] + v * EV[0]
    cy = Y_CENTER + x_stage * SLOPE_Y + u * EU[1] + v * EV[1]
    return np.array([cx, cy], dtype=float)


def draw_iso_quad(
    ax: plt.Axes,
    x_stage: float,
    u0: float,
    v0: float,
    u1: float,
    v1: float,
    facecolor: str,
    edgecolor: str = "none",
    lw: float = 1.0,
    alpha: float = 1.0,
    zorder: float = 2.0,
    linestyle: str = "-",
) -> None:
    """Draws a planar parallelogram strictly in the isometric YZ plane at x_stage."""
    p00 = iso_pt(x_stage, u0, v0)
    p10 = iso_pt(x_stage, u1, v0)
    p11 = iso_pt(x_stage, u1, v1)
    p01 = iso_pt(x_stage, u0, v1)
    poly = Polygon(
        [p00, p10, p11, p01],
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
        joinstyle="round",
    )
    ax.add_patch(poly)


def draw_iso_keycap_tile(
    ax: plt.Axes,
    x_stage: float,
    u0: float,
    v0: float,
    u1: float,
    v1: float,
    extrude_scale: float,
    color_front: str,
    color_top: str,
    color_right: str,
    edgecolor: str,
    lw: float = 0.95,
    alpha: float = 1.0,
    zorder: float = 3.0,
) -> None:
    """
    Draws a plane-aligned 3D isometric tile whose base B = iso_pt(x_stage, u, v) rests flat on
    the glass plane and whose front face F = B + d_out extrudes along the outward plane normal
    d_out = (-dx, -dy) (down-and-left, facing the viewer in the exact same direction as the
    glass plane). Because B = F - d_out lies up-and-right of F, the visible thickness bevels
    are on the Top edge [f01, f11, b11, b01] and Right edge [f10, b10, b11, f11].
    """
    base00 = iso_pt(x_stage, u0, v0)
    base10 = iso_pt(x_stage, u1, v0)
    base11 = iso_pt(x_stage, u1, v1)
    base01 = iso_pt(x_stage, u0, v1)

    d_out = np.array([-0.022 * extrude_scale, -0.023 * extrude_scale], dtype=float)
    b00 = base00 - 0.35 * d_out
    b10 = base10 - 0.35 * d_out
    b11 = base11 - 0.35 * d_out
    b01 = base01 - 0.35 * d_out

    f00 = base00 + 0.65 * d_out
    f10 = base10 + 0.65 * d_out
    f11 = base11 + 0.65 * d_out
    f01 = base01 + 0.65 * d_out

    top_bevel = Polygon(
        [f01, f11, b11, b01],
        closed=True,
        facecolor=color_top,
        edgecolor=edgecolor,
        linewidth=lw * 0.85,
        alpha=alpha,
        zorder=zorder + 0.01,
        joinstyle="round",
    )
    ax.add_patch(top_bevel)

    right_bevel = Polygon(
        [f10, b10, b11, f11],
        closed=True,
        facecolor=color_right,
        edgecolor=edgecolor,
        linewidth=lw * 0.85,
        alpha=alpha,
        zorder=zorder + 0.02,
        joinstyle="round",
    )
    ax.add_patch(right_bevel)

    front_face = Polygon(
        [f00, f10, f11, f01],
        closed=True,
        facecolor=color_front,
        edgecolor=edgecolor,
        linewidth=lw,
        alpha=alpha,
        zorder=zorder + 0.04,
        joinstyle="round",
    )
    ax.add_patch(front_face)


def draw_interstage_bus_streamlines(
    ax: plt.Axes,
    xa: float,
    xb: float,
    frame_half: float,
    stream_specs: list[tuple[float, str, float]],
    port_edge_col: str,
    dark_mode: bool,
    zorder: float,
) -> None:
    """
    Routes smooth S-curve streamlines strictly in the open air between the right edge
    of stage xa (u = +frame_half) and the left edge of stage xb (u = -frame_half).
    """
    for v_port, col, lw in stream_specs:
        p_start = iso_pt(xa, frame_half, v_port)
        p_end = iso_pt(xb, -frame_half, v_port)
        dx = p_end[0] - p_start[0]

        c1 = np.array([p_start[0] + dx * 0.44, p_start[1] - 0.10 * dx], dtype=float)
        c2 = np.array([p_end[0] - dx * 0.44, p_end[1] + 0.10 * dx], dtype=float)

        verts = [p_start, c1, c2, p_end]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
        path = MplPath(verts, codes)

        glow_patch = PathPatch(
            path,
            facecolor="none",
            edgecolor=col,
            linewidth=lw * 2.8,
            alpha=0.16 if not dark_mode else 0.24,
            zorder=zorder - 0.05,
            capstyle="round",
        )
        ax.add_patch(glow_patch)

        core_patch = PathPatch(
            path,
            facecolor="none",
            edgecolor=col,
            linewidth=lw,
            alpha=0.88 if not dark_mode else 0.94,
            zorder=zorder,
            capstyle="round",
        )
        ax.add_patch(core_patch)

        for pt in (p_start, p_end):
            ax.add_patch(
                Circle(
                    _pt_tuple(pt),
                    radius=0.040 if abs(v_port) > 0.05 else 0.050,
                    facecolor=col,
                    edgecolor=port_edge_col,
                    linewidth=0.85,
                    zorder=zorder + 0.2,
                )
            )

        p_mid = 0.125 * p_start + 0.375 * c1 + 0.375 * c2 + 0.125 * p_end
        ax.add_patch(
            Circle(
                _pt_tuple(p_mid),
                radius=0.030 if abs(v_port) > 0.05 else 0.038,
                facecolor=col,
                edgecolor="none",
                alpha=0.95,
                zorder=zorder + 0.1,
            )
        )


def cell_uv(r: int, c: int, n_cells: int = 5, span: float = 0.88) -> tuple[float, float]:
    """Returns isometric plane coordinates (u, v) for the center of cell (r, c)."""
    step = (2.0 * span) / n_cells
    u = -span + (c + 0.5) * step
    v = span - (r + 0.5) * step
    return u, v


def render_cover_theme(
    out_stem: Path,
    dark_mode: bool = False,
    mode: Optional[str] = None,
) -> None:
    """
    Renders the isometric 5-stage architectural diagram in PNG, SVG, and PDF.
    Modes supported:
      - 'light_clean': Pure white background (#FFFFFF), compact height (6.5), without top banner,
                       optimized as an embedded academic figure for the technical monograph PDF.
      - 'dark': Carbon-charcoal dark UI palette with signature warm yellow-gold and cyan accents,
                including full editorial header banners (height 8.0).
      - 'light' (or 'editorial'): Classic warm editorial ivory Distill/Nature theme with full
                editorial header banners (height 8.0).
    """
    if mode is None:
        mode = "dark" if dark_mode else "light_clean"
    is_dark = mode == "dark"
    has_banner = mode in ("dark", "light", "editorial")
    is_editorial_light = mode in ("light", "editorial")

    fig_h = 8.0 if has_banner else 6.5
    fig = plt.figure(figsize=(16.0, fig_h), dpi=175)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(-6.45, 6.45)
    if has_banner:
        ax.set_ylim(-3.225, 3.225)
    else:
        ax.set_ylim(-2.98, 2.02)
    ax.set_aspect("equal")
    ax.axis("off")

    if is_dark:
        bg_inner = np.array([0.106, 0.114, 0.133])
        bg_outer = np.array([0.055, 0.059, 0.071])
        grid_col = "#232730"
        port_edge_col = "#0F172A"

        kicker_col = "#FACC15"
        title_col = "#F8FAFC"
        subtitle_col = "#CBD5E1"
        stage_num_col = "#FACC15"
        stage_title_col = "#F8FAFC"
        stage_sub_col = "#94A3B8"
        oplus_bg = "#272214"
        oplus_edge = "#FACC15"
        oplus_fg = "#FDE047"

        empty_front = "#212631"
        empty_edge = "#384152"

        glass_1_fill = "#182030"
        glass_1_edge = "#60A5FA"
        glass_2_outer = "#1A1E26"
        glass_2_edge = "#94A3B8"
        glass_2a_fill = "#2D2312"
        glass_2a_edge = "#FACC15"
        glass_2b_fill = "#142636"
        glass_2b_edge = "#38BDF8"
        glass_3_fill = "#142634"
        glass_3_edge = "#22D3EE"
        glass_4_fill = "#122826"
        glass_4_edge = "#34D399"
        glass_5_fill = "#2B2213"
        glass_5_edge = "#FACC15"

        palette_tiles = {
            1: ("#3B82F6", "#60A5FA", "#1D4ED8", "#1E3A8A"),
            2: ("#F43F5E", "#FB7185", "#BE123C", "#881337"),
            3: ("#10B981", "#34D399", "#047857", "#064E3B"),
            4: ("#FACC15", "#FEF08A", "#CA8A04", "#713F12"),
            8: ("#22D3EE", "#67E8F9", "#0891B2", "#164E63"),
        }
    elif is_editorial_light:
        bg_inner = np.array([0.988, 0.980, 0.962])
        bg_outer = np.array([0.938, 0.924, 0.892])
        grid_col = "#E5DFD2"
        port_edge_col = "#FFFFFF"

        kicker_col = "#B45309"
        title_col = "#0F172A"
        subtitle_col = "#334155"
        stage_num_col = "#B45309"
        stage_title_col = "#0F172A"
        stage_sub_col = "#475569"
        oplus_bg = "#FEF3C7"
        oplus_edge = "#D97706"
        oplus_fg = "#92400E"

        empty_front = "#E2E8F0"
        empty_edge = "#94A3B8"

        glass_1_fill = "#F0F7FF"
        glass_1_edge = "#3B82F6"
        glass_2_outer = "#F8FAFC"
        glass_2_edge = "#64748B"
        glass_2a_fill = "#FEF3C7"
        glass_2a_edge = "#D97706"
        glass_2b_fill = "#E0F2FE"
        glass_2b_edge = "#0284C7"
        glass_3_fill = "#ECFEFF"
        glass_3_edge = "#0891B2"
        glass_4_fill = "#ECFDF5"
        glass_4_edge = "#0D9488"
        glass_5_fill = "#FFFBEB"
        glass_5_edge = "#D97706"

        palette_tiles = {
            1: ("#2563EB", "#60A5FA", "#1D4ED8", "#1E3A8A"),
            2: ("#E11D48", "#FB7185", "#BE123C", "#881337"),
            3: ("#10B981", "#34D399", "#059669", "#064E3B"),
            4: ("#F59E0B", "#FCD34D", "#D97706", "#78350F"),
            8: ("#06B6D4", "#67E8F9", "#0891B2", "#164E63"),
        }
    else:
        bg_inner = np.array([1.0, 1.0, 1.0])
        bg_outer = np.array([1.0, 1.0, 1.0])
        grid_col = "#E2E8F0"
        port_edge_col = "#FFFFFF"

        kicker_col = "#B45309"
        title_col = "#0F172A"
        subtitle_col = "#334155"
        stage_num_col = "#B45309"
        stage_title_col = "#0F172A"
        stage_sub_col = "#475569"
        oplus_bg = "#FEF3C7"
        oplus_edge = "#D97706"
        oplus_fg = "#92400E"

        empty_front = "#E2E8F0"
        empty_edge = "#94A3B8"

        glass_1_fill = "#F0F7FF"
        glass_1_edge = "#3B82F6"
        glass_2_outer = "#F8FAFC"
        glass_2_edge = "#64748B"
        glass_2a_fill = "#FEF3C7"
        glass_2a_edge = "#D97706"
        glass_2b_fill = "#E0F2FE"
        glass_2b_edge = "#0284C7"
        glass_3_fill = "#ECFEFF"
        glass_3_edge = "#0891B2"
        glass_4_fill = "#ECFDF5"
        glass_4_edge = "#0D9488"
        glass_5_fill = "#FFFBEB"
        glass_5_edge = "#D97706"

        palette_tiles = {
            1: ("#2563EB", "#60A5FA", "#1D4ED8", "#1E3A8A"),
            2: ("#E11D48", "#FB7185", "#BE123C", "#881337"),
            3: ("#10B981", "#34D399", "#059669", "#064E3B"),
            4: ("#F59E0B", "#FCD34D", "#D97706", "#78350F"),
            8: ("#06B6D4", "#67E8F9", "#0891B2", "#164E63"),
        }

    gx = np.linspace(-1.0, 1.0, 400)
    gy = np.linspace(-1.0, 1.0, 200)
    xx, yy = np.meshgrid(gx, gy)
    rr = np.clip(np.sqrt((xx * 0.86) ** 2 + (yy * 1.06) ** 2), 0.0, 1.0) ** 1.45
    bg_img = bg_inner[None, None, :] * (1.0 - rr[:, :, None]) + bg_outer[None, None, :] * rr[:, :, None]
    ax.imshow(bg_img, extent=(-6.45, 6.45, -3.225, 3.225), origin="lower", zorder=0, aspect="auto")

    for k in range(-14, 15):
        x0 = k * 0.72
        ax.plot(
            [x0 - 3.2, x0 + 3.2],
            [-3.3, 3.3],
            color=grid_col,
            lw=0.55,
            alpha=0.50 if not is_dark else 0.55,
            zorder=0.4,
        )
        ax.plot(
            [x0 - 4.8, x0 + 4.8],
            [3.3, -3.3],
            color=grid_col,
            lw=0.55,
            alpha=0.35 if not is_dark else 0.38,
            zorder=0.4,
        )

    if has_banner:
        ax.text(
            0.0,
            2.86,
            "ARC PRIZE 2026   ·   PAPER TRACK WRITEUP",
            ha="center",
            va="center",
            fontsize=15.5,
            fontweight="bold",
            fontfamily="sans-serif",
            color=kicker_col,
            path_effects=[pe.withStroke(linewidth=0.45, foreground=kicker_col)],
            zorder=20.1,
        )

        ax.text(
            0.0,
            2.36,
            "FGNS-TRM: FINSLER GAUGE & SHEAF NETWORK",
            ha="center",
            va="center",
            fontsize=32.0,
            fontweight="bold",
            fontfamily="sans-serif",
            color=title_col,
            path_effects=[pe.withStroke(linewidth=0.65, foreground=title_col)],
            zorder=20.1,
        )

        ax.text(
            0.0,
            1.88,
            "Continuous Differential-Geometric, Stiefel Cellular Sheaf & Spectral-Green Field Architecture",
            ha="center",
            va="center",
            fontsize=15.0,
            fontweight="bold",
            fontfamily="sans-serif",
            color=subtitle_col,
            path_effects=[pe.withStroke(linewidth=0.20, foreground=subtitle_col)],
            zorder=20.1,
        )

    x_stages = [-4.92, -2.46, 0.00, 2.46, 4.92]
    n_cells = 5
    span = 0.88
    step = (2.0 * span) / n_cells
    tile_pad = 0.050
    frame_half = 1.05

    input_grid = [
        [1, 1, 0, 2, 0],
        [1, 4, 0, 0, 2],
        [0, 0, 8, 0, 0],
        [3, 0, 0, 4, 1],
        [0, 3, 0, 1, 1],
    ]

    output_grid = [
        [1, 1, 8, 1, 1],
        [1, 4, 8, 4, 1],
        [8, 8, 4, 8, 8],
        [1, 4, 8, 4, 1],
        [1, 1, 8, 1, 1],
    ]

    stream_specs = [
        (0.70, "#3B82F6" if not dark_mode else "#60A5FA", 1.85),
        (0.35, "#F59E0B" if not dark_mode else "#FACC15", 2.25),
        (0.00, "#0891B2" if not dark_mode else "#22D3EE", 2.70),
        (-0.35, "#F59E0B" if not dark_mode else "#FACC15", 2.25),
        (-0.70, "#10B981" if not dark_mode else "#34D399", 1.85),
    ]

    stage_z = [3.0, 6.0, 9.0, 12.0, 15.0]

    for s_idx in range(4):
        draw_interstage_bus_streamlines(
            ax=ax,
            xa=x_stages[s_idx],
            xb=x_stages[s_idx + 1],
            frame_half=frame_half,
            stream_specs=stream_specs,
            port_edge_col=port_edge_col,
            dark_mode=dark_mode,
            zorder=stage_z[s_idx] + 1.5,
        )

    x0 = x_stages[0]
    z0 = stage_z[0]
    draw_iso_quad(
        ax,
        x0,
        -frame_half,
        -frame_half,
        frame_half,
        frame_half,
        facecolor=glass_1_fill,
        edgecolor=glass_1_edge,
        lw=2.0,
        alpha=0.95 if not dark_mode else 0.92,
        zorder=z0,
    )
    draw_iso_quad(
        ax,
        x0,
        -frame_half + 0.06,
        -frame_half + 0.06,
        frame_half - 0.06,
        frame_half - 0.06,
        facecolor="none",
        edgecolor=glass_1_edge,
        lw=0.80,
        alpha=0.45,
        zorder=z0 + 0.05,
    )

    u_center_shift = 0.0
    v_center_shift = 0.0
    for r in range(n_cells):
        for c in range(n_cells):
            u0 = -span + c * step + tile_pad + u_center_shift
            u1 = -span + (c + 1) * step - tile_pad + u_center_shift
            v1 = span - r * step - tile_pad + v_center_shift
            v0 = span - (r + 1) * step + tile_pad + v_center_shift
            sym = input_grid[r][c]
            z_tile = z0 + 0.20 + r * 0.06 + (n_cells - 1 - c) * 0.06
            draw_iso_quad(
                ax,
                x0,
                u0,
                v0,
                u1,
                v1,
                facecolor=empty_front,
                edgecolor=empty_edge,
                lw=0.70,
                alpha=0.88,
                zorder=z0 + 0.10,
            )
            if sym in palette_tiles:
                cf, ct, cr, ce = palette_tiles[sym]
                draw_iso_keycap_tile(
                    ax,
                    x0,
                    u0,
                    v0,
                    u1,
                    v1,
                    extrude_scale=1.0,
                    color_front=cf,
                    color_top=ct,
                    color_right=cr,
                    edgecolor=ce,
                    lw=0.95,
                    alpha=0.99,
                    zorder=z_tile,
                )

    x1 = x_stages[1]
    z1 = stage_z[1]
    draw_iso_quad(
        ax,
        x1,
        -frame_half,
        -frame_half,
        frame_half,
        frame_half,
        facecolor=glass_2_outer,
        edgecolor=glass_2_edge,
        lw=2.0,
        alpha=0.94 if not dark_mode else 0.90,
        zorder=z1,
    )
    draw_iso_quad(
        ax,
        x1,
        -0.93,
        0.10,
        0.93,
        0.93,
        facecolor=glass_2a_fill,
        edgecolor=glass_2a_edge,
        lw=1.55,
        alpha=0.95 if not dark_mode else 0.92,
        zorder=z1 + 0.10,
    )
    draw_iso_quad(
        ax,
        x1,
        -0.93,
        -0.93,
        0.93,
        -0.10,
        facecolor=glass_2b_fill,
        edgecolor=glass_2b_edge,
        lw=1.55,
        alpha=0.95 if not dark_mode else 0.92,
        zorder=z1 + 0.10,
    )

    u_cols = np.linspace(-0.72, 0.72, 5)
    v_top_rows = [0.72, 0.32]
    v_bot_rows = [-0.32, -0.72]
    color_row_palette = [
        ["#2563EB", "#F59E0B", "#06B6D4", "#E11D48", "#2563EB"],
        ["#F59E0B", "#10B981", "#F59E0B", "#10B981", "#F59E0B"],
    ]
    if dark_mode:
        color_row_palette = [
            ["#3B82F6", "#FACC15", "#22D3EE", "#FB7185", "#3B82F6"],
            ["#FACC15", "#34D399", "#FACC15", "#34D399", "#FACC15"],
        ]

    for c_idx, u_val in enumerate(u_cols):
        p_t = iso_pt(x1, u_val, v_top_rows[1])
        p_b = iso_pt(x1, u_val, v_bot_rows[0])
        ax.plot(
            [p_t[0], p_b[0]],
            [p_t[1], p_b[1]],
            color=kicker_col,
            lw=1.25,
            linestyle=":",
            alpha=0.78,
            zorder=z1 + 0.18,
        )

    for r_idx, v_val in enumerate(v_top_rows):
        for c_idx, u_val in enumerate(u_cols):
            pt = iso_pt(x1, u_val, v_val)
            col_node = color_row_palette[r_idx][c_idx]
            ax.add_patch(
                Circle(
                    _pt_tuple(pt),
                    radius=0.072,
                    facecolor=col_node,
                    edgecolor=port_edge_col,
                    linewidth=0.95,
                    zorder=z1 + 0.30,
                )
            )
            if c_idx + 1 < 5:
                pt_next = iso_pt(x1, u_cols[c_idx + 1], v_val)
                ax.plot(
                    [pt[0], pt_next[0]],
                    [pt[1], pt_next[1]],
                    color=glass_2a_edge,
                    lw=1.30,
                    alpha=0.68,
                    zorder=z1 + 0.22,
                )

    for r_idx, v_val in enumerate(v_bot_rows):
        for c_idx, u_val in enumerate(u_cols):
            pt = iso_pt(x1, u_val, v_val)
            ax.add_patch(
                Circle(
                    _pt_tuple(pt),
                    radius=0.060,
                    facecolor="#0284C7" if not dark_mode else "#38BDF8",
                    edgecolor=port_edge_col,
                    linewidth=0.90,
                    zorder=z1 + 0.30,
                )
            )
            ang = 0.45 * (c_idx - 2) + (0.35 if r_idx == 0 else -0.35)
            du = 0.15 * np.cos(ang)
            dv = 0.15 * np.sin(ang)
            p_tip = iso_pt(x1, u_val + du, v_val + dv)
            ax.plot(
                [pt[0], p_tip[0]],
                [pt[1], p_tip[1]],
                color="#1E3A8A" if not dark_mode else "#FACC15",
                lw=1.65,
                alpha=0.95,
                zorder=z1 + 0.35,
                solid_capstyle="round",
            )

    p_oplus = iso_pt(x1, 0.0, 0.0)
    ax.add_patch(
        Circle(
            _pt_tuple(p_oplus),
            radius=0.104,
            facecolor=oplus_bg,
            edgecolor=oplus_edge,
            linewidth=1.45,
            zorder=z1 + 0.45,
        )
    )
    ax.plot(
        [p_oplus[0] - 0.060, p_oplus[0] + 0.060],
        [p_oplus[1], p_oplus[1]],
        color=oplus_fg,
        lw=1.75,
        zorder=z1 + 0.50,
    )
    ax.plot(
        [p_oplus[0], p_oplus[0]],
        [p_oplus[1] - 0.060, p_oplus[1] + 0.060],
        color=oplus_fg,
        lw=1.75,
        zorder=z1 + 0.50,
    )

    x2 = x_stages[2]
    z2 = stage_z[2]
    draw_iso_quad(
        ax,
        x2,
        -frame_half,
        -frame_half,
        frame_half,
        frame_half,
        facecolor=glass_3_fill,
        edgecolor=glass_3_edge,
        lw=2.0,
        alpha=0.95 if not dark_mode else 0.92,
        zorder=z2,
    )
    draw_iso_quad(
        ax,
        x2,
        -frame_half + 0.06,
        -frame_half + 0.06,
        frame_half - 0.06,
        frame_half - 0.06,
        facecolor="none",
        edgecolor=glass_3_edge,
        lw=0.80,
        alpha=0.45,
        zorder=z2 + 0.05,
    )

    for r in range(n_cells):
        for c in range(n_cells):
            u, v = cell_uv(r, c, n_cells, span)
            p = iso_pt(x2, u, v)
            if c + 1 < n_cells:
                u_n, v_n = cell_uv(r, c + 1, n_cells, span)
                q = iso_pt(x2, u_n, v_n)
                is_barrier = (c in (1, 2)) and (r in (1, 2, 3))
                ax.plot(
                    [p[0], q[0]],
                    [p[1], q[1]],
                    color=("#E11D48" if not dark_mode else "#FB7185")
                    if is_barrier
                    else ("#0284C7" if not dark_mode else "#38BDF8"),
                    lw=2.50 if is_barrier else 1.75,
                    linestyle="--" if is_barrier else "-",
                    alpha=0.95,
                    zorder=z2 + 0.20,
                )
            if r + 1 < n_cells:
                u_n, v_n = cell_uv(r + 1, c, n_cells, span)
                q = iso_pt(x2, u_n, v_n)
                is_barrier = (r in (1, 2)) and (c in (1, 2, 3))
                ax.plot(
                    [p[0], q[0]],
                    [p[1], q[1]],
                    color=("#E11D48" if not dark_mode else "#FB7185")
                    if is_barrier
                    else ("#0284C7" if not dark_mode else "#38BDF8"),
                    lw=2.50 if is_barrier else 1.75,
                    linestyle="--" if is_barrier else "-",
                    alpha=0.95,
                    zorder=z2 + 0.20,
                )

            is_center = r == 2 and c == 2
            is_inner = max(abs(r - 2), abs(c - 2)) == 1
            node_col = (
                ("#F59E0B" if not dark_mode else "#FACC15")
                if is_center
                else (
                    ("#10B981" if not dark_mode else "#34D399")
                    if is_inner
                    else ("#2563EB" if not dark_mode else "#38BDF8")
                )
            )
            ax.add_patch(
                Circle(
                    _pt_tuple(p),
                    radius=0.088 if is_center else 0.064,
                    facecolor=node_col,
                    edgecolor=port_edge_col,
                    linewidth=1.05,
                    zorder=z2 + 0.40,
                )
            )
            e1 = iso_pt(x2, u + 0.12, v)
            e2 = iso_pt(x2, u, v + 0.12)
            e3 = p + np.array([-0.078, -0.036], dtype=float)
            ax.plot([p[0], e1[0]], [p[1], e1[1]], color="#E11D48" if not dark_mode else "#FB7185", lw=1.40, zorder=z2 + 0.50)
            ax.plot([p[0], e2[0]], [p[1], e2[1]], color="#10B981" if not dark_mode else "#34D399", lw=1.40, zorder=z2 + 0.50)
            ax.plot([p[0], e3[0]], [p[1], e3[1]], color="#2563EB" if not dark_mode else "#FACC15", lw=1.40, zorder=z2 + 0.50)

    x3 = x_stages[3]
    z3 = stage_z[3]
    draw_iso_quad(
        ax,
        x3,
        -frame_half,
        -frame_half,
        frame_half,
        frame_half,
        facecolor=glass_4_fill,
        edgecolor=glass_4_edge,
        lw=2.0,
        alpha=0.95 if not dark_mode else 0.92,
        zorder=z3,
    )
    draw_iso_quad(
        ax,
        x3,
        -frame_half + 0.06,
        -frame_half + 0.06,
        frame_half - 0.06,
        frame_half - 0.06,
        facecolor="none",
        edgecolor=glass_4_edge,
        lw=0.80,
        alpha=0.45,
        zorder=z3 + 0.05,
    )

    theta = np.linspace(0.0, 2.0 * np.pi, 180)
    ring_specs = [
        (0.22, "#F59E0B" if not dark_mode else "#FACC15", 2.5, 0.96),
        (0.42, "#10B981" if not dark_mode else "#34D399", 2.1, 0.90),
        (0.63, "#0891B2" if not dark_mode else "#22D3EE", 1.8, 0.84),
        (0.84, "#2563EB" if not dark_mode else "#60A5FA", 1.5, 0.76),
    ]
    for rad, rcol, rlw, ralpha in ring_specs:
        u_ring = rad * np.cos(theta) * (1.0 + 0.085 * np.cos(4.0 * theta))
        v_ring = rad * np.sin(theta) * (1.0 + 0.085 * np.cos(4.0 * theta))
        pts = np.array([iso_pt(x3, uu, vv) for uu, vv in zip(u_ring, v_ring)])
        ax.plot(pts[:, 0], pts[:, 1], color=rcol, lw=rlw, alpha=ralpha, zorder=z3 + 0.25)

    p_spec_center = iso_pt(x3, 0.0, 0.0)
    for k_dir in range(8):
        ang = k_dir * np.pi / 4.0
        u_ray = 0.88 * np.cos(ang)
        v_ray = 0.88 * np.sin(ang)
        p_ray = iso_pt(x3, u_ray, v_ray)
        ax.plot(
            [p_spec_center[0], p_ray[0]],
            [p_spec_center[1], p_ray[1]],
            color=("#D97706" if not dark_mode else "#FACC15")
            if k_dir % 2 == 0
            else ("#0D9488" if not dark_mode else "#22D3EE"),
            lw=1.60 if k_dir % 2 == 0 else 1.20,
            linestyle="-" if k_dir % 2 == 0 else ":",
            alpha=0.88,
            zorder=z3 + 0.30,
        )

    spec_nodes = [
        (0, 0, "#3B82F6" if not dark_mode else "#60A5FA"),
        (0, 4, "#E11D48" if not dark_mode else "#FB7185"),
        (4, 0, "#10B981" if not dark_mode else "#34D399"),
        (4, 4, "#3B82F6" if not dark_mode else "#60A5FA"),
        (1, 1, "#F59E0B" if not dark_mode else "#FACC15"),
        (1, 3, "#F59E0B" if not dark_mode else "#FACC15"),
        (3, 1, "#F59E0B" if not dark_mode else "#FACC15"),
        (3, 3, "#F59E0B" if not dark_mode else "#FACC15"),
        (2, 2, "#06B6D4" if not dark_mode else "#22D3EE"),
    ]
    for r, c, scol in spec_nodes:
        u, v = cell_uv(r, c, n_cells, span)
        p_node = iso_pt(x3, u * 0.88, v * 0.88)
        ax.add_patch(
            Circle(
                _pt_tuple(p_node),
                radius=0.094 if (r == 2 and c == 2) else 0.056,
                facecolor=scol,
                edgecolor=port_edge_col,
                linewidth=1.05,
                zorder=z3 + 0.45,
            )
        )

    x4 = x_stages[4]
    z4 = stage_z[4]
    draw_iso_quad(
        ax,
        x4,
        -frame_half,
        -frame_half,
        frame_half,
        frame_half,
        facecolor=glass_5_fill,
        edgecolor=glass_5_edge,
        lw=2.0,
        alpha=0.95 if not dark_mode else 0.92,
        zorder=z4,
    )
    draw_iso_quad(
        ax,
        x4,
        -frame_half + 0.06,
        -frame_half + 0.06,
        frame_half - 0.06,
        frame_half - 0.06,
        facecolor="none",
        edgecolor=glass_5_edge,
        lw=0.80,
        alpha=0.45,
        zorder=z4 + 0.05,
    )

    for r in range(n_cells):
        for c in range(n_cells):
            u0 = -span + c * step + tile_pad + u_center_shift
            u1 = -span + (c + 1) * step - tile_pad + u_center_shift
            v1 = span - r * step - tile_pad + v_center_shift
            v0 = span - (r + 1) * step + tile_pad + v_center_shift
            sym = output_grid[r][c]
            z_tile = z4 + 0.20 + r * 0.06 + (n_cells - 1 - c) * 0.06
            cf, ct, cr, ce = palette_tiles[sym]
            draw_iso_keycap_tile(
                ax,
                x4,
                u0,
                v0,
                u1,
                v1,
                extrude_scale=1.15 if sym == 4 else 0.95,
                color_front=cf,
                color_top=ct,
                color_right=cr,
                edgecolor=ce,
                lw=0.95,
                alpha=0.99,
                zorder=z_tile,
            )

    stage_labels = [
        ("01   DISCRETE LATTICE", "Input Grid & Conservation"),
        ("02   FIBER BUNDLE", r"Color $\oplus$ Geometry Lifting"),
        ("03   CELLULAR SHEAF", "Stiefel Parallel Transport"),
        ("04   SPECTRAL GREEN", "Harmonic RKHS & Wavefront"),
        ("05   ATTRACTOR GRID", "Crystallized Equilibrium"),
    ]

    y_top_lbl = -2.42
    for idx, (xs, (st_title, st_sub)) in enumerate(zip(x_stages, stage_labels)):
        t_col = stage_num_col if idx in (0, 2, 4) else stage_title_col
        ax.text(
            xs,
            y_top_lbl,
            st_title,
            ha="center",
            va="center",
            fontsize=14.4,
            fontweight="bold",
            fontfamily="sans-serif",
            color=t_col,
            path_effects=[pe.withStroke(linewidth=0.45, foreground=t_col)],
            zorder=20.0,
        )
        ax.text(
            xs,
            y_top_lbl - 0.36,
            st_sub,
            ha="center",
            va="center",
            fontsize=11.8,
            fontweight="bold",
            fontfamily="sans-serif",
            color=stage_sub_col,
            path_effects=[pe.withStroke(linewidth=0.18, foreground=stage_sub_col)],
            zorder=20.0,
        )
        if idx < 4:
            x_mid = 0.5 * (x_stages[idx] + x_stages[idx + 1])
            ax.text(
                x_mid,
                y_top_lbl - 0.14,
                r"$\rightarrow$",
                ha="center",
                va="center",
                fontsize=17.0,
                fontweight="bold",
                color=kicker_col,
                path_effects=[pe.withStroke(linewidth=0.45, foreground=kicker_col)],
                alpha=0.95,
                zorder=20.0,
            )

    out_png = out_stem.with_suffix(".png")
    out_svg = out_stem.with_suffix(".svg")
    out_pdf = out_stem.with_suffix(".pdf")
    out_thumb = out_stem.parent / f"{out_stem.name}_560x280.png"

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=175)
    fig.savefig(out_svg)
    fig.savefig(out_pdf)
    plt.close(fig)

    with Image.open(out_png) as im:
        im_rgb = im.convert("RGB")
        if has_banner and im_rgb.size != (2800, 1400):
            im_rgb = im_rgb.resize((2800, 1400), Image.Resampling.LANCZOS)
            im_rgb.save(out_png, quality=98)
        im_560 = im_rgb.resize((560, 280), Image.Resampling.LANCZOS)
        im_560.save(out_thumb, quality=98)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "artifacts" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mode 1: Classic warm editorial beige/ivory theme with banner (for Kaggle Writeup & Card)
    render_cover_theme(
        out_stem=out_dir / "fig00_kaggle_card_thumbnail",
        mode="editorial",
    )
    render_cover_theme(
        out_stem=out_dir / "fig00_cover_variant_editorial_beige",
        mode="editorial",
    )

    # Mode 2: Carbon-charcoal dark manifold UI theme with yellow-gold & cyan banners
    render_cover_theme(
        out_stem=out_dir / "fig00_cover_variant_dark_manifold",
        mode="dark",
    )

    # Mode 3: Clean academic figure on pure white background (#FFFFFF) without banner (for monograph PDF)
    render_cover_theme(
        out_stem=out_dir / "fig00_pipeline_stages_white",
        mode="light_clean",
    )

    print("Generated all 3 cover modes: Dark Manifold, Warm Editorial Beige, and Pure White Pipeline.")


def generate_fig00() -> None:
    """Master orchestrator hook for scripts/figures/generate_all_figures.py."""
    main()


if __name__ == "__main__":
    main()
