"""
Location: // fgns_engine/fgns/primitives/stiefel_sheaf.py
Description: Continuous Cellular Sheaf Operator on Stiefel Manifolds V_k(R^d),
             Degree-Normalized Positive-Definite Sheaf Laplacian Diffusion,
             Anisotropic Edge-Stopping Perona-Malik-Randers 12-Ray Spatial Sheaf,
             and Gauge-Equivariant NA3D-Finsler-Lie v3.0 Engine with Casimir-Chiral Observables (FGNS v4.3.0).
Language: Python 3.12+ / PyTorch 2.5+ / Cellular Sheaf & Finsler-Lie Geometry sm_120 Aligned
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import math
from typing import Final, List, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from fgns.primitives.swiglu import RMSNorm

DEFAULT_STIEFEL_NS_ITERS: Final[int] = 5
DEFAULT_D_NODE: Final[int] = 128
DEFAULT_D_EDGE: Final[int] = 32
DEFAULT_STIEFEL_EPS: Final[float] = 1e-6
NEWTON_SCHULZ_SAFE_ALPHA: Final[float] = 1.75
MIN_CONDUCTANCE_FLOOR: Final[float] = 0.05
NA3D_CELL_STRIDE: Final[int] = 16


def compute_dynamic_edge_dimension(d_node: int, requested_d_edge: int) -> int:
    if d_node < requested_d_edge:
        quarter_dim = max(4, d_node // 4)
        power_two = 1 << int(math.floor(math.log2(max(1, quarter_dim))))
        return min(requested_d_edge, max(4, min(d_node, power_two)))
    return requested_d_edge


def newton_schulz_stiefel_5th_order(
    w_raw: torch.Tensor,
    iterations: int = DEFAULT_STIEFEL_NS_ITERS,
    eps: float = DEFAULT_STIEFEL_EPS,
) -> torch.Tensor:
    """
    Projects raw matrix onto the compact Stiefel manifold V_{d_edge}(R^{d_node})
    using 5th-order Newton-Schulz iterations in strict IEEE-754 FP32 precision.
    """
    target_dtype = w_raw.dtype
    dev_type = "cuda" if w_raw.is_cuda else "cpu"
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False

    try:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False

        with torch.amp.autocast(device_type=dev_type, enabled=False):
            w_f32 = w_raw.to(torch.float32)

            d_edge = w_f32.shape[-2]
            d_node = w_f32.shape[-1]

            if d_edge > d_node:
                raise ValueError(
                    f"Stiefel manifold requires d_edge ({d_edge}) <= d_node ({d_node})."
                )

            effective_iters = max(3, int(iterations))

            frob_sq = torch.sum(w_f32 * w_f32, dim=(-2, -1), keepdim=True)
            frob_norm = torch.sqrt(frob_sq + eps)

            scale_factor = math.sqrt(NEWTON_SCHULZ_SAFE_ALPHA)
            y_k = (w_f32 / frob_norm) * scale_factor

            eye_d_edge = torch.eye(d_edge, dtype=torch.float32, device=w_raw.device)
            eye = eye_d_edge.view(*([1] * (w_raw.dim() - 2)), d_edge, d_edge)

            for _ in range(effective_iters):
                y_k_t = y_k.transpose(-2, -1)
                a_k = torch.matmul(y_k, y_k_t)
                a_k_sq = torch.matmul(a_k, a_k)

                poly = 0.125 * ((15.0 * eye) - (10.0 * a_k) + (3.0 * a_k_sq))
                y_k = torch.matmul(poly, y_k)

            row_sq = torch.sum(y_k * y_k, dim=-1, keepdim=True)
            row_norm = torch.sqrt(row_sq + eps)
            w_ortho = y_k / row_norm
    finally:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    return w_ortho.to(target_dtype)


def project_stiefel_tangent_vjp(
    w_ortho: torch.Tensor,
    grad_euclidean: torch.Tensor,
) -> torch.Tensor:
    target_dtype = grad_euclidean.dtype
    w_f32 = w_ortho.to(torch.float32)
    g_f32 = grad_euclidean.to(torch.float32)

    gw_t = torch.matmul(g_f32, w_f32.transpose(-2, -1))
    wg_t = torch.matmul(w_f32, g_f32.transpose(-2, -1))

    sym_k_by_k = 0.50 * (gw_t + wg_t)
    normal_component = torch.matmul(sym_k_by_k, w_f32)

    tangent_grad = g_f32 - normal_component
    return tangent_grad.to(target_dtype)


def stiefel_orthogonality_penalty(
    w_sheaf: torch.Tensor,
    eps: float = DEFAULT_STIEFEL_EPS,
) -> torch.Tensor:
    d_edge = w_sheaf.shape[-2]
    w_f32 = w_sheaf.to(torch.float32)

    eye = torch.eye(d_edge, dtype=torch.float32, device=w_sheaf.device)
    ww_t = torch.matmul(w_f32, w_f32.transpose(-2, -1))

    diff = ww_t - eye
    penalty = torch.mean(diff * diff) / float(max(1, d_edge))
    return penalty.to(w_sheaf.dtype)


class NA3DFinslerLieEngine(nn.Module):
    """
    Singularity-Free Isochoric Non-Linear Asymmetric 3D Token Engine v3.0 on Finsler-Lie Bundles.
    Features Gauge-Equivariant C x C SO(3) Frame Mixing, 2x2 Yang-Mills-Wilson Plaquette Curvature,
    Bravais Lattice Dislocation Resonance, and 32D Casimir-Chiral Gauge Observables.
    """

    def __init__(
        self,
        d_node: int = DEFAULT_D_NODE,
        rest_mass_init: float = 0.25,
        eps: float = DEFAULT_STIEFEL_EPS,
    ) -> None:
        super().__init__()
        self.d_node = d_node
        self.num_cells = max(1, d_node // NA3D_CELL_STRIDE)
        self.eps = eps

        self.raw_rest_mass = nn.Parameter(
            torch.full((1, 1, 1, 1, self.num_cells, 1), math.log(math.exp(rest_mass_init) - 1.0))
        )
        self.raw_tau_0 = nn.Parameter(torch.zeros((1, 1, 1, 1, self.num_cells, 1), dtype=torch.float32))

        self.cell_gate_proj = nn.Linear(4 * self.num_cells, self.num_cells * 3, bias=True)
        self.channel_mixer = nn.Linear(d_node, d_node, bias=False)

        self.frame_mix_weight = nn.Parameter(torch.eye(self.num_cells, dtype=torch.float32) * 0.25)
        self.bravais_heal_gate = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

        nn.init.xavier_uniform_(self.cell_gate_proj.weight, gain=0.25)
        nn.init.zeros_(self.cell_gate_proj.bias)
        nn.init.orthogonal_(self.channel_mixer.weight, gain=0.25)

    @staticmethod
    def apply_rodrigues_rotation(
        v: torch.Tensor,
        omega: torch.Tensor,
        eps: float = DEFAULT_STIEFEL_EPS,
    ) -> torch.Tensor:
        omega_norm = torch.sqrt(torch.sum(omega * omega, dim=-1, keepdim=True) + eps)
        u_axis = omega / omega_norm
        theta = math.pi * torch.tanh(omega_norm)

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        u_cross_v = torch.cross(u_axis, v, dim=-1)
        u_dot_v = torch.sum(u_axis * v, dim=-1, keepdim=True)

        v_rot = (v * cos_t) + (u_cross_v * sin_t) + (u_axis * u_dot_v * (1.0 - cos_t))
        return v_rot

    @classmethod
    def extract_casimir_chiral_observables(
        cls,
        h_before: torch.Tensor,
        h_after: torch.Tensor,
        mask: torch.Tensor,
        eps: float = DEFAULT_STIEFEL_EPS,
    ) -> torch.Tensor:
        """
        Extracts the 32-dimensional (4 * C) Non-Linear Asymmetric Casimir-Chiral Gauge Observables:
          I_1^(c) = tanh(det[L_after, C_after, R_after])                  (3D Chiral Volume)
          I_2^(c) = tanh(<beta, L_after x R_after>)                       (Randers Helicity)
          I_3^(c) = tanh(<C_after - C_before, L_after - R_after>)         (Diode Core-Shift Work)
          I_4^(c) = 1 - <C_after, C_before> / (||C_after|| ||C_before||)  (Geodesic Torsion Defect)
        """
        B, H, W, D = h_after.shape
        num_cells = max(1, D // NA3D_CELL_STRIDE)
        if D % NA3D_CELL_STRIDE != 0:
            return torch.zeros((B, H, W, 4 * num_cells), dtype=h_after.dtype, device=h_after.device)

        dev_type = "cuda" if h_after.is_cuda else "cpu"
        with torch.amp.autocast(device_type=dev_type, enabled=False):
            m_5d = mask.to(torch.float32).view(B, H, W, 1, 1)
            b_cells = h_before.to(torch.float32).view(B, H, W, num_cells, NA3D_CELL_STRIDE) * m_5d
            a_cells = h_after.to(torch.float32).view(B, H, W, num_cells, NA3D_CELL_STRIDE) * m_5d

            c_0 = b_cells[..., 3:6]
            l_a = a_cells[..., 0:3]
            c_a = a_cells[..., 3:6]
            r_a = a_cells[..., 6:9]
            beta_a = torch.tanh(a_cells[..., 9:12])

            i1_chiral_vol = torch.tanh(torch.sum(l_a * torch.cross(c_a, r_a, dim=-1), dim=-1, keepdim=True))
            i2_helicity = torch.tanh(torch.sum(beta_a * torch.cross(l_a, r_a, dim=-1), dim=-1, keepdim=True))
            i3_diode_work = torch.tanh(torch.sum((c_a - c_0) * (l_a - r_a), dim=-1, keepdim=True))

            norm_ca = torch.sqrt(torch.sum(c_a * c_a, dim=-1, keepdim=True) + eps)
            norm_c0 = torch.sqrt(torch.sum(c_0 * c_0, dim=-1, keepdim=True) + eps)
            cos_sim = torch.sum(c_a * c_0, dim=-1, keepdim=True) / (norm_ca * norm_c0 + eps)
            i4_torsion_defect = (1.0 - cos_sim.clamp(-1.0, 1.0)) * m_5d

            obs_4c = torch.cat([i1_chiral_vol, i2_helicity, i3_diode_work, i4_torsion_defect], dim=-1) * m_5d
            return obs_4c.view(B, H, W, 4 * num_cells).to(h_after.dtype)

    @classmethod
    def enforce_isochoric_frame_projection(
        cls,
        h_anchor: torch.Tensor,
        h_perturbed: torch.Tensor,
        mask: torch.Tensor,
        blend_gain: float = 0.15,
        eps: float = DEFAULT_STIEFEL_EPS,
    ) -> torch.Tensor:
        orig_shape = h_perturbed.shape
        D = orig_shape[-1]
        if D % NA3D_CELL_STRIDE != 0:
            return h_perturbed

        target_dtype = h_perturbed.dtype
        dev_type = "cuda" if h_perturbed.is_cuda else "cpu"

        with torch.amp.autocast(device_type=dev_type, enabled=False):
            num_cells = D // NA3D_CELL_STRIDE
            mask_f32 = mask.to(torch.float32).unsqueeze(-1)
            mask_cell = mask.to(torch.float32).view(*orig_shape[:-1], 1, 1)

            a_cells = (h_anchor.to(torch.float32) * mask_f32).view(*orig_shape[:-1], num_cells, NA3D_CELL_STRIDE)
            p_cells = (h_perturbed.to(torch.float32) * mask_f32).view(*orig_shape[:-1], num_cells, NA3D_CELL_STRIDE)

            r_a = a_cells[..., 6:9]
            c_a = a_cells[..., 3:6]

            l_p = p_cells[..., 0:3]
            c_p = p_cells[..., 3:6]
            r_p = p_cells[..., 6:9]
            b_p = p_cells[..., 9:12]
            m_p = p_cells[..., 12:16]

            omega_gap = torch.tanh(torch.cross(r_a, l_p, dim=-1))
            c_rot = cls.apply_rodrigues_rotation(c_a, omega_gap * blend_gain, eps=eps)
            c_preserved = ((1.0 - blend_gain) * c_p) + (blend_gain * c_rot)

            b_p_norm = torch.sqrt(torch.sum(b_p * b_p, dim=-1, keepdim=True) + eps)
            b_retracted = (torch.tanh(b_p_norm) / b_p_norm) * b_p

            out_cells = torch.cat([l_p, c_preserved, r_p, b_retracted, m_p], dim=-1) * mask_cell
        return out_cells.view(*orig_shape).to(target_dtype)

    @classmethod
    def extract_na3d_global_invariants(
        cls,
        h_field: torch.Tensor,
        mask: torch.Tensor,
        eps: float = DEFAULT_STIEFEL_EPS,
    ) -> torch.Tensor:
        B, H, W, D = h_field.shape
        num_cells = max(1, D // NA3D_CELL_STRIDE)
        if D % NA3D_CELL_STRIDE != 0:
            return torch.zeros((B, 3 * num_cells), dtype=h_field.dtype, device=h_field.device)

        dev_type = "cuda" if h_field.is_cuda else "cpu"
        with torch.amp.autocast(device_type=dev_type, enabled=False):
            mask_f32 = mask.float().view(B, H, W, 1, 1)
            active_count = mask_f32.sum(dim=(1, 2)).clamp_min(1.0)

            cells = (h_field.to(torch.float32) * mask.float().unsqueeze(-1)).view(
                B, H, W, num_cells, NA3D_CELL_STRIDE
            )
            l_face = cells[..., 0:3]
            c_core = cells[..., 3:6]
            r_face = cells[..., 6:9]
            meta_4 = cells[..., 12:16]

            chiral_vol = torch.tanh(
                torch.sum(l_face * torch.cross(c_core, r_face, dim=-1), dim=-1, keepdim=True)
            )
            mean_vol = (chiral_vol * mask_f32).sum(dim=(1, 2)) / active_count

            wilson_k = meta_4[..., 1:2]
            mean_wilson = (wilson_k * mask_f32).sum(dim=(1, 2)) / active_count

            if H > 1 and W > 1:
                diff_y = c_core[:, 1:, :, :, :] - c_core[:, :-1, :, :, :]
                diff_x = c_core[:, :, 1:, :, :] - c_core[:, :, :-1, :, :]
                dy_c = torch.sqrt(torch.sum(diff_y * diff_y, dim=-1, keepdim=True) + eps)
                dx_c = torch.sqrt(torch.sum(diff_x * diff_x, dim=-1, keepdim=True) + eps)
                mean_dy = dy_c.mean(dim=(1, 2))
                mean_dx = dx_c.mean(dim=(1, 2))
                anisotropy_spin = torch.tanh(mean_dy - mean_dx)
            else:
                anisotropy_spin = torch.zeros_like(mean_vol)

            bundle = torch.cat(
                [mean_vol.squeeze(-1), mean_wilson.squeeze(-1), anisotropy_spin.squeeze(-1)],
                dim=-1,
            )
        return bundle.to(h_field.dtype)

    def forward(self, h_field: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, H, W, D = h_field.shape
        if D % NA3D_CELL_STRIDE != 0:
            return h_field

        target_dtype = h_field.dtype
        dev_type = "cuda" if h_field.is_cuda else "cpu"

        with torch.amp.autocast(device_type=dev_type, enabled=False):
            mask_f32 = mask.float().unsqueeze(-1)
            mask_5d = mask.float().view(B, H, W, 1, 1)

            h_masked = (h_field.to(torch.float32) * mask_f32)
            cells = h_masked.view(B, H, W, self.num_cells, NA3D_CELL_STRIDE)

            l_face = cells[..., 0:3]
            c_core = cells[..., 3:6]
            r_face = cells[..., 6:9]
            v_beta = cells[..., 9:12]

            v_norm = torch.sqrt(torch.sum(v_beta * v_beta, dim=-1, keepdim=True) + self.eps)
            beta_wind = (torch.tanh(v_norm) / v_norm) * v_beta

            spin_local = torch.cross(l_face, r_face, dim=-1)
            chiral_vol = torch.sum(l_face * torch.cross(c_core, r_face, dim=-1), dim=-1, keepdim=True)
            chiral_vol_bounded = torch.tanh(chiral_vol)

            pack_12 = torch.cat([l_face, c_core, beta_wind, spin_local], dim=-1).view(B, H, W, self.num_cells * 12)
            pack_pad = F.pad(pack_12.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="constant", value=0.0).permute(0, 2, 3, 1)

            nb_n = pack_pad[:, :-2, 1:-1, :]
            nb_s = pack_pad[:, 2:, 1:-1, :]
            nb_w = pack_pad[:, 1:-1, :-2, :]
            nb_e = pack_pad[:, 1:-1, 2:, :]

            nb_stack = torch.stack([nb_n, nb_s, nb_w, nb_e], dim=1).view(B, 4, H, W, self.num_cells, 12)

            l_nb = nb_stack[..., 0:3]
            c_nb = nb_stack[..., 3:6]
            beta_nb = nb_stack[..., 6:9]
            spin_nb = nb_stack[..., 9:12]

            r_exp = r_face.unsqueeze(1)
            beta_exp = beta_wind.unsqueeze(1)
            spin_exp = spin_local.unsqueeze(1)

            mu_0 = F.softplus(self.raw_rest_mass) + 0.05
            tau_0 = F.softplus(self.raw_tau_0) + 0.50

            delta_front = l_nb - r_exp
            riemann_dist = torch.sqrt(torch.sum(delta_front * delta_front, dim=-1, keepdim=True) + mu_0.pow(2))
            randers_drift = torch.sum((beta_exp - beta_nb) * delta_front, dim=-1, keepdim=True)

            tau_ab = torch.clamp_min(riemann_dist + randers_drift, 1e-4)
            sigma_cfc = torch.exp(-tau_ab / tau_0)
            w_conductance = sigma_cfc / torch.sum(sigma_cfc, dim=1, keepdim=True).clamp_min(self.eps)

            omega_gap = torch.cross(r_exp.expand_as(l_nb), l_nb, dim=-1)
            comm_1 = torch.cross(spin_exp.expand_as(spin_nb), spin_nb, dim=-1)
            comm_2 = torch.cross(spin_exp.expand_as(comm_1), comm_1, dim=-1) - torch.cross(spin_nb, comm_1, dim=-1)
            omega_bch = omega_gap + (0.50 * comm_1) + ((1.0 / 12.0) * comm_2)

            omega_net = torch.sum(w_conductance * omega_bch, dim=1)
            c_iso_rotated = self.apply_rodrigues_rotation(
                v=c_core,
                omega=omega_net,
                eps=self.eps,
            )

            c_nb_transported = self.apply_rodrigues_rotation(
                v=c_nb,
                omega=-omega_bch * sigma_cfc,
                eps=self.eps,
            )
            c_nb_weighted = torch.sum(w_conductance * c_nb_transported, dim=1)

            c_flat_ch = c_core.reshape(B, H, W, self.num_cells * 3).permute(0, 3, 1, 2)
            c_pad_2 = F.pad(c_flat_ch, (2, 2, 2, 2), mode="replicate").permute(0, 2, 3, 1)
            c_up2 = c_pad_2[:, :-4, 2:-2, :].view(B, H, W, self.num_cells, 3)
            c_dn2 = c_pad_2[:, 4:, 2:-2, :].view(B, H, W, self.num_cells, 3)
            c_lt2 = c_pad_2[:, 2:-2, :-4, :].view(B, H, W, self.num_cells, 3)
            c_rt2 = c_pad_2[:, 2:-2, 4:, :].view(B, H, W, self.num_cells, 3)

            c_bravais_avg = 0.25 * (c_up2 + c_dn2 + c_lt2 + c_rt2)
            bravais_dislocation = torch.tanh(
                torch.sqrt(torch.sum((c_core - c_bravais_avg).pow(2), dim=-1, keepdim=True) + self.eps)
            )

            omega_s = omega_bch[:, 1]
            omega_e = omega_bch[:, 3]

            omega_e_south = F.pad(omega_e[:, 1:, :, :, :], (0, 0, 0, 0, 0, 0, 0, 1), mode="constant", value=0.0)
            omega_s_east = F.pad(omega_s[:, :, 1:, :, :], (0, 0, 0, 0, 0, 1, 0, 0), mode="constant", value=0.0)

            f_plaquette = (
                omega_e
                + omega_s_east
                - omega_e_south
                - omega_s
                + 0.50 * (torch.cross(omega_e, omega_s_east, dim=-1) - torch.cross(omega_s, omega_e_south, dim=-1))
            )
            f_norm = torch.sqrt(torch.sum(f_plaquette * f_plaquette, dim=-1, keepdim=True) + self.eps)
            wilson_curvature = 1.0 - torch.cos(torch.tanh(f_norm) * math.pi)

            beta_nb_n = beta_nb[:, 0]
            beta_nb_s = beta_nb[:, 1]
            beta_nb_w = beta_nb[:, 2]
            beta_nb_e = beta_nb[:, 3]
            div_beta = 0.50 * (
                (beta_nb_s[..., 0:1] - beta_nb_n[..., 0:1])
                + (beta_nb_e[..., 1:2] - beta_nb_w[..., 1:2])
            )
            stokes_barrier = torch.tanh(div_beta + 0.25 * torch.mean(tau_ab, dim=1))

            meta_updated = torch.cat(
                [
                    chiral_vol_bounded,
                    wilson_curvature,
                    bravais_dislocation,
                    stokes_barrier,
                ],
                dim=-1,
            )

            w_f = self.frame_mix_weight.to(torch.float32)
            c_frame_mixed = torch.einsum("cd,bhwdk->bhwck", w_f, c_iso_rotated)

        meta_flat = meta_updated.view(B, H, W, 4 * self.num_cells).to(target_dtype)
        cell_delta_c = torch.tanh(self.cell_gate_proj(meta_flat)).to(torch.float32).view(B, H, W, self.num_cells, 3)

        g_brav = torch.tanh(self.bravais_heal_gate.to(torch.float32)) * 0.15
        c_final = (
            c_iso_rotated
            + (0.20 * (c_nb_weighted - c_iso_rotated))
            + (0.15 * cell_delta_c)
            + (0.15 * c_frame_mixed)
            + (g_brav * bravais_dislocation * (c_bravais_avg - c_iso_rotated))
        )

        cells_updated = torch.cat([l_face, c_final, r_face, beta_wind, meta_updated], dim=-1) * mask_5d
        flat_updated = cells_updated.view(B, H, W, D).to(target_dtype)

        return (h_field + self.channel_mixer(flat_updated)) * mask_f32.to(target_dtype)


class CellularSheafOperator(nn.Module):
    """
    Continuous Cellular Sheaf Operator on Stiefel Manifolds V_k(R^d) for Object Slots.
    """

    def __init__(
        self,
        d_node: int = DEFAULT_D_NODE,
        d_edge: int = DEFAULT_D_EDGE,
        max_slots: int = 12,
        alpha_advection: float = 0.85,
        beta_diffusion: float = 0.15,
        stiefel_iterations: int = DEFAULT_STIEFEL_NS_ITERS,
        eps: float = DEFAULT_STIEFEL_EPS,
    ) -> None:
        super().__init__()
        self.d_node = d_node
        self.d_edge = compute_dynamic_edge_dimension(d_node, d_edge)
        self.max_slots = max_slots
        self.alpha_advection = float(alpha_advection)
        self.beta_diffusion = float(beta_diffusion)
        self.stiefel_iterations = max(3, int(stiefel_iterations))
        self.eps = eps

        self.w_projection = nn.Sequential(
            nn.Linear(self.d_node, self.d_node, bias=False),
            RMSNorm(self.d_node),
            nn.SiLU(),
            nn.Linear(self.d_node, self.d_edge * self.d_node, bias=False),
        )
        self.norm_stalk = RMSNorm(self.d_node)

        r_init = torch.zeros(self.d_edge, self.d_edge, dtype=torch.float32)
        for i in range(self.d_edge):
            j = self.d_edge - 1 - i
            if i != j:
                sign = 1.0 if (i % 2 == 0) else -1.0
                val = sign * math.sin(float(i + 1) / float(self.d_edge))
                r_init[i, j] = val
                r_init[j, i] = -val
        self.register_buffer("canonical_skew_generator", r_init)
        self.register_buffer("gamma_resolvent", torch.tensor(0.10, dtype=torch.float32))

    @property
    def safe_gamma(self) -> torch.Tensor:
        return torch.clamp_min(cast(torch.Tensor, self.gamma_resolvent), 1e-4)

    def project_stiefel_5th_order(
        self,
        w_raw: torch.Tensor,
        iterations: Optional[int] = None,
    ) -> torch.Tensor:
        iters = iterations if iterations is not None else self.stiefel_iterations
        return newton_schulz_stiefel_5th_order(w_raw, iterations=iters, eps=self.eps)

    def generate_sheaf_bundle(
        self,
        z_stalks: torch.Tensor,
    ) -> torch.Tensor:
        b, k, _ = z_stalks.shape
        w_flat = self.w_projection(z_stalks)
        w_raw = w_flat.view(b, k, self.d_edge, self.d_node)
        w_ortho = self.project_stiefel_5th_order(w_raw)
        return w_ortho

    def _compute_normalized_edge_conductance(
        self,
        v_local: torch.Tensor,
        port_conductance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, k, d_e = v_local.shape
        if port_conductance is not None:
            c_raw = port_conductance.to(v_local.dtype)
        else:
            v_k = v_local.unsqueeze(2)
            v_j = v_local.unsqueeze(1)
            dist_sq = torch.sum((v_k - v_j).to(torch.float32).pow(2), dim=-1) / float(max(1, d_e))
            eye_k = torch.eye(k, dtype=torch.bool, device=v_local.device).unsqueeze(0)
            c_raw = torch.exp(-0.50 * dist_sq).masked_fill(eye_k, 0.0).to(v_local.dtype)

        deg = torch.sum(c_raw, dim=-1, keepdim=True).clamp_min(1.0)
        return c_raw / deg

    def compute_sheaf_laplacian(
        self,
        z_stalks: torch.Tensor,
        w_sheaf: torch.Tensor,
        port_conductance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, k, _ = z_stalks.shape
        d_e = self.d_edge

        v_local = torch.matmul(w_sheaf, z_stalks.unsqueeze(-1)).squeeze(-1)
        c_norm = self._compute_normalized_edge_conductance(v_local, port_conductance).unsqueeze(-1)

        v_k = v_local.unsqueeze(2).expand(b, k, k, d_e)
        v_j = v_local.unsqueeze(1).expand(b, k, k, d_e)

        coboundary_kj = v_k - v_j
        diff_sum_edges = torch.sum(coboundary_kj * c_norm, dim=2)

        w_t = w_sheaf.transpose(-2, -1)
        lz = torch.matmul(w_t, diff_sum_edges.unsqueeze(-1)).squeeze(-1)
        return lz

    def compute_dirac_advection(
        self,
        z_stalks: torch.Tensor,
        w_sheaf: torch.Tensor,
        c_relational: torch.Tensor,
        port_conductance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, k, _ = z_stalks.shape
        d_e = self.d_edge

        v_k = torch.matmul(w_sheaf, z_stalks.unsqueeze(-1)).squeeze(-1)
        c_norm = self._compute_normalized_edge_conductance(v_k, port_conductance).unsqueeze(-1)

        r_gen = cast(torch.Tensor, self.canonical_skew_generator).view(1, 1, d_e, d_e).expand(b, k, -1, -1).to(v_k.dtype)
        rv = torch.matmul(r_gen, v_k.unsqueeze(-1)).squeeze(-1)

        rv_j = rv.unsqueeze(1).expand(b, k, k, d_e)
        rv_sum = torch.sum(rv_j * c_norm, dim=2)

        w_t = w_sheaf.transpose(-2, -1)
        az = torch.matmul(w_t, rv_sum.unsqueeze(-1)).squeeze(-1)
        return az

    def compute_dirichlet_energy(
        self,
        z_stalks: torch.Tensor,
        w_sheaf: torch.Tensor,
        port_conductance: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, k, _ = z_stalks.shape
        d_e = self.d_edge

        v_local = torch.matmul(w_sheaf, z_stalks.unsqueeze(-1)).squeeze(-1).to(torch.float32)

        v_k = v_local.unsqueeze(2).expand(b, k, k, d_e)
        v_j = v_local.unsqueeze(1).expand(b, k, k, d_e)
        diff_sq = torch.sum((v_k - v_j).pow(2), dim=-1) / float(max(1, d_e))

        if port_conductance is not None:
            c_mat = port_conductance.to(torch.float32)
        else:
            eye_k = torch.eye(k, dtype=torch.bool, device=z_stalks.device).unsqueeze(0)
            c_mat = torch.ones((b, k, k), dtype=torch.float32, device=z_stalks.device).masked_fill(eye_k, 0.0) / float(
                max(1, k - 1)
            )

        weighted_diff = diff_sq * c_mat
        active_edges = torch.clamp_min(torch.sum(c_mat, dim=(-2, -1)), 1.0)

        abs_energy = 0.50 * torch.sum(weighted_diff, dim=(-2, -1))
        norm_energy = abs_energy / active_edges

        return abs_energy.to(z_stalks.dtype), norm_energy.to(z_stalks.dtype)

    def solve_schur_resolvent(
        self,
        z_boundary: torch.Tensor,
        u_mol_context: torch.Tensor,
        c_relational: torch.Tensor,
        gamma_reg: Optional[float] = None,
        num_unknown_slots: int = 1,
    ) -> torch.Tensor:
        b, k_b, d = z_boundary.shape
        k_u = max(1, min(num_unknown_slots, self.max_slots))
        device = z_boundary.device
        dtype = z_boundary.dtype
        dev_type = "cuda" if z_boundary.is_cuda else "cpu"

        gamma_val = gamma_reg if gamma_reg is not None else float(self.safe_gamma.item())

        u_expanded = u_mol_context.unsqueeze(1).expand(b, k_u, d)
        z_composite = torch.cat([z_boundary, u_expanded], dim=1)

        w_composite = self.generate_sheaf_bundle(z_composite)

        w_b = w_composite[:, :k_b]
        w_u = w_composite[:, k_b:]

        w_u_t = w_u.transpose(-2, -1)
        v_b = torch.matmul(w_b, z_boundary.unsqueeze(-1)).squeeze(-1)

        v_b_mean = torch.mean(v_b, dim=1, keepdim=True).expand(b, k_u, -1)
        l_ub_zb = -torch.matmul(w_u_t, v_b_mean.unsqueeze(-1)).squeeze(-1)

        az = self.compute_dirac_advection(z_composite, w_composite, c_relational)
        a_ub_zb = az[:, k_b:]

        with torch.amp.autocast(device_type=dev_type, enabled=False):
            n_flat = b * k_u
            de = self.d_edge
            gamma_safe = max(float(gamma_val), 1e-3)
            w_u_flat = w_u.to(torch.float32).reshape(n_flat, de, d)

            rhs = (
                gamma_safe * u_expanded.to(torch.float32)
                + (0.20 * self.alpha_advection * a_ub_zb.to(torch.float32))
                - l_ub_zb.to(torch.float32)
            )
            rhs_flat = rhs.reshape(n_flat, d, 1)

            """
            Woodbury-Sherman-Morrison + 8-Step Newton-Schulz SPD Solver in pure cuBLAS GEMM:
              (W_u^T W_u + gamma I_d)^(-1) rhs = (1 / gamma) * [rhs - W_u^T (gamma I_de + W_u W_u^T)^(-1) W_u rhs]
            Completely eliminates MAGMA sgetrf_batched / cuSOLVER cholesky calls on CUDA 13.0!
            """
            gram_raw = torch.bmm(w_u_flat, w_u_flat.transpose(1, 2))
            gram_sym = 0.50 * (gram_raw + gram_raw.transpose(1, 2))
            eye_de = torch.eye(de, dtype=torch.float32, device=device).unsqueeze(0).expand(n_flat, de, de)
            gram_spd = gram_sym + (gamma_safe * eye_de)

            trace_spd = (
                torch.diagonal(gram_spd, dim1=-2, dim2=-1)
                .sum(dim=-1, keepdim=True)
                .unsqueeze(-1)
                .clamp_min(1e-4)
            )
            gram_inv = eye_de / trace_spd
            two_eye = 2.0 * eye_de
            for _ in range(8):
                gram_inv = torch.bmm(gram_inv, two_eye - torch.bmm(gram_spd, gram_inv))

            wu_rhs = torch.bmm(w_u_flat, rhs_flat)
            corr_flat = torch.bmm(w_u_flat.transpose(1, 2), torch.bmm(gram_inv, wu_rhs))
            z_u_flat = ((rhs_flat - corr_flat) / gamma_safe).squeeze(-1)

        z_unknown = z_u_flat.view(b, k_u, d).to(dtype)
        return z_unknown

    def forward(
        self,
        z_stalks: torch.Tensor,
        c_relational: torch.Tensor,
        port_conductance: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w_sheaf = self.generate_sheaf_bundle(z_stalks)

        lz = self.compute_sheaf_laplacian(z_stalks, w_sheaf, port_conductance)
        az = self.compute_dirac_advection(z_stalks, w_sheaf, c_relational, port_conductance)

        z_raw = z_stalks - (self.beta_diffusion * lz) + (0.20 * self.alpha_advection * az)
        z_evolved = self.norm_stalk(z_raw)
        _, norm_energy = self.compute_dirichlet_energy(z_evolved, w_sheaf, port_conductance)

        return z_evolved, w_sheaf, norm_energy


class SpatialMooreSheafDiffusion(nn.Module):
    """
    Anisotropic Edge-Stopping Perona-Malik-Randers 12-Ray Cellular Sheaf Diffusion,
    Heterogeneous Local Frame Restriction, and Isochoric NA3D-Finsler-Lie v3.0 Engine.
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
        (-3, 0, 0.25),
        (3, 0, 0.25),
        (0, -3, 0.25),
        (0, 3, 0.25),
    )

    def __init__(
        self,
        d_node: int = DEFAULT_D_NODE,
        d_edge: int = DEFAULT_D_EDGE,
        diffusion_gain: float = 0.15,
        advection_gain: float = 0.10,
        min_conductance: float = MIN_CONDUCTANCE_FLOOR,
        stiefel_iterations: int = 3,
        eps: float = DEFAULT_STIEFEL_EPS,
    ) -> None:
        super().__init__()
        self.d_node = d_node
        self.d_edge = compute_dynamic_edge_dimension(d_node, d_edge)
        self.diffusion_gain = float(diffusion_gain)
        self.advection_gain = float(advection_gain)
        self.min_conductance = float(min_conductance)
        self.stiefel_iterations = stiefel_iterations
        self.eps = eps
        self.num_stencil_rays = len(self.DIRECTIONS)

        self.w_raw = nn.Parameter(torch.randn(self.d_edge, self.d_node, dtype=torch.float32) / math.sqrt(self.d_node))
        self.local_stalk_modulator = nn.Linear(self.d_node, self.d_edge, bias=False)

        self.conductance_router = nn.Linear(self.d_node, self.num_stencil_rays, bias=True)
        self.permeability_valve = nn.Linear(self.d_node, 1, bias=True)
        self.edge_stop_scale = nn.Parameter(torch.tensor(0.45, dtype=torch.float32))

        self.advection_velocity_proj = nn.Linear(self.d_node, self.num_stencil_rays, bias=True)
        self.skew_rotor_param = nn.Parameter(torch.randn(self.d_edge, self.d_edge, dtype=torch.float32) * 0.05)

        self.na3d_engine = NA3DFinslerLieEngine(d_node=self.d_node, eps=eps)
        self.na3d_gate = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

        metric_w_tensor = torch.tensor(
            [m / float(self.num_stencil_rays) for _, _, m in self.DIRECTIONS],
            dtype=torch.float32,
        ).view(1, 1, 1, self.num_stencil_rays, 1)
        self.register_buffer("metric_ray_weights", metric_w_tensor, persistent=False)

        self.norm = RMSNorm(self.d_node)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.local_stalk_modulator.weight)
        nn.init.xavier_uniform_(self.conductance_router.weight, gain=0.2)
        nn.init.zeros_(self.conductance_router.bias)
        nn.init.xavier_uniform_(self.permeability_valve.weight, gain=0.2)
        nn.init.constant_(self.permeability_valve.bias, 0.25)
        nn.init.xavier_uniform_(self.advection_velocity_proj.weight, gain=0.1)
        nn.init.zeros_(self.advection_velocity_proj.bias)

    def _get_cayley_skew_rotor(self, target_dtype: torch.dtype) -> torch.Tensor:
        a_skew = 0.50 * (self.skew_rotor_param - self.skew_rotor_param.t())
        a_norm = torch.sqrt(torch.sum(a_skew * a_skew) + self.eps).clamp_min(1.0)
        a_bounded = a_skew / a_norm
        eye = torch.eye(self.d_edge, dtype=torch.float32, device=self.w_raw.device)
        r_mat = eye + a_bounded + (0.50 * torch.matmul(a_bounded, a_bounded))
        return r_mat.to(target_dtype)

    def forward(
        self,
        h_field: torch.Tensor,
        mask: torch.Tensor,
        rule_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, W, D = h_field.shape
        d_e = self.d_edge
        mask_f32 = mask.to(h_field.dtype).unsqueeze(-1)

        if D == self.d_node and D % NA3D_CELL_STRIDE == 0:
            h_na3d = self.na3d_engine(h_field, mask)
            h_active = h_field + (torch.tanh(self.na3d_gate.to(h_field.dtype)) * (h_na3d - h_field))
        else:
            h_active = h_field

        w_ortho = newton_schulz_stiefel_5th_order(
            self.w_raw.unsqueeze(0),
            iterations=self.stiefel_iterations,
            eps=self.eps,
        ).squeeze(0).to(h_field.dtype)

        local_frame_gain = 1.0 + 0.20 * torch.tanh(self.local_stalk_modulator(h_active))
        v_edge = (F.linear(h_active, w_ortho) * local_frame_gain) * mask_f32
        r_rotor = self._get_cayley_skew_rotor(h_field.dtype)
        rv_edge = F.linear(v_edge, r_rotor) * mask_f32

        ray_logits = self.conductance_router(h_active)
        ray_simplex = F.softmax(ray_logits, dim=-1)

        raw_valve = torch.sigmoid(self.permeability_valve(h_active))
        g_edge = self.min_conductance + ((1.0 - self.min_conductance) * raw_valve)

        conductances = (float(self.num_stencil_rays) * g_edge * ray_simplex) * mask_f32

        if rule_context is not None:
            b_r = rule_context.shape[0]
            if B != b_r and B % b_r == 0:
                r_aligned = rule_context.repeat_interleave(B // b_r, dim=0)
            else:
                r_aligned = rule_context[:B]
            adv_weights = torch.tanh(self.advection_velocity_proj(r_aligned)).view(B, 1, 1, self.num_stencil_rays, 1)
        else:
            mean_h = (h_active * mask_f32).sum(dim=(1, 2)) / mask_f32.sum(dim=(1, 2)).clamp_min(1.0)
            adv_weights = torch.tanh(self.advection_velocity_proj(mean_h)).view(B, 1, 1, self.num_stencil_rays, 1)

        pack_all = torch.cat([v_edge, rv_edge, mask_f32], dim=-1)
        pad_all = F.pad(pack_all.permute(0, 3, 1, 2), (3, 3, 3, 3), mode="constant", value=0.0).permute(0, 2, 3, 1)

        ray_slices = [
            pad_all[:, (3 - dy) : (3 - dy + H), (3 - dx) : (3 - dx + W), :]
            for dy, dx, _ in self.DIRECTIONS
        ]
        stacked_nb = torch.stack(ray_slices, dim=3)

        v_nb = stacked_nb[..., :d_e]
        rv_nb = stacked_nb[..., d_e : 2 * d_e]
        mask_nb = stacked_nb[..., 2 * d_e :]

        valid_edges = mask_f32.unsqueeze(3) * mask_nb
        metric_w = cast(torch.Tensor, self.metric_ray_weights).to(h_field.dtype)

        """
        Anisotropic Perona-Malik-Randers Edge-Stopping Barrier:
        Prevents foreground contours from bleeding into background across sharp edges!
        """
        diff_edges = (v_edge.unsqueeze(3) - v_nb) * valid_edges
        diff_sq_f32 = torch.sum(diff_edges.to(torch.float32).pow(2), dim=-1, keepdim=True)
        norm_diff_sq = diff_sq_f32 / float(max(1, d_e))

        sigma_edge_sq = F.softplus(self.edge_stop_scale.to(torch.float32)).pow(2) + 0.05
        asym_drift = F.relu(torch.sum(v_edge.unsqueeze(3).to(torch.float32) * diff_edges.to(torch.float32), dim=-1, keepdim=True)) / float(max(1, d_e))
        edge_barrier = torch.exp(-((norm_diff_sq + 0.35 * asym_drift) / sigma_edge_sq)).to(h_field.dtype)

        c_dirs = conductances.unsqueeze(-1) * metric_w * edge_barrier * valid_edges
        w_adv_dirs = adv_weights * metric_w * edge_barrier * valid_edges

        laplacian_edge_accum = torch.sum(c_dirs * diff_edges, dim=3)
        advection_edge_accum = torch.sum(w_adv_dirs * rv_nb, dim=3)

        energy_accum = torch.sum(c_dirs.detach().to(torch.float32) * diff_sq_f32, dim=3)

        delta_f_h = F.linear(laplacian_edge_accum * local_frame_gain, w_ortho.t())
        adv_f_h = F.linear(advection_edge_accum * local_frame_gain, w_ortho.t())

        h_updated = self.norm(
            h_active - (self.diffusion_gain * delta_f_h) + (self.advection_gain * adv_f_h)
        ) * mask_f32

        active_pixels = mask.float().sum(dim=(1, 2)).clamp_min(1.0)
        dirichlet_energy = (0.50 * energy_accum.sum(dim=(1, 2, 3))) / active_pixels

        return h_updated, dirichlet_energy


def verify_stiefel_sheaf_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for stiefel_sheaf.py.
    """
    torch.manual_seed(42)

    d_v, d_e = 64, 16
    u_vec = torch.randn(2, 4, d_e, 1)
    v_vec = torch.randn(2, 4, 1, d_v)
    w_rank1 = torch.matmul(u_vec, v_vec)

    w_ortho_rank1 = newton_schulz_stiefel_5th_order(w_rank1, iterations=5)
    if torch.isnan(w_ortho_rank1).any() or torch.isinf(w_ortho_rank1).any():
        return False

    h_zero = torch.zeros(2, 6, 6, d_v, dtype=torch.float32, requires_grad=True)
    mask_partial = torch.zeros(2, 6, 6, dtype=torch.bool)
    mask_partial[0, :3, :3] = True

    h_iso = NA3DFinslerLieEngine.enforce_isochoric_frame_projection(h_zero, h_zero, mask_partial)
    obs_zero = NA3DFinslerLieEngine.extract_na3d_global_invariants(h_iso, mask_partial)
    casimir_obs = NA3DFinslerLieEngine.extract_casimir_chiral_observables(h_zero, h_iso, mask_partial)
    if casimir_obs.shape != (2, 6, 6, 4 * (d_v // 16)):
        return False

    loss_zero = torch.sum(h_iso.pow(2)) + torch.sum(obs_zero.pow(2)) + torch.sum(casimir_obs.pow(2))
    loss_zero.backward()

    if h_zero.grad is None or torch.isnan(h_zero.grad).any() or torch.isinf(h_zero.grad).any():
        return False

    sheaf_op = CellularSheafOperator(
        d_node=d_v,
        d_edge=d_e,
        max_slots=4,
        alpha_advection=0.0,
        beta_diffusion=0.25,
    )

    z_stalks = torch.randn(2, 4, d_v, requires_grad=True)
    c_rel = torch.randn(2, d_v, requires_grad=True)

    w_init = sheaf_op.generate_sheaf_bundle(z_stalks)
    _, e_before = sheaf_op.compute_dirichlet_energy(z_stalks, w_init)
    lz = sheaf_op.compute_sheaf_laplacian(z_stalks, w_init)
    z_step = z_stalks - 0.25 * lz
    _, e_after = sheaf_op.compute_dirichlet_energy(z_step, w_init)
    if float(e_after.mean().item()) >= float(e_before.mean().item()):
        return False

    z_ev, _, e_dir = sheaf_op(z_stalks, c_rel)
    z_solved = sheaf_op.solve_schur_resolvent(
        z_boundary=z_stalks[:, :3],
        u_mol_context=c_rel,
        c_relational=c_rel,
        gamma_reg=0.10,
        num_unknown_slots=1,
    )

    spatial_sheaf = SpatialMooreSheafDiffusion(d_node=d_v, d_edge=d_e, diffusion_gain=0.15)
    h_canvas = torch.randn(2, 6, 6, d_v, requires_grad=True)
    mask_canvas = torch.ones(2, 6, 6, dtype=torch.bool)
    mask_canvas[:, :, 3:] = False

    h_diffused, e_spatial = spatial_sheaf(h_canvas, mask_canvas, rule_context=c_rel)
    if not torch.all(h_diffused[:, :, 3:, :] == 0.0):
        return False

    loss = torch.sum(z_ev.pow(2)) + torch.sum(z_solved.pow(2)) + torch.sum(h_diffused.pow(2)) + torch.sum(e_spatial)
    loss.backward()

    if z_stalks.grad is None or h_canvas.grad is None or spatial_sheaf.w_raw.grad is None:
        return False
    if torch.isnan(z_stalks.grad).any() or torch.isnan(h_canvas.grad).any():
        return False

    return True
