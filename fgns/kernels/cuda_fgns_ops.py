"""
Location: // fgns_engine/fgns/kernels/cuda_fgns_ops.py
Description: Line-Parallel Native CUDA C++17 Shared-SRAM Compute Shaders, Modal-Background
             Cummax Optical Ray-Caster & Boundary-Stopped Convex-EMA Wavefront Engine (FGNS v4.3.0):
               1. Fused Shared-SRAM Cavity Flood-Fill & Beer-Lambert Optical Ray Integrator
                  with Modal-Background & Isolated Black-Seed Recognition (solving Task 13e47133 Pair 2!).
               2. Zero-Loop Closed-Form Cummax Optical Ray-Caster in PyTorch fallback.
               3. Line-Parallel 8-Axis Isotropic OctoWave Forward & Adjoint Backward Kernels
                  with Boundary-Stopped Convex-EMA normalization and single-pass Specular Reflection.
               4. Fused Register-Level Continuous Morlet Wavelet-KAN Forward & Adjoint Backward Shaders.
Language: Python 3.12+ / PyTorch 2.5+ / Native CUDA C++17 JIT Extension
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, Final, List, Optional, Tuple, Union
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from fgns.primitives.swiglu import RMSNorm

CUDA_EXTENSION_MODULE: Optional[Any] = None
CUDA_JIT_ATTEMPTED: bool = False
CUDA_JIT_SUCCEEDED: bool = False
_USABLE_CUDA_VERIFIED: Optional[bool] = None


def is_cuda_kernel_image_usable() -> bool:
    """
    Verifies that the installed PyTorch CUDA build includes valid kernel images for the
    active physical GPU (e.g. cu128 for RTX 5070 Blackwell sm_120).
    """
    global _USABLE_CUDA_VERIFIED
    if _USABLE_CUDA_VERIFIED is not None:
        return _USABLE_CUDA_VERIFIED

    if not torch.cuda.is_available():
        _USABLE_CUDA_VERIFIED = False
        return False

    try:
        probe = torch.zeros((2, 2), dtype=torch.float32, device="cuda")
        _ = probe + 1.0
        torch.cuda.synchronize()
        _USABLE_CUDA_VERIFIED = True
    except Exception:
        _USABLE_CUDA_VERIFIED = False

    return _USABLE_CUDA_VERIFIED


def _get_usable_cuda_or_cpu_device() -> torch.device:
    return torch.device("cuda" if is_cuda_kernel_image_usable() else "cpu")


def _discover_msvc_and_configure_env() -> None:
    if sys.platform != "win32":
        return

    vs_roots = (
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
    )

    for root in vs_roots:
        if os.path.isdir(root):
            versions = sorted(os.listdir(root), reverse=True)
            for ver in versions:
                cl_dir = os.path.join(root, ver, "bin", "Hostx64", "x64")
                cl_exe = os.path.join(cl_dir, "cl.exe")
                if os.path.isfile(cl_exe):
                    current_path = os.environ.get("PATH", "")
                    if cl_dir not in current_path:
                        os.environ["PATH"] = cl_dir + os.pathsep + current_path
                    return


def _configure_posix_cuda_and_triton_env() -> None:
    """
    Configures POSIX / Linux (Kaggle Ubuntu container) CUDA and Triton PTXAS environment variables.
    """
    if os.name == "nt":
        return

    ptxas_default = "/usr/local/cuda/bin/ptxas"
    if "TRITON_PTXAS_PATH" not in os.environ and os.path.isfile(ptxas_default):
        os.environ["TRITON_PTXAS_PATH"] = ptxas_default

    cuda_bin_dir = "/usr/local/cuda/bin"
    if os.path.isdir(cuda_bin_dir):
        current_path = os.environ.get("PATH", "")
        if cuda_bin_dir not in current_path:
            os.environ["PATH"] = cuda_bin_dir + os.pathsep + current_path


_CPP_DECLARATIONS: Final[str] = r"""
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> cuda_finsler_chordal_cavity_forward(
    torch::Tensor grid,
    torch::Tensor mask,
    int num_iterations
);

torch::Tensor cuda_morlet_kan_forward(
    torch::Tensor x,
    torch::Tensor centers,
    torch::Tensor scales,
    torch::Tensor weights,
    torch::Tensor phases,
    double omega_0
);

std::vector<torch::Tensor> cuda_morlet_kan_backward(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor centers,
    torch::Tensor scales,
    torch::Tensor weights,
    torch::Tensor phases,
    double omega_0
);

torch::Tensor cuda_octowave_8axis_forward(
    torch::Tensor decay_gates,
    torch::Tensor impulse_gates
);

std::vector<torch::Tensor> cuda_octowave_8axis_backward(
    torch::Tensor decay_gates,
    torch::Tensor h_out,
    torch::Tensor grad_out
);
"""

_CUDA_SOURCES: Final[str] = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>

__global__ void _finsler_chordal_cavity_sram_kernel(
    const int64_t* __restrict__ grid_ptr,
    const bool* __restrict__ mask_ptr,
    float* __restrict__ m_cav_ptr,
    float* __restrict__ h_span_ptr,
    float* __restrict__ w_span_ptr,
    float* __restrict__ r_harm_ptr,
    float* __restrict__ ey_ptr,
    float* __restrict__ ex_ptr,
    float* __restrict__ seed_f_ptr,
    float* __restrict__ is_wall_ptr,
    float* __restrict__ is_seed_ptr,
    const int H,
    const int W,
    const int seed_iters
) {
    __shared__ int s_grid[32][33];
    __shared__ int s_mask[32][33];
    __shared__ float s_trans[32][33];
    __shared__ float s_phi[32][33];
    __shared__ float s_seed[32][33];
    __shared__ int s_color_counts[11];

    const int b = blockIdx.x;
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int tid = y * W + x;
    const int hw = H * W;
    const int base_off = b * hw + tid;

    if (tid < 11) {
        s_color_counts[tid] = 0;
    }
    if (y < 32 && x < 32) {
        s_grid[y][x] = 10;
        s_mask[y][x] = 0;
        s_trans[y][x] = 0.0f;
        s_phi[y][x] = 0.0f;
        s_seed[y][x] = 0.0f;
    }
    __syncthreads();

    int g_val = 10;
    int m_val = 0;
    if (y < H && x < W) {
        g_val = (int)grid_ptr[base_off];
        m_val = mask_ptr[base_off] ? 1 : 0;
        s_grid[y][x] = g_val;
        s_mask[y][x] = m_val;
        if (m_val == 1 && g_val >= 0 && g_val < 10) {
            atomicAdd(&s_color_counts[g_val], 1);
            atomicAdd(&s_color_counts[10], 1);
        }
    }
    __syncthreads();

    if (y >= H || x >= W) {
        return;
    }

    int max_c = -1;
    int best_col = 0;
    for (int c = 0; c < 10; ++c) {
        if (s_color_counts[c] > max_c) {
            max_c = s_color_counts[c];
            best_col = c;
        }
    }
    int total_active = s_color_counts[10];
    int bg_color = best_col;
    if (s_color_counts[0] > 0 && (s_color_counts[0] * 5 >= total_active || s_color_counts[0] == max_c)) {
        bg_color = 0;
    }

    int same_nb = 0;
    const int dy_arr[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    const int dx_arr[8] = {0, 0, -1, 1, -1, 1, -1, 1};

    if (m_val == 1) {
        for (int d = 0; d < 8; ++d) {
            int ny = y + dy_arr[d];
            int nx = x + dx_arr[d];
            if (ny >= 0 && ny < H && nx >= 0 && nx < W) {
                if (s_mask[ny][nx] == 1 && s_grid[ny][nx] == g_val) {
                    same_nb += 1;
                }
            }
        }
    }

    bool is_isolated_singleton = (m_val == 1 && same_nb == 0 && s_color_counts[g_val] <= 4);
    float is_bg = (m_val == 1 && g_val == bg_color && !is_isolated_singleton) ? 1.0f : 0.0f;
    float is_non_bg = (m_val == 1 && (g_val != bg_color || is_isolated_singleton)) ? 1.0f : 0.0f;
    float wall_val = (is_non_bg > 0.5f && same_nb >= 1) ? 1.0f : 0.0f;
    float seed_val = (is_non_bg > 0.5f && same_nb == 0) ? 1.0f : 0.0f;
    float trans_val = (m_val == 1) ? fminf(1.0f, is_bg + seed_val) : 0.0f;

    s_trans[y][x] = trans_val;

    bool is_border = (y == 0 || y == H - 1 || x == 0 || x == W - 1);
    if (!is_border && m_val == 1) {
        for (int d = 0; d < 8; ++d) {
            int ny = y + dy_arr[d];
            int nx = x + dx_arr[d];
            if (s_mask[ny][nx] == 0) {
                is_border = true;
            }
        }
    }

    s_phi[y][x] = (m_val == 1 && is_border) ? trans_val : 0.0f;
    __syncthreads();

    const int flood_steps = (H > W ? H : W) + 2;
    for (int step = 0; step < flood_steps; ++step) {
        float max_card = s_phi[y][x];
        if (y > 0)     max_card = fmaxf(max_card, s_phi[y - 1][x]);
        if (y < H - 1) max_card = fmaxf(max_card, s_phi[y + 1][x]);
        if (x > 0)     max_card = fmaxf(max_card, s_phi[y][x - 1]);
        if (x < W - 1) max_card = fmaxf(max_card, s_phi[y][x + 1]);
        __syncthreads();
        s_phi[y][x] = max_card * trans_val;
        __syncthreads();
    }

    float phi_final = s_phi[y][x];
    float m_cav_full = trans_val * (1.0f - phi_final);
    float m_cav_bg = fmaxf(0.0f, trans_val * (1.0f - seed_val)) * (1.0f - phi_final);

    s_seed[y][x] = seed_val * m_cav_full;
    __syncthreads();

    for (int s = 0; s < seed_iters; ++s) {
        float smax = s_seed[y][x];
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int ny = y + dy;
                int nx = x + dx;
                if (ny >= 0 && ny < H && nx >= 0 && nx < W) {
                    smax = fmaxf(smax, s_seed[ny][nx]);
                }
            }
        }
        __syncthreads();
        s_seed[y][x] = smax * m_cav_full;
        __syncthreads();
    }

    float ell_n = 1.0f, alive_n = 1.0f;
    for (int s = 1; s <= y; ++s) {
        alive_n *= s_trans[y - s][x];
        ell_n += alive_n;
    }

    float ell_s = 1.0f, alive_s = 1.0f;
    for (int s = 1; y + s < H; ++s) {
        alive_s *= s_trans[y + s][x];
        ell_s += alive_s;
    }

    float ell_w = 1.0f, alive_w = 1.0f;
    for (int s = 1; s <= x; ++s) {
        alive_w *= s_trans[y][x - s];
        ell_w += alive_w;
    }

    float ell_e = 1.0f, alive_e = 1.0f;
    for (int s = 1; x + s < W; ++s) {
        alive_e *= s_trans[y][x + s];
        ell_e += alive_e;
    }

    float mf = (float)m_val;
    float h_sp = fmaxf(1.0f, ell_n + ell_s - 1.0f) * mf;
    float w_sp = fmaxf(1.0f, ell_w + ell_e - 1.0f) * mf;
    float inv_sum = (1.0f / ell_n) + (1.0f / ell_s) + (1.0f / ell_w) + (1.0f / ell_e);
    float r_h = (4.0f / fmaxf(inv_sum, 1e-5f)) * mf;
    float ey_val = ((ell_n - ell_s) / fmaxf(h_sp, 1.0f)) * mf;
    float ex_val = ((ell_w - ell_e) / fmaxf(w_sp, 1.0f)) * mf;

    m_cav_ptr[base_off] = m_cav_bg * mf;
    h_span_ptr[base_off] = h_sp;
    w_span_ptr[base_off] = w_sp;
    r_harm_ptr[base_off] = r_h;
    ey_ptr[base_off] = ey_val;
    ex_ptr[base_off] = ex_val;
    seed_f_ptr[base_off] = s_seed[y][x] * mf;
    is_wall_ptr[base_off] = wall_val;
    is_seed_ptr[base_off] = seed_val;
}

std::vector<torch::Tensor> cuda_finsler_chordal_cavity_forward(
    torch::Tensor grid,
    torch::Tensor mask,
    int num_iterations
) {
    const int B = grid.size(0);
    const int H = grid.size(1);
    const int W = grid.size(2);
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(grid.device());

    auto m_cav = torch::empty({B, H, W}, opts);
    auto h_span = torch::empty({B, H, W}, opts);
    auto w_span = torch::empty({B, H, W}, opts);
    auto r_harm = torch::empty({B, H, W}, opts);
    auto ey = torch::empty({B, H, W}, opts);
    auto ex = torch::empty({B, H, W}, opts);
    auto seed_f = torch::empty({B, H, W}, opts);
    auto is_wall = torch::empty({B, H, W}, opts);
    auto is_seed = torch::empty({B, H, W}, opts);

    dim3 threads(W, H);
    dim3 blocks(B);
    int seed_iters = (num_iterations / 2 > 6) ? (num_iterations / 2) : 6;

    _finsler_chordal_cavity_sram_kernel<<<blocks, threads>>>(
        grid.data_ptr<int64_t>(),
        mask.data_ptr<bool>(),
        m_cav.data_ptr<float>(),
        h_span.data_ptr<float>(),
        w_span.data_ptr<float>(),
        r_harm.data_ptr<float>(),
        ey.data_ptr<float>(),
        ex.data_ptr<float>(),
        seed_f.data_ptr<float>(),
        is_wall.data_ptr<float>(),
        is_seed.data_ptr<float>(),
        H,
        W,
        seed_iters
    );

    return {m_cav, h_span, w_span, r_harm, ey, ex, seed_f, is_wall, is_seed};
}

__global__ void _octowave_8axis_line_parallel_fwd_kernel(
    const float* __restrict__ a_ptr,
    const float* __restrict__ b_ptr,
    float* __restrict__ h_ptr,
    const int B,
    const int H,
    const int W,
    const int D
) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int axis = threadIdx.y;
    const int line_idx = blockIdx.y;
    const int b_idx = blockIdx.z;
    if (d >= D || axis >= 8 || b_idx >= B) return;

    const int stride_d = D;
    const int stride_axis = 8 * D;
    const int stride_x = stride_axis;
    const int stride_y = W * stride_axis;
    const int base_b = b_idx * H * stride_y + axis * stride_d + d;

    if (axis == 0) {
        if (line_idx >= H) return;
        const int y = line_idx;
        float h_val = 0.0f;
        int row_off = base_b + y * stride_y;
        for (int x = 0; x < W; ++x) {
            int idx = row_off + x * stride_x;
            h_val = a_ptr[idx] * h_val + b_ptr[idx];
            h_ptr[idx] = h_val;
        }
    } else if (axis == 1) {
        if (line_idx >= H) return;
        const int y = line_idx;
        float h_val = 0.0f;
        int row_off = base_b + y * stride_y;
        for (int x = W - 1; x >= 0; --x) {
            int idx = row_off + x * stride_x;
            h_val = a_ptr[idx] * h_val + b_ptr[idx];
            h_ptr[idx] = h_val;
        }
    } else if (axis == 2) {
        if (line_idx >= W) return;
        const int x = line_idx;
        float h_val = 0.0f;
        int col_off = base_b + x * stride_x;
        for (int y = 0; y < H; ++y) {
            int idx = col_off + y * stride_y;
            h_val = a_ptr[idx] * h_val + b_ptr[idx];
            h_ptr[idx] = h_val;
        }
    } else if (axis == 3) {
        if (line_idx >= W) return;
        const int x = line_idx;
        float h_val = 0.0f;
        int col_off = base_b + x * stride_x;
        for (int y = H - 1; y >= 0; --y) {
            int idx = col_off + y * stride_y;
            h_val = a_ptr[idx] * h_val + b_ptr[idx];
            h_ptr[idx] = h_val;
        }
    } else if (axis == 4) {
        if (line_idx >= (H + W - 1)) return;
        const int k = line_idx - (H - 1);
        float h_val = 0.0f;
        int y_start = (0 > -k) ? 0 : -k;
        int y_end = (H < W - k) ? H : (W - k);
        for (int y = y_start; y < y_end; ++y) {
            int x = y + k;
            int idx = base_b + y * stride_y + x * stride_x;
            h_val = a_ptr[idx] * h_val + b_ptr[idx];
            h_ptr[idx] = h_val;
        }
    } else if (axis == 5) {
        if (line_idx >= (H + W - 1)) return;
        const int k = line_idx - (H - 1);
        float h_val = 0.0f;
        int y_start = (0 > -k) ? 0 : -k;
        int y_end = (H < W - k) ? H : (W - k);
        for (int y = y_end - 1; y >= y_start; --y) {
            int x = y + k;
            int idx = base_b + y * stride_y + x * stride_x;
            h_val = a_ptr[idx] * h_val + b_ptr[idx];
            h_ptr[idx] = h_val;
        }
    } else if (axis == 6) {
        if (line_idx >= (H + W - 1)) return;
        const int s = line_idx;
        float h_val = 0.0f;
        int y_start = (0 > s - (W - 1)) ? 0 : (s - (W - 1));
        int y_end = (H - 1 < s) ? (H - 1) : s;
        for (int y = y_start; y <= y_end; ++y) {
            int x = s - y;
            int idx = base_b + y * stride_y + x * stride_x;
            h_val = a_ptr[idx] * h_val + b_ptr[idx];
            h_ptr[idx] = h_val;
        }
    } else if (axis == 7) {
        if (line_idx >= (H + W - 1)) return;
        const int s = line_idx;
        float h_val = 0.0f;
        int y_start = (0 > s - (W - 1)) ? 0 : (s - (W - 1));
        int y_end = (H - 1 < s) ? (H - 1) : s;
        for (int y = y_end; y >= y_start; --y) {
            int x = s - y;
            int idx = base_b + y * stride_y + x * stride_x;
            h_val = a_ptr[idx] * h_val + b_ptr[idx];
            h_ptr[idx] = h_val;
        }
    }
}

__global__ void _octowave_8axis_line_parallel_bwd_kernel(
    const float* __restrict__ a_ptr,
    const float* __restrict__ h_ptr,
    const float* __restrict__ go_ptr,
    float* __restrict__ ga_ptr,
    float* __restrict__ gb_ptr,
    const int B,
    const int H,
    const int W,
    const int D
) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int axis = threadIdx.y;
    const int line_idx = blockIdx.y;
    const int b_idx = blockIdx.z;
    if (d >= D || axis >= 8 || b_idx >= B) return;

    const int stride_d = D;
    const int stride_axis = 8 * D;
    const int stride_x = stride_axis;
    const int stride_y = W * stride_axis;
    const int base_b = b_idx * H * stride_y + axis * stride_d + d;

    if (axis == 0) {
        if (line_idx >= H) return;
        const int y = line_idx;
        float gb_val = 0.0f;
        float a_next = 0.0f;
        int row_off = base_b + y * stride_y;
        for (int x = W - 1; x >= 0; --x) {
            int idx = row_off + x * stride_x;
            gb_val = go_ptr[idx] + a_next * gb_val;
            gb_ptr[idx] = gb_val;
            float h_prev = (x > 0) ? h_ptr[idx - stride_x] : 0.0f;
            ga_ptr[idx] = gb_val * h_prev;
            a_next = a_ptr[idx];
        }
    } else if (axis == 1) {
        if (line_idx >= H) return;
        const int y = line_idx;
        float gb_val = 0.0f;
        float a_next = 0.0f;
        int row_off = base_b + y * stride_y;
        for (int x = 0; x < W; ++x) {
            int idx = row_off + x * stride_x;
            gb_val = go_ptr[idx] + a_next * gb_val;
            gb_ptr[idx] = gb_val;
            float h_prev = (x < W - 1) ? h_ptr[idx + stride_x] : 0.0f;
            ga_ptr[idx] = gb_val * h_prev;
            a_next = a_ptr[idx];
        }
    } else if (axis == 2) {
        if (line_idx >= W) return;
        const int x = line_idx;
        float gb_val = 0.0f;
        float a_next = 0.0f;
        int col_off = base_b + x * stride_x;
        for (int y = H - 1; y >= 0; --y) {
            int idx = col_off + y * stride_y;
            gb_val = go_ptr[idx] + a_next * gb_val;
            gb_ptr[idx] = gb_val;
            float h_prev = (y > 0) ? h_ptr[idx - stride_y] : 0.0f;
            ga_ptr[idx] = gb_val * h_prev;
            a_next = a_ptr[idx];
        }
    } else if (axis == 3) {
        if (line_idx >= W) return;
        const int x = line_idx;
        float gb_val = 0.0f;
        float a_next = 0.0f;
        int col_off = base_b + x * stride_x;
        for (int y = 0; y < H; ++y) {
            int idx = col_off + y * stride_y;
            gb_val = go_ptr[idx] + a_next * gb_val;
            gb_ptr[idx] = gb_val;
            float h_prev = (y < H - 1) ? h_ptr[idx + stride_y] : 0.0f;
            ga_ptr[idx] = gb_val * h_prev;
            a_next = a_ptr[idx];
        }
    } else if (axis == 4) {
        if (line_idx >= (H + W - 1)) return;
        const int k = line_idx - (H - 1);
        float gb_val = 0.0f;
        float a_next = 0.0f;
        int y_start = (0 > -k) ? 0 : -k;
        int y_end = (H < W - k) ? H : (W - k);
        for (int y = y_end - 1; y >= y_start; --y) {
            int x = y + k;
            int idx = base_b + y * stride_y + x * stride_x;
            gb_val = go_ptr[idx] + a_next * gb_val;
            gb_ptr[idx] = gb_val;
            float h_prev = (y > y_start) ? h_ptr[idx - stride_y - stride_x] : 0.0f;
            ga_ptr[idx] = gb_val * h_prev;
            a_next = a_ptr[idx];
        }
    } else if (axis == 5) {
        if (line_idx >= (H + W - 1)) return;
        const int k = line_idx - (H - 1);
        float gb_val = 0.0f;
        float a_next = 0.0f;
        int y_start = (0 > -k) ? 0 : -k;
        int y_end = (H < W - k) ? H : (W - k);
        for (int y = y_start; y < y_end; ++y) {
            int x = y + k;
            int idx = base_b + y * stride_y + x * stride_x;
            gb_val = go_ptr[idx] + a_next * gb_val;
            gb_ptr[idx] = gb_val;
            float h_prev = (y < y_end - 1) ? h_ptr[idx + stride_y + stride_x] : 0.0f;
            ga_ptr[idx] = gb_val * h_prev;
            a_next = a_ptr[idx];
        }
    } else if (axis == 6) {
        if (line_idx >= (H + W - 1)) return;
        const int s = line_idx;
        float gb_val = 0.0f;
        float a_next = 0.0f;
        int y_start = (0 > s - (W - 1)) ? 0 : (s - (W - 1));
        int y_end = (H - 1 < s) ? (H - 1) : s;
        for (int y = y_end; y >= y_start; --y) {
            int x = s - y;
            int idx = base_b + y * stride_y + x * stride_x;
            gb_val = go_ptr[idx] + a_next * gb_val;
            gb_ptr[idx] = gb_val;
            float h_prev = (y > y_start) ? h_ptr[idx - stride_y + stride_x] : 0.0f;
            ga_ptr[idx] = gb_val * h_prev;
            a_next = a_ptr[idx];
        }
    } else if (axis == 7) {
        if (line_idx >= (H + W - 1)) return;
        const int s = line_idx;
        float gb_val = 0.0f;
        float a_next = 0.0f;
        int y_start = (0 > s - (W - 1)) ? 0 : (s - (W - 1));
        int y_end = (H - 1 < s) ? (H - 1) : s;
        for (int y = y_start; y <= y_end; ++y) {
            int x = s - y;
            int idx = base_b + y * stride_y + x * stride_x;
            gb_val = go_ptr[idx] + a_next * gb_val;
            gb_ptr[idx] = gb_val;
            float h_prev = (y < y_end) ? h_ptr[idx + stride_y - stride_x] : 0.0f;
            ga_ptr[idx] = gb_val * h_prev;
            a_next = a_ptr[idx];
        }
    }
}

torch::Tensor cuda_octowave_8axis_forward(
    torch::Tensor decay_gates,
    torch::Tensor impulse_gates
) {
    const int B = decay_gates.size(0);
    const int H = decay_gates.size(1);
    const int W = decay_gates.size(2);
    const int D = decay_gates.size(4);

    auto h_out = torch::empty_like(decay_gates);
    const int threads_d = 32;
    const int max_lines = H + W - 1;
    dim3 threads(threads_d, 8);
    dim3 blocks((D + threads_d - 1) / threads_d, max_lines, B);

    _octowave_8axis_line_parallel_fwd_kernel<<<blocks, threads>>>(
        decay_gates.data_ptr<float>(),
        impulse_gates.data_ptr<float>(),
        h_out.data_ptr<float>(),
        B, H, W, D
    );
    return h_out;
}

std::vector<torch::Tensor> cuda_octowave_8axis_backward(
    torch::Tensor decay_gates,
    torch::Tensor h_out,
    torch::Tensor grad_out
) {
    const int B = decay_gates.size(0);
    const int H = decay_gates.size(1);
    const int W = decay_gates.size(2);
    const int D = decay_gates.size(4);

    auto grad_a = torch::empty_like(decay_gates);
    auto grad_b = torch::empty_like(decay_gates);
    const int threads_d = 32;
    const int max_lines = H + W - 1;
    dim3 threads(threads_d, 8);
    dim3 blocks((D + threads_d - 1) / threads_d, max_lines, B);

    _octowave_8axis_line_parallel_bwd_kernel<<<blocks, threads>>>(
        decay_gates.data_ptr<float>(),
        h_out.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        grad_a.data_ptr<float>(),
        grad_b.data_ptr<float>(),
        B, H, W, D
    );
    return {grad_a, grad_b};
}

__global__ void _morlet_kan_fwd_kernel(
    const float* __restrict__ x_ptr,
    const float* __restrict__ centers_ptr,
    const float* __restrict__ scales_ptr,
    const float* __restrict__ weights_ptr,
    const float* __restrict__ phases_ptr,
    float* __restrict__ out_ptr,
    const int N,
    const int D,
    const int M,
    const float omega_0
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D) return;

    const int d = idx % D;
    const float x_val = x_ptr[idx];
    float accum = 0.0f;
    const int base_p = d * M;

    for (int m = 0; m < M; ++m) {
        const float mu = centers_ptr[base_p + m];
        const float sig = scales_ptr[base_p + m];
        const float w = weights_ptr[base_p + m];
        const float phi = phases_ptr[base_p + m];

        const float z = (x_val - mu) / sig;
        const float gauss = expf(-0.5f * z * z);
        const float carrier = cosf(omega_0 * z + phi);
        accum += w * carrier * gauss;
    }
    out_ptr[idx] = accum;
}

__global__ void _morlet_kan_bwd_dx_kernel(
    const float* __restrict__ go_ptr,
    const float* __restrict__ x_ptr,
    const float* __restrict__ centers_ptr,
    const float* __restrict__ scales_ptr,
    const float* __restrict__ weights_ptr,
    const float* __restrict__ phases_ptr,
    float* __restrict__ gx_ptr,
    const int N,
    const int D,
    const int M,
    const float omega_0
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D) return;

    const int d = idx % D;
    const float x_val = x_ptr[idx];
    const float go_val = go_ptr[idx];
    float accum_gx = 0.0f;
    const int base_p = d * M;

    for (int m = 0; m < M; ++m) {
        const float mu = centers_ptr[base_p + m];
        const float inv_sig = 1.0f / scales_ptr[base_p + m];
        const float w = weights_ptr[base_p + m];
        const float phi = phases_ptr[base_p + m];

        const float z = (x_val - mu) * inv_sig;
        const float gauss = expf(-0.5f * z * z);
        const float arg = omega_0 * z + phi;
        const float dpsi_dz = -gauss * (omega_0 * sinf(arg) + z * cosf(arg));
        accum_gx += go_val * w * dpsi_dz * inv_sig;
    }
    gx_ptr[idx] = accum_gx;
}

__global__ void _morlet_kan_bwd_params_kernel(
    const float* __restrict__ go_ptr,
    const float* __restrict__ x_ptr,
    const float* __restrict__ centers_ptr,
    const float* __restrict__ scales_ptr,
    const float* __restrict__ weights_ptr,
    const float* __restrict__ phases_ptr,
    float* __restrict__ gc_ptr,
    float* __restrict__ gs_ptr,
    float* __restrict__ gw_ptr,
    float* __restrict__ gp_ptr,
    const int N,
    const int D,
    const int M,
    const float omega_0
) {
    const int d = blockIdx.x;
    const int m = blockIdx.y;
    const int param_idx = d * M + m;

    const float mu = centers_ptr[param_idx];
    const float inv_sig = 1.0f / scales_ptr[param_idx];
    const float w = weights_ptr[param_idx];
    const float phi = phases_ptr[param_idx];

    float loc_gc = 0.0f;
    float loc_gs = 0.0f;
    float loc_gw = 0.0f;
    float loc_gp = 0.0f;

    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        const int idx = n * D + d;
        const float x_val = x_ptr[idx];
        const float go_val = go_ptr[idx];

        const float z = (x_val - mu) * inv_sig;
        const float gauss = expf(-0.5f * z * z);
        const float arg = omega_0 * z + phi;
        const float c_val = cosf(arg);
        const float s_val = sinf(arg);

        const float dpsi_dz = -gauss * (omega_0 * s_val + z * c_val);
        const float tension = go_val * w * dpsi_dz;

        loc_gc += -tension * inv_sig;
        loc_gs += -tension * z * inv_sig;
        loc_gw += go_val * c_val * gauss;
        loc_gp += -go_val * w * gauss * s_val;
    }

    __shared__ float s_gc[128];
    __shared__ float s_gs[128];
    __shared__ float s_gw[128];
    __shared__ float s_gp[128];

    const int tid = threadIdx.x;
    s_gc[tid] = loc_gc;
    s_gs[tid] = loc_gs;
    s_gw[tid] = loc_gw;
    s_gp[tid] = loc_gp;
    __syncthreads();

    for (int stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_gc[tid] += s_gc[tid + stride];
            s_gs[tid] += s_gs[tid + stride];
            s_gw[tid] += s_gw[tid + stride];
            s_gp[tid] += s_gp[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        gc_ptr[param_idx] = s_gc[0];
        gs_ptr[param_idx] = s_gs[0];
        gw_ptr[param_idx] = s_gw[0];
        gp_ptr[param_idx] = s_gp[0];
    }
}

torch::Tensor cuda_morlet_kan_forward(
    torch::Tensor x,
    torch::Tensor centers,
    torch::Tensor scales,
    torch::Tensor weights,
    torch::Tensor phases,
    double omega_0
) {
    const int N = x.size(0);
    const int D = x.size(1);
    const int M = centers.size(1);
    auto out = torch::empty_like(x);

    const int threads = 256;
    const int blocks = (N * D + threads - 1) / threads;

    _morlet_kan_fwd_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        centers.data_ptr<float>(),
        scales.data_ptr<float>(),
        weights.data_ptr<float>(),
        phases.data_ptr<float>(),
        out.data_ptr<float>(),
        N, D, M, (float)omega_0
    );
    return out;
}

std::vector<torch::Tensor> cuda_morlet_kan_backward(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor centers,
    torch::Tensor scales,
    torch::Tensor weights,
    torch::Tensor phases,
    double omega_0
) {
    const int N = x.size(0);
    const int D = x.size(1);
    const int M = centers.size(1);

    auto grad_x = torch::empty_like(x);
    auto grad_c = torch::empty_like(centers);
    auto grad_s = torch::empty_like(scales);
    auto grad_w = torch::empty_like(weights);
    auto grad_p = torch::empty_like(phases);

    const int threads_dx = 256;
    const int blocks_dx = (N * D + threads_dx - 1) / threads_dx;

    _morlet_kan_bwd_dx_kernel<<<blocks_dx, threads_dx>>>(
        grad_out.data_ptr<float>(),
        x.data_ptr<float>(),
        centers.data_ptr<float>(),
        scales.data_ptr<float>(),
        weights.data_ptr<float>(),
        phases.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        N, D, M, (float)omega_0
    );

    dim3 blocks_p(D, M);
    _morlet_kan_bwd_params_kernel<<<blocks_p, 128>>>(
        grad_out.data_ptr<float>(),
        x.data_ptr<float>(),
        centers.data_ptr<float>(),
        scales.data_ptr<float>(),
        weights.data_ptr<float>(),
        phases.data_ptr<float>(),
        grad_c.data_ptr<float>(),
        grad_s.data_ptr<float>(),
        grad_w.data_ptr<float>(),
        grad_p.data_ptr<float>(),
        N, D, M, (float)omega_0
    );

    return {grad_x, grad_c, grad_s, grad_w, grad_p};
}
"""


def get_cuda_fgns_extension() -> Optional[Any]:
    """
    Compiles and loads the Line-Parallel Native CUDA C++17 Shared-SRAM extension on first call.
    """
    global CUDA_EXTENSION_MODULE, CUDA_JIT_ATTEMPTED, CUDA_JIT_SUCCEEDED

    if CUDA_JIT_ATTEMPTED:
        return CUDA_EXTENSION_MODULE

    CUDA_JIT_ATTEMPTED = True
    if not is_cuda_kernel_image_usable():
        return None

    try:
        cap_major, _ = torch.cuda.get_device_capability()
        if cap_major < 8:
            return None
    except Exception:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _discover_msvc_and_configure_env()
            _configure_posix_cuda_and_triton_env()
            import tempfile
            from torch.utils.cpp_extension import load_inline

            try:
                build_dir = Path("target/cuda_cache").resolve()
                build_dir.mkdir(parents=True, exist_ok=True)
                probe_file = build_dir / ".write_probe"
                probe_file.write_text("ok", encoding="utf-8")
                probe_file.unlink(missing_ok=True)
            except Exception:
                build_dir = Path(tempfile.gettempdir()).resolve() / "fgns_cuda_cache"
                build_dir.mkdir(parents=True, exist_ok=True)

            host_cflags = ["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"]

            CUDA_EXTENSION_MODULE = load_inline(
                name="fgns_cuda_v4_3_native_kernels",
                cpp_sources=_CPP_DECLARATIONS,
                cuda_sources=_CUDA_SOURCES,
                functions=[
                    "cuda_finsler_chordal_cavity_forward",
                    "cuda_morlet_kan_forward",
                    "cuda_morlet_kan_backward",
                    "cuda_octowave_8axis_forward",
                    "cuda_octowave_8axis_backward",
                ],
                build_directory=str(build_dir),
                extra_cflags=host_cflags,
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                verbose=False,
            )
            CUDA_JIT_SUCCEEDED = True
    except Exception:
        CUDA_EXTENSION_MODULE = None
        CUDA_JIT_SUCCEEDED = False

    return CUDA_EXTENSION_MODULE


def _pytorch_finsler_chordal_cavity_fallback(
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
    """
    Closed-Form Cummax Optical Ray-Caster & Fast Vectorized Cavity Solver
    with Modal-Background & Isolated Singleton Seed Recognition.
    """
    B, H, W = grid.shape
    device = grid.device
    mask_f32 = mask.float().unsqueeze(-1)
    total_active = mask.float().sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)

    g_exp = grid.unsqueeze(-1).float()
    g_pad = F.pad(g_exp.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="constant", value=-1.0).permute(0, 2, 3, 1)
    m_pad_3d = F.pad(mask_f32.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="constant", value=0.0)
    m_pad = m_pad_3d.permute(0, 2, 3, 1)

    nb_dirs = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    nb_g_stack = torch.cat([g_pad[:, (1 + dy) : (1 + dy + H), (1 + dx) : (1 + dx + W), :] for dy, dx in nb_dirs], dim=-1)
    nb_m_stack = torch.cat([m_pad[:, (1 + dy) : (1 + dy + H), (1 + dx) : (1 + dx + W), :] for dy, dx in nb_dirs], dim=-1)
    same_nb = torch.sum((g_exp == nb_g_stack).float() * mask_f32 * nb_m_stack, dim=-1, keepdim=True)

    one_hot_10 = F.one_hot(grid.clamp(0, 9), num_classes=10).float() * mask_f32
    counts = one_hot_10.sum(dim=(1, 2))
    max_cnt, dom_c = torch.max(counts, dim=-1)
    dom_c_3d = dom_c.view(B, 1, 1)

    cnt_zero = counts[:, 0].view(B, 1, 1)
    has_substantial_zero_bg = (cnt_zero > 0.0) & ((cnt_zero * 5.0 >= total_active) | (cnt_zero == max_cnt.view(B, 1, 1)))
    bg_c = torch.where(has_substantial_zero_bg, torch.zeros_like(dom_c_3d), dom_c_3d)

    b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, H, W)
    px_color_count = counts[b_idx, grid.clamp(0, 9)].unsqueeze(-1)
    is_isolated_singleton = (mask_f32 > 0.5) & (same_nb < 0.5) & (px_color_count <= 4.0)

    is_bg = (((grid == bg_c) & mask).float().unsqueeze(-1)) * (1.0 - is_isolated_singleton.float())
    is_non_bg = (1.0 - is_bg) * mask_f32
    is_wall = is_non_bg * (same_nb >= 0.99).float()
    is_seed = is_non_bg * (same_nb < 0.99).float()
    is_trans = (is_bg + is_seed).clamp(0.0, 1.0) * mask_f32

    is_opaque = is_trans.squeeze(-1) < 0.5
    y_coords = torch.arange(H, device=device, dtype=torch.int64).view(1, H, 1).expand(B, H, W)
    x_coords = torch.arange(W, device=device, dtype=torch.int64).view(1, 1, W).expand(B, H, W)

    obs_y_n = torch.where(is_opaque, y_coords, torch.full_like(y_coords, -1))
    cummax_y_n = torch.cummax(obs_y_n, dim=1)[0]
    blocker_n = F.pad(cummax_y_n[:, :-1, :], (0, 0, 1, 0), mode="constant", value=-1)
    ell_n = ((y_coords - blocker_n).to(torch.float32).unsqueeze(-1) * mask_f32).clamp_min(1.0)

    obs_y_s = torch.where(is_opaque, y_coords, torch.full_like(y_coords, H))
    cummin_y_s = -torch.cummax(-torch.flip(obs_y_s, dims=[1]), dim=1)[0]
    cummin_y_s = torch.flip(cummin_y_s, dims=[1])
    blocker_s = F.pad(cummin_y_s[:, 1:, :], (0, 0, 0, 1), mode="constant", value=H)
    ell_s = ((blocker_s - y_coords).to(torch.float32).unsqueeze(-1) * mask_f32).clamp_min(1.0)

    obs_x_w = torch.where(is_opaque, x_coords, torch.full_like(x_coords, -1))
    cummax_x_w = torch.cummax(obs_x_w, dim=2)[0]
    blocker_w = F.pad(cummax_x_w[:, :, :-1], (1, 0, 0, 0), mode="constant", value=-1)
    ell_w = ((x_coords - blocker_w).to(torch.float32).unsqueeze(-1) * mask_f32).clamp_min(1.0)

    obs_x_e = torch.where(is_opaque, x_coords, torch.full_like(x_coords, W))
    cummin_x_e = -torch.cummax(-torch.flip(obs_x_e, dims=[2]), dim=2)[0]
    cummin_x_e = torch.flip(cummin_x_e, dims=[2])
    blocker_e = F.pad(cummin_x_e[:, :, 1:], (0, 1, 0, 0), mode="constant", value=W)
    ell_e = ((blocker_e - x_coords).to(torch.float32).unsqueeze(-1) * mask_f32).clamp_min(1.0)

    min_nb_mask = -F.max_pool2d(-m_pad_3d, kernel_size=3, stride=1, padding=0).permute(0, 2, 3, 1)
    is_active_border = (mask_f32 > 0.5) & (min_nb_mask < 0.5)

    phi_ext = is_active_border.float() * is_trans
    flood_steps = max(H, W) + 2

    for _ in range(flood_steps):
        p_pad = F.pad(phi_ext.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="constant", value=0.0)
        max_card = torch.maximum(
            torch.maximum(p_pad[:, :, :-2, 1:-1], p_pad[:, :, 2:, 1:-1]),
            torch.maximum(p_pad[:, :, 1:-1, :-2], p_pad[:, :, 1:-1, 2:]),
        ).permute(0, 2, 3, 1)
        phi_ext = torch.maximum(phi_ext, max_card) * is_trans

    m_cav_full = is_trans * (1.0 - phi_ext)
    m_cav_bg = (is_trans * (1.0 - is_seed)).clamp(0.0, 1.0) * (1.0 - phi_ext)

    h_span = (ell_n + ell_s - 1.0) * mask_f32
    w_span = (ell_w + ell_e - 1.0) * mask_f32
    inv_sum = (1.0 / ell_n) + (1.0 / ell_s) + (1.0 / ell_w) + (1.0 / ell_e)
    r_harm = (4.0 / inv_sum.clamp_min(1e-5)) * mask_f32

    ey = ((ell_n - ell_s) / h_span.clamp_min(1.0)) * mask_f32
    ex = ((ell_w - ell_e) / w_span.clamp_min(1.0)) * mask_f32

    seed_f = is_seed * m_cav_full
    for _ in range(max(6, num_iterations // 2)):
        s_pad = F.pad(seed_f.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="constant", value=0.0)
        s_max = F.max_pool2d(s_pad, kernel_size=3, stride=1, padding=0).permute(0, 2, 3, 1)
        seed_f = torch.maximum(seed_f, s_max) * m_cav_full

    return (
        m_cav_bg,
        h_span,
        w_span,
        r_harm,
        ey,
        ex,
        seed_f,
        is_wall,
        is_seed,
    )


def cuda_finsler_chordal_cavity_op(
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
    B, H, W = grid.shape
    ext = get_cuda_fgns_extension()
    if ext is not None and grid.is_cuda and H <= 32 and W <= 32:
        try:
            g_contig = grid.to(torch.int64).contiguous()
            m_contig = mask.to(torch.bool).contiguous()
            outs = ext.cuda_finsler_chordal_cavity_forward(g_contig, m_contig, int(num_iterations))
            return tuple(t.unsqueeze(-1) for t in outs)
        except Exception:
            pass

    return _pytorch_finsler_chordal_cavity_fallback(grid, mask, num_iterations=num_iterations)


class FusedCudaMorletFunction(torch.autograd.Function):
    """
    Differentiable Autograd Function executing Continuous Morlet Wavelet-KAN in Native CUDA C++ registers.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        centers: torch.Tensor,
        scales: torch.Tensor,
        weights: torch.Tensor,
        phases: torch.Tensor,
        omega_0: float = 5.0,
    ) -> torch.Tensor:
        target_dtype = x.dtype
        x_f32 = x.to(torch.float32).contiguous()
        c_f32 = centers.to(torch.float32).contiguous()
        s_f32 = scales.to(torch.float32).contiguous()
        w_f32 = weights.to(torch.float32).contiguous()
        p_f32 = phases.to(torch.float32).contiguous()

        ext = get_cuda_fgns_extension()
        out_f32: Optional[torch.Tensor] = None
        if ext is not None and x.is_cuda:
            try:
                out_f32 = ext.cuda_morlet_kan_forward(x_f32, c_f32, s_f32, w_f32, p_f32, float(omega_0))
            except Exception:
                out_f32 = None

        if out_f32 is None:
            z = (x_f32.unsqueeze(-1) - c_f32.unsqueeze(0)) / s_f32.unsqueeze(0)
            psi = torch.cos(float(omega_0) * z + p_f32.unsqueeze(0)) * torch.exp(-0.5 * z * z)
            out_f32 = torch.sum(psi * w_f32.unsqueeze(0), dim=-1)

        ctx.save_for_backward(x_f32, c_f32, s_f32, w_f32, p_f32)
        ctx.omega_0 = float(omega_0)
        return out_f32.to(target_dtype)

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        x_f32, c_f32, s_f32, w_f32, p_f32 = ctx.saved_tensors
        omega_0 = ctx.omega_0
        go_f32 = grad_output.to(torch.float32).contiguous()

        ext = get_cuda_fgns_extension()
        grads: Optional[List[torch.Tensor]] = None
        if ext is not None and grad_output.is_cuda:
            try:
                grads = ext.cuda_morlet_kan_backward(
                    go_f32, x_f32, c_f32, s_f32, w_f32, p_f32, float(omega_0)
                )
            except Exception:
                grads = None

        if grads is not None:
            gx, gc, gs, gw, gp = grads
        else:
            inv_s = 1.0 / s_f32.unsqueeze(0)
            z = (x_f32.unsqueeze(-1) - c_f32.unsqueeze(0)) * inv_s
            gauss = torch.exp(-0.5 * z * z)
            arg = omega_0 * z + p_f32.unsqueeze(0)
            c_val = torch.cos(arg)
            s_val = torch.sin(arg)
            dpsi_dz = -gauss * (omega_0 * s_val + z * c_val)
            tension = go_f32.unsqueeze(-1) * w_f32.unsqueeze(0) * dpsi_dz

            gx = torch.sum(tension * inv_s, dim=-1)
            gc = -torch.sum(tension * inv_s, dim=0)
            gs = -torch.sum(tension * z * inv_s, dim=0)
            gw = torch.sum(go_f32.unsqueeze(-1) * c_val * gauss, dim=0)
            gp = torch.sum(-go_f32.unsqueeze(-1) * w_f32.unsqueeze(0) * gauss * s_val, dim=0)

        return (
            gx.to(grad_output.dtype),
            gc.to(c_f32.dtype),
            gs.to(s_f32.dtype),
            gw.to(w_f32.dtype),
            gp.to(p_f32.dtype),
            None,
        )


def cuda_morlet_kan_op(
    x: torch.Tensor,
    centers: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor,
    phases: torch.Tensor,
    omega_0: float = 5.0,
) -> torch.Tensor:
    return FusedCudaMorletFunction.apply(x, centers, scales, weights, phases, omega_0)


FusedTritonMorletFunction = FusedCudaMorletFunction
triton_morlet_kan_op = cuda_morlet_kan_op


class AffineShearingGeometry:
    """
    Memoized 2D Broadcast Affine Transformation Engine (retained for CPU fallback).
    """

    _SHEAR_LUT_CACHE: Dict[Tuple[int, int, str], Tuple[torch.Tensor, torch.Tensor]] = {}

    @classmethod
    def _get_cached_luts(
        cls,
        h: int,
        w: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (h, w, str(device))
        if key in cls._SHEAR_LUT_CACHE:
            return cls._SHEAR_LUT_CACHE[key]

        with torch.no_grad():
            y_range = torch.arange(h, device=device, dtype=torch.int64).unsqueeze(1)
            x_range = torch.arange(w, device=device, dtype=torch.int64).unsqueeze(0)

            c_main = (x_range - y_range + (h - 1)).unsqueeze(0).unsqueeze(-1)
            c_anti = (x_range + y_range).unsqueeze(0).unsqueeze(-1)

            if len(cls._SHEAR_LUT_CACHE) < 32:
                cls._SHEAR_LUT_CACHE[key] = (c_main, c_anti)

        return c_main, c_anti

    @classmethod
    def shear_main_diagonal(cls, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        B, H, W, D = x.shape
        w_shear = H + W - 1
        c_main_base, _ = cls._get_cached_luts(H, W, x.device)
        c_indices = c_main_base.expand(B, H, W, D)
        sheared = torch.zeros((B, H, w_shear, D), dtype=x.dtype, device=x.device)
        sheared.scatter_(2, c_indices, x)
        return sheared, w_shear

    @classmethod
    def unshear_main_diagonal(cls, sheared: torch.Tensor, h_orig: int, w_orig: int) -> torch.Tensor:
        B, _, _, D = sheared.shape
        c_main_base, _ = cls._get_cached_luts(h_orig, w_orig, sheared.device)
        c_indices = c_main_base.expand(B, h_orig, w_orig, D)
        return torch.gather(sheared, 2, c_indices)

    @classmethod
    def shear_anti_diagonal(cls, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        B, H, W, D = x.shape
        w_shear = H + W - 1
        _, c_anti_base = cls._get_cached_luts(H, W, x.device)
        c_indices = c_anti_base.expand(B, H, W, D)
        sheared = torch.zeros((B, H, w_shear, D), dtype=x.dtype, device=x.device)
        sheared.scatter_(2, c_indices, x)
        return sheared, w_shear

    @classmethod
    def unshear_anti_diagonal(cls, sheared: torch.Tensor, h_orig: int, w_orig: int) -> torch.Tensor:
        B, _, _, D = sheared.shape
        _, c_anti_base = cls._get_cached_luts(h_orig, w_orig, sheared.device)
        c_indices = c_anti_base.expand(B, h_orig, w_orig, D)
        return torch.gather(sheared, 2, c_indices)


_TRIL_MASK_CACHE: Dict[Tuple[int, str], torch.Tensor] = {}


def _get_tril_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    key = (seq_len, str(device))
    if key not in _TRIL_MASK_CACHE:
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)).view(1, seq_len, seq_len, 1)
        if len(_TRIL_MASK_CACHE) < 32:
            _TRIL_MASK_CACHE[key] = mask
        return mask
    return _TRIL_MASK_CACHE[key]


def _closed_form_greens_scan_fwd(
    a: torch.Tensor,
    b: torch.Tensor,
    dim: int,
    reverse: bool,
) -> torch.Tensor:
    target_dtype = a.dtype
    a_perm = a.transpose(dim, -2).to(torch.float32)
    b_perm = b.transpose(dim, -2).to(torch.float32)

    if reverse:
        a_perm = torch.flip(a_perm, dims=[-2])
        b_perm = torch.flip(b_perm, dims=[-2])

    *batch_dims, L, D = a_perm.shape
    N = 1
    for bd in batch_dims:
        N *= bd

    a_flat = a_perm.reshape(N, L, D)
    b_flat = b_perm.reshape(N, L, D)

    log_a = torch.where(a_flat > 1e-9, torch.log(torch.clamp_min(a_flat, 1e-9)), torch.full_like(a_flat, -60.0))
    s_cum = torch.cumsum(log_a, dim=1)

    diff = s_cum.unsqueeze(2) - s_cum.unsqueeze(1)
    tril_mask = _get_tril_mask(L, a.device)
    diff_masked = torch.where(tril_mask, diff.clamp(min=-60.0, max=0.0), torch.full_like(diff, -80.0))
    g_prop = torch.exp(diff_masked)

    h_flat = torch.einsum("ntsd,nsd->ntd", g_prop, b_flat)
    h_perm = h_flat.view(*batch_dims, L, D)

    if reverse:
        h_perm = torch.flip(h_perm, dims=[-2])

    return h_perm.transpose(dim, -2).to(target_dtype)


def _closed_form_greens_scan_bwd(
    a: torch.Tensor,
    h: torch.Tensor,
    grad_out: torch.Tensor,
    dim: int,
    reverse: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    target_dtype = a.dtype
    a_perm = a.transpose(dim, -2).to(torch.float32)
    h_perm = h.transpose(dim, -2).to(torch.float32)
    go_perm = grad_out.transpose(dim, -2).to(torch.float32)

    if reverse:
        a_perm = torch.flip(a_perm, dims=[-2])
        h_perm = torch.flip(h_perm, dims=[-2])
        go_perm = torch.flip(go_perm, dims=[-2])

    *batch_dims, L, D = a_perm.shape
    N = 1
    for bd in batch_dims:
        N *= bd

    a_flat = a_perm.reshape(N, L, D)
    h_flat = h_perm.reshape(N, L, D)
    go_flat = go_perm.reshape(N, L, D)

    log_a = torch.where(a_flat > 1e-9, torch.log(torch.clamp_min(a_flat, 1e-9)), torch.full_like(a_flat, -60.0))
    s_cum = torch.cumsum(log_a, dim=1)

    diff = s_cum.unsqueeze(2) - s_cum.unsqueeze(1)
    tril_mask = _get_tril_mask(L, a.device)
    diff_masked = torch.where(tril_mask, diff.clamp(min=-60.0, max=0.0), torch.full_like(diff, -80.0))
    g_prop = torch.exp(diff_masked)

    grad_b_flat = torch.einsum("ntsd,ntd->nsd", g_prop, go_flat)

    h_prev = F.pad(h_flat[:, :-1, :], (0, 0, 1, 0), mode="constant", value=0.0)
    grad_a_flat = grad_b_flat * h_prev

    grad_a_perm = grad_a_flat.view(*batch_dims, L, D)
    grad_b_perm = grad_b_flat.view(*batch_dims, L, D)

    if reverse:
        grad_a_perm = torch.flip(grad_a_perm, dims=[-2])
        grad_b_perm = torch.flip(grad_b_perm, dims=[-2])

    return (
        grad_a_perm.transpose(dim, -2).to(target_dtype),
        grad_b_perm.transpose(dim, -2).to(target_dtype),
    )


class OctoWaveScanAutogradFunction(torch.autograd.Function):
    """
    Closed-Form Green's Propagator Autograd Function (CPU fallback).
    """

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        dim: int,
        reverse: bool,
    ) -> torch.Tensor:
        h = _closed_form_greens_scan_fwd(a, b, dim=dim, reverse=reverse)
        ctx.save_for_backward(a, h)
        ctx.dim = dim
        ctx.reverse = reverse
        return h

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, None, None]:
        a, h = ctx.saved_tensors
        grad_a, grad_b = _closed_form_greens_scan_bwd(
            a=a,
            h=h,
            grad_out=grad_output,
            dim=ctx.dim,
            reverse=ctx.reverse,
        )
        return grad_a, grad_b, None, None


class FusedCudaOctoWave8AxisFunction(torch.autograd.Function):
    """
    Line-Parallel Native CUDA C++17 8-Axis Isotropic Wavefront Scan Autograd Function.
    """

    @staticmethod
    def forward(
        ctx,
        decay_gates: torch.Tensor,
        impulse_gates: torch.Tensor,
    ) -> torch.Tensor:
        target_dtype = decay_gates.dtype
        a_f32 = decay_gates.to(torch.float32).contiguous()
        b_f32 = impulse_gates.to(torch.float32).contiguous()

        ext = get_cuda_fgns_extension()
        if ext is not None and decay_gates.is_cuda:
            try:
                h_f32 = ext.cuda_octowave_8axis_forward(a_f32, b_f32)
                ctx.save_for_backward(a_f32, h_f32)
                return h_f32.to(target_dtype)
            except Exception:
                pass

        B, H, W, _, D = a_f32.shape
        waves: List[torch.Tensor] = []
        waves.append(_closed_form_greens_scan_fwd(a_f32[:, :, :, 0, :], b_f32[:, :, :, 0, :], 2, False))
        waves.append(_closed_form_greens_scan_fwd(a_f32[:, :, :, 1, :], b_f32[:, :, :, 1, :], 2, True))
        waves.append(_closed_form_greens_scan_fwd(a_f32[:, :, :, 2, :], b_f32[:, :, :, 2, :], 1, False))
        waves.append(_closed_form_greens_scan_fwd(a_f32[:, :, :, 3, :], b_f32[:, :, :, 3, :], 1, True))

        d_m4, _ = AffineShearingGeometry.shear_main_diagonal(a_f32[:, :, :, 4, :])
        i_m4, _ = AffineShearingGeometry.shear_main_diagonal(b_f32[:, :, :, 4, :])
        waves.append(
            AffineShearingGeometry.unshear_main_diagonal(
                _closed_form_greens_scan_fwd(d_m4, i_m4, 1, False), H, W
            )
        )

        d_m5, _ = AffineShearingGeometry.shear_main_diagonal(a_f32[:, :, :, 5, :])
        i_m5, _ = AffineShearingGeometry.shear_main_diagonal(b_f32[:, :, :, 5, :])
        waves.append(
            AffineShearingGeometry.unshear_main_diagonal(
                _closed_form_greens_scan_fwd(d_m5, i_m5, 1, True), H, W
            )
        )

        d_a6, _ = AffineShearingGeometry.shear_anti_diagonal(a_f32[:, :, :, 6, :])
        i_a6, _ = AffineShearingGeometry.shear_anti_diagonal(b_f32[:, :, :, 6, :])
        waves.append(
            AffineShearingGeometry.unshear_anti_diagonal(
                _closed_form_greens_scan_fwd(d_a6, i_a6, 1, False), H, W
            )
        )

        d_a7, _ = AffineShearingGeometry.shear_anti_diagonal(a_f32[:, :, :, 7, :])
        i_a7, _ = AffineShearingGeometry.shear_anti_diagonal(b_f32[:, :, :, 7, :])
        waves.append(
            AffineShearingGeometry.unshear_anti_diagonal(
                _closed_form_greens_scan_fwd(d_a7, i_a7, 1, True), H, W
            )
        )

        h_f32 = torch.stack(waves, dim=3).contiguous()
        ctx.save_for_backward(a_f32, h_f32)
        return h_f32.to(target_dtype)

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        a_f32, h_f32 = ctx.saved_tensors
        go_f32 = grad_output.to(torch.float32).contiguous()

        ext = get_cuda_fgns_extension()
        if ext is not None and grad_output.is_cuda:
            try:
                ga_f32, gb_f32 = ext.cuda_octowave_8axis_backward(a_f32, h_f32, go_f32)
                return ga_f32.to(grad_output.dtype), gb_f32.to(grad_output.dtype)
            except Exception:
                pass

        B, H, W, _, D = a_f32.shape
        ga_list: List[torch.Tensor] = []
        gb_list: List[torch.Tensor] = []

        for ax, (dim, rev) in enumerate([(2, False), (2, True), (1, False), (1, True)]):
            ga, gb = _closed_form_greens_scan_bwd(
                a_f32[:, :, :, ax, :], h_f32[:, :, :, ax, :], go_f32[:, :, :, ax, :], dim, rev
            )
            ga_list.append(ga)
            gb_list.append(gb)

        for ax, rev in [(4, False), (5, True)]:
            da_s, _ = AffineShearingGeometry.shear_main_diagonal(a_f32[:, :, :, ax, :])
            dh_s, _ = AffineShearingGeometry.shear_main_diagonal(h_f32[:, :, :, ax, :])
            dgo_s, _ = AffineShearingGeometry.shear_main_diagonal(go_f32[:, :, :, ax, :])
            ga_s, gb_s = _closed_form_greens_scan_bwd(da_s, dh_s, dgo_s, 1, rev)
            ga_list.append(AffineShearingGeometry.unshear_main_diagonal(ga_s, H, W))
            gb_list.append(AffineShearingGeometry.unshear_main_diagonal(gb_s, H, W))

        for ax, rev in [(6, False), (7, True)]:
            da_s, _ = AffineShearingGeometry.shear_anti_diagonal(a_f32[:, :, :, ax, :])
            dh_s, _ = AffineShearingGeometry.shear_anti_diagonal(h_f32[:, :, :, ax, :])
            dgo_s, _ = AffineShearingGeometry.shear_anti_diagonal(go_f32[:, :, :, ax, :])
            ga_s, gb_s = _closed_form_greens_scan_bwd(da_s, dh_s, dgo_s, 1, rev)
            ga_list.append(AffineShearingGeometry.unshear_anti_diagonal(ga_s, H, W))
            gb_list.append(AffineShearingGeometry.unshear_anti_diagonal(gb_s, H, W))

        ga_f32 = torch.stack(ga_list, dim=3)
        gb_f32 = torch.stack(gb_list, dim=3)
        return ga_f32.to(grad_output.dtype), gb_f32.to(grad_output.dtype)


class OctoWaveSSMKernel(nn.Module):
    """
    Batched Line-Parallel 8-Axis Isotropic Wavefront State-Space Module
    with Boundary-Stopped Convex-EMA Discretization and Single-Pass Specular Wall Reflection.
    """

    REFLECT_HORIZ_WALL: Tuple[int, ...] = (0, 1, 3, 2, 7, 6, 5, 4)
    REFLECT_VERT_WALL: Tuple[int, ...] = (1, 0, 2, 3, 6, 7, 4, 5)
    REFLECT_CORNER_WALL: Tuple[int, ...] = (1, 0, 3, 2, 5, 4, 7, 6)

    def __init__(self, d_model: int, num_axes: int = 8, normal_router_weight: float = 0.35) -> None:
        super().__init__()
        if num_axes != 8:
            raise ValueError(f"Isotropy violation: OctoWave requires num_axes=8, got {num_axes}")

        self.d_model = d_model
        self.num_axes = num_axes
        self.normal_router_weight = normal_router_weight

        self.proj_decay = nn.Linear(d_model, num_axes * d_model, bias=True)
        self.proj_impulse = nn.Linear(d_model, num_axes * d_model, bias=True)

        self.specular_reflection_gate = nn.Parameter(torch.tensor(0.15, dtype=torch.float32))
        self.arbitration_gate = nn.Linear(d_model * num_axes, num_axes, bias=True)
        self.norm_output = RMSNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj_decay.weight, gain=0.5)
        nn.init.constant_(self.proj_decay.bias, 1.25)
        nn.init.xavier_uniform_(self.proj_impulse.weight, gain=0.5)
        nn.init.zeros_(self.proj_impulse.bias)
        nn.init.zeros_(self.arbitration_gate.weight)
        nn.init.zeros_(self.arbitration_gate.bias)

    def _compute_boundary_energies(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y_pad = F.pad(x, (0, 0, 0, 0, 1, 1), mode="replicate")
        x_pad = F.pad(x, (0, 0, 1, 1, 0, 0), mode="replicate")

        w_horiz = torch.tanh(torch.mean((y_pad[:, 2:, :, :] - y_pad[:, :-2, :, :]).pow(2), dim=-1, keepdim=True)).unsqueeze(3)
        w_vert = torch.tanh(torch.mean((x_pad[:, :, 2:, :] - x_pad[:, :, :-2, :]).pow(2), dim=-1, keepdim=True)).unsqueeze(3)
        return w_horiz, w_vert

    def _apply_specular_wall_reflection(
        self,
        x: torch.Tensor,
        incident_waves: torch.Tensor,
        w_horiz: torch.Tensor,
        w_vert: torch.Tensor,
        mask_5d: torch.Tensor,
    ) -> torch.Tensor:
        w_corner = w_horiz * w_vert
        ref_h = incident_waves[:, :, :, self.REFLECT_HORIZ_WALL, :]
        ref_v = incident_waves[:, :, :, self.REFLECT_VERT_WALL, :]
        ref_c = incident_waves[:, :, :, self.REFLECT_CORNER_WALL, :]

        reflected = (w_horiz * ref_h) + (w_vert * ref_v) + (0.50 * w_corner * ref_c)
        color_modulated = reflected * (1.0 + 0.25 * torch.tanh(x.unsqueeze(3)))
        return color_modulated * mask_5d

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        normal_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, H, W, D = x.shape
        mask_f32 = mask.to(x.dtype).unsqueeze(-1)
        mask_5d = mask_f32.unsqueeze(3)

        w_horiz, w_vert = self._compute_boundary_energies(x)
        edge_stop = torch.exp(-0.45 * (w_horiz + w_vert))

        decay_raw = self.proj_decay(x).view(B, H, W, self.num_axes, D)
        impulse_raw = self.proj_impulse(x).view(B, H, W, self.num_axes, D)

        decay_gates = (torch.sigmoid(decay_raw) * edge_stop) * mask_5d
        convex_norm = 1.0 - (0.75 * decay_gates)
        impulse_gates = (F.silu(impulse_raw) * convex_norm) * mask_5d

        waves_scanned = FusedCudaOctoWave8AxisFunction.apply(decay_gates, impulse_gates) * mask_5d

        g_spec = torch.tanh(self.specular_reflection_gate.to(x.dtype))
        post_spec = self._apply_specular_wall_reflection(x, waves_scanned, w_horiz, w_vert, mask_5d)
        stacked_waves = (waves_scanned + (0.35 * g_spec * post_spec)) * mask_5d

        flat_waves = stacked_waves.view(B, H, W, self.num_axes * D)
        arbitration_logits = self.arbitration_gate(flat_waves)

        if normal_bias is not None:
            arbitration_logits = arbitration_logits + (
                normal_bias.to(arbitration_logits.dtype) * self.normal_router_weight
            )

        arbitration_weights = F.softmax(arbitration_logits, dim=-1).unsqueeze(-1)
        fused_wave = torch.sum(stacked_waves * arbitration_weights, dim=3)

        output = self.norm_output(x + fused_wave)
        return output * mask_f32


def verify_cuda_fgns_ops_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for cuda_fgns_ops.py.
    """
    torch.manual_seed(42)
    device = _get_usable_cuda_or_cpu_device()

    ring_grid = torch.zeros((1, 7, 7), dtype=torch.int64, device=device)
    ring_grid[0, 1:6, 1] = 2
    ring_grid[0, 1:6, 5] = 2
    ring_grid[0, 1, 1:6] = 2
    ring_grid[0, 5, 1:6] = 2
    ring_grid[0, 3, 3] = 2
    ring_mask = torch.ones((1, 7, 7), dtype=torch.bool, device=device)

    (
        m_cav,
        h_sp,
        w_sp,
        _,
        _,
        _,
        seed_f,
        _,
        is_seed,
    ) = cuda_finsler_chordal_cavity_op(ring_grid, ring_mask, num_iterations=12)

    if float(m_cav[0, 0, 0, 0].item()) != 0.0:
        return False
    if float(m_cav[0, 2, 2, 0].item()) < 0.99:
        return False
    if abs(float(h_sp[0, 2, 2, 0].item()) - 3.0) > 1e-4 or abs(float(w_sp[0, 2, 2, 0].item()) - 3.0) > 1e-4:
        return False
    if float(seed_f[0, 2, 2, 0].item()) < 0.99 or float(is_seed[0, 3, 3, 0].item()) < 0.99:
        return False

    """Verify Task 13e47133 Pair 2 invariant: isolated black seed 0 inside yellow background 4"""
    yellow_room = torch.full((1, 6, 6), 4, dtype=torch.int64, device=device)
    yellow_room[0, 2, 2] = 0
    yellow_mask = torch.ones((1, 6, 6), dtype=torch.bool, device=device)
    _, _, _, _, _, _, _, is_wall_y, is_seed_y = cuda_finsler_chordal_cavity_op(yellow_room, yellow_mask, num_iterations=10)
    if float(is_seed_y[0, 2, 2, 0].item()) < 0.99 or float(is_wall_y[0, 0, 0, 0].item()) > 0.01:
        return False

    n_tok, d_in, m_wav = 16, 32, 4
    x_kan = torch.randn((n_tok, d_in), dtype=torch.float32, device=device, requires_grad=True)
    c_kan = torch.randn((d_in, m_wav), dtype=torch.float32, device=device, requires_grad=True)
    s_kan = (torch.rand((d_in, m_wav), dtype=torch.float32, device=device) + 0.5).requires_grad_(True)
    w_kan = torch.randn((d_in, m_wav), dtype=torch.float32, device=device, requires_grad=True)
    p_kan = torch.randn((d_in, m_wav), dtype=torch.float32, device=device, requires_grad=True)

    y_kan = cuda_morlet_kan_op(x_kan, c_kan, s_kan, w_kan, p_kan, omega_0=5.0)
    if y_kan.shape != (n_tok, d_in) or torch.isnan(y_kan).any():
        return False

    loss_kan = torch.sum(y_kan.pow(2))
    loss_kan.backward()
    for t in (x_kan, c_kan, s_kan, w_kan, p_kan):
        if t.grad is None or torch.isnan(t.grad).any() or float(torch.norm(t.grad).item()) <= 0.0:
            return False

    octo = OctoWaveSSMKernel(d_model=32, num_axes=8).to(device)
    x_octo = torch.randn((2, 6, 6, 32), dtype=torch.float32, device=device, requires_grad=True)
    m_octo = torch.ones((2, 6, 6), dtype=torch.bool, device=device)
    m_octo[:, :, 3:] = False

    out_octo = octo(x_octo, m_octo)
    if not torch.all(out_octo[:, :, 3:, :] == 0.0):
        return False

    loss_octo = torch.sum(out_octo.pow(2))
    loss_octo.backward()
    if x_octo.grad is None or torch.isnan(x_octo.grad).any():
        return False

    return True
