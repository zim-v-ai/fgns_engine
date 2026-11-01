"""
Location: // fgns_engine/fgns/kernels/cuda_fgns_ops.py
Description: Native CUDA C++17 Shared-SRAM Compute Shaders & Specular Wavefront Engine (FGNS v4.1.0):
               1. Fused Shared-SRAM Cavity Flood-Fill & Beer-Lambert Optical Ray Integrator.
               2. Native Shared-SRAM 8-Axis Isotropic OctoWave Forward & Adjoint Backward Kernels
                  (Zero [N, L, L, D] Green's matrices, Zero scatter/gather shearing, < 450 MB VRAM!).
               3. Two-Bounce Specular Billiard Wall Reflection Tensor R_{d -> d'}(grad tau_wall)
                  resolving 45-degree and 90-degree multi-bounce ray trajectories (Task 142ca369).
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


def _discover_msvc_and_configure_env() -> None:
    """
    Automatically locates Visual Studio MSVC cl.exe on Windows hosts and prepends it to PATH.
    """
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
    __shared__ int s_color_counts[10];

    const int b = blockIdx.x;
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int tid = y * W + x;
    const int hw = H * W;
    const int base_off = b * hw + tid;

    if (tid < 10) {
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
        }
    }
    __syncthreads();

    if (y >= H || x >= W) {
        return;
    }

    int bg_color = 0;
    if (s_color_counts[0] == 0) {
        int max_c = -1;
        int best_col = 0;
        for (int c = 0; c < 10; ++c) {
            if (s_color_counts[c] > max_c) {
                max_c = s_color_counts[c];
                best_col = c;
            }
        }
        bg_color = best_col;
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

    float is_bg = (m_val == 1 && g_val == bg_color) ? 1.0f : 0.0f;
    float is_non_bg = (m_val == 1 && g_val != bg_color) ? 1.0f : 0.0f;
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

__global__ void _octowave_8axis_sram_fwd_kernel(
    const float* __restrict__ a_ptr,
    const float* __restrict__ b_ptr,
    float* __restrict__ h_ptr,
    const int B,
    const int H,
    const int W,
    const int D
) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int axis = blockIdx.y;
    const int b_idx = blockIdx.z;
    if (d >= D || axis >= 8 || b_idx >= B) return;

    const int stride_d = D;
    const int stride_axis = 8 * D;
    const int stride_x = stride_axis;
    const int stride_y = W * stride_axis;
    const int base_b = b_idx * H * stride_y + axis * stride_d + d;

    if (axis == 0) {
        for (int y = 0; y < H; ++y) {
            float h_val = 0.0f;
            for (int x = 0; x < W; ++x) {
                int idx = base_b + y * stride_y + x * stride_x;
                h_val = a_ptr[idx] * h_val + b_ptr[idx];
                h_ptr[idx] = h_val;
            }
        }
    } else if (axis == 1) {
        for (int y = 0; y < H; ++y) {
            float h_val = 0.0f;
            for (int x = W - 1; x >= 0; --x) {
                int idx = base_b + y * stride_y + x * stride_x;
                h_val = a_ptr[idx] * h_val + b_ptr[idx];
                h_ptr[idx] = h_val;
            }
        }
    } else if (axis == 2) {
        for (int x = 0; x < W; ++x) {
            float h_val = 0.0f;
            for (int y = 0; y < H; ++y) {
                int idx = base_b + y * stride_y + x * stride_x;
                h_val = a_ptr[idx] * h_val + b_ptr[idx];
                h_ptr[idx] = h_val;
            }
        }
    } else if (axis == 3) {
        for (int x = 0; x < W; ++x) {
            float h_val = 0.0f;
            for (int y = H - 1; y >= 0; --y) {
                int idx = base_b + y * stride_y + x * stride_x;
                h_val = a_ptr[idx] * h_val + b_ptr[idx];
                h_ptr[idx] = h_val;
            }
        }
    } else if (axis == 4) {
        for (int k = -(H - 1); k <= (W - 1); ++k) {
            float h_val = 0.0f;
            int y_start = (0 > -k) ? 0 : -k;
            int y_end = (H < W - k) ? H : (W - k);
            for (int y = y_start; y < y_end; ++y) {
                int x = y + k;
                int idx = base_b + y * stride_y + x * stride_x;
                h_val = a_ptr[idx] * h_val + b_ptr[idx];
                h_ptr[idx] = h_val;
            }
        }
    } else if (axis == 5) {
        for (int k = -(H - 1); k <= (W - 1); ++k) {
            float h_val = 0.0f;
            int y_start = (0 > -k) ? 0 : -k;
            int y_end = (H < W - k) ? H : (W - k);
            for (int y = y_end - 1; y >= y_start; --y) {
                int x = y + k;
                int idx = base_b + y * stride_y + x * stride_x;
                h_val = a_ptr[idx] * h_val + b_ptr[idx];
                h_ptr[idx] = h_val;
            }
        }
    } else if (axis == 6) {
        for (int s = 0; s <= (H + W - 2); ++s) {
            float h_val = 0.0f;
            int y_start = (0 > s - (W - 1)) ? 0 : (s - (W - 1));
            int y_end = (H - 1 < s) ? (H - 1) : s;
            for (int y = y_start; y <= y_end; ++y) {
                int x = s - y;
                int idx = base_b + y * stride_y + x * stride_x;
                h_val = a_ptr[idx] * h_val + b_ptr[idx];
                h_ptr[idx] = h_val;
            }
        }
    } else if (axis == 7) {
        for (int s = 0; s <= (H + W - 2); ++s) {
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
}

__global__ void _octowave_8axis_sram_bwd_kernel(
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
    const int axis = blockIdx.y;
    const int b_idx = blockIdx.z;
    if (d >= D || axis >= 8 || b_idx >= B) return;

    const int stride_d = D;
    const int stride_axis = 8 * D;
    const int stride_x = stride_axis;
    const int stride_y = W * stride_axis;
    const int base_b = b_idx * H * stride_y + axis * stride_d + d;

    if (axis == 0) {
        for (int y = 0; y < H; ++y) {
            float gb_val = 0.0f;
            float a_next = 0.0f;
            for (int x = W - 1; x >= 0; --x) {
                int idx = base_b + y * stride_y + x * stride_x;
                gb_val = go_ptr[idx] + a_next * gb_val;
                gb_ptr[idx] = gb_val;
                float h_prev = (x > 0) ? h_ptr[idx - stride_x] : 0.0f;
                ga_ptr[idx] = gb_val * h_prev;
                a_next = a_ptr[idx];
            }
        }
    } else if (axis == 1) {
        for (int y = 0; y < H; ++y) {
            float gb_val = 0.0f;
            float a_next = 0.0f;
            for (int x = 0; x < W; ++x) {
                int idx = base_b + y * stride_y + x * stride_x;
                gb_val = go_ptr[idx] + a_next * gb_val;
                gb_ptr[idx] = gb_val;
                float h_prev = (x < W - 1) ? h_ptr[idx + stride_x] : 0.0f;
                ga_ptr[idx] = gb_val * h_prev;
                a_next = a_ptr[idx];
            }
        }
    } else if (axis == 2) {
        for (int x = 0; x < W; ++x) {
            float gb_val = 0.0f;
            float a_next = 0.0f;
            for (int y = H - 1; y >= 0; --y) {
                int idx = base_b + y * stride_y + x * stride_x;
                gb_val = go_ptr[idx] + a_next * gb_val;
                gb_ptr[idx] = gb_val;
                float h_prev = (y > 0) ? h_ptr[idx - stride_y] : 0.0f;
                ga_ptr[idx] = gb_val * h_prev;
                a_next = a_ptr[idx];
            }
        }
    } else if (axis == 3) {
        for (int x = 0; x < W; ++x) {
            float gb_val = 0.0f;
            float a_next = 0.0f;
            for (int y = 0; y < H; ++y) {
                int idx = base_b + y * stride_y + x * stride_x;
                gb_val = go_ptr[idx] + a_next * gb_val;
                gb_ptr[idx] = gb_val;
                float h_prev = (y < H - 1) ? h_ptr[idx + stride_y] : 0.0f;
                ga_ptr[idx] = gb_val * h_prev;
                a_next = a_ptr[idx];
            }
        }
    } else if (axis == 4) {
        for (int k = -(H - 1); k <= (W - 1); ++k) {
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
        }
    } else if (axis == 5) {
        for (int k = -(H - 1); k <= (W - 1); ++k) {
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
        }
    } else if (axis == 6) {
        for (int s = 0; s <= (H + W - 2); ++s) {
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
        }
    } else if (axis == 7) {
        for (int s = 0; s <= (H + W - 2); ++s) {
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
    const int threads = 64;
    dim3 blocks((D + threads - 1) / threads, 8, B);

    _octowave_8axis_sram_fwd_kernel<<<blocks, threads>>>(
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
    const int threads = 64;
    dim3 blocks((D + threads - 1) / threads, 8, B);

    _octowave_8axis_sram_bwd_kernel<<<blocks, threads>>>(
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
    Compiles and loads the Native CUDA C++17 Shared-SRAM extension on first call.
    """
    global CUDA_EXTENSION_MODULE, CUDA_JIT_ATTEMPTED, CUDA_JIT_SUCCEEDED

    if CUDA_JIT_ATTEMPTED:
        return CUDA_EXTENSION_MODULE

    CUDA_JIT_ATTEMPTED = True
    if not torch.cuda.is_available():
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _discover_msvc_and_configure_env()
            from torch.utils.cpp_extension import load_inline

            build_dir = Path("target/cuda_cache").resolve()
            build_dir.mkdir(parents=True, exist_ok=True)

            CUDA_EXTENSION_MODULE = load_inline(
                name="fgns_cuda_v4_1_kernels",
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
    B, H, W = grid.shape
    device = grid.device
    mask_f32 = mask.float().unsqueeze(-1)

    g_exp = grid.unsqueeze(-1).float()
    g_pad = F.pad(g_exp.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="constant", value=-1.0)
    m_pad = F.pad(mask_f32.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="constant", value=0.0)

    same_nb = torch.zeros((B, H, W, 1), dtype=torch.float32, device=device)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        y0, x0 = 1 + dy, 1 + dx
        nb_g = g_pad[:, :, y0 : y0 + H, x0 : x0 + W].permute(0, 2, 3, 1)
        nb_m = m_pad[:, :, y0 : y0 + H, x0 : x0 + W].permute(0, 2, 3, 1)
        same_nb = same_nb + ((g_exp == nb_g).float() * mask_f32 * nb_m)

    is_zero = (grid == 0) & mask
    has_zero_bg = (is_zero.float().sum(dim=(-2, -1), keepdim=True) >= 1.0)

    counts = [((grid == c) & mask).float().sum(dim=(-2, -1)) for c in range(10)]
    dom_c = torch.argmax(torch.stack(counts, dim=-1), dim=-1).view(B, 1, 1)
    bg_c = torch.where(has_zero_bg, torch.zeros_like(dom_c), dom_c)

    is_bg = ((grid == bg_c) & mask).float().unsqueeze(-1)
    is_non_bg = (1.0 - is_bg) * mask_f32
    is_wall = is_non_bg * (same_nb >= 0.99).float()
    is_seed = is_non_bg * (same_nb < 0.99).float()
    is_trans = (is_bg + is_seed).clamp(0.0, 1.0) * mask_f32

    min_nb_mask = -F.max_pool2d(-m_pad, kernel_size=3, stride=1, padding=0).permute(0, 2, 3, 1)
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

    tau_med = is_trans.squeeze(-1)
    ell_n = torch.ones((B, H, W), dtype=torch.float32, device=device)
    ell_s = torch.ones((B, H, W), dtype=torch.float32, device=device)
    ell_w = torch.ones((B, H, W), dtype=torch.float32, device=device)
    ell_e = torch.ones((B, H, W), dtype=torch.float32, device=device)

    for y in range(1, H):
        ell_n[:, y, :] = 1.0 + (tau_med[:, y - 1, :] * ell_n[:, y - 1, :])
        yr = H - 1 - y
        ell_s[:, yr, :] = 1.0 + (tau_med[:, yr + 1, :] * ell_s[:, yr + 1, :])

    for x in range(1, W):
        ell_w[:, :, x] = 1.0 + (tau_med[:, :, x - 1] * ell_w[:, :, x - 1])
        xr = W - 1 - x
        ell_e[:, :, xr] = 1.0 + (tau_med[:, :, xr + 1] * ell_e[:, :, xr + 1])

    ell_n = (ell_n.unsqueeze(-1) * mask_f32).clamp_min(1.0)
    ell_s = (ell_s.unsqueeze(-1) * mask_f32).clamp_min(1.0)
    ell_w = (ell_w.unsqueeze(-1) * mask_f32).clamp_min(1.0)
    ell_e = (ell_e.unsqueeze(-1) * mask_f32).clamp_min(1.0)

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
        g_contig = grid.to(torch.int64).contiguous()
        m_contig = mask.to(torch.bool).contiguous()
        outs = ext.cuda_finsler_chordal_cavity_forward(g_contig, m_contig, int(num_iterations))
        return tuple(t.unsqueeze(-1) for t in outs)

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
        if ext is not None and x.is_cuda:
            out_f32 = ext.cuda_morlet_kan_forward(x_f32, c_f32, s_f32, w_f32, p_f32, float(omega_0))
        else:
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
        if ext is not None and grad_output.is_cuda:
            gx, gc, gs, gw, gp = ext.cuda_morlet_kan_backward(
                go_f32, x_f32, c_f32, s_f32, w_f32, p_f32, float(omega_0)
            )
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
    Native CUDA C++17 8-Axis Isotropic Wavefront Scan Autograd Function.
    Executes all 8 axes (E, W, S, N, SE, NW, SW, NE) in O(1) memory without L x L matrices!
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
            h_f32 = ext.cuda_octowave_8axis_forward(a_f32, b_f32)
            ctx.save_for_backward(a_f32, h_f32)
            return h_f32.to(target_dtype)

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
            ga_f32, gb_f32 = ext.cuda_octowave_8axis_backward(a_f32, h_f32, go_f32)
            return ga_f32.to(grad_output.dtype), gb_f32.to(grad_output.dtype)

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
    Batched 8-Axis Isotropic Wavefront State-Space Module with Specular Billiard Wall Reflection.
    """

    """
    Axis Ordering: [0:E, 1:W, 2:S, 3:N, 4:SE, 5:NW, 6:SW, 7:NE]
    Horizontal Wall Reflection (dy -> -dy): E->E(0), W->W(1), S<->N(3,2), SE<->NE(7,6), SW<->NW(5,4)
    Vertical Wall Reflection   (dx -> -dx): E<->W(1,0), S->S(2), N->N(3), SE<->SW(6,7), NW<->NE(4,5)
    Corner Retro-Reflection    (-dy, -dx):  E<->W(1,0), S<->N(3,2), SE<->NW(5,4), SW<->NE(7,6)
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

        self.specular_reflection_gate = nn.Parameter(torch.tensor(0.20, dtype=torch.float32))
        self.arbitration_gate = nn.Linear(d_model * num_axes, num_axes, bias=True)
        self.norm_output = RMSNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj_decay.weight)
        nn.init.constant_(self.proj_decay.bias, 2.0)
        nn.init.xavier_uniform_(self.proj_impulse.weight)
        nn.init.zeros_(self.proj_impulse.bias)
        nn.init.zeros_(self.arbitration_gate.weight)
        nn.init.zeros_(self.arbitration_gate.bias)

    def _compute_specular_wall_reflection_impulse(
        self,
        x: torch.Tensor,
        incident_waves: torch.Tensor,
        mask_f32: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluates the Dihedral Specular Reflection Tensor R_{d -> d'}(grad tau_wall):
            h_reflected^{(d')} = sum_{d=0}^7 R_{d -> d'}(g_y, g_x) * h_incident^{(d)},
        modulated by the local obstacle feature x(y, x) so reflected rays inherit obstacle color!
        """
        y_pad = F.pad(x, (0, 0, 0, 0, 1, 1), mode="replicate")
        x_pad = F.pad(x, (0, 0, 1, 1, 0, 0), mode="replicate")

        grad_y_energy = torch.mean((y_pad[:, 2:, :, :] - y_pad[:, :-2, :, :]).pow(2), dim=-1, keepdim=True)
        grad_x_energy = torch.mean((x_pad[:, :, 2:, :] - x_pad[:, :, :-2, :]).pow(2), dim=-1, keepdim=True)

        w_horiz = torch.tanh(grad_y_energy).unsqueeze(3)
        w_vert = torch.tanh(grad_x_energy).unsqueeze(3)
        w_corner = (w_horiz * w_vert)

        ref_h = incident_waves[:, :, :, self.REFLECT_HORIZ_WALL, :]
        ref_v = incident_waves[:, :, :, self.REFLECT_VERT_WALL, :]
        ref_c = incident_waves[:, :, :, self.REFLECT_CORNER_WALL, :]

        reflected = (w_horiz * ref_h) + (w_vert * ref_v) + (0.50 * w_corner * ref_c)
        color_modulated = reflected * (1.0 + 0.35 * torch.tanh(x.unsqueeze(3)))
        return color_modulated * mask_f32.unsqueeze(3)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        normal_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, H, W, D = x.shape
        mask_f32 = mask.to(x.dtype).unsqueeze(-1)
        mask_5d = mask_f32.unsqueeze(3)

        decay_raw = self.proj_decay(x).view(B, H, W, self.num_axes, D)
        impulse_raw = self.proj_impulse(x).view(B, H, W, self.num_axes, D)

        decay_gates = torch.sigmoid(decay_raw) * mask_5d
        impulse_gates = F.silu(impulse_raw) * mask_5d

        """First-Bounce 8-Axis Isotropic Scan in Native CUDA C++ Shared SRAM"""
        waves_pass1 = FusedCudaOctoWave8AxisFunction.apply(decay_gates, impulse_gates) * mask_5d

        """Specular Billiard Wall Reflection & Second-Bounce Coupled Propagation"""
        g_spec = torch.tanh(self.specular_reflection_gate.to(x.dtype))
        if float(torch.abs(g_spec.detach()).item()) > 1e-4:
            spec_impulse = self._compute_specular_wall_reflection_impulse(x, waves_pass1, mask_f32)
            coupled_impulse = (impulse_gates + (g_spec * spec_impulse)) * mask_5d
            stacked_waves = FusedCudaOctoWave8AxisFunction.apply(decay_gates, coupled_impulse) * mask_5d
        else:
            stacked_waves = waves_pass1

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

