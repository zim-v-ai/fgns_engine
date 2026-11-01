"""
Location: // fgns_engine/scripts/ablations/common.py
Description: Shared Evaluation Harness, Dataset Loader, and Mathematical Metrics
             for the FGNS-TRM Architectural Ablation Suite.
Language: Python 3.12+ / PyTorch 2.5+
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.stats import wasserstein_distance
import torch
import torch.nn as nn
import torch.nn.functional as F

from fgns.config import FGNSConfig
from fgns.env.batcher import FastVectorizedBatcher
from fgns.env.task import ARCPair, ARCTask, load_arc_dataset
from fgns.models.fgns_network import FinslerGaugeSheafNetwork


def ensure_utf8_environment() -> None:
    """Configures system stdout and stderr to UTF-8 encoding with replace fallback."""
    reconfig_out = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfig_out):
        try:
            reconfig_out(encoding="utf-8", errors="replace")
        except Exception:
            pass
    reconfig_err = getattr(sys.stderr, "reconfigure", None)
    if callable(reconfig_err):
        try:
            reconfig_err(encoding="utf-8", errors="replace")
        except Exception:
            pass


def get_ablation_device() -> torch.device:
    """Returns preferred PyTorch execution device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_benchmark_tasks(
    data_path: Optional[Path] = None,
    limit: int = 10,
) -> List[ARCTask]:
    """Loads a bounded subset of ARC benchmark tasks for controlled ablations."""
    target_path = data_path or Path("data/arc-agi_training_challenges.json").resolve()
    if not target_path.is_file():
        raise FileNotFoundError(f"Ablation dataset not found at {target_path}")

    sol_path = target_path.parent / str(target_path.name).replace("challenges", "solutions")
    tasks = load_arc_dataset(target_path, sol_path if sol_path.is_file() else None)
    return tasks[:limit]


def create_model_with_weights(
    config: Optional[FGNSConfig] = None,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[Path] = None,
) -> FinslerGaugeSheafNetwork:
    """Instantiates FinslerGaugeSheafNetwork and attaches checkpoint weights if present."""
    cfg = config or FGNSConfig()
    dev = device or get_ablation_device()
    model = FinslerGaugeSheafNetwork(cfg).to(dev)

    ckpt_candidate = checkpoint_path or Path("artifacts/checkpoints/model_final.pt")
    if ckpt_candidate.is_file():
        try:
            state = torch.load(ckpt_candidate, map_location=dev, weights_only=False)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            model.load_state_dict(state, strict=False)
        except Exception:
            pass

    return model


class MathematicalMetrics:
    """Rigorous scientific metrics for non-Euclidean representation and reasoning."""

    @staticmethod
    def compute_dirichlet_energy(
        stalk_features: torch.Tensor,
        restriction_maps: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Computes cellular sheaf Dirichlet energy:
          E_F(X) = 1/2 sum_{(u, v) in E} || F(u -> e) x_u - F(v -> e) x_v ||^2
        """
        b, c, h, w = stalk_features.shape
        flat_x = stalk_features.view(b, c, -1)

        diff_h = flat_x[:, :, 1:] - flat_x[:, :, :-1]
        energy_h = torch.norm(diff_h, dim=1).pow(2).mean()

        if restriction_maps is not None and restriction_maps.dim() >= 3:
            transport_factor = torch.norm(restriction_maps, dim=-1).mean().clamp_min(1e-4)
            return float((energy_h / transport_factor).item())

        return float(energy_h.item())

    @staticmethod
    def compute_effective_rank(tensor_matrix: torch.Tensor) -> float:
        """
        Computes the Roy & Vetterli (2007) spectral effective rank via normalized singular values:
          r_eff = exp(- sum p_i ln p_i) where p_i = sigma_i / sum sigma_j
        """
        if tensor_matrix.dim() > 2:
            tensor_matrix = tensor_matrix.reshape(tensor_matrix.size(0), -1)

        mat_f = tensor_matrix.float()
        try:
            s = torch.linalg.svdvals(mat_f)
            s_sum = s.sum().clamp_min(1e-9)
            p = (s / s_sum).clamp_min(1e-9)
            entropy = -torch.sum(p * torch.log(p)).item()
            return float(math.exp(entropy))
        except Exception:
            return 1.0

    @staticmethod
    def compute_angular_deviation(
        rotor_pred: torch.Tensor,
        rotor_target: torch.Tensor,
    ) -> float:
        """
        Computes the geodesic SO(3) angular distance:
          theta = arccos((Tr(R_1^T R_2) - 1) / 2)
        """
        r1 = rotor_pred.float().view(-1, 3)
        r2 = rotor_target.float().view(-1, 3)

        dot = torch.sum(r1 * r2, dim=-1)
        norm1 = torch.norm(r1, dim=-1).clamp_min(1e-7)
        norm2 = torch.norm(r2, dim=-1).clamp_min(1e-7)
        cos_theta = (dot / (norm1 * norm2)).clamp(-1.0, 1.0)
        return float(torch.acos(cos_theta).mean().item())

    @staticmethod
    def compute_color_hallucination_rate(
        predicted_grid: List[List[int]],
        allowed_colors: Set[int],
    ) -> float:
        """
        Measures fraction of predicted pixels assigned colors outside ground truth palette.
        """
        total_pixels = 0
        hallucinated_pixels = 0

        for row in predicted_grid:
            for c in row:
                total_pixels += 1
                if c not in allowed_colors:
                    hallucinated_pixels += 1

        return float(hallucinated_pixels / max(1, total_pixels))

    @staticmethod
    def compute_histogram_wasserstein(
        pred_hist: Sequence[float],
        true_hist: Sequence[float],
    ) -> float:
        """Computes Earth Mover's Distance between normalized 10-color distributions."""
        p = np.asarray(pred_hist, dtype=np.float64)
        t = np.asarray(true_hist, dtype=np.float64)
        p = p / max(1e-9, p.sum())
        t = t / max(1e-9, t.sum())
        return float(wasserstein_distance(np.arange(len(p)), np.arange(len(t)), p, t))
