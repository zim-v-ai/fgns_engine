"""
Location: // fgns_engine/fgns/train/meta_train/metric_tracker.py
Description: SOTA Forensic 8-Channel Multi-Head Telemetry Dashboard with Dual Count/Percentage Reporting.
             Core Metrological Architecture for FGNS v1.3.0 (Omega-NA3D):
               1. Complete 8-channel Kendall multitask loss accumulation including True XOR parity.
               2. Honest Foreground Entity Accuracy (Pxl_FG) completely eliminating background cheating.
               3. Dual Count & Percentage Reporting for Ext_Full, Ext_FG, and Bdg_Acc in training logs.
               4. Active canvas boundary segmentation IoU and Shannon gradient energy entropy tracking.
Invariants: Exact percentage bounds [0.0, 100.0]%, zero division safety on empty foreground masks,
            loss accumulation strictly non-negative, zero hash comment compliance.
Language: Python 3.12+ / Metrological Science sm_120 Aligned
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple, Union

import torch

from fgns.types import BatchData, LossTelemetry, ModelOutput

DEFAULT_TRACKER_EPS: Final[float] = 1e-7


class FullDashboardMetricTracker:
    """
    SOTA Forensic Multi-Head Telemetry Dashboard Engine.
    """

    def __init__(self, eps: float = DEFAULT_TRACKER_EPS) -> None:
        self.eps = eps
        self.reset()

    def reset(self) -> None:
        self.total_grids: int = 0
        self.exact_full_matches: int = 0
        self.exact_fg_matches: int = 0
        self.exact_size_matches: int = 0
        self.exact_budget_matches: int = 0

        self.query_correct_pixels: float = 0.0
        self.query_total_pixels: float = 0.0

        self.query_correct_fg_pixels: float = 0.0
        self.query_total_fg_pixels: float = 0.0

        self.support_correct_pixels: float = 0.0
        self.support_total_pixels: float = 0.0

        self.accum_mask_iou: float = 0.0
        self.mask_eval_count: int = 0

        self.accum_total_loss: float = 0.0
        self.accum_q_loss: float = 0.0
        self.accum_s_loss: float = 0.0
        self.accum_mask_loss: float = 0.0
        self.accum_size_loss: float = 0.0
        self.accum_budget_loss: float = 0.0
        self.accum_anti_col_loss: float = 0.0
        self.accum_mdl_loss: float = 0.0
        self.accum_ortho_loss: float = 0.0
        self.accum_xor_loss: float = 0.0
        self.accum_lorentz_loss: float = 0.0

        self.accum_kendall_entropy: float = 0.0
        self.latest_kendall_weights: Tuple[float, ...] = (0.0,) * 8
        self.steps_count: int = 0

    def update_losses(self, telemetry: LossTelemetry) -> None:
        self.accum_total_loss += float(telemetry.total_loss.detach().item())
        self.accum_q_loss += telemetry.loss_query
        self.accum_s_loss += telemetry.loss_support
        self.accum_mask_loss += telemetry.loss_mask
        self.accum_size_loss += telemetry.loss_size
        self.accum_budget_loss += telemetry.loss_budget
        self.accum_anti_col_loss += telemetry.loss_anti_collapse
        self.accum_mdl_loss += telemetry.loss_program_mdl
        self.accum_ortho_loss += telemetry.loss_orthogonality
        self.accum_xor_loss += telemetry.loss_xor_parity
        self.accum_lorentz_loss += telemetry.loss_lorentz_rule

        self.accum_mask_iou += telemetry.mask_iou
        self.mask_eval_count += 1

        self.accum_kendall_entropy += telemetry.kendall_entropy
        self.latest_kendall_weights = telemetry.kendall_weights
        self.steps_count += 1

    def update_metrics(
        self,
        output: ModelOutput,
        batch: BatchData,
    ) -> None:
        q_logits = output.query_logits
        q_targets = batch.query.targets
        q_masks = batch.query.target_masks
        s_logits = output.support_logits
        s_targets = batch.support.outputs
        s_masks = batch.support.out_masks
        h_pred = output.query_h_pred
        w_pred = output.query_w_pred
        h_target = batch.query.target_heights
        w_target = batch.query.target_widths
        in_masks = batch.query.in_masks

        B, H, W, V = q_logits.shape

        if q_targets is not None and q_masks is not None:
            q_preds = torch.argmax(q_logits, dim=-1)
            q_is_correct = (q_preds == q_targets) & q_masks

            self.query_correct_pixels += float(q_is_correct.sum().item())
            self.query_total_pixels += float(q_masks.sum().item())

            is_fg = (q_targets > 0) & (q_targets < 10) & q_masks
            q_fg_correct = q_is_correct & is_fg

            self.query_correct_fg_pixels += float(q_fg_correct.sum().item())
            self.query_total_fg_pixels += float(is_fg.sum().item())
        else:
            q_is_correct = None
            is_fg = None
            q_fg_correct = None

        if s_targets is not None and s_masks is not None:
            s_preds = torch.argmax(s_logits, dim=-1)
            s_correct = (s_preds == s_targets) & s_masks

            self.support_correct_pixels += float(s_correct.sum().item())
            self.support_total_pixels += float(s_masks.sum().item())

        if h_target is not None and w_target is not None:
            mask_f32 = in_masks.float()
            h_in = mask_f32.max(dim=2)[0].sum(dim=1).to(torch.int64).clamp_min(1)
            w_in = mask_f32.max(dim=1)[0].sum(dim=1).to(torch.int64).clamp_min(1)

            pred_delta_h = torch.argmax(h_pred, dim=-1) - 30
            pred_delta_w = torch.argmax(w_pred, dim=-1) - 30

            pred_h_abs = (h_in + pred_delta_h).clamp(1, 30)
            pred_w_abs = (w_in + pred_delta_w).clamp(1, 30)

            size_matches = (pred_h_abs == h_target) & (pred_w_abs == w_target)
            self.exact_size_matches += int(size_matches.sum().item())
        else:
            size_matches = None

        if (
            q_is_correct is not None
            and q_masks is not None
            and size_matches is not None
            and q_fg_correct is not None
            and is_fg is not None
        ):
            pixel_failures_per_grid = ((~q_is_correct) & q_masks).view(B, -1).sum(dim=-1)
            grid_pixel_perfect = pixel_failures_per_grid == 0
            grid_full_perfect = grid_pixel_perfect & size_matches
            self.exact_full_matches += int(grid_full_perfect.sum().item())

            fg_failures_per_grid = ((~q_fg_correct) & is_fg).view(B, -1).sum(dim=-1)
            grid_fg_perfect = (fg_failures_per_grid == 0) & size_matches
            self.exact_fg_matches += int(grid_fg_perfect.sum().item())

        if output.query_budget is not None and batch.query.target_histograms is not None:
            pred_counts = output.query_budget.predicted_counts.to(torch.int64)
            true_counts = batch.query.target_histograms.to(torch.int64)
            budget_matches = (pred_counts == true_counts).all(dim=-1)
            self.exact_budget_matches += int(budget_matches.sum().item())

        self.total_grids += B

    def get_summary(self) -> Dict[str, Any]:
        t_grids = max(1, self.total_grids)
        total_q_pxl = max(1.0, self.query_total_pixels)
        total_fg_pxl = max(1.0, self.query_total_fg_pixels)
        total_s_pxl = max(1.0, self.support_total_pixels)
        steps = max(1, self.steps_count)
        mask_evals = max(1, self.mask_eval_count)

        return {
            "avg_total_loss": self.accum_total_loss / steps,
            "avg_q_loss": self.accum_q_loss / steps,
            "avg_s_loss": self.accum_s_loss / steps,
            "avg_mask_loss": self.accum_mask_loss / steps,
            "avg_size_loss": self.accum_size_loss / steps,
            "avg_budget_loss": self.accum_budget_loss / steps,
            "avg_anti_col_loss": self.accum_anti_col_loss / steps,
            "avg_mdl_loss": self.accum_mdl_loss / steps,
            "avg_ortho_loss": self.accum_ortho_loss / steps,
            "avg_xor_loss": self.accum_xor_loss / steps,
            "avg_lorentz_loss": self.accum_lorentz_loss / steps,
            "mean_mask_iou": self.accum_mask_iou / mask_evals,
            "avg_kendall_entropy": self.accum_kendall_entropy / steps,
            "latest_kendall_weights": self.latest_kendall_weights,
            "exact_full_pct": (self.exact_full_matches / t_grids) * 100.0,
            "exact_full_count": self.exact_full_matches,
            "exact_fg_pct": (self.exact_fg_matches / t_grids) * 100.0,
            "exact_fg_count": self.exact_fg_matches,
            "exact_budget_pct": (self.exact_budget_matches / t_grids) * 100.0,
            "exact_budget_count": self.exact_budget_matches,
            "size_acc_pct": (self.exact_size_matches / t_grids) * 100.0,
            "pixel_query_pct": (self.query_correct_pixels / total_q_pxl) * 100.0,
            "pixel_fg_pct": (self.query_correct_fg_pixels / total_fg_pxl) * 100.0,
            "pixel_support_pct": (self.support_correct_pixels / total_s_pxl) * 100.0,
            "total_grids": self.total_grids,
        }

    def format_train_progress_line(
        self,
        epoch: int,
        tasks_processed: int,
        total_tasks: int,
        elapsed_time: float,
        vram_mb: float = 0.0,
    ) -> str:
        """
        Synthesizes standardized single-line console progress telemetry displaying both counts and percentages.
        """
        summary = self.get_summary()
        tps = float(tasks_processed) / max(0.1, elapsed_time)

        line = (
            f"    -> [E{epoch:>3} | Tsk {tasks_processed:>4}/{total_tasks}] "
            f"Train Ls: {summary['avg_total_loss']:>5.2f} "
            f"[Q:{summary['avg_q_loss']:.2f} S:{summary['avg_s_loss']:.2f} "
            f"Msk:{summary['avg_mask_loss']:.2f} Sz:{summary['avg_size_loss']:.2f} "
            f"Bdg:{summary['avg_budget_loss']:.2f} Mdl:{summary['avg_mdl_loss']:.2f} "
            f"Ort:{summary['avg_ortho_loss']:.2f} Xor:{summary['avg_xor_loss']:.2f}] | "
            f"H_k: {summary['avg_kendall_entropy']:>4.2f} | "
            f"Pxl_Q: {summary['pixel_query_pct']:>5.2f}% | "
            f"Pxl_FG: {summary['pixel_fg_pct']:>5.2f}% | "
            f"Bdg_Acc: {summary['exact_budget_count']:>2} ({summary['exact_budget_pct']:>4.1f}%) | "
            f"Msk_IoU: {summary['mean_mask_iou']:>4.2f} | "
            f"Ext_FG: {summary['exact_fg_count']:>2} | "
            f"Ext_Full: {summary['exact_full_count']:>2} | "
            f"Sz_Acc: {summary['size_acc_pct']:>5.2f}% | "
            f"VRAM: {vram_mb:>4.0f}MB | "
            f"TPS: {tps:>4.1f}\n"
        )
        return line

    def format_val_report(
        self,
        epoch: int,
        total_val_tasks: int,
    ) -> str:
        summary = self.get_summary()
        report = (
            f"🏆 [VAL REPORT - EPOCH {epoch}] "
            f"Ext_Full: {summary['exact_full_count']}/{total_val_tasks} ({summary['exact_full_pct']:.2f}%) | "
            f"Ext_FG: {summary['exact_fg_count']}/{total_val_tasks} ({summary['exact_fg_pct']:.2f}%) | "
            f"Bdg_Acc: {summary['exact_budget_count']}/{total_val_tasks} ({summary['exact_budget_pct']:.2f}%) | "
            f"Val Loss: {summary['avg_total_loss']:.3f} "
            f"[Q:{summary['avg_q_loss']:.2f} S:{summary['avg_s_loss']:.2f} "
            f"Msk:{summary['avg_mask_loss']:.2f} Sz:{summary['avg_size_loss']:.2f} "
            f"Bdg:{summary['avg_budget_loss']:.2f} Mdl:{summary['avg_mdl_loss']:.2f} "
            f"Ort:{summary['avg_ortho_loss']:.2f} Xor:{summary['avg_xor_loss']:.2f}] | "
            f"H_k: {summary['avg_kendall_entropy']:.2f} | "
            f"Pxl_Q: {summary['pixel_query_pct']:.2f}% | "
            f"Pxl_FG: {summary['pixel_fg_pct']:.2f}% | "
            f"Pxl_S: {summary['pixel_support_pct']:.2f}% | "
            f"Msk_IoU: {summary['mean_mask_iou']:.2f} | "
            f"Sz_Acc: {summary['size_acc_pct']:.2f}%\n"
        )
        return report
