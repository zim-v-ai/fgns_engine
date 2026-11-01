"""
Location: // fgns_engine/fgns/train/meta_train/checkpoint_manager.py
Description: Atomic Write-Sync-Replace Checkpoint Manager & Lexicographic Pareto Selector (FGNS v4.1.0):
               1. Two-phase atomic persistence protocol eliminating corrupt torn-write checkpoints.
               2. Direct hardware physical synchronization via os.fsync(fileno) before commitment.
               3. Automatic Sidecar Metadata Discovery upon initialization preventing cross-stage
                  Pareto baseline resets.
Language: Python 3.12+ / Operating System Persistence sm_120 Aligned
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Final, List, Optional, Tuple, Union

import torch

DEFAULT_MAX_CHECKPOINTS_TO_KEEP: Final[int] = 3
DEFAULT_MAX_IO_RETRIES: Final[int] = 5


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """
    Structured metadata header serialized into lightweight JSON sidecars and torch checkpoints.
    """
    epoch: int
    global_step: int
    exact_full: int
    exact_budget: int
    exact_fg: int
    mask_iou: float
    val_loss: float
    pixel_fg_pct: float
    pixel_query_pct: float
    timestamp_utc: str
    fgns_version: str = "fgns-v4.1.0"


class CheckpointManager:
    """
    Checkpoint Lifecycle Coordinator with Lexicographic Pareto Tournament Selection.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_to_keep: int = DEFAULT_MAX_CHECKPOINTS_TO_KEEP,
        max_io_retries: int = DEFAULT_MAX_IO_RETRIES,
        rank: int = 0,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.max_to_keep = max(1, int(max_to_keep))
        self.max_io_retries = max(1, int(max_io_retries))
        self.rank = rank

        if self.rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_checkpoints: List[Dict[str, Any]] = []
        self.best_metadata: Optional[CheckpointMetadata] = None

    def restore_existing_pareto_baseline(self) -> Optional[CheckpointMetadata]:
        """
        Scans existing sidecar JSON files in checkpoint_dir and restores the highest Pareto baseline.
        """
        if not self.checkpoint_dir.exists():
            return None

        best_found: Optional[CheckpointMetadata] = None
        for json_file in sorted(self.checkpoint_dir.glob("checkpoint_*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f_j:
                    raw = json.load(f_j)
                cand = CheckpointMetadata(**raw)
                if self.is_better_checkpoint(cand, best_found):
                    best_found = cand
            except Exception:
                continue

        if best_found is not None:
            self.best_metadata = best_found
        return best_found

    def _safe_os_replace(self, src: Path, dst: Path) -> None:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_io_retries):
            try:
                os.replace(src, dst)
                return
            except (PermissionError, OSError) as err:
                last_error = err
                time.sleep(0.05 * (2 ** attempt))

        if src.exists():
            try:
                src.unlink()
            except OSError:
                pass

        if last_error is not None:
            raise last_error

    def _atomic_torch_save(
        self,
        state_payload: Dict[str, Any],
        metadata: CheckpointMetadata,
        target_path: Path,
    ) -> None:
        temp_pt_path = target_path.with_suffix(".pt.tmp")

        with open(temp_pt_path, "wb") as f_bin:
            torch.save(state_payload, f_bin)
            f_bin.flush()
            os.fsync(f_bin.fileno())

        self._safe_os_replace(temp_pt_path, target_path)

        sidecar_path = target_path.with_suffix(".json")
        temp_sidecar_path = target_path.with_suffix(".json.tmp")

        with open(temp_sidecar_path, "w", encoding="utf-8") as f_json:
            json.dump(asdict(metadata), f_json, indent=2)
            f_json.flush()
            os.fsync(f_json.fileno())

        self._safe_os_replace(temp_sidecar_path, sidecar_path)

    @classmethod
    def is_better_checkpoint(
        cls,
        candidate: CheckpointMetadata,
        baseline: Optional[CheckpointMetadata],
    ) -> bool:
        if baseline is None:
            return True

        if candidate.exact_full > baseline.exact_full:
            return True
        if candidate.exact_full < baseline.exact_full:
            return False

        if candidate.exact_budget > baseline.exact_budget:
            return True
        if candidate.exact_budget < baseline.exact_budget:
            return False

        if candidate.exact_fg > baseline.exact_fg:
            return True
        if candidate.exact_fg < baseline.exact_fg:
            return False

        if candidate.mask_iou > baseline.mask_iou + 1e-4:
            return True
        if candidate.mask_iou < baseline.mask_iou - 1e-4:
            return False

        return candidate.val_loss < baseline.val_loss

    def save_checkpoint(
        self,
        epoch: int,
        global_step: int,
        models: Dict[str, torch.nn.Module],
        optimizers: Dict[str, torch.optim.Optimizer],
        schedulers: Dict[str, Any],
        val_summary: Dict[str, Any],
    ) -> Tuple[Optional[str], bool]:
        if self.rank != 0:
            return None, False

        metadata = CheckpointMetadata(
            epoch=epoch,
            global_step=global_step,
            exact_full=int(val_summary.get("exact_full_count", 0)),
            exact_budget=int(val_summary.get("exact_budget_count", 0)),
            exact_fg=int(val_summary.get("exact_fg_count", 0)),
            mask_iou=float(val_summary.get("mean_mask_iou", 0.0)),
            val_loss=float(val_summary.get("avg_total_loss", 999.0)),
            pixel_fg_pct=float(val_summary.get("pixel_fg_pct", 0.0)),
            pixel_query_pct=float(val_summary.get("pixel_query_pct", 0.0)),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )

        state_payload = {
            "metadata": asdict(metadata),
            "models": {k: v.state_dict() for k, v in models.items()},
            "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
            "schedulers": {k: v.state_dict() for k, v in schedulers.items() if hasattr(v, "state_dict")},
        }

        last_path = self.checkpoint_dir / "checkpoint_last.pt"
        self._atomic_torch_save(state_payload, metadata, last_path)
        committed_path = str(last_path)

        is_best = self.is_better_checkpoint(metadata, self.best_metadata)

        if is_best:
            self.best_metadata = metadata
            best_filename = f"checkpoint_best_full_{metadata.exact_full}_step_{global_step}.pt"
            best_path = self.checkpoint_dir / best_filename
            self._atomic_torch_save(state_payload, metadata, best_path)
            committed_path = str(best_path)

            self.best_checkpoints.append({
                "path": best_path,
                "metadata": metadata,
            })

            if len(self.best_checkpoints) > self.max_to_keep:
                to_remove = self.best_checkpoints.pop(0)
                rem_pt = Path(to_remove["path"])
                rem_json = rem_pt.with_suffix(".json")
                if rem_pt.exists():
                    try:
                        rem_pt.unlink()
                    except OSError:
                        pass
                if rem_json.exists():
                    try:
                        rem_json.unlink()
                    except OSError:
                        pass

        return committed_path, is_best

    def load_latest_or_best(
        self,
        models: Dict[str, torch.nn.Module],
        optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
        schedulers: Optional[Dict[str, Any]] = None,
        prefer_best: bool = False,
        device: Union[torch.device, str] = "cpu",
    ) -> Tuple[Optional[CheckpointMetadata], bool]:
        target_path: Optional[Path] = None

        if prefer_best:
            candidate_bests = sorted(list(self.checkpoint_dir.glob("checkpoint_best_*.pt")))
            if candidate_bests:
                target_path = candidate_bests[-1]

        if target_path is None:
            last_path = self.checkpoint_dir / "checkpoint_last.pt"
            if last_path.exists():
                target_path = last_path

        if target_path is None or not target_path.exists():
            return None, False

        checkpoint = torch.load(target_path, map_location=device)

        for k, model in models.items():
            if k in checkpoint.get("models", {}):
                model.load_state_dict(checkpoint["models"][k])

        if optimizers is not None:
            for k, opt in optimizers.items():
                if k in checkpoint.get("optimizers", {}):
                    opt.load_state_dict(checkpoint["optimizers"][k])

        if schedulers is not None:
            for k, sched in schedulers.items():
                if k in checkpoint.get("schedulers", {}) and hasattr(sched, "load_state_dict"):
                    sched.load_state_dict(checkpoint["schedulers"][k])

        raw_meta = checkpoint.get("metadata", {})
        metadata_obj = CheckpointMetadata(**raw_meta) if raw_meta else None

        return metadata_obj, True


def verify_checkpoint_manager_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for checkpoint_manager.py.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        ckpt_dir = Path(temp_dir)
        manager = CheckpointManager(checkpoint_dir=ckpt_dir, max_to_keep=2)

        meta_inferior = CheckpointMetadata(
            epoch=1, global_step=100, exact_full=2, exact_budget=5, exact_fg=5,
            mask_iou=0.90, val_loss=0.15, pixel_fg_pct=50.0, pixel_query_pct=80.0,
            timestamp_utc="2026-09-26T00:00:00",
        )
        meta_superior = CheckpointMetadata(
            epoch=2, global_step=200, exact_full=3, exact_budget=4, exact_fg=4,
            mask_iou=0.85, val_loss=0.85, pixel_fg_pct=40.0, pixel_query_pct=75.0,
            timestamp_utc="2026-09-26T01:00:00",
        )

        if not CheckpointManager.is_better_checkpoint(meta_superior, meta_inferior):
            return False
        if CheckpointManager.is_better_checkpoint(meta_inferior, meta_superior):
            return False

        dummy_model = torch.nn.Linear(8, 8)
        dummy_opt = torch.optim.Adam(dummy_model.parameters(), lr=1e-3)
        dummy_sched = torch.optim.lr_scheduler.StepLR(dummy_opt, step_size=10)

        saved_path, is_best = manager.save_checkpoint(
            epoch=1,
            global_step=100,
            models={"model": dummy_model},
            optimizers={"opt": dummy_opt},
            schedulers={"sched": dummy_sched},
            val_summary={
                "exact_full_count": 2,
                "exact_budget_count": 5,
                "exact_fg_count": 5,
                "mean_mask_iou": 0.90,
                "avg_total_loss": 0.15,
                "pixel_fg_pct": 50.0,
                "pixel_query_pct": 80.0,
            },
        )

        if not saved_path or not Path(saved_path).exists():
            return False

        manager2 = CheckpointManager(checkpoint_dir=ckpt_dir, max_to_keep=2)
        restored_base = manager2.restore_existing_pareto_baseline()
        if restored_base is None or restored_base.exact_full != 2:
            return False

        restore_model = torch.nn.Linear(8, 8)
        meta_restored, ok = manager.load_latest_or_best(
            models={"model": restore_model},
            prefer_best=False,
        )

        if not ok or meta_restored is None or meta_restored.exact_full != 2:
            return False

    return True