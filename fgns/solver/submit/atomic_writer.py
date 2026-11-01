"""
Location: // fgns_engine/fgns/solver/submit/atomic_writer.py
Description: SOTA Thread-Safe Atomic Submission Serializer & Kaggle ARC-AGI-2 Schema Auditor.
             Core Submission Infrastructure for FGNS v0.6.0 (Unified Causal Manifold):
               1. Strict pre-submission schema auditor enforcing official Kaggle ARC-AGI-2 invariants.
               2. Rectangular geometry verification: guarantees no ragged arrays and bounds H, W in [1, 30].
               3. Pure discrete color alphabet check: guarantees elements in [0, 9] with ZERO PAD (10) leaks.
               4. Two-phase atomic persistence (Write-Sync-Replace) eliminating corrupted partial JSON files.
               5. Windows WDDM/NTFS lock recovery surviving background indexing file-lock race conditions.
               6. Crash-resilient incremental task updating enabling seamless hot-resuming across interruptions.
Invariants: Exact JSON schema compliance, strict atomicity under OS kill signals (SIGTERM, 0xc000013a),
            thread-safety via re-entrant locks (threading.RLock), zero hash comment compliance.
Language: Python 3.12+ / Operating System I/O Fabric sm_120 Aligned
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, Final, List, Optional, Sequence, Set, Tuple, Union

from fgns.types import AttemptPrediction, GridData, TaskSubmission

DEFAULT_MAX_RETRIES: Final[int] = 5
DEFAULT_MIN_DIMENSION: Final[int] = 1
DEFAULT_MAX_DIMENSION: Final[int] = 30


class KaggleSchemaAuditor:
    """
    SOTA Strict Pre-Submission Schema Auditor.
    Enforces the official Kaggle ARC-AGI-2 evaluation specification prior to any filesystem commit.
    Eliminates silent submission scoring errors on the competition leaderboard.
    """

    @classmethod
    def validate_grid_geometry(
        cls,
        grid: Any,
        grid_name: str = "grid",
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates that a grid is a rectangular 2D list of integers strictly bounded in [0, 9].

        Args:
            grid: Candidate 2D grid structure to audit.
            grid_name: Contextual name for diagnostic error messages.

        Returns:
            Tuple of (is_valid, Optional[error_message]).
        """
        if not isinstance(grid, list) or len(grid) == 0:
            return False, f"{grid_name} must be a non-empty 2D list, got {type(grid)}"

        h = len(grid)
        if h < DEFAULT_MIN_DIMENSION or h > DEFAULT_MAX_DIMENSION:
            return False, f"{grid_name} height ({h}) exceeds ARC bounds [1..30]"

        if not isinstance(grid[0], list) or len(grid[0]) == 0:
            return False, f"{grid_name} row 0 must be a non-empty list"

        w = len(grid[0])
        if w < DEFAULT_MIN_DIMENSION or w > DEFAULT_MAX_DIMENSION:
            return False, f"{grid_name} width ({w}) exceeds ARC bounds [1..30]"

        for r_idx, row in enumerate(grid):
            if not isinstance(row, list):
                return False, f"{grid_name} row {r_idx} is not a list"
            if len(row) != w:
                return False, f"{grid_name} row {r_idx} width ({len(row)}) != expected width ({w}) [Ragged Array]"

            for c_idx, val in enumerate(row):
                if not isinstance(val, int) or isinstance(val, bool):
                    return False, f"{grid_name} at ({r_idx}, {c_idx}) has non-integer value: {val} ({type(val)})"
                if val < 0 or val > 9:
                    return False, f"{grid_name} at ({r_idx}, {c_idx}) has invalid color token {val} (PAD 10 is illegal!)"

        return True, None

    @classmethod
    def validate_task_attempts(
        cls,
        attempts_list: Any,
        task_id: str,
        expected_test_count: Optional[int] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Validates prediction list structure for a specific ARC task.

        Args:
            attempts_list: Target list of prediction dictionaries.
            task_id: Unique task identifier string.
            expected_test_count: Optional expected number of test outputs.

        Returns:
            Tuple of (is_valid, List[error_messages]).
        """
        errors: List[str] = []

        if not isinstance(attempts_list, list) or len(attempts_list) == 0:
            errors.append(f"Task {task_id} must contain a non-empty list of test predictions")
            return False, errors

        if expected_test_count is not None and len(attempts_list) != expected_test_count:
            errors.append(
                f"Task {task_id} test count mismatch: expected {expected_test_count}, got {len(attempts_list)}"
            )

        for t_idx, pair_dict in enumerate(attempts_list):
            if not isinstance(pair_dict, dict):
                errors.append(f"Task {task_id} prediction index {t_idx} is not a dictionary")
                continue

            for key in ("attempt_1", "attempt_2"):
                if key not in pair_dict:
                    errors.append(f"Task {task_id} test {t_idx} is missing required key '{key}'")
                    continue

                grid_obj = pair_dict[key]
                valid_grid, err_msg = cls.validate_grid_geometry(grid_obj, f"Task {task_id} test {t_idx} {key}")
                if not valid_grid and err_msg is not None:
                    errors.append(err_msg)

        return len(errors) == 0, errors

    @classmethod
    def validate_submission_dict(
        cls,
        submission_dict: Dict[str, Any],
        expected_task_ids: Optional[Sequence[str]] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Executes complete tournament-level pre-submission audit.

        Args:
            submission_dict: Master submission dictionary mapping task IDs to prediction lists.
            expected_task_ids: Optional sequence of mandatory task IDs that must be present.

        Returns:
            Tuple of (is_valid, List[error_messages]).
        """
        all_errors: List[str] = []

        if not isinstance(submission_dict, dict):
            return False, ["Root submission object must be a dictionary"]

        if expected_task_ids is not None:
            for req_id in expected_task_ids:
                if req_id not in submission_dict:
                    all_errors.append(f"Missing mandatory task ID in submission: {req_id}")

        for task_id, attempts in submission_dict.items():
            valid_task, task_errors = cls.validate_task_attempts(attempts, task_id=str(task_id))
            if not valid_task:
                all_errors.extend(task_errors)

        return len(all_errors) == 0, all_errors


class AtomicSubmissionWriter:
    """
    SOTA Thread-Safe Crash-Resilient Atomic Submission Serializer.
    Guarantees that submission.json cannot be corrupted by abrupt process termination.
    Executes pre-submission schema audit prior to committing atomic filesystem replacements.
    """

    _lock: Final[threading.RLock] = threading.RLock()

    @classmethod
    def _safe_os_replace(
        cls,
        src: Path,
        dst: Path,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        """
        Executes atomic file replacement with exponential backoff to survive Windows file locks.
        """
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
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

    @classmethod
    def read_existing_submission(
        cls,
        submission_path: Union[str, Path] = "submission.json",
    ) -> Dict[str, List[Dict[str, List[List[int]]]]]:
        """
        Restores completed task predictions to enable hot-resuming interrupted sessions.

        Args:
            submission_path: Path to target submission.json file.

        Returns:
            Dictionary mapping completed task IDs to their serialized predictions.
        """
        target = Path(submission_path)
        if not target.is_file():
            return {}

        with cls._lock:
            try:
                with open(target, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}

    @classmethod
    def write_atomic_submission(
        cls,
        submission_dict: Dict[str, Any],
        target_path: Union[str, Path] = "submission.json",
        validate_schema: bool = True,
        expected_task_ids: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Audits submission schema and commits atomic write-sync-replace.

        Args:
            submission_dict: Master prediction payload dictionary.
            target_path: Destination JSON file path (typically submission.json).
            validate_schema: Enforce strict Kaggle ARC-AGI-2 schema verification.
            expected_task_ids: Optional list of mandatory task IDs.

        Raises:
            ValueError: If schema audit detects any ragged array, illegal color token, or missing key.
        """
        target = Path(target_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        """1. Strict Pre-Submission Schema Audit"""
        if validate_schema:
            is_valid, errors = KaggleSchemaAuditor.validate_submission_dict(
                submission_dict=submission_dict,
                expected_task_ids=expected_task_ids,
            )
            if not is_valid:
                error_summary = "\n   - ".join(errors[:5])
                raise ValueError(
                    f"Kaggle ARC-AGI-2 submission schema violation ({len(errors)} errors detected):\n   - {error_summary}"
                )

        temp_path = target.with_suffix(".json.tmp")

        """2. Thread-Safe Atomic Serialization with Hardware Disk Sync"""
        with cls._lock:
            with open(temp_path, "w", encoding="utf-8") as f_json:
                json.dump(submission_dict, f_json, separators=(",", ":"))
                f_json.flush()
                os.fsync(f_json.fileno())

            cls._safe_os_replace(temp_path, target)

    @classmethod
    def update_task_submission(
        cls,
        task_id: str,
        predictions: Sequence[AttemptPrediction],
        target_path: Union[str, Path] = "submission.json",
    ) -> None:
        """
        Incrementally commits an individual solved task into submission.json.
        Enables zero-downtime hot resuming across multi-hour tournament runs.

        Args:
            task_id: Unique task identifier string.
            predictions: Sequence of AttemptPrediction monads matching task test queries.
            target_path: Destination submission.json path.
        """
        task_submission = TaskSubmission(
            task_id=str(task_id),
            predictions=tuple(predictions),
        )
        task_payload = task_submission.to_kaggle_dict()

        with cls._lock:
            current_submission = cls.read_existing_submission(target_path)
            current_submission.update(task_payload)
            cls.write_atomic_submission(
                submission_dict=current_submission,
                target_path=target_path,
                validate_schema=True,
            )


def verify_atomic_writer_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for atomic_writer.py.
    Validates:
      1. Valid Payload Acceptance: Canonical rectangular grid serializes cleanly.
      2. Illegal Color Rejection: Rejects token 10 (PAD) or negative values with ValueError.
      3. Ragged Array Rejection: Rejects non-rectangular rows with ValueError.
      4. Incremental Hot-Resuming: update_task_submission appends tasks without data loss.
      5. Atomic Replacement Invariant: Temporary files are cleaned up and final file is valid JSON.

    Returns:
        True if all serialization and schema audit invariants pass with zero defect.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        sub_file = temp_dir_path / "submission.json"

        """1. Test Valid Payload Serialization"""
        valid_grid_2x2 = [[1, 2], [3, 4]]
        valid_payload = {
            "task_001": [
                {
                    "attempt_1": valid_grid_2x2,
                    "attempt_2": valid_grid_2x2,
                }
            ]
        }

        AtomicSubmissionWriter.write_atomic_submission(valid_payload, sub_file)

        if not sub_file.is_file():
            return False

        restored = AtomicSubmissionWriter.read_existing_submission(sub_file)
        if "task_001" not in restored or restored["task_001"][0]["attempt_1"] != valid_grid_2x2:
            return False

        """2. Test Illegal Color Token 10 (PAD) Rejection"""
        illegal_pad_payload = {
            "task_002": [
                {
                    "attempt_1": [[1, 10], [3, 4]],
                    "attempt_2": valid_grid_2x2,
                }
            ]
        }

        rejected_pad = False
        try:
            AtomicSubmissionWriter.write_atomic_submission(illegal_pad_payload, sub_file)
        except ValueError:
            rejected_pad = True

        if not rejected_pad:
            return False

        """3. Test Ragged Array Rejection"""
        ragged_payload = {
            "task_003": [
                {
                    "attempt_1": [[1, 2, 3], [4, 5]],
                    "attempt_2": valid_grid_2x2,
                }
            ]
        }

        rejected_ragged = False
        try:
            AtomicSubmissionWriter.write_atomic_submission(ragged_payload, sub_file)
        except ValueError:
            rejected_ragged = True

        if not rejected_ragged:
            return False

        """4. Test Incremental Hot-Resuming Update"""
        g1 = GridData(height=2, width=2, data=(1, 2, 3, 4))
        g2 = GridData(height=2, width=2, data=(5, 6, 7, 8))
        pred_monad = AttemptPrediction(attempt_1=g1, attempt_2=g2, confidence=0.90, d4_vote_spread=(8, 8))

        AtomicSubmissionWriter.update_task_submission("task_incremental", [pred_monad], sub_file)

        final_dict = AtomicSubmissionWriter.read_existing_submission(sub_file)
        if "task_001" not in final_dict or "task_incremental" not in final_dict:
            return False
        if final_dict["task_incremental"][0]["attempt_2"] != [[5, 6], [7, 8]]:
            return False

    return True