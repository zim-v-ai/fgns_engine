"""
Location: // fgns_engine/fgns/solver/submit/path_resolver.py
Description: SOTA Autonomous Reality Path Discovery Engine across Kaggle Environments & Local Storage.
             Core Ingestion Infrastructure for FGNS v0.6.0 (Unified Causal Manifold):
               1. Complete multi-platform path resolution transparently handling Windows NTFS and Linux VFS.
               2. Autonomous reality discovery across Kaggle input datasets (/kaggle/input/**/) and local trees.
               3. Intelligent pairing: automatically resolves matching ground-truth solution files when available.
               4. Safe test-mode decoupling: gracefully returns None for solutions in hidden competition runs.
               5. In-memory resolution caching eliminating redundant recursive filesystem traversal overhead.
               6. Autonomous checkpoint discovery across local artifact directories and external dataset mounts.
Invariants: Strict idempotency, zero unhandled FileNotFoundError exceptions on valid datasets,
            POSIX and Win32 path normalization, zero hash comment compliance.
Language: Python 3.12+ / Operating System VFS sm_120 Aligned
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Final, List, Optional, Sequence, Tuple, Union

DEFAULT_SEARCH_ROOTS: Final[Tuple[str, ...]] = (
    "data",
    "../data",
    "/kaggle/input",
    ".",
    "..",
    "artifacts",
    "artifacts/checkpoints",
    "../artifacts",
)


class DatasetPathResolver:
    """
    SOTA Autonomous Reality Path Discovery and Storage Topology Engine.
    Discovers challenge files, ground truth solutions, synthetic directory shards,
    and neural checkpoints seamlessly across local workstations and Kaggle cloud sandboxes.
    """

    """In-memory cache mapping requested filenames to resolved physical paths"""
    _resolved_path_cache: Dict[str, Path] = {}

    @classmethod
    def clear_cache(cls) -> None:
        """Clears the in-memory path resolution cache."""
        cls._resolved_path_cache.clear()

    @classmethod
    def resolve_path(
        cls,
        filename: Union[str, Path],
        candidate_roots: Sequence[Union[str, Path]] = DEFAULT_SEARCH_ROOTS,
        use_cache: bool = True,
    ) -> Path:
        """
        Locates the physical absolute path of the target file across environment roots.

        Args:
            filename: Target file name or relative path to discover.
            candidate_roots: Sequence of candidate root directories to probe.
            use_cache: Memoize resolved path in memory for O(1) subsequent queries.

        Returns:
            Resolved absolute or normalized Path instance.
        """
        filename_str = str(filename).strip()

        if use_cache and filename_str in cls._resolved_path_cache:
            return cls._resolved_path_cache[filename_str]

        target_as_path = Path(filename_str)

        """1. Direct Existence Verification"""
        if target_as_path.is_file():
            resolved = target_as_path.resolve()
            if use_cache:
                cls._resolved_path_cache[filename_str] = resolved
            return resolved

        pure_name = target_as_path.name

        """2. Probing Direct Sub-paths in Candidate Roots"""
        for root_str in candidate_roots:
            root_path = Path(root_str)
            if not root_path.exists():
                continue

            direct_candidate = root_path / filename_str
            if direct_candidate.is_file():
                resolved = direct_candidate.resolve()
                if use_cache:
                    cls._resolved_path_cache[filename_str] = resolved
                return resolved

            direct_by_name = root_path / pure_name
            if direct_by_name.is_file():
                resolved = direct_by_name.resolve()
                if use_cache:
                    cls._resolved_path_cache[filename_str] = resolved
                return resolved

        """3. Recursive Deep Globbing across Candidate Roots (Kaggle Dataset Mounts)"""
        for root_str in candidate_roots:
            root_path = Path(root_str)
            if not root_path.exists() or not root_path.is_dir():
                continue

            try:
                for match in root_path.glob(f"**/{pure_name}"):
                    if match.is_file():
                        resolved = match.resolve()
                        if use_cache:
                            cls._resolved_path_cache[filename_str] = resolved
                        return resolved
            except (PermissionError, OSError):
                continue

        """Fallback: Return normalized path even if missing to allow caller handling"""
        fallback_path = target_as_path.resolve()
        if use_cache:
            cls._resolved_path_cache[filename_str] = fallback_path
        return fallback_path

    @classmethod
    def resolve_pair(
        cls,
        challenges_filename: Union[str, Path],
        use_cache: bool = True,
    ) -> Tuple[Path, Optional[Path]]:
        """
        Resolves paired challenges and solutions files.
        Automatically deduces solutions filename by substituting 'challenges' -> 'solutions'.
        Gracefully returns None for the solution file if executing in hidden test mode.

        Args:
            challenges_filename: Name or path of the challenges JSON file.
            use_cache: Use in-memory resolution cache.

        Returns:
            Tuple of (challenges_path, Optional[solutions_path]).
        """
        challenges_path = cls.resolve_path(challenges_filename, use_cache=use_cache)

        base_name = challenges_path.name
        if "challenges" in base_name:
            solutions_name = base_name.replace("challenges", "solutions")
        else:
            solutions_name = f"{challenges_path.stem}_solutions{challenges_path.suffix}"

        solutions_path: Optional[Path] = None

        """Probe adjacent location first"""
        adjacent_candidate = challenges_path.parent / solutions_name
        if adjacent_candidate.is_file():
            solutions_path = adjacent_candidate.resolve()
        else:
            discovered_sol = cls.resolve_path(solutions_name, use_cache=use_cache)
            if discovered_sol.is_file():
                solutions_path = discovered_sol

        return challenges_path, solutions_path

    @classmethod
    def resolve_directory(
        cls,
        dir_name: Union[str, Path],
        candidate_roots: Sequence[Union[str, Path]] = DEFAULT_SEARCH_ROOTS,
        use_cache: bool = True,
    ) -> Optional[Path]:
        """
        Discovers physical directory paths (such as NVARC synthetic sharded folders).

        Args:
            dir_name: Directory name or relative path to discover.
            candidate_roots: Candidate search roots.
            use_cache: Use in-memory resolution cache.

        Returns:
            Resolved Path instance if directory exists, else None.
        """
        dir_str = str(dir_name).strip()

        if use_cache and dir_str in cls._resolved_path_cache:
            cached = cls._resolved_path_cache[dir_str]
            if cached.is_dir():
                return cached

        target_dir = Path(dir_str)
        if target_dir.is_dir():
            resolved = target_dir.resolve()
            if use_cache:
                cls._resolved_path_cache[dir_str] = resolved
            return resolved

        pure_name = target_dir.name

        for root_str in candidate_roots:
            root_path = Path(root_str)
            if not root_path.exists() or not root_path.is_dir():
                continue

            direct_candidate = root_path / dir_str
            if direct_candidate.is_dir():
                resolved = direct_candidate.resolve()
                if use_cache:
                    cls._resolved_path_cache[dir_str] = resolved
                return resolved

            direct_by_name = root_path / pure_name
            if direct_by_name.is_dir():
                resolved = direct_by_name.resolve()
                if use_cache:
                    cls._resolved_path_cache[dir_str] = resolved
                return resolved

            try:
                for match in root_path.glob(f"**/{pure_name}"):
                    if match.is_dir():
                        resolved = match.resolve()
                        if use_cache:
                            cls._resolved_path_cache[dir_str] = resolved
                        return resolved
            except (PermissionError, OSError):
                continue

        return None

    @classmethod
    def resolve_checkpoint(
        cls,
        checkpoint_name: str = "model_final.pt",
        candidate_roots: Optional[Sequence[Union[str, Path]]] = None,
    ) -> Optional[Path]:
        """
        Discovers pre-trained neural network checkpoints across local artifacts and Kaggle datasets.

        Args:
            checkpoint_name: File name of target checkpoint.
            candidate_roots: Optional root override.

        Returns:
            Resolved Path if checkpoint exists, else None.
        """
        roots = candidate_roots if candidate_roots is not None else DEFAULT_SEARCH_ROOTS
        resolved = cls.resolve_path(checkpoint_name, candidate_roots=roots, use_cache=False)
        return resolved if resolved.is_file() else None


def verify_path_resolver_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for path_resolver.py.
    Validates:
      1. Direct Path Resolution: Successfully discovers existing local test files.
      2. Recursive Globbing: Finds deeply nested files across mock directory hierarchies.
      3. Challenge/Solution Pairing: Correctly pairs challenges with solutions or detects test mode.
      4. Memory Cache Integrity: In-memory dictionary accelerates repeated lookups.
      5. Directory Resolution: Successfully resolves directory paths.

    Returns:
        True if all path discovery invariants pass without error.
    """
    import tempfile

    DatasetPathResolver.clear_cache()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        """Construct nested directory structure simulating Kaggle dataset mounts"""
        nested_dir = temp_path / "mock_dataset" / "nested_challenges"
        nested_dir.mkdir(parents=True, exist_ok=True)

        mock_chal_file = nested_dir / "arc-agi_training_challenges.json"
        mock_sol_file = nested_dir / "arc-agi_training_solutions.json"

        mock_chal_file.write_text("{}", encoding="utf-8")
        mock_sol_file.write_text("{}", encoding="utf-8")

        candidate_roots = [temp_path]

        """1. Test Recursive Deep Resolution"""
        resolved_chal = DatasetPathResolver.resolve_path(
            "arc-agi_training_challenges.json",
            candidate_roots=candidate_roots,
            use_cache=True,
        )

        if not resolved_chal.is_file() or resolved_chal != mock_chal_file.resolve():
            return False

        """2. Test In-Memory Cache Hit"""
        if "arc-agi_training_challenges.json" not in DatasetPathResolver._resolved_path_cache:
            return False

        """3. Test Challenge / Solution Pairing"""
        chal_p, sol_p = DatasetPathResolver.resolve_pair(resolved_chal, use_cache=True)
        if sol_p is None or not sol_p.is_file() or sol_p != mock_sol_file.resolve():
            return False

        """4. Test Hidden Test Mode (Missing Solutions)"""
        hidden_test_file = nested_dir / "arc-agi_test_challenges.json"
        hidden_test_file.write_text("{}", encoding="utf-8")

        test_chal_p, test_sol_p = DatasetPathResolver.resolve_pair(hidden_test_file, use_cache=False)
        if not test_chal_p.is_file():
            return False
        if test_sol_p is not None and test_sol_p.is_file():
            return False

        """5. Test Directory Resolution"""
        resolved_dir = DatasetPathResolver.resolve_directory(
            "nested_challenges",
            candidate_roots=candidate_roots,
            use_cache=False,
        )
        if resolved_dir is None or not resolved_dir.is_dir() or resolved_dir != nested_dir.resolve():
            return False

    DatasetPathResolver.clear_cache()
    return True

