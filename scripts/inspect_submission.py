"""
Location: // fgns_engine/scripts/inspect_submission.py
Description: SOTA Autonomous Multi-Corpus ARC-AGI Forensic Inspector & Leaderboard Metric Auditor.
             Features automatic multi-manifest ingestion (simultaneously resolving tasks across
             training, evaluation, and test JSON corpora), eradicating [EMPTY OR INVALID GRID]
             errors when inspecting mixed or evaluation submissions!
Invariants: Strict compliance with zero hash comment rule, zero external API dependencies,
            C-contiguous buffer inspection, resilient multi-corpus path discovery.
Language: Python 3.12+ / Forensic Metrology sm_120 Aligned
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple, Union

DEFAULT_SUBMISSION_PATH: Final[str] = "submission.json"
DEFAULT_CHALLENGES_PATH: Final[str] = "arc-agi_training_challenges.json"
DEFAULT_SOLUTIONS_PATH: Final[str] = "arc-agi_training_solutions.json"

CANDIDATE_CHALLENGE_FILES: Final[Tuple[str, ...]] = (
    "arc-agi_training_challenges.json",
    "arc-agi_evaluation_challenges.json",
    "arc-agi_test_challenges.json",
    "arc-agi_training-challenges.json",
    "arc-agi_evaluation-challenges.json",
    "arc-agi_test-challenges.json",
)

CANDIDATE_SOLUTION_FILES: Final[Tuple[str, ...]] = (
    "arc-agi_training_solutions.json",
    "arc-agi_evaluation_solutions.json",
    "arc-agi_training-solutions.json",
    "arc-agi_evaluation-solutions.json",
)


def find_file_in_roots(
    filename: str,
    search_dirs: Sequence[Union[str, Path]] = ("data", "data/synthetic", "../data", "/kaggle/input", "."),
) -> Optional[Path]:
    """
    Discovers files across local folders, synthetic directories, and Kaggle inputs.
    """
    cand = Path(filename)
    if cand.is_file():
        return cand.resolve()

    for d in search_dirs:
        p = Path(d)
        if p.exists():
            direct = p / filename
            if direct.is_file():
                return direct.resolve()
            try:
                for sub in p.glob(f"**/{filename}"):
                    if sub.is_file():
                        return sub.resolve()
            except (PermissionError, OSError):
                continue
    return None


def load_json_file(path: Optional[Path]) -> Dict[str, Any]:
    """
    Loads JSON file safely, returning empty dictionary on absence or parse error.
    """
    if not path or not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_merged_corpora(
    primary_challenges: str,
    primary_solutions: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Path], List[Path]]:
    """
    Loads the primary challenges/solutions files AND automatically merges any auxiliary
    ARC-AGI-1/2 manifests (training, evaluation, test) so every task_id in submission.json is resolved!
    """
    merged_chal: Dict[str, Any] = {}
    merged_sol: Dict[str, Any] = {}
    chal_paths: List[Path] = []
    sol_paths: List[Path] = []

    all_chal_names = [primary_challenges] + [c for c in CANDIDATE_CHALLENGE_FILES if c != primary_challenges]
    all_sol_names = [primary_solutions] + [s for s in CANDIDATE_SOLUTION_FILES if s != primary_solutions]

    for c_name in all_chal_names:
        c_path = find_file_in_roots(c_name)
        if c_path is not None and c_path not in chal_paths:
            data = load_json_file(c_path)
            if data:
                for k, v in data.items():
                    if k not in merged_chal:
                        merged_chal[k] = v
                chal_paths.append(c_path)

    for s_name in all_sol_names:
        s_path = find_file_in_roots(s_name)
        if s_path is not None and s_path not in sol_paths:
            data = load_json_file(s_path)
            if data:
                for k, v in data.items():
                    if k not in merged_sol:
                        merged_sol[k] = v
                sol_paths.append(s_path)

    return merged_chal, merged_sol, chal_paths, sol_paths


def render_grid_compact(grid: Any) -> str:
    """
    Renders 2D grid into visually indented plain text with dot zero-suppression.
    """
    if not grid or not isinstance(grid, list):
        return "  [EMPTY OR INVALID GRID]"
    lines = []
    for row in grid:
        lines.append("  " + " ".join(str(c) if c != 0 else "." for c in row))
    return "\n".join(lines)


def compare_grid_pair(
    pred: Any,
    truth: Any,
) -> Tuple[float, int, int, List[Tuple[Tuple[int, int], int, int]], str, bool]:
    """
    Compares prediction against ground-truth grid.
    Evaluates pixel accuracy percentage, mismatch coordinates, and diff map.
    """
    if not pred or not truth or not isinstance(pred, list) or not isinstance(truth, list):
        return 0.0, 0, 0, [], "  [NO COMPARISON: MISSING DATA]", True

    h_p = len(pred)
    w_p = len(pred[0]) if h_p > 0 and isinstance(pred[0], list) else 0

    h_t = len(truth)
    w_t = len(truth[0]) if h_t > 0 and isinstance(truth[0], list) else 0

    if h_p != h_t or w_p != w_t:
        return 0.0, 0, h_t * w_t, [], f"  [SIZE MISMATCH: Pred {w_p}x{h_p} vs True {w_t}x{h_t}]", True

    correct = 0
    total = h_t * w_t
    mismatches: List[Tuple[Tuple[int, int], int, int]] = []
    diff_map_lines = []

    for y in range(h_t):
        diff_row = []
        row_p = pred[y]
        row_t = truth[y]
        for x in range(w_t):
            p_val = row_p[x] if x < len(row_p) else -1
            t_val = row_t[x] if x < len(row_t) else -2
            if p_val == t_val:
                correct += 1
                diff_row.append(".")
            else:
                mismatches.append(((y, x), p_val, t_val))
                diff_row.append(f"{p_val}")
        diff_map_lines.append("  " + " ".join(diff_row))

    pct = (float(correct) / float(max(1, total))) * 100.0
    return pct, correct, total, mismatches, "\n".join(diff_map_lines), False


def extract_ground_truth_for_task(
    task_id: str,
    chal_dict: Dict[str, Any],
    sol_dict: Dict[str, Any],
) -> List[Any]:
    """
    Extracts ground truth test outputs across both standalone solutions.json
    and natively embedded NVARC test dictionaries.
    """
    truths = []

    if task_id in sol_dict:
        raw_sols = sol_dict[task_id]
        if isinstance(raw_sols, list):
            return raw_sols

    task_data = chal_dict.get(task_id, {})
    test_entries = task_data.get("test", [])

    for entry in test_entries:
        if isinstance(entry, dict) and "output" in entry and entry["output"] is not None:
            truths.append(entry["output"])

    return truths


def main() -> None:
    """
    Master CLI entry point for the SOTA Forensic Inspector.
    """
    parser = argparse.ArgumentParser(description="FGNS SOTA Forensic Submission Inspector")
    parser.add_argument("--sub", default=DEFAULT_SUBMISSION_PATH, help="Path to submission.json")
    parser.add_argument("--challenges", default=DEFAULT_CHALLENGES_PATH, help="Primary Challenges JSON file")
    parser.add_argument("--solutions", default=DEFAULT_SOLUTIONS_PATH, help="Primary Solutions JSON file")
    parser.add_argument("--task", default=None, help="Inspect specific Task ID only")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks to inspect")
    parser.add_argument("--failed-only", action="store_true", help="Display only unsolved tasks")
    parser.add_argument("--output", default=None, help="Save report to file (e.g. report.txt)")
    args = parser.parse_args()

    sub_path = find_file_in_roots(args.sub)
    if not sub_path:
        print(f"[!] Critical Error: Prediction file '{args.sub}' not discovered.")
        sys.exit(1)

    sub_data = load_json_file(sub_path)
    if not sub_data:
        print(f"[!] Critical Error: Failed to parse '{sub_path}'.")
        sys.exit(1)

    chal_data, sol_data, chal_paths, sol_paths = load_merged_corpora(
        args.challenges, args.solutions
    )

    output_lines: List[str] = []

    def log_line(line: str = "") -> None:
        output_lines.append(line)
        if not args.output:
            print(line)

    divider = "=" * 80
    sub_divider = "-" * 80

    chal_str = ", ".join(str(p.name) for p in chal_paths) if chal_paths else "NOT FOUND"
    sol_str = ", ".join(str(p.name) for p in sol_paths) if sol_paths else "NOT FOUND"

    log_line(divider)
    log_line("   FGNS v2.0.0 AUTOMATED FORENSIC SUBMISSION INSPECTOR")
    log_line(f"   Submission Path : {sub_path}")
    log_line(f"   Challenges Pool : {chal_str} ({len(chal_data)} tasks indexed)")
    log_line(f"   Solutions Pool  : {sol_str} ({len(sol_data)} solutions indexed)")
    log_line(f"   Tasks in Sub    : {len(sub_data)} total submissions")
    log_line(divider)

    total_tasks_count = 0
    solved_tasks_count = 0
    total_test_instances = 0
    solved_test_instances = 0

    task_ids = list(sub_data.keys())
    if args.task:
        task_ids = [t for t in task_ids if t == args.task]
    if args.limit:
        task_ids = task_ids[: args.limit]

    for t_idx, task_id in enumerate(task_ids):
        task_sub = sub_data.get(task_id, [])
        task_chal = chal_data.get(task_id, {})
        ground_truths = extract_ground_truth_for_task(task_id, chal_data, sol_data)

        train_pairs = task_chal.get("train", [])
        test_inputs = task_chal.get("test", [])

        task_all_passed = True
        test_results = []

        for test_i, pred_dict in enumerate(task_sub):
            att1 = pred_dict.get("attempt_1", [])
            att2 = pred_dict.get("attempt_2", [])
            truth = ground_truths[test_i] if test_i < len(ground_truths) else None

            pct1, c1, tot1, mis1, diff1, sz_mis1 = compare_grid_pair(att1, truth)
            pct2, c2, tot2, mis2, diff2, sz_mis2 = compare_grid_pair(att2, truth)

            best_pct = max(pct1, pct2)
            passed = (pct1 == 100.0) or (pct2 == 100.0)

            if truth is not None:
                total_test_instances += 1
                if passed:
                    solved_test_instances += 1
                else:
                    task_all_passed = False
            else:
                task_all_passed = False

            test_results.append({
                "test_idx": test_i,
                "att1": att1,
                "att2": att2,
                "truth": truth,
                "pct1": pct1,
                "pct2": pct2,
                "best_pct": best_pct,
                "passed": passed,
                "mis1": mis1,
                "diff1": diff1,
                "sz_mis1": sz_mis1,
                "sz_mis2": sz_mis2,
            })

        has_truth = len(ground_truths) > 0
        if has_truth:
            total_tasks_count += 1
            if task_all_passed:
                solved_tasks_count += 1

        if args.failed_only and task_all_passed and has_truth:
            continue

        if has_truth:
            status_text = "EXACT MATCH 100%" if task_all_passed else "MISMATCH / FAILED"
        else:
            status_text = "TEST INFERENCE (NO TRUTH AVAILABLE)"

        log_line("\n" + sub_divider)
        log_line(f"TASK [{t_idx + 1}/{len(task_ids)}] ID: {task_id} | Status: {status_text}")
        log_line(sub_divider)

        if train_pairs:
            log_line(f"\n[DEMONSTRATIONS: {len(train_pairs)} pairs]")
            for p_idx, pair in enumerate(train_pairs):
                in_g = pair.get("input", [])
                out_g = pair.get("output", [])
                h_i = len(in_g)
                w_i = len(in_g[0]) if h_i > 0 and isinstance(in_g[0], list) else 0
                h_o = len(out_g)
                w_o = len(out_g[0]) if h_o > 0 and isinstance(out_g[0], list) else 0

                log_line(f"  --- Train Pair {p_idx + 1} (In: {w_i}x{h_i} -> Out: {w_o}x{h_o}) ---")
                log_line("  Input:")
                log_line(render_grid_compact(in_g))
                log_line("  Output:")
                log_line(render_grid_compact(out_g))

        for res in test_results:
            t_no = res["test_idx"] + 1
            raw_test_entry = test_inputs[res["test_idx"]] if res["test_idx"] < len(test_inputs) else {}
            t_input = raw_test_entry.get("input", []) if isinstance(raw_test_entry, dict) else []

            h_ti = len(t_input)
            w_ti = len(t_input[0]) if h_ti > 0 and isinstance(t_input[0], list) else 0

            log_line(f"\n[TEST INSTANCE {t_no}] Input Dimensions: {w_ti}x{h_ti}")
            log_line("  Test Input:")
            log_line(render_grid_compact(t_input))

            if res["truth"] is not None:
                h_tr = len(res["truth"])
                w_tr = len(res["truth"][0]) if h_tr > 0 else 0
                log_line(f"  Ground Truth Output ({w_tr}x{h_tr}):")
                log_line(render_grid_compact(res["truth"]))

            h_a1 = len(res["att1"])
            w_a1 = len(res["att1"][0]) if h_a1 > 0 and isinstance(res["att1"][0], list) else 0
            log_line(f"  Attempt 1 ({w_a1}x{h_a1}) [Match: {res['pct1']:.2f}%]:")
            log_line(render_grid_compact(res["att1"]))

            h_a2 = len(res["att2"])
            w_a2 = len(res["att2"][0]) if h_a2 > 0 and isinstance(res["att2"][0], list) else 0
            log_line(f"  Attempt 2 ({w_a2}x{h_a2}) [Match: {res['pct2']:.2f}%]:")
            log_line(render_grid_compact(res["att2"]))

            if res["truth"] is not None and res["pct1"] < 100.0:
                log_line("\n  Diff Map (Attempt 1 vs Truth; '.'=Match, Numbers=Predictions):")
                log_line(res["diff1"])
                if res["mis1"]:
                    sample_mis = res["mis1"][:8]
                    coord_strs = [f"(y={y},x={x}): Pred={p}, True={t}" for (y, x), p, t in sample_mis]
                    log_line(f"  Mismatched Pixels ({len(res['mis1'])} total): " + "; ".join(coord_strs))

    log_line("\n" + divider)
    log_line("   FGNS v2.0.0 LEADERBOARD AUDIT SUMMARY")
    log_line(divider)
    log_line(f"   Evaluated Tasks Total    : {total_tasks_count}")

    if total_tasks_count > 0:
        task_score_pct = (float(solved_tasks_count) / float(total_tasks_count)) * 100.0
        log_line(f"   ARC Task Exact Match     : {solved_tasks_count} / {total_tasks_count} ({task_score_pct:.2f}%)")

    if total_test_instances > 0:
        kaggle_score_pct = (float(solved_test_instances) / float(total_test_instances)) * 100.0
        log_line(f"   Kaggle Official Score    : {solved_test_instances} / {total_test_instances} ({kaggle_score_pct:.2f}%)")

    log_line(divider)

    if args.output:
        out_file = Path(args.output).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f_out:
            f_out.write("\n".join(output_lines) + "\n")
        print(f"[+] Complete forensic report committed successfully to: {out_file}")


if __name__ == "__main__":
    main()
