"""
Location: // fgns_engine/scripts/ablations/generate_all_figures.py
Description: Entrypoint in scripts/ablations/ that invokes the full 7-figure Nature Physics /
             PRX / Quanta Magazine visualization suite and synchronizes artifacts/figures/.
Author: Vyacheslav Zimovsky (v.zimovsky.ai@gmail.com)
License: MIT-0
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.figures.generate_all_figures import generate_all

if __name__ == "__main__":
    generate_all()
