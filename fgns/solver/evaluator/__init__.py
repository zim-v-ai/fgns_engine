"""
Location: // fgns_engine/fgns/solver/evaluator/__init__.py
Description: Decision-Theoretic Dual Evaluation & D4 Reynolds Group Marginalization Package Facade.
Language: Python 3.12+ / Pure Docstrings / Zero Hash Compliance
"""

from __future__ import annotations

from fgns.solver.evaluator.d4_ensemble import D4ReynoldsEnsembleEngine
from fgns.solver.evaluator.dual_evaluator import DualAttemptEvaluator
from fgns.solver.evaluator.plackett_luce import PlackettLuceModalSynthesizer

__all__ = [
    "D4ReynoldsEnsembleEngine",
    "DualAttemptEvaluator",
    "PlackettLuceModalSynthesizer",
]