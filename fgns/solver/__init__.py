"""
Location: // fgns_engine/fgns/solver/__init__.py
Description: Master Competition Solver, Evaluator & Refiner Package Facade for FGNS v0.6.0.
Language: Python 3.12+ / Pure Docstrings / Zero Hash Compliance
"""

from __future__ import annotations

from fgns.solver.evaluator import (
    DualAttemptEvaluator,
    D4ReynoldsEnsembleEngine,
    PlackettLuceModalSynthesizer,
)
from fgns.solver.refiner import (
    TransductiveOperatorAdapter,
    SpectralLoRALinear,
    inject_spectral_adapters,
)
from fgns.solver.submit import (
    AtomicSubmissionWriter,
    EvaluationPipelineRunner,
    KaggleSchemaAuditor,
    LeaderboardSummary,
    OfficialArcScorer,
    DatasetPathResolver,
)

__all__ = [
    "DualAttemptEvaluator",
    "D4ReynoldsEnsembleEngine",
    "PlackettLuceModalSynthesizer",
    "TransductiveOperatorAdapter",
    "SpectralLoRALinear",
    "inject_spectral_adapters",
    "AtomicSubmissionWriter",
    "EvaluationPipelineRunner",
    "KaggleSchemaAuditor",
    "LeaderboardSummary",
    "OfficialArcScorer",
    "DatasetPathResolver",
]