"""
Location: // fgns_engine/fgns/solver/submit/__init__.py
Description: Tournament Submission Serialization, Reality Path Discovery & Scorer Package Facade.
Language: Python 3.12+ / Pure Docstrings / Zero Hash Compliance
"""

from __future__ import annotations

from fgns.solver.submit.arc_scorer import (
    InstanceEvaluationResult,
    LeaderboardSummary,
    OfficialArcScorer,
    TaskEvaluationReport,
)
from fgns.solver.submit.atomic_writer import (
    AtomicSubmissionWriter,
    KaggleSchemaAuditor,
)
from fgns.solver.submit.eval_runner import EvaluationPipelineRunner
from fgns.solver.submit.path_resolver import DatasetPathResolver

__all__ = [
    "InstanceEvaluationResult",
    "LeaderboardSummary",
    "OfficialArcScorer",
    "TaskEvaluationReport",
    "AtomicSubmissionWriter",
    "KaggleSchemaAuditor",
    "EvaluationPipelineRunner",
    "DatasetPathResolver",
]