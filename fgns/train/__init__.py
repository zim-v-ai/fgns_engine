"""
Location: // fgns_engine/fgns/train/__init__.py
Description: Master Training & Optimization Package Facade for FGNS v0.6.0.
Language: Python 3.12+ / Pure Docstrings / Zero Hash Compliance
"""

from __future__ import annotations

from fgns.train.losses import (
    BoundedKendallMultitaskLoss,
    DifferentiableActiveMaskLoss,
    ActiveDomainFocalLoss,
    SpectralRankMDLObjective,
    SpectralProgramComplexity,
    SymplecticXORRegularizationLoss,
    SchattenNuclearMDLObjective,
    UniversalTargetSizeLoss,
    XORParityRegularizationLoss,
)
from fgns.train.meta_train import (
    CheckpointManager,
    CheckpointMetadata,
    DualStreamSessionLogger,
    FullDashboardMetricTracker,
    MetaTrainingPipeline,
    StructuredTelemetryLogger,
    initialize_session_logging,
)

__all__ = [
    "BoundedKendallMultitaskLoss",
    "DifferentiableActiveMaskLoss",
    "ActiveDomainFocalLoss",
    "SpectralRankMDLObjective",
    "SpectralProgramComplexity",
    "SymplecticXORRegularizationLoss",
    "SchattenNuclearMDLObjective",
    "UniversalTargetSizeLoss",
    "XORParityRegularizationLoss",
    "CheckpointManager",
    "CheckpointMetadata",
    "DualStreamSessionLogger",
    "FullDashboardMetricTracker",
    "MetaTrainingPipeline",
    "StructuredTelemetryLogger",
    "initialize_session_logging",
]