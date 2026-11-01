"""
Location: // fgns_engine/fgns/train/meta_train/__init__.py
Description: Offline Meta-Learning, Telemetry Tracking & Checkpoint Persistence Package Facade.
Language: Python 3.12+ / Pure Docstrings / Zero Hash Compliance
"""

from __future__ import annotations

from fgns.train.meta_train.checkpoint_manager import (
    CheckpointManager,
    CheckpointMetadata,
)
from fgns.train.meta_train.logger import (
    DualStreamSessionLogger,
    StructuredTelemetryLogger,
    initialize_session_logging,
)
from fgns.train.meta_train.metric_tracker import FullDashboardMetricTracker
from fgns.train.meta_train.pipeline import MetaTrainingPipeline

__all__ = [
    "CheckpointManager",
    "CheckpointMetadata",
    "DualStreamSessionLogger",
    "StructuredTelemetryLogger",
    "initialize_session_logging",
    "FullDashboardMetricTracker",
    "MetaTrainingPipeline",
]