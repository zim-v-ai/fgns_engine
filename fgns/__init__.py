"""
Location: // fgns_engine/fgns/__init__.py
Description: Master Root Package Facade for FGNS v0.6.0 (Unified Causal Manifold).
             Exposes core configurations, mathematical state monads, and the apex transductive network.
Language: Python 3.12+ / Pure Docstrings / Zero Hash Compliance
"""

from __future__ import annotations

from fgns.config import (
    AccroConfig,
    FGNSConfig,
    GeometryConfig,
    HardwareConfig,
    KendallMultitaskConfig,
    LorentzManifoldConfig,
    MultiscaleLieConfig,
    OctoWaveSSMConfig,
    PaletteBudgetConfig,
    PlackettLuceConfig,
    PottsLatchingConfig,
    SizeOracleConfig,
    SlotAttentionConfig,
    SpectralMDLConfig,
    TopologyConfig,
    TTTRefinementConfig,
    XORInvolutionConfig,
)
from fgns.models.fgns_network import FinslerGaugeSheafNetwork
from fgns.types import (
    ARCPair,
    ARCTask,
    AttemptPrediction,
    BatchData,
    EntitySlots,
    GridData,
    HyperbolicRuleState,
    LieTransitionState,
    LossTelemetry,
    ModelOutput,
    PaletteBudgetPrediction,
    PaletteColor,
    TaskMetadata,
    TaskSubmission,
    XORParityState,
)

__all__ = [
    "AccroConfig",
    "FGNSConfig",
    "GeometryConfig",
    "HardwareConfig",
    "KendallMultitaskConfig",
    "LorentzManifoldConfig",
    "MultiscaleLieConfig",
    "OctoWaveSSMConfig",
    "PaletteBudgetConfig",
    "PlackettLuceConfig",
    "PottsLatchingConfig",
    "SizeOracleConfig",
    "SlotAttentionConfig",
    "SpectralMDLConfig",
    "TopologyConfig",
    "TTTRefinementConfig",
    "XORInvolutionConfig",
    "FinslerGaugeSheafNetwork",
    "ARCPair",
    "ARCTask",
    "AttemptPrediction",
    "BatchData",
    "EntitySlots",
    "GridData",
    "HyperbolicRuleState",
    "LieTransitionState",
    "LossTelemetry",
    "ModelOutput",
    "PaletteBudgetPrediction",
    "PaletteColor",
    "TaskMetadata",
    "TaskSubmission",
    "XORParityState",
]