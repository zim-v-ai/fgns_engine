"""
Location: // fgns_engine/fgns/models/__init__.py
Description: Master Neural Models & Cognitive Mechanisms Package Facade for FGNS v0.6.0.
Language: Python 3.12+ / Pure Docstrings / Zero Hash Compliance
"""

from __future__ import annotations

from fgns.models.cauchy_field_decoder import (
    DualStateCauchyRecurrentDecoder,
    RelationalReasoningForceSynthesizer,
)
from fgns.models.fgns_network import FinslerGaugeSheafNetwork
from fgns.models.memory_bank import TaskLorentzMemoryBank
from fgns.models.octo_ssm import (
    BoundaryNormalGuidanceStencil,
    OctoWaveBlock,
    OctoWaveSSM,
)
from fgns.models.palette_histogram import (
    EquivariantPaletteHistogramPredictor,
    PaletteBudgetLossEngine,
    SpatialPaletteAttentionScanner,
    evaluate_budget_loss,
)
from fgns.models.patch_projector import TopologicalCoordinateEncoder
from fgns.models.size_predictor import (
    CartanRuleModulator,
    HarmonicGeometricFiber,
    BoundaryMetricSizePredictor,
    SpatialEnvelopeAttentionPooler,
    SupportDemonstrationSizeInductor,
)
from fgns.models.slot_binding import (
    AsymmetricDirectedHypergraphTransformer,
    DifferentiableSpatialMoments,
    RelationalSlotBinder,
    SpatialMomentsBundle,
)
from fgns.models.transition_lie import (
    DifferentiableRayleighPowerIteration,
    DifferentiableSpectralDecompositionModule,
    LieAlgebraTransitionExtractor,
)

__all__ = [
    "DualStateCauchyRecurrentDecoder",
    "RelationalReasoningForceSynthesizer",
    "FinslerGaugeSheafNetwork",
    "TaskLorentzMemoryBank",
    "BoundaryNormalGuidanceStencil",
    "OctoWaveBlock",
    "OctoWaveSSM",
    "EquivariantPaletteHistogramPredictor",
    "PaletteBudgetLossEngine",
    "SpatialPaletteAttentionScanner",
    "evaluate_budget_loss",
    "TopologicalCoordinateEncoder",
    "CartanRuleModulator",
    "HarmonicGeometricFiber",
    "BoundaryMetricSizePredictor",
    "SpatialEnvelopeAttentionPooler",
    "SupportDemonstrationSizeInductor",
    "AsymmetricDirectedHypergraphTransformer",
    "DifferentiableSpatialMoments",
    "RelationalSlotBinder",
    "SpatialMomentsBundle",
    "DifferentiableRayleighPowerIteration",
    "DifferentiableSpectralDecompositionModule",
    "LieAlgebraTransitionExtractor",
]