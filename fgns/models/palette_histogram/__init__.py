"""
Location: // fgns_engine/fgns/models/palette_budget/__init__.py
Description: Neural Palette Capacity Oracle Package Facade (Equivariant Palette Histogram Predictor).
Language: Python 3.12+ / Pure Docstrings / Zero Hash Compliance
"""

from __future__ import annotations

from fgns.models.palette_histogram.budget_loss import (
    PaletteBudgetLossEngine,
    evaluate_budget_loss,
)
from fgns.models.palette_histogram.histogram_predictor import EquivariantPaletteHistogramPredictor
from fgns.models.palette_histogram.spatial_scanner import SpatialPaletteAttentionScanner

__all__ = [
    "PaletteBudgetLossEngine",
    "evaluate_budget_loss",
    "EquivariantPaletteHistogramPredictor",
    "SpatialPaletteAttentionScanner",
]