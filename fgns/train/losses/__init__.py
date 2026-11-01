"""
Location: // fgns_engine/fgns/train/losses/__init__.py
Description: Multitask Loss Landscapes & Bounded Kendall Bayesian Optimization Package Facade.
Language: Python 3.12+ / Pure Docstrings / Zero Hash Compliance
"""

from __future__ import annotations

from fgns.train.losses.focal_canvas import ActiveDomainFocalLoss
from fgns.train.losses.kendall_multitask import (
    TASK_NAMES,
    BoundedKendallMultitaskLoss,
)
from fgns.train.losses.mask_segmentation import DifferentiableActiveMaskLoss
from fgns.train.losses.mdl_complexity import (
    SpectralRankMDLObjective,
    SpectralProgramComplexity,
)
from fgns.train.losses.size_loss import UniversalTargetSizeLoss
from fgns.train.losses.true_spectral_mdl import SchattenNuclearMDLObjective
from fgns.train.losses.xor_regularization import (
    SymplecticXORRegularizationLoss,
    XORParityRegularizationLoss,
)

__all__ = [
    "ActiveDomainFocalLoss",
    "TASK_NAMES",
    "BoundedKendallMultitaskLoss",
    "DifferentiableActiveMaskLoss",
    "SpectralRankMDLObjective",
    "SpectralProgramComplexity",
    "UniversalTargetSizeLoss",
    "SchattenNuclearMDLObjective",
    "SymplecticXORRegularizationLoss",
    "XORParityRegularizationLoss",
]