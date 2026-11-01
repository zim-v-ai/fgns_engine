"""
Location: // fgns_engine/fgns/solver/refiner/__init__.py
Description: High-Throughput Test-Time Training (TTT) Swarm & Spectral LoRA Package Facade.
Language: Python 3.12+ / Pure Docstrings / Zero Hash Compliance
"""

from __future__ import annotations

from fgns.solver.refiner.spectral_lora import (
    SpectralLoRALinear,
    inject_spectral_adapters,
)
from fgns.solver.refiner.operator_adapter import TransductiveOperatorAdapter

__all__ = [
    "SpectralLoRALinear",
    "inject_spectral_adapters",
    "TransductiveOperatorAdapter",
]