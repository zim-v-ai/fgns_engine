"""
Location: // fgns_engine/fgns/kernels/__init__.py
Description: Native CUDA C++17 Shared-SRAM Compute Shaders & Spectral Graph Kernels Package Facade.
Language: Python 3.12+ / Pure Docstrings / Zero Hash Compliance
"""

from __future__ import annotations

from fgns.kernels.cuda_fgns_ops import (
    AffineShearingGeometry,
    FusedCudaMorletFunction,
    FusedTritonMorletFunction,
    OctoWaveScanAutogradFunction,
    OctoWaveSSMKernel,
    cuda_finsler_chordal_cavity_op,
    cuda_morlet_kan_op,
    get_cuda_fgns_extension,
    triton_morlet_kan_op,
    verify_cuda_fgns_ops_diagnostics,
)
from fgns.kernels.log_sinkhorn import (
    LogDomainSinkhornBirkhoffProjector,
    log_sinkhorn_iterations,
)
from fgns.kernels.spectral_laplacian import (
    DifferentiableSpectralGreenOperator,
    PoissonDirichletCavitySolver,
    SpectralGraphTopology,
    SpectralGreenWoodburyFunction,
    StencilGraphLaplacianOps,
    apply_spectral_green_woodbury,
    verify_spectral_laplacian_diagnostics,
)

__all__ = [
    "AffineShearingGeometry",
    "FusedCudaMorletFunction",
    "FusedTritonMorletFunction",
    "OctoWaveScanAutogradFunction",
    "OctoWaveSSMKernel",
    "cuda_finsler_chordal_cavity_op",
    "cuda_morlet_kan_op",
    "get_cuda_fgns_extension",
    "triton_morlet_kan_op",
    "verify_cuda_fgns_ops_diagnostics",
    "LogDomainSinkhornBirkhoffProjector",
    "log_sinkhorn_iterations",
    "DifferentiableSpectralGreenOperator",
    "PoissonDirichletCavitySolver",
    "SpectralGraphTopology",
    "SpectralGreenWoodburyFunction",
    "StencilGraphLaplacianOps",
    "apply_spectral_green_woodbury",
    "verify_spectral_laplacian_diagnostics",
]
