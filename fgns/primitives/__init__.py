"""
Location: // fgns_engine/fgns/primitives/__init__.py
Description: Mathematical Atoms, Riemannian Differential Geometry, Cellular Sheaves,
             and Invariant Primitives Package Facade for FGNS v1.0.0 (Sovereign).
Language: Python 3.12+ / Pure Docstrings / Zero Hash Compliance
"""

from __future__ import annotations

from fgns.primitives.accro_resolvent import (
    ACCROCauchyResolvent,
    ACCROFlowEngine,
    compute_cauchy_kernels,
)
from fgns.primitives.cartan_vielbein import CartanVielbeinFrame
from fgns.primitives.hare_lrq import (
    StraightThroughHareQuota,
    continuous_quota_surplus,
    differentiable_hare_lrq,
)
from fgns.primitives.holographic_tape import (
    HolographicKeySynthesizer,
    HolographicValueSynthesizer,
    SupportCrossAttentionMemory,
)
from fgns.primitives.lie_coords import (
    AsymmetricKroneckerLieFiber,
    MultiscaleLieResonator,
    ScaleResonanceRouter,
)
from fgns.primitives.lorentz_manifold import (
    LorentzManifold,
    decompose_minkowski_coordinates,
    lorentz_inner_product,
    lorentz_norm_squared,
    project_to_hyperboloid,
    project_to_tangent_space,
)
from fgns.primitives.potts_latching import (
    OrthonormalPaletteFrame,
    PottsLatchingLayer,
    StraightThroughPottsCrystallizer,
)
from fgns.primitives.safe_math import (
    canonical_phase_wrap_2pi,
    hyperbolic_saturating_norm,
    safe_divide,
    safe_hyperbolic_sqrt,
    safe_log,
    safe_log_add_exp,
    safe_reciprocal,
)
from fgns.primitives.schatten_norm import (
    SchattenComplexityLoss,
    compute_gram_matrix,
    extract_regularized_singular_values,
    schatten_gram_trace,
    von_neumann_spectral_entropy,
)
from fgns.primitives.stiefel_sheaf import (
    CellularSheafOperator,
    SpatialMooreSheafDiffusion,
    compute_dynamic_edge_dimension,
    newton_schulz_stiefel_5th_order,
    project_stiefel_tangent_vjp,
    stiefel_orthogonality_penalty,
)
from fgns.primitives.swiglu import (
    RMSNorm,
    SwiGLU,
    TopologicalBypassFeedForward,
    TopologicalSwiGLU,
)
from fgns.primitives.symplectic_involution import (
    apply_symplectic_involution,
    pairwise_symplectic_form_matrix,
    symplectic_bilinear_form,
    symplectic_rotation_flow,
)
from fgns.primitives.wavelet_kan import FactoredMorletKANLinear
from fgns.primitives.xor_involution import (
    AnalyticXORInvolutionEngine,
    BilinearGrassmannParityGate,
    SymplecticVorticityCore,
)

__all__ = [
    "ACCROCauchyResolvent",
    "ACCROFlowEngine",
    "compute_cauchy_kernels",
    "CartanVielbeinFrame",
    "StraightThroughHareQuota",
    "continuous_quota_surplus",
    "differentiable_hare_lrq",
    "HolographicKeySynthesizer",
    "HolographicValueSynthesizer",
    "SupportCrossAttentionMemory",
    "AsymmetricKroneckerLieFiber",
    "MultiscaleLieResonator",
    "ScaleResonanceRouter",
    "LorentzManifold",
    "decompose_minkowski_coordinates",
    "lorentz_inner_product",
    "lorentz_norm_squared",
    "project_to_hyperboloid",
    "project_to_tangent_space",
    "OrthonormalPaletteFrame",
    "PottsLatchingLayer",
    "StraightThroughPottsCrystallizer",
    "canonical_phase_wrap_2pi",
    "hyperbolic_saturating_norm",
    "safe_divide",
    "safe_hyperbolic_sqrt",
    "safe_log",
    "safe_log_add_exp",
    "safe_reciprocal",
    "SchattenComplexityLoss",
    "compute_gram_matrix",
    "extract_regularized_singular_values",
    "schatten_gram_trace",
    "von_neumann_spectral_entropy",
    "CellularSheafOperator",
    "SpatialMooreSheafDiffusion",
    "compute_dynamic_edge_dimension",
    "newton_schulz_stiefel_5th_order",
    "project_stiefel_tangent_vjp",
    "stiefel_orthogonality_penalty",
    "RMSNorm",
    "SwiGLU",
    "TopologicalBypassFeedForward",
    "TopologicalSwiGLU",
    "apply_symplectic_involution",
    "pairwise_symplectic_form_matrix",
    "symplectic_bilinear_form",
    "symplectic_rotation_flow",
    "FactoredMorletKANLinear",
    "AnalyticXORInvolutionEngine",
    "BilinearGrassmannParityGate",
    "SymplecticVorticityCore",
]