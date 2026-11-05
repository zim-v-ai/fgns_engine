"""
Location: // fgns_engine/fgns/config.py
Description: Master Configuration and Physical Invariant Specification for the Finsler-Gauge
             Cellular Sheaf Network (FGNS v4.1.0 Academic Standard):
               1. Platform-Adaptive CUDA Memory Virtualization (Windows WDDM vs Linux VMM).
               2. Native CUDA C++17 Shared-SRAM Kernel Configuration targeting sm_120 & sm_89.
               3. Static Support Horizon (K_static=6) eliminating allocator fragmentation.
               4. 4-Step Dual-State Cauchy Recurrent Decoder (Separate Thought Z & Draft Y).
               5. Universal Riemannian-Finsler Proper-Time Metric Frame Integrator.
               6. Chamber-Local Bravais Lattice Cohomology (T_max=8) & Chebyshev Eikonal Level-Sets.
               7. Two-Bounce Specular Billiard Wavefront Reflection in 8-Axis OctoWave SSM.
               8. Full-Network Geodesic Test-Time Adaptation under D4 x S9 Automorphism Group.
Language: Python 3.12+ / PyTorch 2.5+ / Native CUDA C++17 sm_120 & sm_89 Engine
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import sys
from typing import Final, Sequence, Tuple

if sys.platform == "win32":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"
else:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["TORCH_LOGS"] = "-all"
os.environ["TRITON_CACHE_DIR"] = os.path.abspath("target/triton_cache")
os.environ["TORCH_EXTENSIONS_DIR"] = os.path.abspath("target/cuda_cache")


@dataclass(frozen=True, slots=True)
class HardwareConfig:
    """
    Hardware execution parameters for NVIDIA GeForce RTX 5070 (Blackwell sm_120) and Kaggle L4x4 (sm_89).
    """
    device_name: str = "cuda"
    target_arch: str = "sm_120"
    enable_bf16_amp: bool = True
    enable_tf32: bool = True
    enable_torch_compile: bool = False
    compile_mode: str = "max-autotune"
    max_vram_budget_mb: int = 2250
    enable_native_cuda_jit: bool = True
    cuda_cache_directory: str = "target/cuda_cache"
    triton_block_size_spatial: int = 32
    triton_block_size_channel: int = 64
    triton_cache_directory: str = "target/triton_cache"


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    """
    Discrete Euclidean canvas invariants and size classification parameters.
    """
    max_grid_size: int = 30
    canvas_channels: int = 1
    vocab_size: int = 11
    pad_token_id: int = 10
    relative_dim_classes: int = 61
    zero_shift_index: int = 30
    min_active_pixels: int = 1
    max_support_pairs: int = 6


@dataclass(frozen=True, slots=True)
class RecursiveReasoningConfig:
    """
    4-Step Dual-State Cauchy Recurrent Reasoning Parameters.
    """
    reasoning_steps: int = 4
    delta_t: float = 0.25
    tau_cooling_max: float = 1.00
    tau_cooling_min: float = 0.05
    tau_decay_power: float = 1.50
    triangular_supervision: bool = True
    enable_draft_re_embedding: bool = True
    enable_intra_step_spatial_mixing: bool = True


@dataclass(frozen=True, slots=True)
class NA3Dv2Config:
    """
    Gauge-Equivariant NA3D-Finsler-Lie v2.0 3D Simplex Bundle Configuration.
    """
    num_cells: int = 8
    cell_stride: int = 16
    rest_mass_init: float = 0.25
    tau_0_init: float = 0.50
    wilson_plaquette_gain: float = 0.35
    bravais_dislocation_gain: float = 0.40
    max_bravais_period: int = 8
    periodicity_threshold: float = 0.70
    defect_heal_gain: float = 2.50
    enable_eikonal_level_sets: bool = True
    enable_chamber_folding: bool = True


@dataclass(frozen=True, slots=True)
class SheafConfig:
    """
    Continuous Cellular Sheaf Bundle Parameters on Stiefel Manifolds V_k(R^d).
    """
    d_node: int = 128
    d_edge: int = 32
    stiefel_iterations: int = 5
    alpha_advection: float = 0.85
    beta_diffusion: float = 0.15
    spatial_sheaf_gain: float = 0.20
    schur_gamma_reg: float = 0.10
    eps: float = 1e-7


@dataclass(frozen=True, slots=True)
class SizeOracleConfig:
    """
    Universal Riemannian-Finsler Proper-Time Metric Frame Integrator Parameters.
    """
    d_model: int = 128
    d_harmonic_fiber: int = 32
    num_harmonic_frequencies: int = 4
    num_envelope_heads: int = 4
    relative_classes: int = 61
    zero_shift_index: int = 30
    label_smoothing: float = 0.05
    uncertainty_threshold: float = 0.40
    stokes_temperature: float = 0.05
    proper_time_sigma: float = 0.45


@dataclass(frozen=True, slots=True)
class PaletteBudgetConfig:
    """
    S10-Equivariant Structural Role Palette Histogram Predictor Parameters.
    """
    enable_budget_oracle: bool = True
    d_budget_hidden: int = 128
    num_scanner_heads: int = 4
    num_role_features: int = 8
    hare_quota_tau: float = 0.10
    anti_collapse_margin: float = 0.50
    anti_collapse_weight: float = 0.25
    smooth_l1_beta: float = 1.00
    bias_gain: float = 2.00
    chemical_feedback_gain: float = 1.00


@dataclass(frozen=True, slots=True)
class AccroConfig:
    """
    Analytical Closed-Form Cellular Response Operator (Cauchy Resolvent Flow).
    """
    feature_dim: int = 128
    base_friction: float = 0.35
    barrier_friction: float = 8.00
    taylor_expansion_threshold: float = 1e-4
    max_flow_speed: float = 4.00
    eps: float = 1e-6


@dataclass(frozen=True, slots=True)
class LorentzManifoldConfig:
    """
    Pseudo-Riemannian Lorentz Hyperboloid Geometry Parameters (H^D_kappa).
    """
    curvature: float = 1.00
    manifold_dim: int = 128
    eps: float = 1e-7
    margin: float = 0.05


@dataclass(frozen=True, slots=True)
class EpisodicMemoryConfig:
    """
    Hyperbolic Vector-Quantized Codebook / Episodic Rule Memory Configuration.
    """
    num_prototypes: int = 256
    commitment_cost: float = 0.25
    cluster_decay: float = 0.99
    initial_tau: float = 0.10


@dataclass(frozen=True, slots=True)
class XORInvolutionConfig:
    """
    Closed-Form Analytic Nonlinear Asymmetric Parity & Involution System.
    """
    d_model: int = 128
    parity_alpha: float = 0.25
    involution_rank: int = 64
    eps: float = 1e-7


@dataclass(frozen=True, slots=True)
class OctoWaveSSMConfig:
    """
    Parameters for the 8-axis isotropic state-space wavefront scanner with Specular Reflection.
    """
    num_axes: int = 8
    d_model: int = 128
    d_state: int = 16
    ssm_dt_min: float = 0.001
    ssm_dt_max: float = 0.10
    absorbing_zero_boundary: bool = True
    directional_arbitration_heads: int = 4
    normal_router_weight: float = 0.35
    enable_specular_reflection: bool = True


@dataclass(frozen=True, slots=True)
class TopologyConfig:
    """
    Differentiable Graph Laplacian, Finsler-Chordal, Eikonal & Bravais Lattice Topology Parameters.
    """
    chebyshev_order: int = 4
    num_eigenmodes: int = 8
    moore_connectivity: int = 8
    laplacian_regularization: float = 1e-5
    spectral_scale: float = 0.25
    cavity_relaxation_iters: int = 12


@dataclass(frozen=True, slots=True)
class MultiscaleLieConfig:
    """
    Kronecker-Factored Multiscale Lie Coordinate Resonator Parameters.
    """
    supported_scale_factors: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)
    num_harmonic_phases: int = 2
    resonance_temperature: float = 0.25


@dataclass(frozen=True, slots=True)
class PottsLatchingConfig:
    """
    Annealed Potts-Ginzburg-Landau Multi-Well Energy Landscape & Unified Sphere Readout Configuration.
    """
    tau_max: float = 1.00
    tau_min: float = 0.05
    tau_decay_power: float = 1.50
    energy_sigma_sq: float = 0.25
    restorative_force_gain: float = 0.35
    cosine_logit_scale_init: float = 12.00
    straight_through_latching: bool = True
    prototype_dimension: int = 128


@dataclass(frozen=True, slots=True)
class SlotAttentionConfig:
    """
    Asymmetric Directed Relational Hypergraph Object-Centric Slot Binder.
    """
    num_slots: int = 12
    d_slot: int = 128
    slot_iterations: int = 3
    hypergraph_heads: int = 4
    d_geo_rel: int = 32
    slot_sparsity_top_k: int = 6


@dataclass(frozen=True, slots=True)
class SpectralMDLConfig:
    """
    Schatten-1 Nuclear Norm & Von Neumann Spectral Entropy Minimum Description Length Parameters.
    """
    fg_class_weight: float = 1.25
    bg_class_weight: float = 1.00
    focal_gamma: float = 1.25
    lambda_nuclear_rank: float = 1e-4
    lambda_spectral_entropy: float = 1e-3
    lora_adaptation_rank: int = 2
    schatten_p_order: float = 1.00
    schatten_epsilon: float = 1e-5


@dataclass(frozen=True, slots=True)
class KendallMultitaskConfig:
    """
    Strictly Positive Liebel-Koerner Kendall Homoscedastic Bayesian Multitask Balancer.
    """
    num_tasks: int = 8
    log_var_bound_max: float = 1.50
    min_query_weight_floor: float = 0.35
    governor_weight: float = 0.15
    init_log_vars: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class PlackettLuceConfig:
    """
    Two-Scale Plackett-Luce Modal Decision Engine for Attempt 1 and Attempt 2 Resolution.
    """
    num_hypotheses: int = 8
    temperature_beta: float = 0.10
    top2_divergence_threshold: float = 0.15
    entropy_bifurcation_threshold: float = 0.35


@dataclass(frozen=True, slots=True)
class TTTRefinementConfig:
    """
    Full-Network Geodesic Test-Time Adaptation Configuration under D4 x S9 Automorphisms.
    """
    canonical_time_budget_secs: int = 333
    min_time_budget_secs: int = 120
    max_boosted_budget_secs: int = 480
    max_gradient_steps: int = 30000
    base_learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    cosine_lr_floor_ratio: float = 0.05
    validation_interval_steps: int = 20
    post_consensus_polish_steps: int = 60
    momentum_reset_threshold_pct: float = 95.0
    use_augmented_loocv: bool = True
    enable_s9_color_augmentation: bool = True
    s9_augmentation_probability: float = 0.65
    pttsa_steps: int = 25
    pttsa_step_size: float = 0.05


@dataclass(frozen=True, slots=True)
class FGNSConfig:
    """
    Master Configuration Monad for the Finsler-Gauge Cellular Sheaf Network (FGNS v4.1.0).
    """
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    reasoning: RecursiveReasoningConfig = field(default_factory=RecursiveReasoningConfig)
    na3d: NA3Dv2Config = field(default_factory=NA3Dv2Config)
    sheaf: SheafConfig = field(default_factory=SheafConfig)
    size_oracle: SizeOracleConfig = field(default_factory=SizeOracleConfig)
    budget: PaletteBudgetConfig = field(default_factory=PaletteBudgetConfig)
    accro: AccroConfig = field(default_factory=AccroConfig)
    lorentz: LorentzManifoldConfig = field(default_factory=LorentzManifoldConfig)
    memory: EpisodicMemoryConfig = field(default_factory=EpisodicMemoryConfig)
    xor: XORInvolutionConfig = field(default_factory=XORInvolutionConfig)
    octowave: OctoWaveSSMConfig = field(default_factory=OctoWaveSSMConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    lie: MultiscaleLieConfig = field(default_factory=MultiscaleLieConfig)
    potts: PottsLatchingConfig = field(default_factory=PottsLatchingConfig)
    slots: SlotAttentionConfig = field(default_factory=SlotAttentionConfig)
    mdl: SpectralMDLConfig = field(default_factory=SpectralMDLConfig)
    kendall: KendallMultitaskConfig = field(default_factory=KendallMultitaskConfig)
    plackett_luce: PlackettLuceConfig = field(default_factory=PlackettLuceConfig)
    ttt: TTTRefinementConfig = field(default_factory=TTTRefinementConfig)

    def __post_init__(self) -> None:
        if self.octowave.d_model % 32 != 0:
            raise ValueError(
                f"Tensor Core alignment violation: d_model ({self.octowave.d_model}) "
                f"must be divisible by 32."
            )

        d = self.octowave.d_model
        if (
            self.slots.d_slot != d
            or self.size_oracle.d_model != d
            or self.accro.feature_dim != d
            or self.sheaf.d_node != d
        ):
            raise ValueError("Latent dimension mismatch across network submodules.")

        if self.sheaf.d_edge > self.sheaf.d_node:
            raise ValueError("Stiefel manifold violation: d_edge cannot exceed d_node.")

        if len(self.kendall.init_log_vars) != self.kendall.num_tasks:
            raise ValueError("Kendall multitask mismatch: init_log_vars length must equal num_tasks.")

        if self.potts.tau_min >= self.potts.tau_max:
            raise ValueError("Potts cooling violation: tau_min must be strictly less than tau_max.")

        if self.octowave.num_axes != 8:
            raise ValueError("Spatial isotropy violation: num_axes must be 8.")

        if self.kendall.log_var_bound_max > 2.0:
            raise ValueError("Multitask balance violation: log_var_bound_max must not exceed 2.0.")

        if self.reasoning.reasoning_steps < 2:
            raise ValueError("Recurrent reasoning depth violation: reasoning_steps must be >= 2.")

    @property
    def d_model(self) -> int:
        return self.octowave.d_model

    @property
    def vocab_size(self) -> int:
        return self.geometry.vocab_size

    @property
    def max_grid_size(self) -> int:
        return self.geometry.max_grid_size

    @property
    def max_support_pairs(self) -> int:
        return self.geometry.max_support_pairs

    @property
    def reasoning_steps(self) -> int:
        return self.reasoning.reasoning_steps

    @property
    def weight_decay(self) -> float:
        return self.ttt.weight_decay

    def compute_cooling_temperature(self, step: int, total_steps: int) -> float:
        progress = min(max(float(step) / max(float(total_steps), 1.0), 0.0), 1.0)
        decay = math.pow(1.0 - progress, self.potts.tau_decay_power)
        return self.potts.tau_min + (self.potts.tau_max - self.potts.tau_min) * decay

    def compute_recurrent_step_temperature(self, tau_step: int) -> float:
        total = self.reasoning.reasoning_steps
        if total <= 1:
            return self.reasoning.tau_cooling_min
        ratio = float(tau_step - 1) / float(total - 1)
        decay = math.pow(1.0 - ratio, self.reasoning.tau_decay_power)
        return self.reasoning.tau_cooling_min + (self.reasoning.tau_cooling_max - self.reasoning.tau_cooling_min) * decay

    def estimate_vram_footprint_mb(self, batch_size: int = 1) -> float:
        spatial_size = self.geometry.max_grid_size * self.geometry.max_grid_size
        k_static = self.geometry.max_support_pairs
        d_model = self.octowave.d_model
        dtype_bytes = 2 if self.hardware.enable_bf16_amp else 4

        weights_mb = 6.574171 * 1000 * 1000 * dtype_bytes / (1024 * 1024)
        tokens_count = batch_size * (1 + k_static) * spatial_size
        octowave_activation_mb = tokens_count * d_model * dtype_bytes / (1024 * 1024)
        accro_activation_mb = tokens_count * d_model * self.reasoning.reasoning_steps * dtype_bytes / (1024 * 1024)
        adam_states_mb = weights_mb * 2.0

        return weights_mb + adam_states_mb + octowave_activation_mb + accro_activation_mb + 64.0

