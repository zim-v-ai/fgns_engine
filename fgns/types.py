"""
Location: // fgns_engine/fgns/types.py
Description: Mathematical State Monads, Strongly-Typed Data Contracts, and Tensor Containers
             for the Finsler-Gauge Cellular Sheaf Network (FGNS v4.1.0 Academic Standard):
               1. Discrete lattice primitives with exact invariant verification.
               2. Analytical Closed-Form Cellular Response Operator (Cauchy Resolvent) flow monads.
               3. Pseudo-Riemannian Lorentz Hyperboloid rule representations (H^D_kappa).
               4. Lie Transition State Monad capturing Cartan decomposition gl(D) = Sym(D) + so(D).
               5. Universal Proper-Time Frame Boundary Monad & Full D4 Orbit Inpainting State.
               6. S10-Equivariant 8D Color Structural Role Monad & Diode Sink Mask.
               7. Chamber-Local 2D Bravais Lattice Periodicity & Eikonal Level-Set State Monads.
               8. Static shape-bucketed multi-task batch tensors with pair_valid_mask.
Language: Python 3.12+ / PyTorch 2.5+ / High-Performance Typed Monads
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import enum
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


class PaletteColor(enum.IntEnum):
    """
    Canonical discrete color alphabet of the Abstraction and Reasoning Corpus (0..9 + PAD=10).
    """
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    MAGENTA = 6
    ORANGE = 7
    CYAN = 8
    MAROON = 9
    PAD = 10


@dataclass(frozen=True, slots=True)
class GridData:
    """
    Immutable zero-copy Euclidean grid representation.
    """
    height: int
    width: int
    data: Tuple[int, ...]

    def __post_init__(self) -> None:
        expected_len = self.height * self.width
        if len(self.data) != expected_len:
            raise ValueError(
                f"Grid dimension mismatch: height={self.height}, width={self.width} "
                f"requires {expected_len} elements, got {len(self.data)}."
            )

    @classmethod
    def from_2d(cls, raw_grid: Sequence[Sequence[int]]) -> GridData:
        h = len(raw_grid)
        w = len(raw_grid[0]) if h > 0 else 0
        flattened: List[int] = []
        for row in raw_grid:
            if len(row) != w:
                raise ValueError("Inconsistent row widths in ARC grid sequence.")
            flattened.extend(row)
        return cls(height=h, width=w, data=tuple(flattened))

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> GridData:
        if tensor.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got shape {tensor.shape}.")
        h, w = tensor.shape
        flat_list = tensor.detach().cpu().to(torch.int64).contiguous().view(-1).tolist()
        return cls(height=h, width=w, data=tuple(flat_list))

    def to_2d(self) -> List[List[int]]:
        return [
            list(self.data[r * self.width : (r + 1) * self.width])
            for r in range(self.height)
        ]

    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        arr = np.array(self.data, dtype=np.int64).reshape((self.height, self.width))
        tensor = torch.from_numpy(arr)
        return tensor.to(device) if device is not None else tensor

    def get_pixel(self, y: int, x: int) -> int:
        if 0 <= y < self.height and 0 <= x < self.width:
            return self.data[y * self.width + x]
        return int(PaletteColor.PAD)

    def compute_histogram(self) -> Tuple[int, ...]:
        counts = [0] * 10
        for val in self.data:
            if 0 <= val < 10:
                counts[val] += 1
        return tuple(counts)


@dataclass(frozen=True, slots=True)
class ARCPair:
    """
    Atomic representation of an input-to-output demonstration or test challenge pair.
    """
    input_grid: GridData
    output_grid: Optional[GridData] = None

    @property
    def is_test(self) -> bool:
        return self.output_grid is None

    @property
    def is_isometry(self) -> bool:
        if self.output_grid is None:
            return False
        return (
            self.input_grid.height == self.output_grid.height
            and self.input_grid.width == self.output_grid.width
        )

    @property
    def scale_multipliers(self) -> Optional[Tuple[int, int]]:
        if self.output_grid is None:
            return None
        in_h, in_w = self.input_grid.height, self.input_grid.width
        out_h, out_w = self.output_grid.height, self.output_grid.width
        if in_h > 0 and in_w > 0 and out_h % in_h == 0 and out_w % in_w == 0:
            return (out_h // in_h, out_w // in_w)
        return None

    def compute_instance_delta_histogram(self) -> Optional[Tuple[int, ...]]:
        if self.output_grid is None:
            return None
        hist_in = self.input_grid.compute_histogram()
        hist_out = self.output_grid.compute_histogram()
        return tuple(hist_out[c] - hist_in[c] for c in range(10))


@dataclass(frozen=True, slots=True)
class TaskMetadata:
    """Analytical geometric invariants extracted prior to neural inference."""
    max_input_h: int
    max_input_w: int
    max_output_h: int
    max_output_w: int
    num_train_pairs: int
    num_test_pairs: int
    is_scale_multiple: bool
    scale_factor_h: int
    scale_factor_w: int


@dataclass(frozen=True, slots=True)
class ARCTask:
    """
    Complete representation of an abstract reasoning task instance.
    """
    task_id: str
    train: Tuple[ARCPair, ...]
    test: Tuple[ARCPair, ...]

    def extract_metadata(self) -> TaskMetadata:
        max_in_h = max(p.input_grid.height for p in self.train + self.test)
        max_in_w = max(p.input_grid.width for p in self.train + self.test)

        valid_out_h = [p.output_grid.height for p in self.train if p.output_grid is not None]
        valid_out_w = [p.output_grid.width for p in self.train if p.output_grid is not None]

        max_out_h = max(valid_out_h) if valid_out_h else max_in_h
        max_out_w = max(valid_out_w) if valid_out_w else max_in_w

        scale_h = 1
        scale_w = 1
        is_mult = False

        if valid_out_h and self.train[0].output_grid is not None:
            first_in = self.train[0].input_grid
            first_out = self.train[0].output_grid
            if first_in.height > 0 and first_in.width > 0:
                if first_out.height % first_in.height == 0 and first_out.width % first_in.width == 0:
                    kh = first_out.height // first_in.height
                    kw = first_out.width // first_in.width
                    consistent = all(
                        p.output_grid.height == p.input_grid.height * kh
                        and p.output_grid.width == p.input_grid.width * kw
                        for p in self.train
                        if p.output_grid is not None
                    )
                    if consistent:
                        scale_h = kh
                        scale_w = kw
                        is_mult = True

        return TaskMetadata(
            max_input_h=max_in_h,
            max_input_w=max_in_w,
            max_output_h=max_out_h,
            max_output_w=max_out_w,
            num_train_pairs=len(self.train),
            num_test_pairs=len(self.test),
            is_scale_multiple=is_mult,
            scale_factor_h=scale_h,
            scale_factor_w=scale_w,
        )


@dataclass(slots=True)
class FrameBoundaryState:
    """
    Universal Riemannian-Finsler Cumulative Metric Time & Boundary Flux State.
    """
    y_min: torch.Tensor
    y_max: torch.Tensor
    x_min: torch.Tensor
    x_max: torch.Tensor
    tau_y: torch.Tensor
    tau_x: torch.Tensor
    d4_orbit_scores: Optional[torch.Tensor] = None
    d4_orbit_valid: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> FrameBoundaryState:
        return FrameBoundaryState(
            y_min=self.y_min.to(device, non_blocking=True),
            y_max=self.y_max.to(device, non_blocking=True),
            x_min=self.x_min.to(device, non_blocking=True),
            x_max=self.x_max.to(device, non_blocking=True),
            tau_y=self.tau_y.to(device, non_blocking=True),
            tau_x=self.tau_x.to(device, non_blocking=True),
            d4_orbit_scores=(
                self.d4_orbit_scores.to(device, non_blocking=True)
                if self.d4_orbit_scores is not None
                else None
            ),
            d4_orbit_valid=(
                self.d4_orbit_valid.to(device, non_blocking=True)
                if self.d4_orbit_valid is not None
                else None
            ),
        )


@dataclass(slots=True)
class ColorRoleState:
    """
    S10-Equivariant 8D Structural Role Descriptor & Diode Annihilation State.
    """
    role_vectors: torch.Tensor
    diode_sink_mask: torch.Tensor
    divider_color: torch.Tensor
    occluder_color: torch.Tensor
    seed_color: torch.Tensor

    def to(self, device: torch.device) -> ColorRoleState:
        return ColorRoleState(
            role_vectors=self.role_vectors.to(device, non_blocking=True),
            diode_sink_mask=self.diode_sink_mask.to(device, non_blocking=True),
            divider_color=self.divider_color.to(device, non_blocking=True),
            occluder_color=self.occluder_color.to(device, non_blocking=True),
            seed_color=self.seed_color.to(device, non_blocking=True),
        )


@dataclass(slots=True)
class BravaisPeriodicityState:
    """
    Chamber-Local 2D Bravais Lattice Translational Periodicity & Coset Consensus State.
    """
    period_y: torch.Tensor
    period_x: torch.Tensor
    period_confidence: torch.Tensor
    healed_logits_10: torch.Tensor

    def to(self, device: torch.device) -> BravaisPeriodicityState:
        return BravaisPeriodicityState(
            period_y=self.period_y.to(device, non_blocking=True),
            period_x=self.period_x.to(device, non_blocking=True),
            period_confidence=self.period_confidence.to(device, non_blocking=True),
            healed_logits_10=self.healed_logits_10.to(device, non_blocking=True),
        )


@dataclass(slots=True)
class EikonalLevelSetState:
    """
    Chebyshev Eikonal Wall-Distance d_partial(y, x) = min(ell_N, ell_S, ell_W, ell_E) - 1
    and Chamber Concentric Level-Set Propagation State.
    """
    wall_distance: torch.Tensor
    level_set_probs_10: torch.Tensor
    chamber_seed_count: torch.Tensor

    def to(self, device: torch.device) -> EikonalLevelSetState:
        return EikonalLevelSetState(
            wall_distance=self.wall_distance.to(device, non_blocking=True),
            level_set_probs_10=self.level_set_probs_10.to(device, non_blocking=True),
            chamber_seed_count=self.chamber_seed_count.to(device, non_blocking=True),
        )


@dataclass(slots=True)
class PaletteBudgetPrediction:
    """
    Instance-conditioned palette histogram prediction monad.
    """
    predicted_counts: torch.Tensor
    simplex_proportions: torch.Tensor
    input_counts: torch.Tensor
    target_capacity: torch.Tensor

    def to(self, device: torch.device) -> PaletteBudgetPrediction:
        return PaletteBudgetPrediction(
            predicted_counts=self.predicted_counts.to(device, non_blocking=True),
            simplex_proportions=self.simplex_proportions.to(device, non_blocking=True),
            input_counts=self.input_counts.to(device, non_blocking=True),
            target_capacity=self.target_capacity.to(device, non_blocking=True),
        )


@dataclass(slots=True)
class AccroFlowState:
    """
    Analytical Closed-Form Cellular Response Operator (Cauchy Resolvent) flow state.
    """
    c_relaxed: torch.Tensor
    v_tangent: torch.Tensor
    psi_decay: torch.Tensor
    psi_resolvent: torch.Tensor
    kinetic_energy: torch.Tensor

    def to(self, device: torch.device) -> AccroFlowState:
        return AccroFlowState(
            c_relaxed=self.c_relaxed.to(device, non_blocking=True),
            v_tangent=self.v_tangent.to(device, non_blocking=True),
            psi_decay=self.psi_decay.to(device, non_blocking=True),
            psi_resolvent=self.psi_resolvent.to(device, non_blocking=True),
            kinetic_energy=self.kinetic_energy.to(device, non_blocking=True),
        )


@dataclass(slots=True)
class TargetCoordinateBundle:
    """
    Target Canvas Coordinate Bundle Monad.
    """
    coords: torch.Tensor
    mask: torch.Tensor
    scale_ratio_h: torch.Tensor
    scale_ratio_w: torch.Tensor
    u_macro: torch.Tensor
    v_macro: torch.Tensor

    def to(self, device: torch.device) -> TargetCoordinateBundle:
        return TargetCoordinateBundle(
            coords=self.coords.to(device, non_blocking=True),
            mask=self.mask.to(device, non_blocking=True),
            scale_ratio_h=self.scale_ratio_h.to(device, non_blocking=True),
            scale_ratio_w=self.scale_ratio_w.to(device, non_blocking=True),
            u_macro=self.u_macro.to(device, non_blocking=True),
            v_macro=self.v_macro.to(device, non_blocking=True),
        )


@dataclass(slots=True)
class HyperbolicRuleState:
    """
    Lorentz Hyperboloid Rule Representation Monad on H^D_kappa.
    """
    point_hyperboloid: torch.Tensor
    tangent_velocity: torch.Tensor
    symmetric_invariant: torch.Tensor
    anti_symmetric_rot: torch.Tensor
    commutator: torch.Tensor
    chirality_ratio: torch.Tensor

    def to(self, device: torch.device) -> HyperbolicRuleState:
        return HyperbolicRuleState(
            point_hyperboloid=self.point_hyperboloid.to(device, non_blocking=True),
            tangent_velocity=self.tangent_velocity.to(device, non_blocking=True),
            symmetric_invariant=self.symmetric_invariant.to(device, non_blocking=True),
            anti_symmetric_rot=self.anti_symmetric_rot.to(device, non_blocking=True),
            commutator=self.commutator.to(device, non_blocking=True),
            chirality_ratio=self.chirality_ratio.to(device, non_blocking=True),
        )


@dataclass(slots=True)
class LieTransitionState:
    """
    Bilinear Lie algebra decomposition tensor monad.
    """
    symmetric_invariant: torch.Tensor
    anti_symmetric_rot: torch.Tensor
    commutator: torch.Tensor
    chirality_ratio: torch.Tensor

    def to(self, device: torch.device) -> LieTransitionState:
        return LieTransitionState(
            symmetric_invariant=self.symmetric_invariant.to(device, non_blocking=True),
            anti_symmetric_rot=self.anti_symmetric_rot.to(device, non_blocking=True),
            commutator=self.commutator.to(device, non_blocking=True),
            chirality_ratio=self.chirality_ratio.to(device, non_blocking=True),
        )


@dataclass(slots=True)
class XORParityState:
    """
    Analytic Nonlinear Asymmetric Involution Parity State.
    """
    vorticity_flux: torch.Tensor
    parity_selector: torch.Tensor
    involution_norm: torch.Tensor

    def to(self, device: torch.device) -> XORParityState:
        return XORParityState(
            vorticity_flux=self.vorticity_flux.to(device, non_blocking=True),
            parity_selector=self.parity_selector.to(device, non_blocking=True),
            involution_norm=self.involution_norm.to(device, non_blocking=True),
        )


@dataclass(slots=True)
class EntitySlots:
    """
    Factorized object-centric slot representations.
    """
    geometry: torch.Tensor
    color: torch.Tensor
    position: torch.Tensor
    euler: torch.Tensor
    mask: torch.Tensor

    def pack(self) -> torch.Tensor:
        return torch.cat([self.geometry, self.color, self.position, self.euler], dim=-1)

    def to(self, device: torch.device) -> EntitySlots:
        return EntitySlots(
            geometry=self.geometry.to(device, non_blocking=True),
            color=self.color.to(device, non_blocking=True),
            position=self.position.to(device, non_blocking=True),
            euler=self.euler.to(device, non_blocking=True),
            mask=self.mask.to(device, non_blocking=True),
        )


@dataclass(frozen=True, slots=True)
class PlackettLuceCandidates:
    """
    Two-Scale Plackett-Luce Modal Decision Container for Attempt 1 and Attempt 2.
    """
    mode_1_grid: GridData
    mode_2_grid: GridData
    hypotheses_grids: Tuple[GridData, ...]
    log_likelihoods: Tuple[float, ...]
    modal_margin: float


@dataclass(slots=True)
class SupportBatch:
    """
    Tensors representing the few-shot demonstration context with static K_static capacity.
    """
    inputs: torch.Tensor
    in_masks: torch.Tensor
    outputs: torch.Tensor
    out_masks: torch.Tensor
    target_histograms: Optional[torch.Tensor] = None
    target_capacities: Optional[torch.Tensor] = None
    pair_valid_mask: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> SupportBatch:
        return SupportBatch(
            inputs=self.inputs.to(device, non_blocking=True),
            in_masks=self.in_masks.to(device, non_blocking=True),
            outputs=self.outputs.to(device, non_blocking=True),
            out_masks=self.out_masks.to(device, non_blocking=True),
            target_histograms=(
                self.target_histograms.to(device, non_blocking=True)
                if self.target_histograms is not None
                else None
            ),
            target_capacities=(
                self.target_capacities.to(device, non_blocking=True)
                if self.target_capacities is not None
                else None
            ),
            pair_valid_mask=(
                self.pair_valid_mask.to(device, non_blocking=True)
                if self.pair_valid_mask is not None
                else None
            ),
        )


@dataclass(slots=True)
class QueryBatch:
    """
    Tensors representing the test-time query with static 30x30 canvas bounds.
    """
    inputs: torch.Tensor
    in_masks: torch.Tensor
    targets: Optional[torch.Tensor] = None
    target_masks: Optional[torch.Tensor] = None
    target_heights: Optional[torch.Tensor] = None
    target_widths: Optional[torch.Tensor] = None
    target_histograms: Optional[torch.Tensor] = None
    target_capacities: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> QueryBatch:
        return QueryBatch(
            inputs=self.inputs.to(device, non_blocking=True),
            in_masks=self.in_masks.to(device, non_blocking=True),
            targets=(
                self.targets.to(device, non_blocking=True)
                if self.targets is not None
                else None
            ),
            target_masks=(
                self.target_masks.to(device, non_blocking=True)
                if self.target_masks is not None
                else None
            ),
            target_heights=(
                self.target_heights.to(device, non_blocking=True)
                if self.target_heights is not None
                else None
            ),
            target_widths=(
                self.target_widths.to(device, non_blocking=True)
                if self.target_widths is not None
                else None
            ),
            target_histograms=(
                self.target_histograms.to(device, non_blocking=True)
                if self.target_histograms is not None
                else None
            ),
            target_capacities=(
                self.target_capacities.to(device, non_blocking=True)
                if self.target_capacities is not None
                else None
            ),
        )


@dataclass(slots=True)
class BatchData:
    """
    Unified multi-task training and inference batch container.
    """
    support: SupportBatch
    query: QueryBatch

    def to(self, device: torch.device) -> BatchData:
        return BatchData(
            support=self.support.to(device),
            query=self.query.to(device),
        )


@dataclass(slots=True)
class ModelOutput:
    """
    Comprehensive output trajectory emitted by the Finsler-Gauge Cellular Sheaf Network.
    """
    query_logits: torch.Tensor
    trajectory_logits: Tuple[torch.Tensor, ...]
    support_logits: torch.Tensor
    all_query_logits: List[torch.Tensor]
    all_support_logits: List[torch.Tensor]
    query_h_pred: torch.Tensor
    query_w_pred: torch.Tensor
    pred_capacity_ste: torch.Tensor
    crystallized_tokens: torch.Tensor
    p_color_matrix: torch.Tensor
    slots: EntitySlots
    hyperbolic_rule: Optional[HyperbolicRuleState] = None
    accro_flow: Optional[AccroFlowState] = None
    xor_state: Optional[XORParityState] = None
    query_budget: Optional[PaletteBudgetPrediction] = None
    support_budgets: Optional[List[PaletteBudgetPrediction]] = None
    dynamic_chemical_biases: Optional[Tuple[torch.Tensor, ...]] = None
    target_coords: Optional[TargetCoordinateBundle] = None
    carrier_psi_logits: Optional[torch.Tensor] = None
    sheaf_dirichlet_energy: Optional[torch.Tensor] = None
    frame_boundary: Optional[FrameBoundaryState] = None
    color_roles: Optional[ColorRoleState] = None
    bravais_state: Optional[BravaisPeriodicityState] = None
    eikonal_state: Optional[EikonalLevelSetState] = None


@dataclass(frozen=True, slots=True)
class LossTelemetry:
    """
    Fine-grained multi-head loss breakdown across all 8 Kendall channels.
    """
    total_loss: torch.Tensor
    loss_query: float
    loss_support: float
    loss_mask: float
    loss_size: float
    loss_budget: float
    loss_anti_collapse: float
    loss_program_mdl: float
    loss_orthogonality: float
    loss_xor_parity: float
    loss_lorentz_rule: float
    mask_iou: float
    spectral_entropy: float
    effective_rank: float
    kendall_weights: Tuple[float, ...]
    kendall_entropy: float


@dataclass(slots=True)
class TTTSnapshot:
    """
    State tracking Test-Time Adaptation convergence under Spectral MDL.
    """
    step: int
    loss: float
    true_mdl_score: float
    singular_spectrum: torch.Tensor
    birkhoff_entropy: float
    temperature: float
    folds_solved_mask: Tuple[bool, ...]
    is_consensus_achieved: bool


@dataclass(frozen=True, slots=True)
class AttemptPrediction:
    """Deterministic dual prediction for an individual test instance."""
    attempt_1: GridData
    attempt_2: GridData
    confidence: float
    d4_vote_spread: Tuple[int, int]


@dataclass(frozen=True, slots=True)
class TaskSubmission:
    """Official Kaggle-compliant submission container for a complete ARC task."""
    task_id: str
    predictions: Tuple[AttemptPrediction, ...]

    def to_kaggle_dict(self) -> Dict[str, List[Dict[str, List[List[int]]]]]:
        attempts_list = []
        for pred in self.predictions:
            attempts_list.append({
                "attempt_1": pred.attempt_1.to_2d(),
                "attempt_2": pred.attempt_2.to_2d(),
            })
        return {self.task_id: attempts_list}
