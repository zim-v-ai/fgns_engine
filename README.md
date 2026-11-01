# FGNS-TRM: Finsler-Gauge Cellular Sheaf Network with Spectral MDL Test-Time Adaptation for Abstract Reasoning

(Клеточная пучковая сеть с калибровочной связностью Финслера и спектральной MDL-адаптацией во время тестирования для задач абстрактного мышления)

## Overview

FGNS-TRM is a pure neural reasoning engine designed for the ARC Prize 2026.
It completely abandons combinatorial Domain-Specific Languages (DSL) in favor of:

1. 8-Axis Isotropic Associative State Space Scanning (OctoWave Triton Kernel).
2. Differentiable Graph Laplacian Spectral Topology (Moore N8 Connectedness).
3. Multiscale Lie Group Coordinate Fibers (Analytic Kronecker X (x) X Invariance).
4. Annealed Potts-Ginzburg-Landau STE Latching (Eliminating the 99.5% Trap).
5. True Kolmogorov-Solomonoff Spectral Rank Minimum Description Length (MDL) TTT.

## Academic Standard Nomenclature & Components

| Component | Academic Standard Name | Scientific Foundation |
| --- | --- | --- |
| Main Model | `FinslerGaugeSheafNetwork` (FGNS) | Transductive network based on non-abelian cellular sheaf and Finsler metric |
| Lattice Dimension | `BoundaryMetricSizePredictor` | Output lattice dimension predictor based on boundary flux and metric density |
| Palette Budget | `EquivariantPaletteHistogramPredictor` | S_10-equivariant color multiset predictor with Haar projection (LRQ) |
| Recurrent Decoder | `DualStateCauchyRecurrentDecoder` | Dual-state recurrent decoder (Z^(tau), Y^(tau)) with analytic Cauchy resolvent |
| Active Loss | `ActiveDomainFocalLoss` | Focal loss computed strictly within the active domain lattice mask |
| Multi-task Rank Loss | `SchattenNuclearMDLObjective` | Multitask objective with effective rank regularization via Schatten-1 nuclear norm |
| TTT Adaptation | `TransductiveOperatorAdapter` | Test-Time Training module in low-rank operator and rule geodesic vector space |
| Evaluation Pipeline | `EvaluationPipelineRunner` | Inference, cross-validation, and submission serialization pipeline |
| Path Resolution | `DatasetPathResolver` | Path resolution utility for local and containerized environments |
| Coordinate Encoder | `TopologicalCoordinateEncoder` | Input encoder unifying graph Laplacian spectral modes, color role invariants, and SO(3) triads |
| Cross-Attention Memory | `SupportCrossAttentionMemory` | Non-local associative support memory with Rodrigues transport in SO(3) |

## Hardware Target

- Native support for NVIDIA RTX 5070 (Blackwell 12GB GDDR7).
- Operational peak VRAM ceiling: < 2.5 GB.
- Test-Time Training throughput: > 65 TPS (~22,000 steps per 333s budget).

## Execution Protocol

### 3-Stage Meta-Learning, Transduction, and Evaluation

 **Этап I (Офлайн-предобучение):**

  ```powershell
  python -m fgns.train.meta_train.pipeline --data data/arc-agi_training_challenges.json --val-data data/arc-agi_training_challenges.json --val-size 100 --epochs 5 --lr 0.0003
  ```

 **Этап II (Трансдуктивное LOO мета-обучение на `evaluation` с безградиентной проверкой на тесте):**

  ```powershell
  python -m fgns.train.meta_train.pipeline --data data/arc-agi_evaluation_challenges.json --val-data data/arc-agi_evaluation_challenges.json --mode loo --epochs 3 --lr 0.00015
  ```

 **Этап III (Адаптация во время тестирования с $D_4 \times S_{10}$ и инспекция решений):**

  ```powershell
  python -m fgns.solver.submit.eval_runner --data data/arc-agi_evaluation_challenges.json --tasks 5 --time 333
  python scripts/inspect_submission.py --limit 5
  ```
