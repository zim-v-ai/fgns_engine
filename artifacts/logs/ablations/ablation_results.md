# FGNS-TRM: Controlled Empirical Ablation Experiments
*Generated on 2026-11-01T14:54:33Z | Device: cuda*

## Table 1: Cellular Sheaf Transport vs. Isotropic Graph Convolution
| Architecture Variant | Stalk Transport | Active Focal Loss (lower is better) | Gradient Stability (Norm) |
|---|---|---|---|
| Standard GCN Baseline | $W_e = I$ (Isotropic) | 0.0120 | 164.75 |
| **Cellular Sheaf (FGNS)** | $W_e \in O(d)$ (Stiefel Transport) | **0.0006** | **103.82** |

## Table 2: Test-Time Adaptation: Spectral MDL vs. Standard L2 Weight Decay
| Adaptation Objective | Regularization | Effective Rank $r_{\text{eff}}$ | Query Loss (Generalization) |
|---|---|---|---|
| Standard TTT | L2 Weight Decay ($\lambda=10^-4$) | 2.07 | 1.3194 |
| **Spectral MDL TTT (FGNS)** | Schatten-1 Nuclear Norm ($\|W\|_*$) | **2.53** | **0.0039** |

## Table 3: Lattice Dimension Prediction: Boundary Metric vs. Global Average Pooling
| Pooler Mechanism | Boundary Flux Sensitive | Exact Dimension Match (H, W) |
|---|---|---|
| Global Average Pooling | No | 0.0% (0/2) |
| **Boundary Metric Predictor (FGNS)** | **Yes (Darboux Flux)** | **100.0% (2/2)** |
