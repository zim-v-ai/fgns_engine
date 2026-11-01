# FGNS-TRM: Complete Architectural Ablation Battery
*Author: Vyacheslav Zimovsky (v.zimovsky.ai@gmail.com)*
*Benchmark Timestamp: 2026-11-01T16:00:26Z | Device: cuda*

## Table 1: Representation & Orientation: NA3D Stiefel Frame Transport vs. Baselines
| Architecture Variant | Frame Bundle | Query Loss (lower is better) | Angular Error (rad) |
|---|---|---|---|
| Isotropic Scalar | Identity ($I$) | 1.5389 | 1.5708 |
| Planar SO(2) | $SO(2)$ Circle | 1.4818 | 0.0000 |
| **NA3D Stiefel (FGNS-TRM)** | **$\text{St}(3, d)$ Lie Rotor** | **1.9243** | **0.0821** |

## Table 2: Non-Abelian Cellular Sheaf Connection vs. Isotropic GCN vs. Attention
| Graph Diffusion Mechanism | Restriction Maps | Query Loss | Dirichlet Energy $\mathcal{E}_\mathcal{F}$ (higher resists oversmoothing) |
|---|---|---|---|
| Standard GCN | $W = I$ (Isotropic) | 1.7402 | 3.0232 |
| Graph Attention (GAT) | $\alpha \in [0, 1]$ (Scalar) | 1.7119 | 3.0091 |
| **Cellular Sheaf (FGNS-TRM)** | **$W \in O(d)$ (Orthogonal)** | **1.8548** | **22.1520** |

## Table 3: Context Aggregation: OctoWave 8-Axis SSM vs. Conv2D vs. 2D Attention
| Mechanism | Scan Directions | Query Loss | Latency (ms) | Complexity |
|---|---|---|---|---|
| Dilated Conv2D | Local 3x3 | 1.8234 | 249.8 | $\mathcal{O}(HW)$ |
| 2-Axis Mamba | 2 Cardinal | 1.6173 | 294.1 | $\mathcal{O}(HW)$ |
| Full 2D Attention | Global Pairwise | 1.7588 | 614.9 | $\mathcal{O}((HW)^2)$ |
| **OctoWave SSM (FGNS-TRM)** | **8 Directions (Triton)** | **1.7554** | **331.0** | **$\mathcal{O}(HW)$** |

## Table 4: Iterative Dynamics: Cauchy Resolvent vs. Discrete ResNet vs. GRU
| State Transition | Integration Type | Query Loss | Monotonic Descent (%) |
|---|---|---|---|
| Feedforward Single-Pass | None | 1.8818 | 0.0% |
| Discrete ResNet | $x_{t+1} = x_t + f(x_t)$ | 1.9667 | 62.5% |
| Recurrent GRU | Gated Discrete | 2.2995 | 75.0% |
| **Cauchy Resolvent (FGNS-TRM)** | **$(zI - A)^{-1}$ Complex Contour** | **1.7709** | **96.5%** |

## Table 5: Test-Time Adaptation: Spectral MDL (Schatten-1) vs. Standard L2 Weight Decay
| TTT Regularization Objective | Mathematical Form | Effective Rank $r_{{\text{{eff}}}}$ | Query Generalization Loss |
|---|---|---|---|
| Zero TTT | No adaptation | 2.55 | 1.6334 |
| Standard L2 | $\lambda \|\Delta W\|_2^2$ | 2.51 | 0.6007 |
| Schatten-2 Frobenius | $\|\Delta W\|_F^2$ | 2.51 | 1.7963 |
| **Spectral MDL (FGNS-TRM)** | **Schatten-1 $\|\Delta W\|_*$** | **2.30** | **0.3073** |

## Table 6: Lattice Dimension Prediction: Darboux Boundary Flux vs. Global Pooling
| Dimension Predictor | Mechanism | Exact Match (H, W) (%) | MAE ($|\Delta H| + |\Delta W|$) |
|---|---|---|---|
| Identity Baseline | Trivial ($H_{in}, W_{in}$) | 60.0% | 4.00 |
| Global Avg Pooling | Spatial Mean + Linear | 100.0% | 0.00 |
| Spatial Flattened MLP | Dense Projection | 60.0% | 2.00 |
| **Boundary Metric (FGNS-TRM)** | **Darboux Boundary Flux** | **100.0%** | **0.00** |

## Table 7: Color Conservation: S_10 Haar LRQ vs. Unconstrained Softmax
| Palette Estimator | Equivariance | Color Hallucination Rate (%) | Earth Mover's Distance |
|---|---|---|---|
| Unconstrained Softmax | None | 22.00% | 1.9344 |
| Non-Equivariant MLP | None | 12.00% | 1.3534 |
| **Haar LRQ (FGNS-TRM)** | **$S_{10}$ Exact Equivariant** | **0.00%** | **0.0240** |

## Table 8: Test-Time Optimizer: SO(3) Rodrigues Geodesic vs. Euclidean LoRA
| Adaptation Method | Parameter Space | Steps to Convergence | Query Loss |
|---|---|---|---|
| Full Fine-Tuning | Full Euclidean ($W$) | 13.4 | 0.1301 |
| Euclidean LoRA | Low-Rank Euclidean ($BA$) | 10.0 | 0.3248 |
| **Geodesic SO(3) (FGNS-TRM)** | **Lie Group Geodesic ($\omega \in \mathfrak{so}(3)$)** | **6.0** | **0.3753** |
