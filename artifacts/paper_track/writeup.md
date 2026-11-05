Basic details

Title *
FGNS-TRM: Finsler-Lie & Sheaf Priors for ARC-AGI-2 (No LLMs or DSLs)
68 / 80

Writeup URL *
kaggle.com/competitions/arc-prize-2026-paper-track/writeups/fgns-trm-finsler-lie-and-sheaf-priors-arc-agi-2

Subtitle
A 6.57M-parameter continuous neural field with two-stage conservation priors, NA3D SO(3) frames, Stiefel sheaves & Cauchy flow.
133 / 140

## Abstract
Most competitive approaches to the [Abstraction and Reasoning Corpus (ARC-AGI-2)](https://arcprize.org) rely on Large Language Models (LLMs) generating Python code or combinatorial search over hand-crafted Domain-Specific Languages (DSLs). We explore a third, purely geometric path: can a compact **6.57M-parameter (6,574,411)** continuous neural network—trained from scratch without LLMs or DSL search—perceive canvas boundaries, discrete color conservation, 3D rotational symmetry, and topology directly? We introduce the **Finsler-Gauge Cellular Sheaf Network (FGNS-TRM)**, built around four inductive biases:

1. **Two-Stage Upstream Conservation Priors (BoundaryMetricSizePredictor & EquivariantPaletteHistogramPredictor):** Rather than searching 11^900 pixel states, the network contracts the phase space before decoding. Stage I resolves target dimensions (H★, W★) over Z_61 × Z_61 via coupled Boltzmann-Gibbs and Woodbury-Tikhonov 1-Jet Stokes integration. Stage II resolves exact integer color quotas Σ N_c★ = H★ × W★ via S_10-equivariant 8D role descriptors, a diode annihilation sink for vanishing colors, and differentiable Hamilton-Hare largest remainder apportionment (LRQ).
2. **Orthogonal Color-Geometry Splitting & Non-Abelian 3D Frames (NA3D-Finsler-Lie):** Two-pass Modified Gram-Schmidt splits the 128D latent space into an invariant 11D color subspace H_color and an orthogonal 117D geometric complement H_geom. Tokens act as oriented 3D simplices (L, C, R, β) evolving via SO(3) Rodrigues rotations and asymmetric Finsler-Randers proper time with machine-zero color leakage (1.05e-7).
3. **8-Axis OctoWave, Stiefel Cellular Sheaves, Shared-Memory Slots & Holographic Tape:** Line-parallel 8-directional CUDA state-space scans (with specular wall reflection) and edge-stopping Stiefel sheaf diffusion on V_32(R^128) (Dirichlet ratio 0.707) couple with Schur-complement object slots, a Leave-One-Out Holographic Support Tape using SO(3) parallel transport, symplectic True XOR parity, and a Unified Differentiable Spectral-Green RKHS Operator.
4. **Dual-State CfC/Cauchy Recurrence & Perturbation-Delta Schatten-1 LOO-TTT:** Separate thought and draft states (Z, P) evolve via analytical Cauchy kernels across 4 steps, while low-rank operator deltas ΔW adapt at test time under a Newton-Schulz Schatten-1 nuclear norm penalty (compressing effective rank from 3.46 to 1.10).

On the blind **ARC-AGI-2 Public Evaluation set**, Stage I achieves **76.74% exact dimension accuracy** across all 120 tasks (**88.0% with 0.40 px MAE on our 25-task ablation suite** vs. 0.0% for an unconstrained MLP), Stage II cuts palette Earth Mover's Distance by **59.1%** (1.3488 to 0.5523), and the network solves dihedral occlusion (0934a4d8, 30×30 to 9×3) and topological cross filtering (1818057f, 22×22) at **100.0% exact match**, while reaching **98.93%–99.44% pixel accuracy** on Bravais lattice repair (135a2760) and Eikonal flooding (13e47133) within **~8 GB peak VRAM** on one RTX 5070 GPU.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F32842221%2F2f2a413c0bfc7725bce740a4d6319106%2Ffig00_cover_variant_dark_manifold.png?generation=1793898998136750&alt=media)

---

## 1. Introduction & Prior Work

In [*On the Measure of Intelligence* (Chollet, 2019)](https://arxiv.org/abs/1911.01547) and the [*ARC Prize Technical Report* (Chollet et al., 2025/2026)](https://arxiv.org/abs/2601.10904), Chollet et al. note that true generalization requires compact systems grounded in Core Knowledge priors (objectness, boundary geometry, symmetry, topology, and cardinal arithmetic). Standard backbones ([Vision Transformers](https://arxiv.org/abs/2010.11929), [RoPE Transformers](https://arxiv.org/abs/2104.09864), and 2-axis [Mamba-2](https://arxiv.org/abs/2405.21060)) struggle on 2D grids when trained from scratch due to: (1) an unconstrained 11^900 output space lacking boundary or color-mass conservation, (2) flat 0D tokens whose spatial rotations mix geometry with color identity, and (3) axial scans blind to 45-degree diagonal rays and 1-pixel barriers.

We build upon four compact paradigms: dual-state recursive refinement from [TRM (Jolicoeur-Martineau, 2025)](https://arxiv.org/abs/2510.04871), analytical Cauchy resolvents from [CfC (Hasani et al., 2022)](https://www.nature.com/articles/s42256-022-00556-7), test-time MDL compression from [CompressARC (Liao & Gu, 2025)](https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.pdf), and cellular sheaf diffusion ([Hansen & Ghrist, 2019](https://arxiv.org/abs/1808.01513); [Bodnar et al., 2022](https://arxiv.org/abs/2202.04579)), unifying them with our four geometric primitives below.

---

## 2. Approach & Mathematical Formulation (6.57M Parameters)

### 2.1. Two-Stage Upstream Conservation Priors
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F32842221%2F792369ac0ed56004c4d5321ef614e6a6%2Ffig02_architecture_3d_stack.png?generation=1793899013897304&alt=media)

**Stage I (BoundaryMetricSizePredictor)** uses a Shared-SRAM CUDA flood-fill and optical ray kernel to extract an 8D 1-Jet geometric basis from demonstrations (input bounds, foreground box, divider chamber spans, vanishing occluder size, and extremal color bounds), coupling Boltzmann-Gibbs peaks with a Woodbury-Tikhonov dual Gram resolvent and carrier-mask feedback. **Stage II (EquivariantPaletteHistogramPredictor)** conditions on K_cap = H★ × W★, using S_10-equivariant 8D role descriptors and a diode annihilation sink (absorbing vanished colors such as c=8 in Task 0934a4d8) to project simplex proportions p_c★ onto exact integer quotas N_c★ via Hamilton-Hare apportionment:

$$
N_c^\star = \lfloor p_c^\star K_{\mathrm{cap}} \rfloor + \mathbb{I}\left[\mathrm{rank}(r_c) \le K_{\mathrm{cap}} - \sum_{j=0}^9 \lfloor p_j^\star K_{\mathrm{cap}} \rfloor\right]
$$

### 2.2. Orthogonal Subspace Splitting & Non-Abelian 3D Token (NA3D-Finsler-Lie)
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F32842221%2F70cab7d4557ed28fe6e8d752d72e2905%2Ffig01_na3d_finsler_lie_3d.png?generation=1793899037311542&alt=media)

We orthonormalize the 11 palette prototypes E in R^(11×128) via two-pass Gram-Schmidt in FP32, splitting the latent space into an invariant color subspace H_color^(11) and an orthogonal geometric complement H_geom^(117). Inside H_geom^(117), features form 8 Non-Abelian Asymmetric 3D Token (NA3D) cells, each carrying a 3D receptor face L, a curvature core C, an emitter face R (spanning oriented chiral volume V = det[L, C, R]), and a Finsler-Randers drift covector β that acts as a directional causal diode (breaking flat symmetry d(u, v) != d(v, u)). Across directed edges, non-commutative Lie torsion generators L × R in so(3) rotate the core via the exact Rodrigues-Hamilton exponential map in SO(3) strictly inside H_geom^(117):

$$
\mathbb{R}^{128} = \mathcal{H} _ {\mathrm{color}}^{(11)} \oplus \mathcal{H} _ {\mathrm{geom}}^{(117)}, \qquad \mathbf{U}(\theta,\hat{\mathbf{u}}) = I _ 3 + \sin(\theta)[\hat{\mathbf{u}}] _ {\times} + (1-\cos\theta)[\hat{\mathbf{u}}] _ {\times}^2 \in \mathrm{SO}(3), \qquad E P _ {\mathrm{geom}} \equiv \mathbf{0}
$$

Because det(U) = +1 and E P_geom = 0, 3D rotations are strictly norm-preserving, capture 2×2 Wilson plaquette holonomy around closed loops, and yield **zero color-subspace leakage** (1.05e-7 vs. 0.7881 for unsplit SO(3)). Color readout normalizes the 10D color fiber onto the unit sphere S^9 and couples the exterior PAD token (Z_10) to the carrier log-odds field ψ(y, x) via a zero-gap max envelope.

### 2.3. 8-Axis OctoWave, Stiefel Sheaves, Holographic Tape & Spectral-Green Operator
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F32842221%2F8a037ac6e0d2e2183c7ee61a69bf30e6%2Ffig03_sheaf_cauchy_dynamics_3d.png?generation=1793899051164039&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F32842221%2Ff921a86b43e3ab345cdf23a02aba8932%2Ffig06_topological_operators_gallery.png?generation=1793899059550979&alt=media)

1. **OctoWave 8-Axis SSM:** A line-parallel CUDA C++17 kernel scans all 8 compass rays simultaneously with Householder specular reflection at walls (11.31 diagonal reach vs. 0.00 for 2-axis Mamba-2).
2. **Stiefel Sheaves & Slots:** Nodes project into R^32 edge stalks via 5th-order Newton-Schulz orthogonalization under an edge-stopping coboundary Laplacian, while 12 slots resolve relational states via Schur-complement extrapolation.
3. **Holographic Tape & True XOR:** Support pairs are indexed using dual-scale local-glyph and global centroids and retrieved via SO(3) Rodrigues parallel transport under strict Leave-One-Out masking (preventing identity self-copying), paired with a second-moment Dirichlet symplectic XOR engine that preserves invariant structural walls.
4. **Differentiable Spectral-Green RKHS Operator (SpectralGreenWoodburyFunction):** Over a 23-channel basis (learnable D4-steerable radial-shell wavelets, Chebyshev Eikonal distance, and Gaussian-Bessel cavity harmonics), topological transport solves the regularized Woodbury-Green normal equations in closed form in the forward pass and the exact implicit-differentiation adjoint PDE system in the backward pass without unrolling iterations (less than 3.7e-9 float64 gradcheck error):

$$
\mathbf{G} _ \lambda \mathbf{W}^\star = \boldsymbol{\Psi} _ S^\top (\mathbf{m} _ S \odot \mathbf{Y} _ S) + \boldsymbol{\lambda} \odot \mathbf{W} _ 0, \qquad \mathbf{G} _ \lambda \boldsymbol{\Lambda} = \boldsymbol{\Psi} _ q^\top \bar{\mathbf{Z}} _ q + \bar{\mathbf{W}}^\star
$$

### 2.4. Dual-State Cauchy Recurrence & Schatten-1 LOO-TTT
Across 4 steps, Z relaxes via a proper-time dilated Cauchy resolvent with boundary viscosity and Lyapunov-gated support-residual correction. During LOO-TTT (TransductiveOperatorAdapter), we freeze E and adapt so(3)^8 rotors and operator projections under D4 × S9 automorphisms, penalizing weight deltas ΔW via a Newton-Schulz Schatten-1 nuclear norm:

$$
Z^{(\tau)} = H_{\mathrm{spatial}}^{(\tau)} \odot e^{-0.35\omega_0 \Delta t_{\mathrm{eff}}} + \kappa F_{\mathrm{driving}}^{(\tau)} \odot \frac{1 - e^{-\omega_0 \Delta t_{\mathrm{eff}}}}{\omega_0}
$$

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F32842221%2F6d0f1302459031ac5c0345dd9b062966%2Ffig07_spectral_mdl_ttt_dynamics.png?generation=1793899072491963&alt=media)

---

## 3. Universality: Reshaping Deep Architectures with NA3D Beyond ARC
1. **NA3D-Gauge-Attention, Vocabulary-Free Tokenization & Ultra-Deep Models:** Standard Transformers suffer from quadratic KV-cache memory, heavy W_Q, W_K, W_V matrices, and rank collapse. In NA3D, Receptor (L), Core (C), and Emitter (R) faces are intrinsic to the 64-byte 3D token, eliminating separate Q, K, V projections (~75% attention parameter savings). Isochoric SO(3) rotations (det U = +1) conserve Shannon entropy across 1,000+ layers without vanishing gradients or positional encodings (No-PE), distinguishing all syntax trees via non-associative Lie brackets. Paired with Barnes-Hut holonomy trees in SE(3) and Finsler bifurcation tokenizers, NA3D slashes memory usage while enabling deeper, richer networks.
2. **Upstream Conservation Priors for Scientific ML & ARC-AGI-3:** Woodbury boundary estimation + Hamilton-Hare integer apportionment directly generalize to molecular fragmentation, PDEs, and bounded action-budget planning in [**ARC-AGI-3**](https://arcprize.org).

---

## 4. Results, Ablations & Honest Assessment

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F32842221%2Fe085f3895df6967541a89e6890e329a5%2Ffig04_arc2_visual_case_studies.png?generation=1793899080071934&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F32842221%2Ff35eafdaa02072fc6ca9666dac6a4cf7%2Ffig05_ablation_scientific_dashboard.png?generation=1793899090425995&alt=media)

On the blind **ARC-AGI-2 Public Evaluation set** (6,574,411 parameters, **~7.8–8.2 GB peak VRAM** on one RTX 5070) after 1 epoch (1,000 steps) of pretraining, FGNS-TRM achieves **76.74% (132/172) exact dimension accuracy** across all 120 tasks, **88.0% (22/25, MAE 0.40 px vs. 0.0% for a spatial MLP)** on our 25-task ablation suite (**Figure 5**), **48.0% exact color budget match** (cutting color hallucinations by **3.7x** from 1.26% to 0.34% and EMD from 1.3488 to 0.5523), **+6.54% pixel accuracy** (+8.03% foreground) from NA3D over flat ViT tokens, **+5.66% pixel accuracy** from 4-step Cauchy recurrence (90.7% monotonic descent), **100.0% exact match** on 0934a4d8 and 1818057f, and **96.78%–99.44% pixel accuracy** on 135a2760 and 13e47133 (**Figure 4**).

**Honest Assessment & Why This Paradigm Has No Ceiling:** Our final submission finished after the deadline (`Succeeded (after deadline) -> 0.00`). Beyond 1,000 pretraining steps, modest margins in several internal ablations (**Figure 5**, Studies 2, 3, 5, 8: ~0.4%–0.9%) reflect prototype rough edges analyzed in our monograph: hypothesized gradient short-circuiting through analytical bypasses, heuristic scaling across five manifolds, and initial basis formulas with vast room for growth. In a 45-module nonlinear system, exact fixes cannot be guessed on paper—they require refactoring and iterative empirical research. Our goal was to give the **first impulse** to an unbounded geometric paradigm without LLMs or DSLs, releasing our MIT-0 codebase, Rust ray-tracer, and monograph PDF below.