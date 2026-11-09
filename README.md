# FGNS-TRM v5.0.0: Finsler-Gauge Cellular Sheaf Network for ARC-AGI-2

[![Monograph](https://img.shields.io/badge/Monograph-PDF-blue.svg)](paper/FGNS_TRM_Monograph.pdf)
[![License: MIT-0](https://img.shields.io/badge/License-MIT--0-green.svg)](LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11%2B-brightgreen.svg)]()

## Abstract & Research Program in Structural Computation

**FGNS-TRM v5.0.0** is an experimental **6.57M-parameter (`6,574,411`)** hybrid geometric-analytical neural prototype developed for the [ARC Prize 2026 Paper Track](https://arcprize.org). Rather than relying on Large Language Model (LLM) program synthesis or combinatorial search over hand-crafted Domain-Specific Languages (DSLs), this project investigates a foundational question in **Structural Computation**: *can compact adaptive systems embed structural constraints, disentangled geometric representations, local-to-global topological propagation, associative memory, and complexity-controlled adaptation directly into their differentiable operations, rather than reconstructing all structure statistically from flat latent vectors?*

We use `ARC-AGI-2` as a concrete experimental testbed to explore four coupled mechanisms:

1. **Two-Stage Upstream Structural Priors (`BoundaryMetricSizePredictor` & `EquivariantPaletteHistogramPredictor`):** Contracts the unconstrained padded phase space ($11^{900} \approx 10^{937}$) before pixel decoding. *Stage I* (`Method 1`) resolves target canvas dimensions $(H^\star, W^\star)$ over $\mathbb{Z}_{61} \times \mathbb{Z}_{61}$ via a Shared-SRAM CUDA flood-fill/ray kernel, Boltzmann-Gibbs hypothesis voting, and a Woodbury-Tikhonov 1-Jet Stokes resolvent. *Stage II* estimates integer color quotas $\sum_{c=0}^9 N_c^\star = K_{\mathrm{cap}}$ ($K_{\mathrm{cap}} = H^\star W^\star$) via $S_{10}$-equivariant 8D role descriptors, a diode annihilation sink ($>80\%$ support disappearance threshold), and straight-through Hamilton-Hare Largest Remainder Quota (`LRQ`) apportionment, injecting $\mathbf{N}^\star$ into the decoder as a **soft Potts chemical potential bias**.
2. **Orthogonal Color-Geometry Splitting & Non-Abelian 3D Frames (`NA3D-Finsler-Lie`):** Two-pass Modified Gram-Schmidt (`MGS2`) in IEEE-754 FP32 splits the 128D latent space into an 11D palette subspace $\mathcal{H}_{\mathrm{color}}^{(11)}$ and an orthogonal 117D geometric complement $\mathcal{H}_{\mathrm{geom}}^{(117)}$ ($Q P_{\mathrm{geom}} \equiv \mathbf{0}$, `Lemma 1`), reducing measured rotational color-subspace leakage from $0.7881$ to $1.05 \times 10^{-7}$ across our 25-task suite (and from $0.2739$ to $1.54 \times 10^{-8}$ on unit-prototype patch tests). Within $\mathcal{H}_{\mathrm{geom}}^{(117)}$, features act as 8 Non-Abelian Asymmetric 3D Token (`NA3D`) simplices $(\mathbf{L}, \mathbf{C}, \mathbf{R}, \boldsymbol{\beta})$ evolving via $\mathrm{SO}(3)$ Rodrigues rotations and asymmetric Finsler-Randers proper time.
3. **8-Axis `OctoWave`, Stiefel Cellular Sheaves, Object Slots & Spectral-Green Operator:** Combines line-parallel 8-compass CUDA C++17 state-space scans (`OctoWave`, achieving a diagonal impulse-response norm of $11.3104$ at the opposite corner of an $8 \times 8$ probe vs. $0.0000$ for a horizontal 2-axis scan), edge-stopping Stiefel cellular sheaf diffusion on $\mathcal{V}_{32}(\mathbb{R}^{128})$ (`Lemma 2`, empirical Dirichlet energy ratio $0.707$ vs. $7.840$ for sign-inverted anti-diffusion), 12 object slots with Woodbury-Schur extrapolation, a Leave-One-Out Holographic Support Tape with $\mathrm{SO}(3)$ parallel transport, an antisymmetric $J$-involutive `True XOR` operator, and a 23-channel `DifferentiableSpectralGreenOperator` (`SpectralGreenWoodburyFunction`, verified to $7.76 \times 10^{-10}$ max relative Jacobian error in `float64` `torch.autograd.gradcheck`).
4. **Dual-State Cauchy Recurrence & Perturbation-Delta Schatten-1 `LOO-TTT`:** Separate thought and draft states $(Z^{(\tau)}, P^{(\tau)})$ evolve across 4 steps via proper-time dilated Cauchy resolvents with Lyapunov-gated support-residual correction (+5.66% pixel accuracy over $T=1$), while transductive test-time adaptation (`Method 2`) optimizes task-specific weight perturbations $\Delta W = W - W_0$ under a Newton-Schulz Schatten-1 nuclear norm penalty, compressing the Schatten-1/Frobenius participation effective rank $r_{\mathrm{eff}}$ from $3.46$ to $1.10$ and reducing 5-task continuous 8-step query loss from $0.4981$ to $0.4736$.

---

## Technical Monograph & Empirical Audit Artifacts

- **Primary Monograph PDF:** [`FGNS_TRM_Monograph.pdf`](FGNS_TRM_Monograph.pdf)
- **Paper Directory PDF:** [`paper/FGNS_TRM_Monograph.pdf`](paper/FGNS_TRM_Monograph.pdf)
- **Markdown Source:** [`paper/FGNS_TRM_Monograph.md`](paper/FGNS_TRM_Monograph.md)
- **Kaggle Paper Track Writeup (<1,500 words):** [`artifacts/paper_track/writeup.md`](artifacts/paper_track/writeup.md)
- **25-Task 8-Subsystem Ablation Telemetry:** [`artifacts/logs/ablations/master_ablation_report.json`](artifacts/logs/ablations/master_ablation_report.json) & [`.md`](artifacts/logs/ablations/master_ablation_report.md)
- **Live Figure Empirical Telemetry & 2D Grids (`fig01`–`fig07`):** [`artifacts/logs/ablations/figures_empirical_audit.json`](artifacts/logs/ablations/figures_empirical_audit.json) & [`.md`](artifacts/logs/ablations/figures_empirical_audit.md)

---

## Architectural Components (`6,574,411` Parameters)

| Component | Class / Module | Mathematical Role & Verified Diagnostic |
| :--- | :--- | :--- |
| **Stage I Size Prior** | `BoundaryMetricSizePredictor` | Coupled Boltzmann-Gibbs & Woodbury-Tikhonov 1-Jet Stokes solver (`76.74%` on 120 Public Eval tasks; `88.0%`, `0.40 px` MAE on 25-task suite) |
| **Stage II Quota Prior** | `EquivariantPaletteHistogramPredictor` | $S_{10}$-equivariant 8D role descriptors, $>80\%$ diode sink & straight-through Hamilton-Hare `LRQ` (`-59.1%` EMD, `-3.7x` color hallucinations) |
| **Subspace Split (`Lemma 1`)** | `_get_exact_orthonormal_palette_basis` | Two-pass FP32 `MGS2` orthogonal split $\mathbb{R}^{128} = \mathcal{H}_{\mathrm{color}}^{(11)} \oplus \mathcal{H}_{\mathrm{geom}}^{(117)}$ ($Q P_{\mathrm{geom}} \equiv \mathbf{0}$, rotational color leakage `1.05e-7` vs `0.7881`) |
| **3D Frame Dynamics** | `NA3DFinslerLieEngine` | $\mathrm{SO}(3)^8$ Rodrigues rotation of 3D simplices $(\mathbf{L}, \mathbf{C}, \mathbf{R}, \boldsymbol{\beta})$ (`+6.54%` pixel acc, `+8.03%` FG acc over flat 0D ViT tokens) |
| **8-Axis Wavefront Scan** | `OctoWaveSSM` | Line-parallel CUDA C++17 8-compass scan with Householder specular wall reflection (`11.3104` diagonal impulse-response norm vs `0.0000` 2-axis) |
| **Stiefel Cellular Sheaf (`Lemma 2`)** | `SpatialMooreSheafDiffusion` & `CellularSheafOperator` | 5th-order Newton-Schulz Stiefel $\mathcal{V}_{32}(\mathbb{R}^{128})$ restriction maps & Perona-Malik-Randers barrier (`0.707` Dirichlet ratio vs `7.840` anti-diffusion) |
| **Spectral-Green Operator** | `SpectralGreenWoodburyFunction` | Closed-form forward & implicit-differentiation backward solve over 23-channel basis (`7.76e-10` max relative Jacobian error in FP64 `gradcheck`) |
| **Recurrent Decoder** | `DualStateCauchyRecurrentDecoder` | 4-step $(Z^{(\tau)}, P^{(\tau)})$ proper-time dilated Cauchy resolvent flow (`+5.66%` pixel acc over $T=1$, `90.7%` monotonic step descent) |
| **Test-Time Adapter (`Method 2`)** | `TransductiveOperatorAdapter` | Geodesic $\mathfrak{so}(3)^8$ + Perturbation-Delta Schatten-1 `LOO-TTT` ($r_{\mathrm{eff}}: 3.46 \to 1.10$, 5-task query loss `0.4981 -> 0.4736`, `100%` on $D_4\times S_9$ stress test) |

---

## Reproducing Figures, Monograph PDF & Verification Tests

```powershell
# 1. Compile Rust 3D GGX ray-tracer, generate all 8 figures (fig00-fig07), and export empirical audit JSON/MD
python scripts/figures/generate_all_figures.py

# 2. Compile and audit the academic monograph PDF (Playwright + MathJax 3 + PyMuPDF verification)
python scripts/build_monograph_pdf.py

# 3. Run the 27 diagnostic verification gates (19 engine gates + 8 ablation gates)
pytest tests/ -v
