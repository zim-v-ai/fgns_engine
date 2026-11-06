# FGNS-TRM: Finsler-Gauge Cellular Sheaf Network for ARC-AGI-2

[![Monograph](https://img.shields.io/badge/Monograph-PDF-blue.svg)](paper/FGNS_TRM_Monograph.pdf)
[![License: MIT-0](https://img.shields.io/badge/License-MIT--0-green.svg)](LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11%2B-brightgreen.svg)]()

## Abstract & Overview

**FGNS-TRM** is an exploratory, continuous neural network architecture designed as an initial proof-of-concept for the [ARC Prize 2026 Paper Track](https://arcprize.org). Rather than relying on Large Language Model (LLM) code generation or combinatorial Domain-Specific Language (DSL) search, this project investigates a geometric alternative: whether canvas boundary estimation, discrete color-mass conservation, 3D spatial symmetry, and topological connectivity can be directly embedded into a compact **6.57M-parameter** continuous model trained from scratch.

The model unifies a universal geometric core with continuous differentiable operator relaxations across four primary architectural principles:

1. **Two-Stage Upstream Conservation Priors:** Contracts the unconstrained combinatorial phase space ($11^{900} \approx 10^{937}$) prior to spatial decoding. *Stage I* (`BoundaryMetricSizePredictor`) predicts canvas dimensions $(H^\star, W^\star)$ over $\mathbb{Z}_{61} \times \mathbb{Z}_{61}$, pruning exterior cells to $10^{K_{\mathrm{cap}}}$ ($K_{\mathrm{cap}} = H^\star W^\star \le 900$). *Stage II* (`EquivariantPaletteHistogramPredictor`) predicts exact integer color quotas $\sum N_c^\star = K_{\mathrm{cap}}$ via $S_{10}$-equivariant descriptors, an $80\%$ diode sink, and Hamilton-Hare apportionment, contracting states to multinomial quotas $\frac{K_{\mathrm{cap}}!}{\prod_c (N_c^\star)!}$ and injecting them as a soft Potts chemical potential bias.
2. **Orthogonal Color-Geometry Splitting (NA3D-Finsler-Lie):** Two-pass Modified Gram-Schmidt separates the 128D latent space into an invariant 11D color subspace $\mathcal{H}_{\mathrm{color}}$ and an orthogonal 117D geometric complement $\mathcal{H}_{\mathrm{geom}}$, ensuring machine-zero color leakage ($1.05 \times 10^{-7}$). Within $\mathcal{H}_{\mathrm{geom}}$, features act as 8 Non-Abelian Asymmetric 3D Token (NA3D) cells with oriented chiral volume $V = \det[L, C, R]$ and proper-time asymmetric Finsler-Randers drift.
3. **8-Axis Scanning, Cellular Sheaves & Spectral-Green Operators:** Combines line-parallel 8-compass CUDA state-space scans (OctoWave), Stiefel cellular sheaf diffusion on $V_{32}(\mathbb{R}^{128})$ with edge-stopping coboundary Laplacians, Schur-complement relational slots, a Leave-One-Out Holographic Support Tape with $\mathrm{SO}(3)$ parallel transport, symplectic True XOR parity, and a 23-channel differentiable Spectral-Green RKHS operator.
4. **Dual-State Cauchy Recurrence & Schatten-1 LOO-TTT:** Thought and draft states $(Z, P)$ evolve across 4 steps via analytical Cauchy resolvents with Lyapunov support gating, while transductive test-time adaptation optimizes low-rank weight deltas $\Delta W$ regularized by a Newton-Schulz Schatten-1 nuclear norm penalty.

---

## Technical Monograph

The comprehensive 22-page mathematical monograph, complete with proofs, proposition derivations, empirical ablations, and figures, is available in the repository:

- **Primary PDF:** [`FGNS_TRM_Monograph.pdf`](FGNS_TRM_Monograph.pdf)
- **Paper Directory:** [`paper/FGNS_TRM_Monograph.pdf`](paper/FGNS_TRM_Monograph.pdf)
- **Markdown Source:** [`paper/FGNS_TRM_Monograph.md`](paper/FGNS_TRM_Monograph.md)
- **Kaggle Paper Track Writeup:** [`artifacts/paper_track/writeup.md`](artifacts/paper_track/writeup.md)

---

## Architectural Components

| Component | Class / Module | Mathematical Role |
| --- | --- | --- |
| **Lattice Dimension** | `BoundaryMetricSizePredictor` | 8D 1-Jet geometric basis Stokes integration with Woodbury-Tikhonov Gram resolvent |
| **Palette Budget** | `EquivariantPaletteHistogramPredictor` | $S_{10}$-equivariant role descriptors with Hamilton-Hare integer quota apportionment |
| **Subspace Split** | `ModifiedGramSchmidt2` | Orthogonal projection $\mathbb{R}^{128} = \mathcal{H}_{\mathrm{color}}^{(11)} \oplus \mathcal{H}_{\mathrm{geom}}^{(117)}$ ($E P_{\mathrm{geom}} \equiv \mathbf{0}$) |
| **3D Frame Dynamics** | `NA3DFinslerLie` | $\mathrm{SO}(3)$ Rodrigues exponential rotation of curvature core $\mathbf{C}_u$ with Finsler drift |
| **State-Space Scan** | `OctoWaveSSM` | 8-compass line-parallel CUDA scan with Householder specular wall reflection |
| **Cellular Sheaves** | `StiefelCellularSheafDiffusion` | Edge-stopping coboundary Laplacian on Stiefel manifold $V_{32}(\mathbb{R}^{128})$ |
| **Green Operator** | `SpectralGreenWoodburyFunction` | Implicit adjoint PDE solver over 23-channel topological Green-RKHS basis |
| **Recurrent Decoder** | `DualStateCauchyRecurrentDecoder` | 4-step dual-state $(Z, P)$ relaxation via proper-time dilated Cauchy resolvents |
| **Test-Time Adaptation** | `TransductiveOperatorAdapter` | Low-rank perturbation adaptation under Newton-Schulz Schatten-1 nuclear penalty |

---

## Repository Structure

```text
fgns_engine/
├── FGNS_TRM_Monograph.pdf      # Complete 22-page academic monograph (PDF)
├── paper/
│   ├── FGNS_TRM_Monograph.md   # Mathematical monograph source (Markdown + KaTeX)
│   └── FGNS_TRM_Monograph.pdf  # Canonical PDF build
├── artifacts/
│   ├── figures/                # High-resolution academic figures (fig00–fig07)
│   └── paper_track/
│       └── writeup.md          # ARC Prize 2026 Paper Track writeup (<1,500 words)
├── fgns/
│   ├── models/                 # PyTorch modules (NA3D, Sheaf, Cauchy, Size, Palette)
│   ├── kernels/                # Triton and CUDA custom operators
│   ├── primitives/             # Geometric and group-theoretic primitives
│   └── solver/                 # Transductive inference and evaluation pipeline
├── scripts/
│   ├── build_monograph_pdf.py  # Automated PDF build pipeline with PyMuPDF verification
│   └── figures/                # Standalone reproduction scripts for figures fig00–fig07
├── tests/                      # PyTest unit and integration test suite
└── LICENSE                     # MIT-0 License
```

---

## Building the Monograph PDF

The publication-grade PDF is built using Playwright Chromium and MathJax 3, including an automated PyMuPDF audit for zero unrendered LaTeX expressions and verified figure placement:

```bash
# Ensure dependencies are installed
pip install playwright pymupdf markdown
playwright install chromium

# Compile and audit monograph PDF
python scripts/build_monograph_pdf.py
```

---

## Running Verification Tests

To verify component invariants (subspace orthogonality, unit Jacobian determinant, Stiefel constraints, and Cauchy decay):

```bash
pytest tests/ -v
```

---

## Citation & License

This research is released under the [MIT-0 License](LICENSE).

```bibtex
@article{zimovsky2026fgnstrm,
  title   = {Finsler-Gauge Cellular Sheaf Network (FGNS-TRM): Geometric and Conservation Inductive Biases for ARC-AGI-2 Reasoning Without Program Synthesis},
  author  = {Zimovsky, V.},
  journal = {ARC Prize 2026 Paper Track Technical Monograph},
  year    = {2026}
}
```
