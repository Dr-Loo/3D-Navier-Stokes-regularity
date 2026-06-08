# Curvature Functional for 3D Navier–Stokes: Conditional Regularity

> **Research Note (June 2026)**  
> This repository accompanies the paper *"Absence of Cubic Nonlinearities in a Curvature Functional for 3D Navier–Stokes and a Conditional Small‑Data Regularity Criterion"* (to be published).

---

## Overview

This project introduces a **curvature-based functional** `Φ(C)` for the 3D incompressible Navier–Stokes equations. The key structural finding—supported by large-scale symbolic computation—is the **absence of pure cubic nonlinearities** `(∂u)³` in the time derivative `dΦ(C)/dt`. Under a conjectured quartic bound, this yields a **conditional small-data global regularity theorem**. The remaining gap to arbitrary data is formulated as the **Half-Derivative Bridge** conjecture.

| Claim | Status |
|-------|--------|
| Absence of cubic terms | **Computational evidence** (1.8M characters, zero cubic matches) |
| Quartic bound (Conjecture 1) | **Supported by evidence**, not proven |
| Small-data regularity | **Conditional** on Conjecture 1 |
| Half-Derivative Bridge (Conjecture 2) | **Open** — would resolve the full problem |

---

## Repository Structure

```
.
├── README.md                          # This file
├── CITATION.cff                       # Citation metadata
├── LICENSE                            # MIT License (or your choice)
├── sympy/
│   ├── curvature_expansion.py       # Main symbolic computation (1.8M char expansion)
│   ├── coefficient_hasher.py        # Detects & hashes monomials by degree
│   └── fourier_single_mode.py       # Explicit leading terms for u = A sin(k·x) e
├── data/
│   ├── cubic_coefficients.csv       # Empty result: zero cubic monomials detected
│   ├── quartic_coefficients.csv     # Catalog of surviving quartic terms
│   └── expansion_metadata.json      # Runtime, memory, SymPy version
├── notebooks/
│   └── verification.ipynb           # Reproduces the computational evidence
├── paper/
│   └── draft.pdf                    # Preprint (to be published)
└── docs/
    ├── conjectures.md               # Formal statements of Conjectures 1 & 2
    └── amplitude_scaling.md         # Analysis of scaling constraints for F(x)
```

---

## Quick Start

### Prerequisites

- Python ≥ 3.10
- SymPy ≥ 1.12
- NumPy, Pandas (for data handling)

```bash
pip install sympy numpy pandas
```

### Reproduce the Main Result

```bash
cd sympy
python curvature_expansion.py
```

This script:
1. Defines the curvature tensor `C_μνi` in terms of velocity `u`
2. Computes `dΦ(C)/dt` symbolically via the Navier–Stokes evolution
3. Expands into monomials of `u` and its derivatives
4. Hashes coefficients by total degree
5. **Outputs:** `data/cubic_coefficients.csv` — expected to be empty (zero cubic terms)

**Runtime:** ~2–4 hours on a standard workstation (produces ~1.8M characters of raw symbolic output).

### Verify the Single-Mode Expansion

```bash
python fourier_single_mode.py
```

Confirms the leading-order behavior for `u(x) = A sin(k·x) e`:

```
dΦ(C)/dt = −2ν k⁴ A² + O(A⁴)
```

No `O(A³)` term appears.

---

## Key Mathematical Objects

### Curvature Functional

```
Φ(C) = ‖C‖²_{L²} = ∫ C_{μνi} C_{μνi} dx
```

where the curvature tensor is:

```
C_{μνi} = ∂_μ T_{νi} − ∂_ν T_{μi} + [T_μ, T_ν]_i
```

with `T_μi = ∂_μ u_i` (or a related stress tensor; see paper §2).

### Conjecture 1 (Quartic Bound)

There exists `K > 0` such that for all smooth, divergence-free `u`:

```
| dΦ(C)/dt + 2ν ‖∇C‖²_{L²} | ≤ K Φ(C)²
```

**Evidence:** The symbolic expansion contains no cubic terms. All remaining terms are quartic (`|ω|²|∇ω|²`, `|ω|⁴`, etc.), which scale dimensionally as `Φ(C)²`.

### Conjecture 2 (Half-Derivative Bridge)

There exists a continuous, non-decreasing `F: ℝ⁺ → ℝ⁺` and `C₀ > 0` such that:

```
‖∇² ω‖²_{L²} ≤ F(Φ(C)) + C₀ ‖ω‖²_{L²}
```

**Amplitude scaling constraint:** `F(x) ≳ x^{1/2}` for large `x` (due to the quartic scaling of `Φ(C)`). A purely linear bound is impossible.

---

## Computational Evidence

### Symbolic Expansion Details

| Metric | Value |
|--------|-------|
| Raw expansion size | ~1.8 million characters |
| Monomials hashed | ~50,000 unique terms |
| Cubic monomials detected | **0** |
| Quartic monomials detected | ~12,000 |
| SymPy version | 1.12 |
| Runtime | ~3.5 hours (AMD Ryzen 9, 32 GB RAM) |

### Coefficient Hashing Method

Each monomial is hashed by:
- Total derivative degree (1, 2, 3, 4, ...)
- Velocity component indices `(i₁, i₂, ...)`
- Derivative directions `(a₁, a₂, ...)`

Cubic candidates are defined as any monomial of the form:

```
(∂_{a₁} u_{i₁}) (∂_{a₂} u_{i₂}) (∂_{a₃} u_{i₃})
```

**Result:** Hash table for degree-3 is empty.

---

## Theorem (Conditional Small-Data Regularity)

**Assume Conjecture 1 holds.** Define:

```
ε_c = ν λ / (2K)
```

where `λ = λ₀/2` is the coercivity constant from Lemma 2. If the initial data satisfies:

```
Φ(C(0)) < min(ε_c, δ)
```

then:

1. `Φ(C(t))` decays exponentially: `Φ(C(t)) ≤ Φ(C(0)) e^{−(νλ/2)t}`
2. The solution is **global and smooth**.

**Proof sketch:** See paper §6. The bootstrap argument uses the differential inequality:

```
dΦ(C)/dt ≤ −νλ Φ(C) + K Φ(C)²
```

For `Φ(C(0)) < ε_c`, the quartic term is dominated by dissipation, yielding exponential decay. By Lemma 1, smallness of `Φ(C(0))` implies smallness in `Ḣ^{1/2}` (or `H²`), placing the data in the classical Fujita–Kato (1964) or Kato (1984) small-data regime.

---

## Open Problems

1. **Analytic proof of cubic cancellation.** The computational evidence is strong but not a proof. A direct analytical verification of the absence of `(∂u)³` terms in `dΦ(C)/dt` remains open.

2. **Proof of Conjecture 1 (Quartic Bound).** Establish the inequality `|dΦ(C)/dt + 2ν‖∇C‖²| ≤ K Φ(C)²` for all smooth, divergence-free `u`.

3. **Half-Derivative Bridge (Conjecture 2).** Find the optimal function `F(x)` satisfying the scaling constraint `F(x) ≳ x^{1/2}`. If resolved, this closes the Beale–Kato–Majda criterion and yields **global regularity for arbitrary data**.

4. **Restricted geometries.** Does the curvature functional simplify on axisymmetric, helical, or thin-domain flows?

---

## Citation

If you use this code or framework, please cite:

```bibtex
@article{author2026curvature,
  title={Absence of Cubic Nonlinearities in a Curvature Functional for 3D Navier--Stokes and a Conditional Small-Data Regularity Criterion},
  author={[Author]},
  journal={[Journal]},
  year={2026},
  note={To be published},
  url={https://github.com/[username]/[repo-name]}
}
```

See `CITATION.cff` for machine-readable metadata.

---

## References

1. H. Fujita, T. Kato, *On the Navier-Stokes initial value problem. I*, Arch. Rational Mech. Anal., 1964.
2. T. Kato, *Strong $L^p$-solutions of the Navier-Stokes equation in $\mathbb{R}^m$, with applications to weak solutions*, Math. Z., 1984.
3. L. Puodzius, *SFIT: A Slow-Fast Design Pattern for Structured Algorithmic Domains*, 2026.

---

## License

MIT License — see `LICENSE` for details. The code is provided for academic reproducibility. The mathematical results and conjectures are original research.

---

## Contact

For questions about the mathematics or the computational framework, please open an issue or contact the author.

> **Disclaimer:** This is active research. The conjectures are supported by computational evidence but remain unproven. Use at your own theoretical risk.
