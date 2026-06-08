#!/usr/bin/env python3
"""
Triadic Percolation Self-Consistency Solver (Fixed)

Fixes:
- numpy has no attribute 'math'  -> removed np.math usage
- argparse --help should not run solver
- NoneType formatting crash in p_c printing
- indentation bug around critical point functions
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
from scipy.optimize import root
from scipy.special import factorial


# -----------------------------
# Utilities: generating functions
# -----------------------------

def poisson_pk(k_avg: float, k_max: int) -> np.ndarray:
    """Poisson degree distribution truncated at k_max."""
    k = np.arange(0, k_max + 1, dtype=np.int64)
    pk = np.exp(-k_avg) * (k_avg ** k) / factorial(k, exact=False)
    pk = pk / pk.sum()
    return pk

def gf_from_pk(pk: np.ndarray):
    """Return G0(x) and G0'(x) from pk array over k=0..k_max."""
    k = np.arange(len(pk), dtype=np.float64)

    def G0(x: float) -> float:
        return float(np.sum(pk * (x ** k)))

    def G0_prime(x: float) -> float:
        # derivative: sum_k pk * k * x^(k-1), with k=0 term 0
        if x == 0.0:
            return float(np.sum(pk[1:] * k[1:] * (0.0 ** (k[1:] - 1))))
        return float(np.sum(pk[1:] * k[1:] * (x ** (k[1:] - 1))))

    return G0, G0_prime


# -----------------------------
# Core solver
# -----------------------------

@dataclass
class Solution:
    R: float
    pL: float
    success: bool
    message: str
    nfev: int


class TriadicPercolationSolver:
    """
    Solves:
      Exact:
        R  = 1 - G0(1 - pL)
        pL = p * R * (1 - R)

      Bianconi:
        R  = 1 - G0(1 - pL)
        pL = p * (1 - G0_plus(1 - R)) * G0_minus(1 - R)

    For ER-like demos, we set G0_plus = G0_minus = G0 by default.
    """

    def __init__(self, model: str, params: Dict):
        self.model = model.upper()
        self.params = params

        self.G0 = None
        self.G0p = None
        self.G0_plus = None
        self.G0_minus = None
        self.set_degree_distributions()

    def set_degree_distributions(self):
        if self.model == "ER":
            k_avg = float(self.params.get("k_avg", 3.0))
            if k_avg <= 0:
                raise ValueError("k_avg must be > 0")
            k_max = int(self.params.get("k_max", max(50, int(10 * k_avg + 50))))
            pk = poisson_pk(k_avg, k_max)
            G0, G0p = gf_from_pk(pk)
            self.G0, self.G0p = G0, G0p

            # default: same for regulators
            self.G0_plus = G0
            self.G0_minus = G0

        else:
            raise NotImplementedError(f"Model {self.model} not implemented in fixed script.")

    # -------- residuals --------

    def residuals_exact(self, x: np.ndarray, p: float) -> np.ndarray:
        R, pL = float(x[0]), float(x[1])
        return np.array([
            R - (1.0 - self.G0(1.0 - pL)),
            pL - p * R * (1.0 - R)
        ], dtype=np.float64)

    def residuals_bianconi(self, x: np.ndarray, p: float) -> np.ndarray:
        R, pL = float(x[0]), float(x[1])
        return np.array([
            R - (1.0 - self.G0(1.0 - pL)),
            pL - p * (1.0 - self.G0_plus(1.0 - R)) * self.G0_minus(1.0 - R)
        ], dtype=np.float64)

    # -------- solve --------

    def solve_exact(self, p: float, R0: float = 0.1, pL0: float = 0.05) -> Solution:
        x0 = np.array([R0, pL0], dtype=np.float64)
        res = root(lambda z: self.residuals_exact(z, p), x0, method="hybr", tol=1e-12)

        R, pL = res.x
        R = float(np.clip(R, 0.0, 1.0))
        pL = float(np.clip(pL, 0.0, 1.0))

        return Solution(R=R, pL=pL, success=bool(res.success), message=str(res.message), nfev=int(res.nfev))

    def solve_bianconi(self, p: float, R0: float = 0.1, pL0: float = 0.05) -> Solution:
        x0 = np.array([R0, pL0], dtype=np.float64)
        res = root(lambda z: self.residuals_bianconi(z, p), x0, method="hybr", tol=1e-12)

        R, pL = res.x
        R = float(np.clip(R, 0.0, 1.0))
        pL = float(np.clip(pL, 0.0, 1.0))

        return Solution(R=R, pL=pL, success=bool(res.success), message=str(res.message), nfev=int(res.nfev))

    # -------- critical points --------

    def critical_point_exact(self) -> Dict:
        """
        Linear stability at R~0:
          Exact: p_c = 1 / G0'(1)
        For ER Poisson: G0'(1)=<k> so p_c=1/<k>
        """
        g1 = float(self.G0p(1.0))
        if g1 <= 0 or not np.isfinite(g1):
            return {"p_c": None, "reason": "invalid G0'(1)"}
        return {"p_c": 1.0 / g1, "reason": "p_c = 1/G0'(1)"}

    def critical_point_bianconi(self) -> Dict:
        """
        If G0_plus = G0_minus = G0 (default ER demo),
        then same p_c formula. For general regulator distributions,
        you'd need derivatives of G0_plus and G0_minus at 1 and the
        correct linearization.
        """
        # For the ER demo in this script, use the same estimate.
        return self.critical_point_exact()

    # -------- phase diagram --------

    def phase_diagram(self, p_min: float, p_max: float, n: int = 50) -> Dict:
        p_vals = np.linspace(p_min, p_max, n)

        out = {"p": p_vals}

        # exact
        R_list, pL_list = [], []
        R_prev, pL_prev = 0.01, 0.005
        for p in p_vals:
            sol = self.solve_exact(float(p), R0=R_prev, pL0=pL_prev)
            R_list.append(sol.R)
            pL_list.append(sol.pL)
            R_prev, pL_prev = sol.R, sol.pL
        out["exact"] = {"R": np.array(R_list), "pL": np.array(pL_list)}

        # bianconi
        R_list, pL_list = [], []
        R_prev, pL_prev = 0.01, 0.005
        for p in p_vals:
            sol = self.solve_bianconi(float(p), R0=R_prev, pL0=pL_prev)
            R_list.append(sol.R)
            pL_list.append(sol.pL)
            R_prev, pL_prev = sol.R, sol.pL
        out["bianconi"] = {"R": np.array(R_list), "pL": np.array(pL_list)}

        return out


# -----------------------------
# CLI
# -----------------------------

def build_argparser():
    ap = argparse.ArgumentParser(
        description="Triadic Percolation Self-Consistency Solver (Fixed)"
    )
    ap.add_argument("--model", type=str, default="ER", choices=["ER"],
                    help="Degree model (only ER implemented in this fixed script).")
    ap.add_argument("--k-avg", type=float, default=3.0, help="Average degree <k> for ER/Poisson.")
    ap.add_argument("--p", type=float, default=0.5, help="Single p value to solve at.")
    ap.add_argument("--phase", action="store_true", help="Compute a small phase diagram.")
    ap.add_argument("--p-min", type=float, default=0.0)
    ap.add_argument("--p-max", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=30)
    return ap


def main():
    args = build_argparser().parse_args()

    print("Triadic Percolation Self-Consistency Solver (Fixed)")
    print("=" * 50)

    solver = TriadicPercolationSolver(args.model, {"k_avg": args.k_avg})

    p = float(args.p)
    sol_ex = solver.solve_exact(p, R0=0.1, pL0=0.05)
    sol_bi = solver.solve_bianconi(p, R0=0.1, pL0=0.05)

    print(f"\nAt p={p:.3f}:")
    print(f"  Exact:    R={sol_ex.R:.4f}, pL={sol_ex.pL:.4f}   (ok={sol_ex.success})")
    print(f"  Bianconi: R={sol_bi.R:.4f}, pL={sol_bi.pL:.4f}   (ok={sol_bi.success})")

    crit_ex = solver.critical_point_exact()
    crit_bi = solver.critical_point_bianconi()

    print("\nCritical points (linear stability):")
    if crit_ex["p_c"] is None:
        print(f"  Exact:    p_c = None  ({crit_ex.get('reason','')})")
    else:
        print(f"  Exact:    p_c = {crit_ex['p_c']:.6f}  ({crit_ex.get('reason','')})")

    if crit_bi["p_c"] is None:
        print(f"  Bianconi: p_c = None  ({crit_bi.get('reason','')})")
    else:
        print(f"  Bianconi: p_c = {crit_bi['p_c']:.6f}  ({crit_bi.get('reason','')})")

    if args.phase:
        print("\nComputing phase diagram...")
        ph = solver.phase_diagram(args.p_min, args.p_max, n=args.n)
        # print a few points
        for idx in np.linspace(0, len(ph["p"]) - 1, num=min(5, len(ph["p"]))).astype(int):
            pp = ph["p"][idx]
            Rex = ph["exact"]["R"][idx]
            Rbi = ph["bianconi"]["R"][idx]
            print(f"  p={pp:.3f}  R_exact={Rex:.4f}  R_bianconi={Rbi:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
