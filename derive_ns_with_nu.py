#!/usr/bin/env python3
"""
Re-run the symbolic derivation for Navier-Stokes WITH viscosity.
This script generates a new expression that includes 'nu' terms.
"""

import sympy as sp
from sympy import symbols, Function, Matrix, diff, simplify

print("=" * 80)
print("SYMBOLIC DERIVATION OF dΦ(C)/dt FOR NAVIER-STOKES (with viscosity)")
print("=" * 80)

# Spatial coordinates
x, y, z = symbols('x y z', real=True)
coords = [x, y, z]
t = symbols('t', real=True)

# Viscosity (explicit)
nu = symbols('nu', real=True, positive=True)

# Velocity components
u1 = Function('u_x')(x, y, z, t)
u2 = Function('u_y')(x, y, z, t)
u3 = Function('u_z')(x, y, z, t)
u_vec = [u1, u2, u3]

print("\n[1] Computing vorticity tensor T_{ij}...")

# Vorticity tensor
T = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
for i in range(3):
    for j in range(3):
        T[i, j] = diff(u_vec[j], coords[i]) - diff(u_vec[i], coords[j])

print("[2] Computing curvature C_{μνi}...")

# Helper functions
def partial(f, var):
    return diff(f, var)

def lie_bracket(mu, nu, i):
    result = 0
    for j in range(3):
        result += T[mu, j] * partial(T[nu, i], coords[j])
        result -= T[nu, j] * partial(T[mu, i], coords[j])
    return result

# Compute C
C = {}
for mu in range(3):
    for nu in range(3):
        for i in range(3):
            dT_mu_nu_i = partial(T[nu, i], coords[mu])
            dT_nu_mu_i = partial(T[mu, i], coords[nu])
            C[mu, nu, i] = dT_mu_nu_i - dT_nu_mu_i + lie_bracket(mu, nu, i)

print("[3] Computing ∂_t T from Navier-Stokes (WITH viscosity)...")

# Compute (u·∇)u
u_grad_u = [0, 0, 0]
for i in range(3):
    for j in range(3):
        u_grad_u[i] += u_vec[j] * partial(u_vec[i], coords[j])

# ∂_t u from NS with explicit nu
partial_t_u = [0, 0, 0]
for i in range(3):
    # ∂_t u = - (u·∇)u - ∇p + ν Δu
    # The pressure gradient will cancel in T, so we omit it
    partial_t_u[i] = -u_grad_u[i] + nu * (partial(partial(u_vec[i], x), x) + 
                                           partial(partial(u_vec[i], y), y) + 
                                           partial(partial(u_vec[i], z), z))

# ∂_t T
partial_t_T = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
for i in range(3):
    for j in range(3):
        partial_t_T[i, j] = partial(partial_t_u[j], coords[i]) - partial(partial_t_u[i], coords[j])

print("[4] Computing ∂_t C...")

def lie_bracket_general(X, Y, i):
    result = 0
    for j in range(3):
        result += X[j] * partial(Y[i], coords[j])
        result -= Y[j] * partial(X[i], coords[j])
    return result

partial_t_C = {}
for mu in range(3):
    for nu in range(3):
        for i in range(3):
            # ∂_μ (∂_t T_ν) - ∂_ν (∂_t T_μ)
            term1 = partial(partial_t_T[nu, i], coords[mu]) - partial(partial_t_T[mu, i], coords[nu])
            
            # [∂_t T_μ, T_ν]_i
            dT_mu = [partial_t_T[mu, k] for k in range(3)]
            T_nu_vec = [T[nu, k] for k in range(3)]
            term2 = lie_bracket_general(dT_mu, T_nu_vec, i)
            
            # [T_μ, ∂_t T_ν]_i
            T_mu_vec = [T[mu, k] for k in range(3)]
            dT_nu = [partial_t_T[nu, k] for k in range(3)]
            term3 = lie_bracket_general(T_mu_vec, dT_nu, i)
            
            partial_t_C[mu, nu, i] = term1 + term2 + term3

print("[5] Computing dΦ/dt = Σ C·∂_t C...")

dPhi_dt = 0
for mu in range(3):
    for nu in range(3):
        for i in range(3):
            dPhi_dt += C[mu, nu, i] * partial_t_C[mu, nu, i]

print(f"    Raw expression length: {len(str(dPhi_dt)):,} characters")

# Try to collect terms by power of nu
print("[6] Collecting terms by viscosity power...")

# Expand and collect nu terms
dPhi_dt_expanded = sp.expand(dPhi_dt)

# Separate inviscid (nu^0) and viscous (nu^1, nu^2) parts
dPhi_dt_inviscid = dPhi_dt_expanded.subs(nu, 0)
dPhi_dt_viscous = dPhi_dt_expanded - dPhi_dt_inviscid

print(f"    Inviscid part length: {len(str(dPhi_dt_inviscid)):,} characters")
print(f"    Viscous part length: {len(str(dPhi_dt_viscous)):,} characters")

# Save to files
with open("dPhi_dt_inviscid.txt", "w", encoding='utf-8') as f:
    f.write(str(dPhi_dt_inviscid))
print("\n[7] Saved inviscid part to 'dPhi_dt_inviscid.txt'")

with open("dPhi_dt_viscous.txt", "w", encoding='utf-8') as f:
    f.write(str(dPhi_dt_viscous))
print("    Saved viscous part to 'dPhi_dt_viscous.txt'")

with open("dPhi_dt_full_with_nu.txt", "w", encoding='utf-8') as f:
    f.write(str(dPhi_dt_expanded))
print("    Saved full expression to 'dPhi_dt_full_with_nu.txt'")

# Quick check for cubic terms
import re
expr_str = str(dPhi_dt_expanded)
cubic_pattern = r'Derivative\([^,]+,\s*[xyz]\)\s*\*\s*Derivative\([^,]+,\s*[xyz]\)\s*\*\s*Derivative\([^,]+,\s*[xyz]\)'
cubic_matches = re.findall(cubic_pattern, expr_str)
print(f"\n[8] Cubic term check: {len(cubic_matches)} pure cubic (∂u)³ terms")

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)
print("""
The full expression with viscosity has been saved to:
  - dPhi_dt_full_with_nu.txt

Next steps:
  1. Verify that the inviscid part matches your previous result
  2. Check that the viscous part contains dissipative terms
  3. Confirm that cubic terms remain absent even with viscosity
""")