#!/usr/bin/env python3
"""
Symbolic expansion of d/dt Φ(C) for 3D Navier-Stokes equations.
Computes the time derivative of the entropic potential based on vortex curvature.
"""

import sympy as sp
from sympy import symbols, Function, Matrix, diff, simplify, expand
import sys

print("=" * 80)
print("SFIT: Symbolic Computation of d/dt Φ(C) for 3D Navier-Stokes")
print("=" * 80)

# ============================================================================
# 1. Define basic symbols and coordinates
# ============================================================================

# Spatial coordinates
x, y, z = symbols('x y z', real=True)
coords = [x, y, z]

# Time
t = symbols('t', real=True)

# Physical parameters
nu = symbols('nu', real=True, positive=True)  # viscosity

# Velocity components as functions of space and time
u1 = Function('u_x')(x, y, z, t)
u2 = Function('u_y')(x, y, z, t)
u3 = Function('u_z')(x, y, z, t)

u_vec = [u1, u2, u3]

print("\n[1] Velocity field defined:")
for i, u in enumerate(u_vec):
    print(f"    u_{i+1} = {u}")

# ============================================================================
# 2. Compute partial derivatives
# ============================================================================

def partial(f, var):
    """Compute partial derivative of f with respect to variable."""
    return diff(f, var)

def grad(f):
    """Gradient of f."""
    return [partial(f, coord) for coord in coords]

def div(u):
    """Divergence of vector field u."""
    return sum(partial(u[i], coords[i]) for i in range(3))

def curl(u):
    """Curl of vector field u."""
    return [
        partial(u[2], coords[1]) - partial(u[1], coords[2]),
        partial(u[0], coords[2]) - partial(u[2], coords[0]),
        partial(u[1], coords[0]) - partial(u[0], coords[1])
    ]

def laplacian(f):
    """Laplacian of f."""
    return sum(partial(partial(f, coord), coord) for coord in coords)

print("\n[2] Differential operators defined.")

# ============================================================================
# 3. Define vorticity tensor T_{ij} = ∂_i u_j - ∂_j u_i
# ============================================================================

print("\n[3] Computing vorticity tensor T_{ij}...")

T = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

for i in range(3):
    for j in range(3):
        T[i, j] = partial(u_vec[j], coords[i]) - partial(u_vec[i], coords[j])

print("    T_{ij} = ∂_i u_j - ∂_j u_i (antisymmetric)")

# Display non-zero components
for i in range(3):
    for j in range(i+1, 3):
        print(f"    T_{i+1}{j+1} = {simplify(T[i, j])}")

# Compute vorticity vector omega = curl(u)
omega = curl(u_vec)
print("\n    Vorticity vector ω = ∇ × u:")
for i in range(3):
    print(f"    ω_{i+1} = {simplify(omega[i])}")

# ============================================================================
# 4. Define curvature C_{μνi}
# ============================================================================

print("\n[4] Computing curvature C_{μνi}...")

# Pre-compute partial derivatives of T components
dT = {}  # dT[mu, nu, i] = ∂_μ T_{νi}
for mu in range(3):
    for nu in range(3):
        for i in range(3):
            dT[mu, nu, i] = partial(T[nu, i], coords[mu])

# Lie bracket [T_μ, T_ν]_i = T_{μj} ∂_j T_{νi} - T_{νj} ∂_j T_{μi}
def lie_bracket(mu, nu, i):
    """Compute [T_mu, T_nu]_i"""
    result = 0
    for j in range(3):
        result += T[mu, j] * partial(T[nu, i], coords[j])
        result -= T[nu, j] * partial(T[mu, i], coords[j])
    return result

C = {}  # dictionary keyed by (mu, nu, i)

for mu in range(3):
    for nu in range(3):
        for i in range(3):
            C[mu, nu, i] = dT[mu, nu, i] - dT[nu, mu, i] + lie_bracket(mu, nu, i)

print("    C_{μνi} = ∂_μ T_{νi} - ∂_ν T_{μi} + [T_μ, T_ν]_i")
print("    (Computing all 27 components...)")

# Display a few components (convert to string for truncation)
print("\n    Sample components:")
for (mu, nu, i) in [(0,1,0), (0,1,1), (0,1,2), (1,2,0)]:
    c_str = str(simplify(C[mu, nu, i]))
    if len(c_str) > 150:
        c_str = c_str[:150] + "..."
    print(f"    C_{mu+1}{nu+1}{i+1} = {c_str}")

# ============================================================================
# 5. Define entropic potential Φ(C)
# ============================================================================

print("\n[5] Computing entropic potential Φ(C) = ½ ∑_{μ,ν,i} C_{μνi}²")

Phi = 0
for mu in range(3):
    for nu in range(3):
        for i in range(3):
            Phi += C[mu, nu, i] * C[mu, nu, i]

Phi = Phi / 2

print(f"    Φ(C) = ½ * (sum of 27 components²)")

# ============================================================================
# 6. Compute ∂_t T from Navier-Stokes (vorticity formulation)
# ============================================================================

print("\n[6] Computing ∂_t T_{ij} from Navier-Stokes...")

# Compute (u·∇)u
u_grad_u = [0, 0, 0]
for i in range(3):
    for j in range(3):
        u_grad_u[i] += u_vec[j] * partial(u_vec[i], coords[j])

# ∂_t u from NS (without pressure - pressure gradient cancels in T)
# ∂_t u = - (u·∇)u + ν Δu
partial_t_u_no_p = [0, 0, 0]
for i in range(3):
    partial_t_u_no_p[i] = -u_grad_u[i] + nu * laplacian(u_vec[i])

# ∂_t T_{ij} = ∂_i (∂_t u_j) - ∂_j (∂_t u_i)
partial_t_T = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
for i in range(3):
    for j in range(3):
        partial_t_T[i, j] = partial(partial_t_u_no_p[j], coords[i]) - partial(partial_t_u_no_p[i], coords[j])

print("    ∂_t T_{ij} computed (pressure eliminated).")

# ============================================================================
# 7. Compute ∂_t C_{μνi}
# ============================================================================

print("\n[7] Computing ∂_t C_{μνi}...")

def lie_bracket_general(X, Y, i):
    """Compute [X, Y]_i where X, Y are vector fields."""
    result = 0
    for j in range(3):
        result += X[j] * partial(Y[i], coords[j])
        result -= Y[j] * partial(X[i], coords[j])
    return result

partial_t_C = {}

print("    Computing 27 components (this may take a moment)...")

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

print("    ∂_t C_{μνi} computed.")

# ============================================================================
# 8. Compute dΦ/dt integrand
# ============================================================================

print("\n[8] Computing dΦ/dt integrand = Σ C_{μνi} * ∂_t C_{μνi}")

dPhi_dt_integrand = 0
for mu in range(3):
    for nu in range(3):
        for i in range(3):
            dPhi_dt_integrand += C[mu, nu, i] * partial_t_C[mu, nu, i]

print(f"    Integrand expression size: {len(str(dPhi_dt_integrand))} characters")

# ============================================================================
# 9. Simplify and expand
# ============================================================================

print("\n[9] Simplifying...")

try:
    dPhi_dt_simplified = simplify(dPhi_dt_integrand)
    print(f"    Simplified size: {len(str(dPhi_dt_simplified))} characters")
except Exception as e:
    print(f"    Simplification failed: {e}")
    dPhi_dt_simplified = dPhi_dt_integrand

print("\n[10] Expanding...")

try:
    dPhi_dt_expanded = expand(dPhi_dt_integrand)
    print(f"    Expanded size: {len(str(dPhi_dt_expanded))} characters")
except Exception as e:
    print(f"    Expansion failed: {e}")
    dPhi_dt_expanded = dPhi_dt_integrand

# ============================================================================
# 10. Write output to file
# ============================================================================

output_file = "sfit_dPhi_dt_output.txt"
print(f"\n[11] Writing full expressions to '{output_file}'...")

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("SFIT: dΦ/dt Computation for 3D Navier-Stokes\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("Velocity components:\n")
    for i, u in enumerate(u_vec):
        f.write(f"  u_{i+1} = {u}\n")
    
    f.write("\nVorticity tensor T_{ij} (non-zero components):\n")
    for i in range(3):
        for j in range(i+1, 3):
            f.write(f"  T_{i+1}{j+1} = {simplify(T[i,j])}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("dΦ/dt Integrand (expanded):\n")
    f.write("=" * 80 + "\n\n")
    f.write(str(dPhi_dt_expanded))
    
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("dΦ/dt Integrand (simplified):\n")
    f.write("=" * 80 + "\n\n")
    f.write(str(dPhi_dt_simplified))

print(f"    Output written to '{output_file}'")

# ============================================================================
# 11. Summary
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
The symbolic computation has produced expressions for:
  1. Vorticity tensor T_{ij}
  2. Curvature C_{μνi}
  3. ∂_t C_{μνi} from Navier-Stokes
  4. dΦ/dt integrand = Σ C·∂_t C

The full expression is in the output file.

To determine if there is a coercive estimate:
  - Inspect the output file for term structure
  - Look for cancellation of highest-order derivatives
  - Or run numerical simulations to test boundedness
""")

print("\nScript complete.")