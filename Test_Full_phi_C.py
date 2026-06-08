#!/usr/bin/env python3
"""
Test the balance for the FULL curvature functional Φ(C) = ∫ |C|² dx.
This computes the actual curvature tensor numerically.
"""

import numpy as np
from scipy.fft import fftn, ifftn
import matplotlib.pyplot as plt

def compute_full_phi_and_balance(u, nu, dx):
    """
    Compute the full Φ(C) and its balance dΦ/dt = N_full - D_full.
    
    C_{μνi} = ∂_μ T_{νi} - ∂_ν T_{μi} + [T_μ, T_ν]_i
    where T_{ij} = ∂_i u_j - ∂_j u_i (vorticity tensor)
    """
    N = u.shape[0]
    
    # Compute first derivatives using FFT
    k = np.fft.fftfreq(N, dx) * 2*np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    
    u_hat = fftn(u, axes=(0,1,2))
    
    # Compute ∂_i u_j
    grad_u_hat = np.zeros((N, N, N, 3, 3), dtype=complex)
    grad_u_hat[:,:,:,0,0] = 1j * kx * u_hat[:,:,:,0]
    grad_u_hat[:,:,:,0,1] = 1j * ky * u_hat[:,:,:,0]
    grad_u_hat[:,:,:,0,2] = 1j * kz * u_hat[:,:,:,0]
    grad_u_hat[:,:,:,1,0] = 1j * kx * u_hat[:,:,:,1]
    grad_u_hat[:,:,:,1,1] = 1j * ky * u_hat[:,:,:,1]
    grad_u_hat[:,:,:,1,2] = 1j * kz * u_hat[:,:,:,1]
    grad_u_hat[:,:,:,2,0] = 1j * kx * u_hat[:,:,:,2]
    grad_u_hat[:,:,:,2,1] = 1j * ky * u_hat[:,:,:,2]
    grad_u_hat[:,:,:,2,2] = 1j * kz * u_hat[:,:,:,2]
    
    grad_u = np.real(ifftn(grad_u_hat, axes=(0,1,2)))
    
    # Compute T_{ij} = ∂_i u_j - ∂_j u_i
    T = np.zeros((N, N, N, 3, 3))
    for i in range(3):
        for j in range(3):
            T[:,:,:,i,j] = grad_u[:,:,:,j,i] - grad_u[:,:,:,i,j]
    
    # Compute ∂_μ T_{νi} (second derivatives)
    # For simplicity, approximate with finite differences
    T_hat = fftn(T, axes=(0,1,2))
    grad_T_hat = np.zeros((N, N, N, 3, 3, 3), dtype=complex)
    for mu in range(3):
        for nu in range(3):
            for i in range(3):
                k_vec = [kx, ky, kz][mu]
                grad_T_hat[:,:,:,mu,nu,i] = 1j * k_vec * T_hat[:,:,:,nu,i]
    grad_T = np.real(ifftn(grad_T_hat, axes=(0,1,2)))
    
    # Compute Lie bracket [T_μ, T_ν]_i = T_{μj} ∂_j T_{νi} - T_{νj} ∂_j T_{μi}
    # First compute ∂_j T_{νi} (already have grad_T)
    lie_bracket = np.zeros((N, N, N, 3, 3, 3))
    for mu in range(3):
        for nu in range(3):
            for i in range(3):
                for j in range(3):
                    lie_bracket[:,:,:,mu,nu,i] += T[:,:,:,mu,j] * grad_T[:,:,:,j,nu,i]
                    lie_bracket[:,:,:,mu,nu,i] -= T[:,:,:,nu,j] * grad_T[:,:,:,j,mu,i]
    
    # Compute C_{μνi} = ∂_μ T_{νi} - ∂_ν T_{μi} + [T_μ, T_ν]_i
    C = np.zeros((N, N, N, 3, 3, 3))
    for mu in range(3):
        for nu in range(3):
            for i in range(3):
                C[:,:,:,mu,nu,i] = grad_T[:,:,:,mu,nu,i] - grad_T[:,:,:,nu,mu,i] + lie_bracket[:,:,:,mu,nu,i]
    
    # Φ(C) = ∫ |C|² dx
    phi = np.sum(C**2) / N**3
    
    # Compute ∂_t u for dΦ/dt (simplified: just return phi for now)
    # Full dΦ/dt computation requires ∂_t C, which is heavy
    
    return phi, 0, 0, 0

def run_full_phi_test(N=24, nu=0.001, T_max=0.1):
    """Test the full Φ(C) evolution."""
    
    print("=" * 80)
    print("FULL CURVATURE FUNCTIONAL Φ(C) TEST")
    print("=" * 80)
    
    dx = 2*np.pi / N
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    # Initial condition
    u = np.zeros((N, N, N, 3))
    u[:,:,:,0] = np.sin(X) * np.cos(Y) * np.cos(Z)
    u[:,:,:,1] = -np.cos(X) * np.sin(Y) * np.cos(Z)
    
    print(f"\nComputing Φ(C) for Taylor-Green vortex on {N}³ grid...")
    phi, _, _, _ = compute_full_phi_and_balance(u, nu, dx)
    print(f"  Φ(C) = {phi:.6e}")
    
    # Test with perturbation
    for amp in [0.1, 0.2, 0.5]:
        u_pert = u.copy()
        u_pert[:,:,:,0] += amp * np.sin(2*X) * np.cos(Y) * np.cos(Z)
        phi_pert, _, _, _ = compute_full_phi_and_balance(u_pert, nu, dx)
        print(f"  Amp {amp}: Φ(C) = {phi_pert:.6e} (Δ = {phi_pert - phi:.6e})")
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
    The full curvature functional Φ(C) is computationally intensive.
    However, the symbolic derivation showed NO cubic terms.
    
    The numerical test for the standard ∫|Δu|² shows N/D >> 1,
    confirming the classical obstruction.
    
    For Φ(C), the nonlinear term is structurally different.
    The next step is to implement the full ∂_t C computation
    or to prove analytically that N_full ≤ 0.
    """)

if __name__ == "__main__":
    run_full_phi_test()