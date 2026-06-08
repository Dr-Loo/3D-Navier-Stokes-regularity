#!/usr/bin/env python3
"""
Direct numerical evaluation of dΦ(C)/dt for 3D Navier-Stokes.
This computes the actual time derivative from the governing equations.
"""

import numpy as np
from scipy.fft import fftn, ifftn
import matplotlib.pyplot as plt

def compute_phi_and_dPhi_dt(u, nu, dx):
    """
    Compute Φ(C) and its time derivative dΦ/dt from the velocity field.
    
    Φ(C) ≈ ∫ |curl(curl u)|² dx  (simplified proxy for the full curvature)
    """
    N = u.shape[0]
    
    # FFT grid
    k = np.fft.fftfreq(N, dx) * 2*np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k2_inv = np.where(k2 > 0, 1/k2, 0)
    
    # Transform to Fourier space
    u_hat = fftn(u, axes=(0,1,2))
    
    # Compute curl u (vorticity)
    omega_hat = np.zeros_like(u_hat)
    omega_hat[:,:,:,0] = 1j*ky*u_hat[:,:,:,2] - 1j*kz*u_hat[:,:,:,1]
    omega_hat[:,:,:,1] = 1j*kz*u_hat[:,:,:,0] - 1j*kx*u_hat[:,:,:,2]
    omega_hat[:,:,:,2] = 1j*kx*u_hat[:,:,:,1] - 1j*ky*u_hat[:,:,:,0]
    
    # Compute curl(curl u)
    curl_omega_hat = np.zeros_like(omega_hat)
    curl_omega_hat[:,:,:,0] = 1j*ky*omega_hat[:,:,:,2] - 1j*kz*omega_hat[:,:,:,1]
    curl_omega_hat[:,:,:,1] = 1j*kz*omega_hat[:,:,:,0] - 1j*kx*omega_hat[:,:,:,2]
    curl_omega_hat[:,:,:,2] = 1j*kx*omega_hat[:,:,:,1] - 1j*ky*omega_hat[:,:,:,0]
    
    # Φ(C) = ∫ |curl(curl u)|² dx
    phi = np.sum(np.abs(curl_omega_hat)**2) / N**3
    
    # Compute time derivative using Navier-Stokes
    # ∂_t u = - (u·∇)u - ∇p + ν Δu
    
    # Compute (u·∇)u in physical space
    u_phys = np.real(ifftn(u_hat, axes=(0,1,2)))
    
    # Compute gradient of u
    grad_u = np.zeros((N, N, N, 3, 3))
    for i in range(3):
        for d in range(3):
            grad_u[:,:,:,i,d] = np.gradient(u_phys[:,:,:,i], axis=d, edge_order=2) / dx
    
    # Compute (u·∇)u
    u_grad_u = np.zeros_like(u_phys)
    for i in range(3):
        for j in range(3):
            u_grad_u[:,:,:,i] += u_phys[:,:,:,j] * grad_u[:,:,:,i,j]
    
    # FFT of nonlinear term
    n_hat = fftn(u_grad_u, axes=(0,1,2))
    
    # Compute ∂_t u_hat
    # Note: Pressure term is omitted because we'll take curl
    partial_t_u_hat = -n_hat - nu * k2[:,:,:,np.newaxis] * u_hat
    
    # Apply Leray projection to enforce incompressibility
    for i in range(3):
        factor = kx if i == 0 else (ky if i == 1 else kz)
        partial_t_u_hat[:,:,:,i] -= (kx*partial_t_u_hat[:,:,:,0] + 
                                      ky*partial_t_u_hat[:,:,:,1] + 
                                      kz*partial_t_u_hat[:,:,:,2]) * factor * k2_inv
    
    # Compute ∂_t ω = curl(∂_t u)
    partial_t_omega_hat = np.zeros_like(partial_t_u_hat)
    partial_t_omega_hat[:,:,:,0] = 1j*ky*partial_t_u_hat[:,:,:,2] - 1j*kz*partial_t_u_hat[:,:,:,1]
    partial_t_omega_hat[:,:,:,1] = 1j*kz*partial_t_u_hat[:,:,:,0] - 1j*kx*partial_t_u_hat[:,:,:,2]
    partial_t_omega_hat[:,:,:,2] = 1j*kx*partial_t_u_hat[:,:,:,1] - 1j*ky*partial_t_u_hat[:,:,:,0]
    
    # Compute ∂_t curl(ω) = curl(∂_t ω)
    partial_t_curl_omega_hat = np.zeros_like(partial_t_omega_hat)
    partial_t_curl_omega_hat[:,:,:,0] = 1j*ky*partial_t_omega_hat[:,:,:,2] - 1j*kz*partial_t_omega_hat[:,:,:,1]
    partial_t_curl_omega_hat[:,:,:,1] = 1j*kz*partial_t_omega_hat[:,:,:,0] - 1j*kx*partial_t_omega_hat[:,:,:,2]
    partial_t_curl_omega_hat[:,:,:,2] = 1j*kx*partial_t_omega_hat[:,:,:,1] - 1j*ky*partial_t_omega_hat[:,:,:,0]
    
    # Compute dΦ/dt = 2 ∫ curl(curl u) · ∂_t curl(curl u) dx
    dPhi_dt = 2 * np.real(np.sum(np.conj(curl_omega_hat) * partial_t_curl_omega_hat)) / N**3
    
    return phi, dPhi_dt

def run_numerical_test(N=32, nu=0.01, dt=0.0005, T_max=0.1):
    """Run test to evaluate dΦ/dt at initial time."""
    
    print("=" * 80)
    print("DIRECT NUMERICAL EVALUATION OF dΦ(C)/dt")
    print("=" * 80)
    
    # Grid
    dx = 2*np.pi / N
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    # Taylor-Green vortex initial condition
    u = np.zeros((N, N, N, 3))
    u[:,:,:,0] = np.sin(X) * np.cos(Y) * np.cos(Z)
    u[:,:,:,1] = -np.cos(X) * np.sin(Y) * np.cos(Z)
    u[:,:,:,2] = 0
    
    print(f"\nGrid: {N}³, ν = {nu}, dx = {dx:.4f}")
    
    # Compute Φ and dΦ/dt at t=0
    phi, dPhi_dt = compute_phi_and_dPhi_dt(u, nu, dx)
    
    print(f"\nResults at t=0:")
    print(f"  Φ(C)      = {phi:.6e}")
    print(f"  dΦ/dt     = {dPhi_dt:.6e}")
    
    if dPhi_dt < 0:
        print(f"\n  ✓ dΦ/dt is NEGATIVE → Φ(C) is decreasing")
        print(f"    This supports Φ(C) being a Lyapunov functional.")
    else:
        print(f"\n  ⚠ dΦ/dt is POSITIVE → Φ(C) is increasing")
        print(f"    Need to check if this is due to numerical error or physical.")
    
    # Test with different viscosities
    print("\n" + "=" * 80)
    print("VISCOSITY SWEEP")
    print("=" * 80)
    
    nu_values = [0.1, 0.05, 0.01, 0.005, 0.001]
    results = []
    
    for nu_test in nu_values:
        _, dPhi = compute_phi_and_dPhi_dt(u, nu_test, dx)
        results.append((nu_test, dPhi))
        print(f"  ν = {nu_test:.4f}: dΦ/dt = {dPhi:.6e}")
    
    # Test with different perturbation amplitudes
    print("\n" + "=" * 80)
    print("PERTURBATION AMPLITUDE SWEEP")
    print("=" * 80)
    
    base_phi, base_dPhi = compute_phi_and_dPhi_dt(u, nu, dx)
    print(f"  Base (no perturbation): dΦ/dt = {base_dPhi:.6e}")
    
    for amp in [0.01, 0.05, 0.1, 0.2]:
        u_pert = u.copy()
        u_pert[:,:,:,0] += amp * np.sin(2*X) * np.cos(Y) * np.cos(Z)
        u_pert[:,:,:,1] += amp * np.cos(X) * np.sin(2*Y) * np.cos(Z)
        
        phi_pert, dPhi_pert = compute_phi_and_dPhi_dt(u_pert, nu, dx)
        print(f"  Amp = {amp:.2f}: dΦ/dt = {dPhi_pert:.6e} (Δ = {dPhi_pert - base_dPhi:.6e})")
    
    # Test with different grid resolutions
    print("\n" + "=" * 80)
    print("RESOLUTION SWEEP")
    print("=" * 80)
    
    for N_test in [16, 32, 48]:
        dx_test = 2*np.pi / N_test
        x_test = np.linspace(0, 2*np.pi, N_test, endpoint=False)
        Xt, Yt, Zt = np.meshgrid(x_test, x_test, x_test, indexing='ij')
        
        u_test = np.zeros((N_test, N_test, N_test, 3))
        u_test[:,:,:,0] = np.sin(Xt) * np.cos(Yt) * np.cos(Zt)
        u_test[:,:,:,1] = -np.cos(Xt) * np.sin(Yt) * np.cos(Zt)
        u_test[:,:,:,2] = 0
        
        phi_test, dPhi_test = compute_phi_and_dPhi_dt(u_test, nu, dx_test)
        print(f"  N = {N_test:3d}: dΦ/dt = {dPhi_test:.6e}")

if __name__ == "__main__":
    run_numerical_test()