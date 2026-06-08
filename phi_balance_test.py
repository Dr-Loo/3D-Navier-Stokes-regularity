#!/usr/bin/env python3
"""
Compare nonlinear term N and viscous dissipation in the evolution of Φ(C).
Simplified using Fourier space to avoid broadcasting errors.
"""

import numpy as np
from scipy.fft import fftn, ifftn
import matplotlib.pyplot as plt

def compute_phi_balance_fourier(u_hat, nu, kx, ky, kz, k2, N):
    """
    Compute Φ(C) and balance in Fourier space.
    
    For Φ ≈ ∫ |Δu|² dx:
        Φ = Σ |k² u_hat|²
        N = Σ (k² u_hat*) · (k² (u·∇)u_hat)  [nonlinear]
        D = ν Σ |k³ u_hat|²  [dissipation]
    """
    
    # Φ = ∫ |Δu|² dx = Σ |k² u_hat|²
    k2_u = k2[:,:,:,np.newaxis] * u_hat
    phi = np.sum(np.abs(k2_u)**2) / N**3
    
    # Dissipation: D = ν ∫ |∇Δu|² dx = ν Σ |k|² |k² u_hat|² = ν Σ |k³ u_hat|²
    k_magnitude = np.sqrt(k2)
    k3_u = k_magnitude[:,:,:,np.newaxis] * k2_u
    D_term = nu * np.sum(np.abs(k3_u)**2) / N**3
    
    # Need nonlinear term in physical space to compute (u·∇)u
    # This we still do in physical space
    u_phys = np.real(ifftn(u_hat, axes=(0,1,2)))
    
    # Compute (u·∇)u in physical space using finite differences
    dx = 2*np.pi / N
    grad_u = np.zeros((N, N, N, 3, 3))
    for i in range(3):
        for d in range(3):
            grad_u[:,:,:,i,d] = np.gradient(u_phys[:,:,:,i], axis=d, edge_order=2) / dx
    
    u_grad_u = np.zeros_like(u_phys)
    for i in range(3):
        for j in range(3):
            u_grad_u[:,:,:,i] += u_phys[:,:,:,j] * grad_u[:,:,:,i,j]
    
    # Compute Δ((u·∇)u) in Fourier space
    u_grad_u_hat = fftn(u_grad_u, axes=(0,1,2))
    delta_nonlin_hat = -k2[:,:,:,np.newaxis] * u_grad_u_hat
    
    # Nonlinear term: N = ∫ Δ((u·∇)u) · Δu dx = Σ (k² u_hat*) · (k² (u·∇)u_hat)
    # Note: careful with complex conjugate
    N_term = np.real(np.sum(np.conj(k2_u) * delta_nonlin_hat)) / N**3
    
    # dΦ/dt = 2*(N_term - D_term)
    dPhi_dt = 2 * (N_term - D_term)
    
    return phi, N_term, D_term, dPhi_dt

def run_balance_test(N=32, nu=0.001, T_max=0.3, dt=0.0002):
    """Run time evolution to track N vs D balance."""
    
    print("=" * 80)
    print("NONLINEAR vs VISCOUS BALANCE FOR Φ(C)")
    print("=" * 80)
    
    # Grid
    dx = 2*np.pi / N
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    # Taylor-Green vortex
    u = np.zeros((N, N, N, 3))
    u[:,:,:,0] = np.sin(X) * np.cos(Y) * np.cos(Z)
    u[:,:,:,1] = -np.cos(X) * np.sin(Y) * np.cos(Z)
    u[:,:,:,2] = 0
    
    # Amplitude
    A = 2.0
    u = A * u
    
    # Fourier space
    k = np.fft.fftfreq(N, dx) * 2*np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k2_inv = np.where(k2 > 0, 1/k2, 0)
    k2_inv[k2 == 0] = 0
    
    u_hat = fftn(u, axes=(0,1,2))
    
    # Enforce incompressibility
    for i in range(3):
        factor = kx if i == 0 else (ky if i == 1 else kz)
        u_hat[:,:,:,i] -= (kx*u_hat[:,:,:,0] + ky*u_hat[:,:,:,1] + kz*u_hat[:,:,:,2]) * factor * k2_inv
    
    times = []
    phi_vals = []
    N_vals = []
    D_vals = []
    dPhi_vals = []
    
    t = 0
    step = 0
    
    print(f"\nN = {N}³, ν = {nu}, dt = {dt}, T_max = {T_max}")
    print(f"Initial amplitude A = {A}")
    print("-" * 80)
    print(f"{'Step':>6} {'t':>8} {'Φ(C)':>12} {'N_term':>12} {'D_term':>12} {'dΦ/dt':>12} {'N/D':>8}")
    print("-" * 80)
    
    # Initial computation
    phi, N_term, D_term, dPhi_dt = compute_phi_balance_fourier(u_hat, nu, kx, ky, kz, k2, N)
    print(f"{step:6d}   {t:8.4f}   {phi:12.6e}   {N_term:12.6e}   {D_term:12.6e}   {dPhi_dt:12.6e}   {N_term/D_term:8.4f}")
    
    while t < T_max:
        # Compute (u·∇)u in physical space
        u_phys = np.real(ifftn(u_hat, axes=(0,1,2)))
        
        # Compute gradient using FFT (more accurate)
        grad_u_hat = np.zeros((N, N, N, 3, 3), dtype=complex)
        for i in range(3):
            grad_u_hat[:,:,:,i,0] = 1j * kx * u_hat[:,:,:,i]
            grad_u_hat[:,:,:,i,1] = 1j * ky * u_hat[:,:,:,i]
            grad_u_hat[:,:,:,i,2] = 1j * kz * u_hat[:,:,:,i]
        grad_u = np.real(ifftn(grad_u_hat, axes=(0,1,2)))
        
        u_grad_u = np.zeros_like(u_phys)
        for i in range(3):
            for j in range(3):
                u_grad_u[:,:,:,i] += u_phys[:,:,:,j] * grad_u[:,:,:,i,j]
        
        # FFT of nonlinear term
        u_grad_u_hat = fftn(u_grad_u, axes=(0,1,2))
        
        # Compute ∂_t u_hat
        partial_t_u_hat = -u_grad_u_hat - nu * k2[:,:,:,np.newaxis] * u_hat
        
        # Leray projection
        proj = (kx*partial_t_u_hat[:,:,:,0] + ky*partial_t_u_hat[:,:,:,1] + kz*partial_t_u_hat[:,:,:,2]) * k2_inv
        proj = np.nan_to_num(proj)
        for i in range(3):
            factor = kx if i == 0 else (ky if i == 1 else kz)
            partial_t_u_hat[:,:,:,i] -= proj * factor
        
        # Update (Euler method for simplicity, small dt)
        u_hat = u_hat + dt * partial_t_u_hat
        t += dt
        step += 1
        
        # Sample
        if step % 20 == 0:
            phi, N_term, D_term, dPhi_dt = compute_phi_balance_fourier(u_hat, nu, kx, ky, kz, k2, N)
            times.append(t)
            phi_vals.append(phi)
            N_vals.append(N_term)
            D_vals.append(D_term)
            dPhi_vals.append(dPhi_dt)
            
            ratio = N_term / D_term if D_term != 0 else 0
            print(f"{step:6d}   {t:8.4f}   {phi:12.6e}   {N_term:12.6e}   {D_term:12.6e}   {dPhi_dt:12.6e}   {ratio:8.4f}")
            
            # Check for blow-up
            if phi > 1e10 or np.isnan(phi):
                print("\n⚠️ Numerical blow-up detected!")
                break
    
    # Plot results
    if len(times) == 0:
        print("No data collected.")
        return times, phi_vals, N_vals, D_vals
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    ax1.plot(times, phi_vals, 'b-', linewidth=1.5)
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Φ(C)')
    ax1.set_title('Φ(C) vs Time')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(times, N_vals, 'r-', label='N (nonlinear)', linewidth=1.5)
    ax2.plot(times, D_vals, 'g-', label='D (viscous dissipation)', linewidth=1.5)
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Value')
    ax2.set_title('Nonlinear vs Dissipation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    ax3.plot(times, dPhi_vals, 'm-', linewidth=1.5)
    ax3.set_xlabel('Time t')
    ax3.set_ylabel('dΦ/dt')
    ax3.set_title('Time Derivative of Φ(C)')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    ratio_arr = [N_vals[i]/max(D_vals[i], 1e-12) for i in range(len(times))]
    ax4.plot(times, ratio_arr, 'c-', linewidth=1.5)
    ax4.set_xlabel('Time t')
    ax4.set_ylabel('N / D')
    ax4.set_title('Nonlinear/Dissipation Ratio')
    ax4.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='N = D (threshold)')
    ax4.set_ylim(0, max(2, max(ratio_arr)*1.1) if ratio_arr else 2)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phi_balance.png', dpi=150)
    plt.show()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Initial Φ(C):    {phi_vals[0]:.6e}")
    print(f"  Final Φ(C):      {phi_vals[-1]:.6e}")
    print(f"  Max N/D ratio:   {max(ratio_arr):.4f}")
    
    if max(ratio_arr) < 1:
        print("\n  ✓ DISSIPATION DOMINATES: N/D < 1 for all t")
        print("    This suggests Φ(C) is a strict Lyapunov functional.")
    else:
        print("\n  ⚠ NONLINEAR DOMINATES temporarily: N/D > 1")
        print("    Need higher resolution or lower viscosity to test criticality.")
    
    return times, phi_vals, N_vals, D_vals

if __name__ == "__main__":
    # Test at moderately low viscosity
    times, phi, N_vals, D_vals = run_balance_test(N=32, nu=0.0005, T_max=0.2, dt=0.0001)