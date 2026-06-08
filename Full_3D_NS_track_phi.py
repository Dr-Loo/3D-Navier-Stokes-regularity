#!/usr/bin/env python3
"""
Full 3D Navier-Stokes simulation tracking Φ(C(t)).
Tests whether Φ(C) remains bounded under actual dynamics.
"""

import numpy as np
from scipy.fft import fftn, ifftn
import matplotlib.pyplot as plt

def compute_Phi(u):
    """Compute Φ(C) = ∫ |curl(curl u)|² dx."""
    N = u.shape[0]
    k = np.fft.fftfreq(N, 1/N) * 2*np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    
    u_hat = fftn(u, axes=(0,1,2))
    
    # curl u
    curl_hat = np.zeros_like(u_hat)
    curl_hat[:,:,:,0] = 1j*ky*u_hat[:,:,:,2] - 1j*kz*u_hat[:,:,:,1]
    curl_hat[:,:,:,1] = 1j*kz*u_hat[:,:,:,0] - 1j*kx*u_hat[:,:,:,2]
    curl_hat[:,:,:,2] = 1j*kx*u_hat[:,:,:,1] - 1j*ky*u_hat[:,:,:,0]
    
    # curl(curl u)
    curl2_hat = np.zeros_like(curl_hat)
    curl2_hat[:,:,:,0] = 1j*ky*curl_hat[:,:,:,2] - 1j*kz*curl_hat[:,:,:,1]
    curl2_hat[:,:,:,1] = 1j*kz*curl_hat[:,:,:,0] - 1j*kx*curl_hat[:,:,:,2]
    curl2_hat[:,:,:,2] = 1j*kx*curl_hat[:,:,:,1] - 1j*ky*curl_hat[:,:,:,0]
    
    return np.sum(np.abs(curl2_hat)**2) / N**3

def compute_enstrophy(u):
    """Compute enstrophy = ∫ |ω|² dx."""
    N = u.shape[0]
    k = np.fft.fftfreq(N, 1/N) * 2*np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    u_hat = fftn(u, axes=(0,1,2))
    
    omega_hat = np.zeros_like(u_hat)
    omega_hat[:,:,:,0] = 1j*ky*u_hat[:,:,:,2] - 1j*kz*u_hat[:,:,:,1]
    omega_hat[:,:,:,1] = 1j*kz*u_hat[:,:,:,0] - 1j*kx*u_hat[:,:,:,2]
    omega_hat[:,:,:,2] = 1j*kx*u_hat[:,:,:,1] - 1j*ky*u_hat[:,:,:,0]
    
    return np.sum(np.abs(omega_hat)**2) / N**3

def run_ns_simulation(N=32, T_max=2.0, dt=0.001, nu=0.01, save_interval=50):
    """Run 3D NS simulation tracking Φ(C)."""
    
    # Grid
    k = np.fft.fftfreq(N, 1/N) * 2*np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    # Handle k=0 safely
    with np.errstate(divide='ignore', invalid='ignore'):
        k2_inv = np.where(k2 > 0, 1/k2, 0)
    k2_inv = np.nan_to_num(k2_inv)
    
    # Initial condition: Taylor-Green vortex
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    u = np.zeros((N, N, N, 3))
    u[:,:,:,0] = np.sin(X) * np.cos(Y) * np.cos(Z)
    u[:,:,:,1] = -np.cos(X) * np.sin(Y) * np.cos(Z)
    u[:,:,:,2] = 0
    
    u_hat = fftn(u, axes=(0,1,2))
    
    # Enforce incompressibility
    for i in range(3):
        factor = kx if i == 0 else (ky if i == 1 else kz)
        u_hat[:,:,:,i] -= (kx*u_hat[:,:,:,0] + ky*u_hat[:,:,:,1] + kz*u_hat[:,:,:,2]) * factor * k2_inv
    
    times = []
    phi_vals = []
    enstrophy_vals = []
    
    t = 0
    step = 0
    
    print(f"Running 3D NS on {N}³ grid")
    print(f"ν = {nu}, dt = {dt}, T_max = {T_max}")
    print("-" * 70)
    print(f"{'Step':>6} {'t':>8} {'Φ(C)':>14} {'Enstrophy':>14} {'dΦ/dt':>14}")
    print("-" * 70)
    
    prev_phi = None
    
    while t < T_max:
        # Compute nonlinear term (u·∇)u in physical space
        u_phys = np.real(ifftn(u_hat, axes=(0,1,2)))
        
        # Compute (u·∇)u
        u_grad_u = np.zeros_like(u_phys)
        for i in range(3):
            for j in range(3):
                u_grad_u[:,:,:,i] += u_phys[:,:,:,j] * np.gradient(u_phys[:,:,:,i], axis=j)
        
        n_hat = fftn(u_grad_u, axes=(0,1,2))
        
        # Euler step with viscosity
        u_hat_new = (u_hat - dt * n_hat) / (1 + nu * dt * k2[:,:,:,np.newaxis])
        
        # Enforce incompressibility
        for i in range(3):
            factor = kx if i == 0 else (ky if i == 1 else kz)
            u_hat_new[:,:,:,i] -= (kx*u_hat_new[:,:,:,0] + ky*u_hat_new[:,:,:,1] + kz*u_hat_new[:,:,:,2]) * factor * k2_inv
        
        u_hat = u_hat_new
        t += dt
        step += 1
        
        # Sample at intervals
        if step % save_interval == 0:
            u_phys = np.real(ifftn(u_hat, axes=(0,1,2)))
            phi = compute_Phi(u_phys)
            enstrophy = compute_enstrophy(u_phys)
            
            times.append(t)
            phi_vals.append(phi)
            enstrophy_vals.append(enstrophy)
            
            # Compute approximate dΦ/dt
            if prev_phi is not None:
                dPhi_dt = (phi - prev_phi) / (dt * save_interval)
            else:
                dPhi_dt = 0
            prev_phi = phi
            
            print(f"{step:6d}   {t:8.4f}   {phi:14.6e}   {enstrophy:14.6e}   {dPhi_dt:14.6e}")
            
            # Early stopping if blow-up detected
            if phi > 1e12 or np.isnan(phi) or np.isinf(phi):
                print("\n⚠️  Numerical blow-up detected!")
                break
    
    return times, phi_vals, enstrophy_vals

if __name__ == "__main__":
    # Run simulation
    times, phi_vals, enstrophy_vals = run_ns_simulation(
        N=32,        # Grid size (32³ = 32,768 points)
        T_max=1.0,   # Maximum time
        dt=0.0005,   # Time step (smaller for stability)
        nu=0.01,     # Viscosity
        save_interval=20  # Save every 20 steps
    )
    
    if len(times) == 0:
        print("No data collected.")
        exit()
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Φ(C) vs time
    ax1 = axes[0, 0]
    ax1.plot(times, phi_vals, 'b-', linewidth=1.5)
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Φ(C)')
    ax1.set_title('Entropic Potential Φ(C) vs Time')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Enstrophy vs time
    ax2 = axes[0, 1]
    ax2.plot(times, enstrophy_vals, 'r-', linewidth=1.5)
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('∫|ω|² dx')
    ax2.set_title('Enstrophy vs Time')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Φ(C) vs enstrophy (phase plot)
    ax3 = axes[1, 0]
    ax3.plot(enstrophy_vals, phi_vals, 'g-', linewidth=1.5)
    ax3.set_xlabel('Enstrophy')
    ax3.set_ylabel('Φ(C)')
    ax3.set_title('Φ(C) vs Enstrophy')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Ratio Φ/Enstrophy
    ax4 = axes[1, 1]
    ratio = [phi_vals[i] / max(enstrophy_vals[i], 1e-10) for i in range(len(times))]
    ax4.plot(times, ratio, 'm-', linewidth=1.5)
    ax4.set_xlabel('Time t')
    ax4.set_ylabel('Φ(C) / Enstrophy')
    ax4.set_title('Ratio Φ/Enstrophy (measures coherence)')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phi_C_time_evolution.png', dpi=150)
    plt.show()
    
    print("\n" + "=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)
    print(f"Initial Φ(C):        {phi_vals[0]:.6e}")
    print(f"Final Φ(C):          {phi_vals[-1]:.6e}")
    print(f"Max Φ(C):            {max(phi_vals):.6e}")
    print(f"Min Φ(C):            {min(phi_vals):.6e}")
    print(f"Φ(C) ratio (final/initial): {phi_vals[-1]/phi_vals[0]:.4f}")
    print(f"Initial enstrophy:   {enstrophy_vals[0]:.6e}")
    print(f"Final enstrophy:     {enstrophy_vals[-1]:.6e}")
    print(f"Enstrophy ratio:     {enstrophy_vals[-1]/enstrophy_vals[0]:.4f}")
    
    # Determine trend
    if phi_vals[-1] < phi_vals[0]:
        print("\n✓ Φ(C) DECREASED over time → dissipative, suggests boundedness")
        print("  This is consistent with Φ(C) being a Lyapunov functional.")
    elif phi_vals[-1] < 2 * phi_vals[0]:
        print("\n? Φ(C) increased slightly → need longer simulation")
    else:
        print("\n⚠️  Φ(C) increased significantly → may need higher resolution")
    
    # Check if enstrophy decreased (viscous dissipation)
    if enstrophy_vals[-1] < enstrophy_vals[0]:
        print("✓ Enstrophy decreased → viscous dissipation active")
    else:
        print("? Enstrophy increased → inviscid dynamics dominant")