#!/usr/bin/env python3
"""
Numerical test: Does Φ(C) remain bounded for 3D NS?
"""

import numpy as np
from scipy.fft import fftn, ifftn

def compute_Phi(u):
    """Compute Φ(C) ≈ ∫ |curl(curl u)|² dx (the curvature term)."""
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

# Taylor-Green vortex initial condition
N = 32
x = np.linspace(0, 2*np.pi, N, endpoint=False)
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
u = np.zeros((N, N, N, 3))
u[:,:,:,0] = np.sin(X) * np.cos(Y) * np.cos(Z)
u[:,:,:,1] = -np.cos(X) * np.sin(Y) * np.cos(Z)

Phi_initial = compute_Phi(u)
print(f"Initial Φ(C) = {Phi_initial:.6e}")

# Simple test: apply a large strain to see if Φ grows
# (This is a proxy — full NS simulation would be better)
strain = 10.0
u_strained = u.copy()
u_strained[:,:,:,0] *= strain
u_strained[:,:,:,1] /= strain

Phi_strained = compute_Phi(u_strained)
print(f"After strain (×{strain}): Φ(C) = {Phi_strained:.6e}")
print(f"Ratio: {Phi_strained/Phi_initial:.2f}")