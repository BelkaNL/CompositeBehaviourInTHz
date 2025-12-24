import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#  MAXWELL-GARNETT MODEL
# ============================================================

def maxwell_garnett(eps_m, eps_f, Vf):
    """
    Maxwell-Garnett model to calculate the effective permittivity of a composite material.
    
    Parameters:
    eps_m : complex or float
        The permittivity of the matrix material (can be complex for lossy materials).
    eps_f : complex or float
        The permittivity of the filler or fiber material (can be complex for lossy materials).
    Vf : float
        The volume fraction of the fiber material in the composite (0 <= Vf <= 1).
        
    Returns:
    eps_eff : complex
        The effective permittivity of the composite material.
    """
    # Implementing the Maxwell-Garnett equation
    numerator = eps_f + 2 * eps_m + 2 * Vf * (eps_f - eps_m)
    denominator = eps_f + 2 * eps_m - Vf * (eps_f - eps_m)
    
    eps_eff = eps_m * (numerator / denominator)
    return eps_eff

# ============================================================
#  ANISOTROPIC MODEL FOR UNIDIRECTIONAL LAMINATES
# ============================================================

def eps_parallel(eps_m, eps_f, Vf):
    """
    Parallel permittivity for unidirectional laminates.
    """
    return Vf * eps_f + (1 - Vf) * eps_m

def eps_perpendicular(eps_m, eps_f, Vf):
    """
    Perpendicular permittivity for unidirectional laminates.
    """
    return eps_m * (eps_f + eps_m + Vf * (eps_f - eps_m)) / \
           (eps_f + eps_m - Vf * (eps_f - eps_m))

# ============================================================
#  CONSTANTS AND PARAMETERS
# ============================================================

eps0 = 8.854e-12  # Permittivity of free space (F/m)
c = 3e8  # Speed of light (m/s)
f = np.linspace(30e9, 3e12, 2000)  # Frequency range from 30 GHz to 3 THz
w = 2 * np.pi * f  # Angular frequency
Vf = 0.55  # Fiber volume fraction

# Material Properties (Example)
eps_m = 3.0 - 1j * 0.02  # Epoxy matrix (with loss)
epsE = 6.0 - 1j * 0.015  # E-glass dielectric (with loss)
epsS = 4.8 - 1j * 0.008  # S-glass dielectric (with loss)

# Assume `eps_cf_PAN` and `eps_cf_pitch` are defined with conductivity for PAN and Pitch carbon fibers
sigma_PAN = 5e4  # Conductivity of PAN-based carbon fiber
eps_cf_PAN = 10 - 1j * (10 * 0.05 + sigma_PAN / (w * eps0))

sigma_pitch = 1e5  # Conductivity of pitch-based carbon fiber
eps_cf_pitch = 12 - 1j * (12 * 0.05 + sigma_pitch / (w * eps0))

# ============================================================
#  CALCULATE EFFECTIVE PERMITTIVITIES (Maxwell-Garnett)
# ============================================================

# Ensure that the permittivities are frequency-dependent (arrays)
eps_eff_E = maxwell_garnett(eps_m, epsE * np.ones_like(f), Vf)
eps_eff_S = maxwell_garnett(eps_m, epsS * np.ones_like(f), Vf)
eps_eff_PAN = maxwell_garnett(eps_m, eps_cf_PAN, Vf)
eps_eff_pitch = maxwell_garnett(eps_m, eps_cf_pitch, Vf)

# ============================================================
#  CALCULATE ANISOTROPIC CFRP PERMITTIVITIES
# ============================================================

# Parallel and perpendicular permittivities for PAN-based CFRP
eps_para_PAN = eps_parallel(eps_m, eps_cf_PAN, Vf)
eps_perp_PAN = eps_perpendicular(eps_m, eps_cf_PAN, Vf)

# ============================================================
#  REFLECTIVITY FUNCTION
# ============================================================

def reflection(eps):
    # Reflectivity calculation for a material with permittivity eps
    return np.abs((np.sqrt(eps) - 1) / (np.sqrt(eps) + 1)) ** 2

R_E = reflection(eps_eff_E)
R_S = reflection(eps_eff_S)
R_PAN = reflection(eps_eff_PAN)
R_pitch = reflection(eps_eff_pitch)

# ============================================================
#  PLOTTING
# ============================================================

# Plot 1: Real Permittivity (ε′) vs Frequency
plt.figure(figsize=(12, 7))
plt.plot(f * 1e-9, np.real(eps_eff_E), label="E-glass ε′")
plt.plot(f * 1e-9, np.real(eps_eff_S), label="S-glass ε′")
plt.plot(f * 1e-9, np.real(eps_eff_PAN), label="PAN-CF ε′")
plt.plot(f * 1e-9, np.real(eps_eff_pitch), label="Pitch-CF ε′")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Real Permittivity ε′")
plt.legend()
plt.grid()
plt.title("Real Permittivity ε′(f)")
plt.show()

# Plot 2: Imaginary Permittivity (ε″) vs Frequency
plt.figure(figsize=(12, 7))
plt.plot(f * 1e-9, np.imag(eps_eff_E), label="E-glass ε″")
plt.plot(f * 1e-9, np.imag(eps_eff_S), label="S-glass ε″")
plt.plot(f * 1e-9, np.imag(eps_eff_PAN), label="PAN-CF ε″")
plt.plot(f * 1e-9, np.imag(eps_eff_pitch), label="Pitch-CF ε″")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Imaginary Permittivity ε″")
plt.legend()
plt.grid()
plt.title("Imaginary Permittivity ε″(f)")
plt.show()

# Plot 3: Reflectivity (R) vs Frequency
plt.figure(figsize=(12, 7))
plt.plot(f * 1e-9, R_E, label="E-glass R")
plt.plot(f * 1e-9, R_S, label="S-glass R")
plt.plot(f * 1e-9, R_PAN, label="PAN-CF R")
plt.plot(f * 1e-9, R_pitch, label="Pitch-CF R")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Reflectivity R")
plt.legend()
plt.grid()
plt.title("Reflectivity vs Frequency")
plt.show()

# Plot 4: Anisotropic CFRP Permittivity (ε∥ and ε⊥) vs Frequency
plt.figure(figsize=(12, 7))
plt.plot(f * 1e-9, np.real(eps_para_PAN), label="CFRP ε∥")
plt.plot(f * 1e-9, np.real(eps_perp_PAN), label="CFRP ε⊥")
plt.xlabel("Frequency (GHz)")
plt.ylabel("ε∥ and ε⊥")
plt.legend()
plt.grid()
plt.title("Anisotropic CFRP Permittivity")
plt.show()
