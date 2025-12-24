import numpy as np
import matplotlib.pyplot as plt

# Frequency range: 30 GHz – 3 THz
f = np.linspace(30e9, 3e12, 2000)
w = 2*np.pi*f
eps0 = 8.854e-12
c = 3e8

# ============================================================
#  MATERIAL PARAMETERS (literature-based typical values)
# ============================================================

# Epoxy (matrix)
eps_m = 3.0
tan_m = 0.02
epsm = eps_m - 1j * eps_m * tan_m

# E-glass (dielectric)
eps_E = 6.0
tan_E = 0.015
epsE = eps_E - 1j * eps_E * tan_E

# S-glass (dielectric)
eps_S = 4.8
tan_S = 0.008
epsS = eps_S - 1j * eps_S * tan_S

# PAN-based carbon fiber (conductive + dielectric loss)
sigma_PAN = 5e4
eps_cf_PAN = 10 - 1j*(10*0.05 + sigma_PAN/(w*eps0))

# Pitch-based carbon fiber (higher conductivity)
sigma_pitch = 1e5
eps_cf_pitch = 12 - 1j*(12*0.05 + sigma_pitch/(w*eps0))

# Fiber volume fraction
Vf = 0.55

# ============================================================
#  EFFECTIVE MEDIUM MODEL (Maxwell Garnett)
# ============================================================

def maxwell_garnett(eps_m, eps_f, Vf):
    return eps_m * (eps_f + 2*eps_m + 2*Vf*(eps_f - eps_m)) / \
           (eps_f + 2*eps_m - Vf*(eps_f - eps_m))

# Effective composite permittivities
eps_eff_E     = maxwell_garnett(epsm, epsE, Vf)
eps_eff_S     = maxwell_garnett(epsm, epsS, Vf)
eps_eff_PAN   = maxwell_garnett(epsm, eps_cf_PAN, Vf)
eps_eff_pitch = maxwell_garnett(epsm, eps_cf_pitch, Vf)

# ============================================================
#  ANISOTROPIC MODEL (Unidirectional Laminates)
# ============================================================
def eps_parallel(eps_m, eps_f, Vf):
    return Vf*eps_f + (1-Vf)*eps_m

def eps_perpendicular(eps_m, eps_f, Vf):
    return eps_m * (eps_f + eps_m + Vf*(eps_f - eps_m)) / \
           (eps_f + eps_m - Vf*(eps_f - eps_m))

# Example: anisotropic CFRP (PAN fiber)
eps_para_PAN = eps_parallel(epsm, eps_cf_PAN, Vf)
eps_perp_PAN = eps_perpendicular(epsm, eps_cf_PAN, Vf)

# ============================================================
#  REFLECTIVITY
# ============================================================
def reflection(eps):
    return np.abs((np.sqrt(eps) - 1)/(np.sqrt(eps) + 1))**2

R_E       = reflection(eps_eff_E)
R_S       = reflection(eps_eff_S)
R_PAN     = reflection(eps_eff_PAN)
R_pitch   = reflection(eps_eff_pitch)

# ============================================================
#  PLOTTING
# ============================================================
plt.figure(figsize=(12,7))
plt.plot(f*1e-9, np.real(eps_eff_E),     label="E-glass ε′")
plt.plot(f*1e-9, np.real(eps_eff_S),     label="S-glass ε′")
plt.plot(f*1e-9, np.real(eps_eff_PAN),   label="PAN-CF ε′")
plt.plot(f*1e-9, np.real(eps_eff_pitch), label="Pitch-CF ε′")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Real Permittivity ε′")
plt.legend(); plt.grid(); plt.title("Real Permittivity ε′(f)")
plt.show()

plt.figure(figsize=(12,7))
plt.plot(f*1e-9, np.imag(eps_eff_E),     label="E-glass ε″")
plt.plot(f*1e-9, np.imag(eps_eff_S),     label="S-glass ε″")
plt.plot(f*1e-9, np.imag(eps_eff_PAN),   label="PAN-CF ε″")
plt.plot(f*1e-9, np.imag(eps_eff_pitch), label="Pitch-CF ε″")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Imaginary Permittivity ε″")
plt.legend(); plt.grid(); plt.title("Imaginary Permittivity ε″(f)")
plt.show()

plt.figure(figsize=(12,7))
plt.plot(f*1e-9, R_E,       label="E-glass R")
plt.plot(f*1e-9, R_S,       label="S-glass R")
plt.plot(f*1e-9, R_PAN,     label="PAN-CF R")
plt.plot(f*1e-9, R_pitch,   label="Pitch-CF R")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Reflectivity R")
plt.legend(); plt.grid(); plt.title("Reflectivity vs Frequency")
plt.show()

plt.figure(figsize=(12,7))
plt.plot(f*1e-9, np.real(eps_para_PAN), label="CFRP ε∥")
plt.plot(f*1e-9, np.real(eps_perp_PAN), label="CFRP ε⊥")
plt.xlabel("Frequency (GHz)")
plt.ylabel("ε∥ and ε⊥")
plt.legend(); plt.grid(); plt.title("Anisotropic CFRP Permittivity")
plt.show()
