# ============================================================
#  EFFECTIVE MEDIUM MODEL (Maxwell Garnett)
# ============================================================

# Ensure that the permittivities are frequency-dependent (arrays)
eps_eff_E = maxwell_garnett(epsm, epsE * np.ones_like(f), Vf)  # Correct this line
eps_eff_S = maxwell_garnett(epsm, epsS * np.ones_like(f), Vf)  # Correct this line
eps_eff_PAN = maxwell_garnett(epsm, eps_cf_PAN, Vf)
eps_eff_pitch = maxwell_garnett(epsm, eps_cf_pitch, Vf)

# ============================================================
#  REFLECTIVITY
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
