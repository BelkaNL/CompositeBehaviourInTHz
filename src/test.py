import numpy as np
import matplotlib
matplotlib.use("Agg")  # Safe for headless environments
import matplotlib.pyplot as plt

# ============================================================
# MAXWELL–GARNETT MODEL
# ============================================================

def maxwell_garnett(eps_m, eps_f, Vf):
    """
    Maxwell–Garnett effective permittivity model
    (valid for dielectric inclusions and Vf <= ~0.35)
    """
    numerator = eps_f + 2*eps_m + 2*Vf*(eps_f - eps_m)
    denominator = eps_f + 2*eps_m - Vf*(eps_f - eps_m)
    return eps_m * numerator / denominator


# ============================================================
# REFLECTIVITY (NORMAL INCIDENCE, ISOTROPIC)
# ============================================================

def reflectivity(eps):
    n = np.sqrt(eps)
    return np.abs((n - 1) / (n + 1))**2


# ============================================================
# FREQUENCY RANGE
# ============================================================

f = np.linspace(30e9, 3e12, 2000)   # 30 GHz – 3 THz
f_GHz = f * 1e-9

# ============================================================
# MATERIAL PARAMETERS (PHYSICALLY VALID)
# ============================================================

Vf = 0.30  # Fiber volume fraction (MG-valid)

# Epoxy matrix
eps_m = 3.0 - 1j*0.02

# Glass fibers (weakly lossy dielectrics)
eps_E_glass = 6.0 - 1j*0.015
eps_S_glass = 4.8 - 1j*0.008

# ============================================================
# EFFECTIVE PERMITTIVITY (MAXWELL–GARNETT)
# ============================================================

eps_eff_E = maxwell_garnett(eps_m, eps_E_glass, Vf)
eps_eff_S = maxwell_garnett(eps_m, eps_S_glass, Vf)

# ============================================================
# REFLECTIVITY
# ============================================================

R_E = reflectivity(eps_eff_E)
R_S = reflectivity(eps_eff_S)

# ============================================================
# PLOTTING
# ============================================================

# --- Real part ε′ ---
plt.figure(figsize=(10, 6))
plt.plot(f_GHz, np.real(eps_eff_E), label="E-glass composite")
plt.plot(f_GHz, np.real(eps_eff_S), label="S-glass composite")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Real Permittivity ε′")
plt.title("Real Part of Effective Permittivity")
plt.legend()
plt.grid()
plt.savefig("real_permittivity_glass.png")
plt.close()

# --- Imaginary part ε″ ---
plt.figure(figsize=(10, 6))
plt.plot(f_GHz, np.imag(eps_eff_E), label="E-glass composite")
plt.plot(f_GHz, np.imag(eps_eff_S), label="S-glass composite")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Imaginary Permittivity ε″")
plt.title("Imaginary Part of Effective Permittivity")
plt.legend()
plt.grid()
plt.savefig("imaginary_permittivity_glass.png")
plt.close()

# --- Reflectivity ---
plt.figure(figsize=(10, 6))
plt.plot(f_GHz, R_E, label="E-glass composite")
plt.plot(f_GHz, R_S, label="S-glass composite")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Reflectivity R")
plt.title("Normal-Incidence Reflectivity")
plt.legend()
plt.grid()
plt.savefig("reflectivity_glass.png")
plt.close()

print("Glass-fiber composite plots saved successfully.")
