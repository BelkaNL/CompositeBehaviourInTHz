import numpy as np
import matplotlib
matplotlib.use("Agg")   # Headless-safe
import matplotlib.pyplot as plt
import os

# ============================================================
# MODELS
# ============================================================

def maxwell_garnett(eps_m, eps_f, Vf):
    num = eps_f + 2*eps_m + 2*Vf*(eps_f - eps_m)
    den = eps_f + 2*eps_m - Vf*(eps_f - eps_m)
    return eps_m * num / den

def eps_parallel(eps_m, eps_f, Vf):
    return Vf * eps_f + (1 - Vf) * eps_m

def eps_perpendicular(eps_m, eps_f, Vf):
    return eps_m * (eps_f + eps_m + Vf*(eps_f - eps_m)) / \
                   (eps_f + eps_m - Vf*(eps_f - eps_m))

def orientation_average(eps_para, eps_perp):
    return (1/3)*eps_para + (2/3)*eps_perp

def loss_tangent(eps):
    return np.imag(eps) / np.real(eps)

def reflectivity(eps):
    n = np.sqrt(eps)
    return np.abs((n - 1)/(n + 1))**2

# ============================================================
# FREQUENCY AXIS
# ============================================================

f = np.linspace(30e9, 3e12, 2000)
f_GHz = f * 1e-9

# ============================================================
# MATERIAL PARAMETERS
# ============================================================

Vf = 0.30  # Fiber volume fraction (MG-valid)

# Epoxy matrix (array)
eps_m_scalar = 3.0 - 1j*0.02
eps_m = eps_m_scalar * np.ones_like(f)

# E-glass fiber
eps_f_real_0 = 6.0
eps_f_imag_0 = 0.015
eps_f = (eps_f_real_0 - 1j*eps_f_imag_0) * np.ones_like(f)

# ============================================================
# ORIENTATION-DEPENDENT PERMITTIVITY
# ============================================================

eps_para = eps_parallel(eps_m, eps_f, Vf)
eps_perp = eps_perpendicular(eps_m, eps_f, Vf)
eps_eff = orientation_average(eps_para, eps_perp)

# ============================================================
# LOSS TANGENT & REFLECTIVITY
# ============================================================

tan_delta = loss_tangent(eps_eff)
R = reflectivity(eps_eff)

# ============================================================
# CREATE PLOTS FOLDER
# ============================================================

save_dir = os.path.join(os.getcwd(), "plots")
os.makedirs(save_dir, exist_ok=True)
print("Plots folder created at:", save_dir)

# ============================================================
# PLOTTING
# ============================================================

plots = {}

# --- Real and Imaginary Permittivity ---
plt.figure(figsize=(10,6))
plt.plot(f_GHz, np.real(eps_eff), label="ε′ (avg)")
plt.plot(f_GHz, np.imag(eps_eff), label="ε″ (avg)")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Permittivity")
plt.legend()
plt.grid()
plt.title("Orientation-Averaged Effective Permittivity (Glass Composite)")
file_eps = os.path.join(save_dir, "eps_effective.png")
plt.savefig(file_eps)
plt.close()
plots['Permittivity'] = file_eps

# --- Loss Tangent ---
plt.figure(figsize=(10,6))
plt.plot(f_GHz, tan_delta)
plt.xlabel("Frequency (GHz)")
plt.ylabel("tan δ")
plt.title("Loss Tangent vs Frequency")
plt.grid()
file_tand = os.path.join(save_dir, "loss_tangent.png")
plt.savefig(file_tand)
plt.close()
plots['Loss Tangent'] = file_tand

# --- Reflectivity ---
plt.figure(figsize=(10,6))
plt.plot(f_GHz, R)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Reflectivity R")
plt.title("Normal-Incidence Reflectivity")
plt.grid()
file_ref = os.path.join(save_dir, "reflectivity.png")
plt.savefig(file_ref)
plt.close()
plots['Reflectivity'] = file_ref

# ============================================================
# CONFIRMATION
# ============================================================

print("Plots saved successfully!")
for name, path in plots.items():
    print(f"{name} plot: {path}")
