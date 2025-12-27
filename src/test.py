import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless-safe
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

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
# MATERIAL PARAMETERS (MG-VALID)
# ============================================================

Vf = 0.30  # valid for Maxwell–Garnett

# Epoxy matrix (array!)
eps_m_scalar = 3.0 - 1j*0.02
eps_m = eps_m_scalar * np.ones_like(f)

# E-glass fiber (initial / nominal values)
eps_f_real_0 = 6.0
eps_f_imag_0 = 0.015

# Fiber permittivity AS ARRAY (CRITICAL FIX)
eps_f = (eps_f_real_0 - 1j*eps_f_imag_0) * np.ones_like(f)

# ============================================================
# ORIENTATION-DEPENDENT EFFECTIVE PERMITTIVITY
# ============================================================

eps_para = eps_parallel(eps_m, eps_f, Vf)
eps_perp = eps_perpendicular(eps_m, eps_f, Vf)

# Randomly oriented glass fiber composite
eps_eff = orientation_average(eps_para, eps_perp)

# ============================================================
# LOSS TANGENT & REFLECTIVITY
# ============================================================

tan_delta = loss_tangent(eps_eff)
R = ref
