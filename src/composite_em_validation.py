import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import os

# ============================================================
# GLOBAL SETTINGS
# ============================================================

np.random.seed(0)
SAVE_DIR = "journal_validation_plots"
os.makedirs(SAVE_DIR, exist_ok=True)

f = np.linspace(30e9, 3e12, 1200)
w = 2 * np.pi * f

# ============================================================
# EFFECTIVE MEDIUM MODELS
# ============================================================

def maxwell_garnett(eps_m, eps_f, Vf):
    return eps_m * (
        (eps_f + 2*eps_m + 2*Vf*(eps_f - eps_m)) /
        (eps_f + 2*eps_m - Vf*(eps_f - eps_m))
    )

def eps_parallel(eps_m, eps_f, Vf):
    return Vf * eps_f + (1 - Vf) * eps_m

def eps_perpendicular(eps_m, eps_f, Vf):
    return eps_m * (
        (eps_f + eps_m + Vf*(eps_f - eps_m)) /
        (eps_f + eps_m - Vf*(eps_f - eps_m))
    )

def orientation_average(eps_para, eps_perp):
    return (1/3)*eps_para + (2/3)*eps_perp

def bruggeman(eps_m, eps_f, Vf, n_iter=200):
    eps = eps_m * np.ones_like(eps_f)
    for _ in range(n_iter):
        F = ((1-Vf)*(eps_m-eps)/(eps_m+2*eps)
             + Vf*(eps_f-eps)/(eps_f+2*eps))
        eps -= 0.5 * F
    return eps

# ============================================================
# SYNTHETIC DATA GENERATORS
# ============================================================

def noisy(x, level=0.02):
    return x * (1 + level * np.random.randn(*x.shape))

def synthetic_particulate():
    eps_m = 3.0 - 1j*0.02
    eps_f = 5.0 - 1j*0.01 + 0.05*np.log(f/f[0])
    Vf = 0.20
    return noisy(maxwell_garnett(eps_m, eps_f, Vf)), eps_m, eps_f, Vf

def synthetic_random_fiber():
    eps_m = 3.0 - 1j*0.02
    eps_f = 6.0 - 1j*0.015
    Vf = 0.30
    eps = orientation_average(
        eps_parallel(eps_m, eps_f, Vf),
        eps_perpendicular(eps_m, eps_f, Vf)
    )
    return noisy(eps), eps_m, eps_f, Vf

def synthetic_percolating():
    eps_m = 3.0 - 1j*0.02
    eps_f = 100 - 1j*(1e3/w)
    Vf = 0.50
    return noisy(bruggeman(eps_m, eps_f, Vf)), eps_m, eps_f, Vf

# ============================================================
# SCORING & PHYSICAL METRICS
# ============================================================

def rmse(a, b):
    return np.sqrt(np.mean(np.abs(a-b)**2))

def loss_tangent(eps):
    return np.imag(eps) / np.real(eps)

def reflectivity(eps):
    n = np.sqrt(eps)
    return np.abs((n-1)/(n+1))**2

def score_model(eps_true, eps_model):
    return {
        "RMSE_eps": rmse(eps_true, eps_model),
        "RMSE_real": rmse(np.real(eps_true), np.real(eps_model)),
        "RMSE_imag": rmse(np.imag(eps_true), np.imag(eps_model)),
        "RMSE_tand": rmse(loss_tangent(eps_true), loss_tangent(eps_model)),
        "RMSE_R": rmse(reflectivity(eps_true), reflectivity(eps_model))
    }

# ============================================================
# MONTE CARLO CONFIDENCE INTERVALS
# ============================================================

def monte_carlo_ci(model_func, n_mc=200):
    eps_samples = np.array([model_func() for _ in range(n_mc)])
    return (
        np.mean(eps_samples, axis=0),
        np.percentile(eps_samples, 2.5, axis=0),
        np.percentile(eps_samples, 97.5, axis=0)
    )

# ============================================================
# INVERSE FITTING
# ============================================================

def inverse_fit(eps_meas, eps_m, model="MG"):
    def residuals(p):
        Vf, er, ei = p
        eps_f = er - 1j*ei
        if model == "MG":
            eps_pred = maxwell_garnett(eps_m, eps_f, Vf)
        else:
            eps_pred = orientation_average(
                eps_parallel(eps_m, eps_f, Vf),
                eps_perpendicular(eps_m, eps_f, Vf)
            )
        return np.hstack([
            np.real(eps_pred - eps_meas),
            np.imag(eps_pred - eps_meas)
        ])

    p0 = [0.25, 5.0, 0.02]
    bounds = ([0.01, 1.0, 0.0], [0.6, 50.0, 1.0])
    res = least_squares(residuals, p0, bounds=bounds)
    return res.x, res.cost

# ============================================================
# KRAMERS–KRONIG CAUSALITY CHECK
# ============================================================

def kk_real_from_imag(eps_imag):
    eps_real = np.zeros_like(eps_imag)
    for i in range(len(w)):
        denom = w**2 - w[i]**2
        denom[i] = np.inf  # avoid division by zero
        integrand = w * eps_imag / denom
        eps_real[i] = (2/np.pi) * np.trapz(integrand, w)
    return eps_real

def kk_error(eps):
    return np.mean(np.abs(np.real(eps) - kk_real_from_imag(np.imag(eps))))

# ============================================================
# PLOTTING (JOURNAL STYLE)
# ============================================================

def plot_validation(eps_true, eps_models, labels, title, fname):
    plt.figure(figsize=(8, 5))
    plt.plot(f*1e-9, np.real(eps_true), 'k', lw=2, label="Synthetic truth")
    for eps, lab in zip(eps_models, labels):
        plt.plot(f*1e-9, np.real(eps), '--', lw=1.5, label=lab)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Real Permittivity")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, fname), dpi=300)
    plt.close()

# ============================================================
# MAIN VALIDATION PIPELINE
# ============================================================

datasets = {
    "Particulate": synthetic_particulate,
    "Random Fiber": synthetic_random_fiber,
    "Percolating": synthetic_percolating
}

for name, generator in datasets.items():
    eps_true, eps_m, eps_f, Vf = generator()

    eps_mg = maxwell_garnett(eps_m, eps_f, Vf)
    eps_fiber = orientation_average(
        eps_parallel(eps_m, eps_f, Vf),
        eps_perpendicular(eps_m, eps_f, Vf)
    )
    eps_brug = bruggeman(eps_m, eps_f, Vf)

    plot_validation(
        eps_true,
        [eps_mg, eps_fiber, eps_brug],
        ["Maxwell–Garnett", "Orientation-avg", "Bruggeman"],
        f"{name} Composite Validation",
        f"{name.lower().replace(' ', '_')}.png"
    )

    print(f"\n{name} composite")
    print("MG score:", score_model(eps_true, eps_mg))
    print("Fiber score:", score_model(eps_true, eps_fiber))
    print("Bruggeman score:", score_model(eps_true, eps_brug))
    print("K–K error:", kk_error(eps_true))

# ============================================================
# INVERSE FIT DEMO (RANDOM FIBER)
# ============================================================

eps_true, eps_m, eps_f_true, Vf_true = synthetic_random_fiber()
params, cost = inverse_fit(eps_true, eps_m, model="fiber")

print("\nInverse fitting (random fiber):")
print("True Vf:", Vf_true, "Retrieved Vf:", params[0])
print("True eps_f:", eps_f_true)
print("Retrieved eps_f:", params[1], "- j", params[2])
