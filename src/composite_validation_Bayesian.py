import numpy as np
from numpy import trapz
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import os
import csv

# ============================================================
# GLOBAL SETTINGS
# ============================================================

np.random.seed(0)
SAVE_DIR = "journal_validation_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

f = np.linspace(30e9, 3e12, 1200)
w = 2 * np.pi * f
eps0 = 8.854e-12

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
    eps = np.full_like(eps_f, eps_m, dtype=complex)
    for _ in range(n_iter):
        F = ((1-Vf)*(eps_m-eps)/(eps_m+2*eps)
             + Vf*(eps_f-eps)/(eps_f+2*eps))
        eps -= 0.5 * F
    return eps

# ============================================================
# DISPERSIVE FILLER MODELS
# ============================================================

def debye_eps(eps_inf, delta_eps, tau, sigma=0.0):
    return eps_inf + delta_eps / (1 + 1j*w*tau) + sigma/(1j*w*eps0)

def lorentz_eps(eps_inf, f0, gamma, strength):
    w0 = 2 * np.pi * f0
    return eps_inf + strength * w0**2 / (w0**2 - w**2 - 1j*gamma*w)

# ============================================================
# SYNTHETIC DATA GENERATORS
# ============================================================

def noisy(x, level=0.02):
    return x * (1 + level * np.random.randn(*x.shape))

def synthetic_debye_particulate():
    eps_m = 2.8 - 1j*0.01
    eps_f = debye_eps(4.0, 3.0, 3e-13)
    Vf = 0.25
    eps_true = maxwell_garnett(eps_m, eps_f, Vf)
    return noisy(eps_true), eps_m, eps_f, Vf

def synthetic_lorentz_percolating():
    eps_m = 3.2 - 1j*0.02
    eps_f = lorentz_eps(2.5, 0.9e12, 1.2e11, 6.0)
    Vf = 0.45
    eps_true = bruggeman(
        np.full_like(f, eps_m, dtype=complex),
        eps_f,
        Vf
    )
    return noisy(eps_true), eps_m, eps_f, Vf

# ============================================================
# METRICS
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
# KRAMERS–KRONIG CHECK
# ============================================================

def kk_real_from_imag(eps_imag):
    eps_real = np.zeros_like(eps_imag)
    for i in range(len(w)):
        denom = w**2 - w[i]**2
        denom[i] = np.inf
        eps_real[i] = (2/np.pi) * trapz(w * eps_imag / denom, w)
    return eps_real

def kk_error(eps):
    return np.mean(np.abs(np.real(eps) - kk_real_from_imag(np.imag(eps))))

# ============================================================
# CSV EXPORT
# ============================================================

def export_csv(fname, eps_true, eps_models, labels):
    path = os.path.join(SAVE_DIR, fname)
    header = ["Frequency_Hz",
              "eps_real_true", "eps_imag_true", "tan_delta_true"]
    for lab in labels:
        header += [f"eps_real_{lab}", f"eps_imag_{lab}", f"tan_delta_{lab}"]

    with open(path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(header)
        for i in range(len(f)):
            row = [f[i],
                   np.real(eps_true[i]),
                   np.imag(eps_true[i]),
                   loss_tangent(eps_true[i])]
            for eps in eps_models:
                row += [np.real(eps[i]),
                        np.imag(eps[i]),
                        loss_tangent(eps[i])]
            writer.writerow(row)

# ============================================================
# BAYESIAN INVERSE FITTING
# ============================================================

def log_likelihood(eps_meas, eps_pred, sigma=0.05):
    err = np.abs(eps_meas - eps_pred)
    return -0.5 * np.sum((err/sigma)**2)

def bayesian_inverse_fit(eps_meas, eps_m, n_steps=6000):
    Vf, er, ei = 0.3, 5.0, 0.05
    chain = []

    def forward(Vf, er, ei):
        return maxwell_garnett(eps_m, er - 1j*ei, Vf)

    logp = log_likelihood(eps_meas, forward(Vf, er, ei))

    for _ in range(n_steps):
        Vf_p = np.clip(Vf + 0.02*np.random.randn(), 0.01, 0.6)
        er_p = er + 0.3*np.random.randn()
        ei_p = abs(ei + 0.02*np.random.randn())

        logp_p = log_likelihood(eps_meas, forward(Vf_p, er_p, ei_p))

        if np.log(np.random.rand()) < (logp_p - logp):
            Vf, er, ei, logp = Vf_p, er_p, ei_p, logp_p

        chain.append([Vf, er, ei])

    return np.array(chain)

# ============================================================
# PLOTTING
# ============================================================

def plot_validation(eps_true, eps_models, labels, title, fname):
    plt.figure(figsize=(8,5))
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
# MAIN PIPELINE
# ============================================================

datasets = {
    "Debye_Particulate": synthetic_debye_particulate,
    "Lorentz_Percolating": synthetic_lorentz_percolating
}

for name, gen in datasets.items():
    eps_true, eps_m, eps_f, Vf = gen()

    eps_mg = maxwell_garnett(
        np.full_like(f, eps_m, dtype=complex),
        eps_f,
        Vf
    )
    eps_br = bruggeman(
        np.full_like(f, eps_m, dtype=complex),
        eps_f,
        Vf
    )

    plot_validation(
        eps_true,
        [eps_mg, eps_br],
        ["MG", "Bruggeman"],
        name,
        f"{name}.png"
    )

    export_csv(
        f"{name}.csv",
        eps_true,
        [eps_mg, eps_br],
        ["MG", "Bruggeman"]
    )

    print(f"\n{name}")
    print("MG:", score_model(eps_true, eps_mg))
    print("Bruggeman:", score_model(eps_true, eps_br))
    print("K–K error:", kk_error(eps_true))

# ============================================================
# BAYESIAN DEMO
# ============================================================

eps_true, eps_m, _, _ = synthetic_debye_particulate()
chain = bayesian_inverse_fit(eps_true, eps_m)

burn = int(0.3 * len(chain))
post = chain[burn:]

print("\nBayesian posterior (mean ± std)")
print("Vf:", np.mean(post[:,0]), "±", np.std(post[:,0]))
print("eps_f real:", np.mean(post[:,1]), "±", np.std(post[:,1]))
print("eps_f imag:", np.mean(post[:,2]), "±", np.std(post[:,2]))
