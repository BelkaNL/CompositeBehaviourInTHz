#!/usr/bin/env python3
"""
Composite Validation Framework
DOI-ready | CI-enforced | Reproducible
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import trapz
from scipy.optimize import least_squares
import yaml, sys, os, csv, subprocess, datetime

# ============================================================
# LOAD CONFIG
# ============================================================

if len(sys.argv) != 2:
    raise RuntimeError("Usage: python composite_validation.py config.yml")

with open(sys.argv[1], "r") as f:
    cfg = yaml.safe_load(f)

np.random.seed(cfg["project"]["random_seed"])
SAVE_DIR = cfg["project"]["output_directory"]
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# METADATA (DOI-READY)
# ============================================================

def write_metadata(cfg):
    meta = {
        "project": cfg["project"],
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "git_commit": subprocess.getoutput("git rev-parse HEAD")
    }
    with open(os.path.join(SAVE_DIR, "metadata.yml"), "w") as f:
        yaml.dump(meta, f)

write_metadata(cfg)

# ============================================================
# FREQUENCY GRID
# ============================================================

f = np.linspace(
    cfg["frequency"]["start_hz"],
    cfg["frequency"]["stop_hz"],
    cfg["frequency"]["num_points"]
)
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

def bruggeman(eps_m, eps_f, Vf, n_iter=200):
    eps = np.full_like(eps_f, eps_m, dtype=complex)
    for _ in range(n_iter):
        F = ((1-Vf)*(eps_m-eps)/(eps_m+2*eps)
             + Vf*(eps_f-eps)/(eps_f+2*eps))
        eps -= 0.5 * F
    return eps

# ============================================================
# DISPERSIVE FILLERS
# ============================================================

def debye_eps(eps_inf, delta_eps, tau, sigma=0.0):
    return eps_inf + delta_eps/(1 + 1j*w*tau) + sigma/(1j*w*eps0)

def lorentz_eps(eps_inf, f0, gamma, strength):
    w0 = 2*np.pi*f0
    return eps_inf + strength*w0**2/(w0**2 - w**2 - 1j*gamma*w)

# ============================================================
# METRICS
# ============================================================

def rmse(a, b):
    return np.sqrt(np.mean(np.abs(a-b)**2))

def loss_tangent(eps):
    return np.imag(eps)/np.real(eps)

def kk_real_from_imag(eps_imag):
    eps_real = np.zeros_like(eps_imag)
    for i in range(len(w)):
        denom = w**2 - w[i]**2
        denom[i] = np.inf
        eps_real[i] = (2/np.pi)*trapz(w*eps_imag/denom, w)
    return eps_real

def kk_error(eps):
    return np.mean(np.abs(np.real(eps) - kk_real_from_imag(np.imag(eps))))

# ============================================================
# CSV EXPORT
# ============================================================

def export_csv(fname, eps):
    with open(os.path.join(SAVE_DIR, fname), "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["frequency_hz", "eps_real", "eps_imag", "tan_delta"])
        for i in range(len(f)):
            writer.writerow([
                f[i],
                np.real(eps[i]),
                np.imag(eps[i]),
                loss_tangent(eps[i])
            ])

# ============================================================
# PLOTTING
# ============================================================

def plot_eps(fname, eps):
    plt.figure(figsize=(8,5))
    plt.plot(f*1e-9, np.real(eps), lw=2)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Real Permittivity")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, fname), dpi=300)
    plt.close()

# ============================================================
# PARAMETER SWEEP + CI ENFORCEMENT
# ============================================================

results = []

for Vf in cfg["parameter_sweeps"]["volume_fraction"]:
    for tau in cfg["parameter_sweeps"]["debye_tau_seconds"]:

        eps_m = 2.8 - 1j*0.01
        eps_f = debye_eps(
            cfg["fillers"]["debye"]["eps_inf"],
            cfg["fillers"]["debye"]["delta_eps"],
            tau
        )

        eps_eff = maxwell_garnett(eps_m, eps_f, Vf)

        kk = kk_error(eps_eff)
        r = rmse(eps_eff, eps_m)

        results.append({
            "Vf": Vf,
            "tau": tau,
            "kk_error": kk,
            "rmse": r
        })

        # CI FAILURE CONDITIONS
        if kk > cfg["ci_thresholds"]["max_kk_error"]:
            raise RuntimeError(
                f"CI FAIL: K–K error {kk:.3f} exceeds threshold"
            )

        if r > cfg["ci_thresholds"]["max_rmse_eps"]:
            raise RuntimeError(
                f"CI FAIL: RMSE {r:.3f} exceeds threshold"
            )

        tag = f"Vf{Vf}_tau{tau:.1e}"
        export_csv(f"{tag}.csv", eps_eff)
        plot_eps(f"{tag}.png", eps_eff)

# ============================================================
# SAVE SWEEP SUMMARY
# ============================================================

with open(os.path.join(SAVE_DIR, "parameter_sweep_summary.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\nAll parameter sweeps completed successfully.")
print("CI thresholds satisfied.")
print(f"Results written to: {SAVE_DIR}")
