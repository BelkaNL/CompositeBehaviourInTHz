import numpy as np
import matplotlib.pyplot as plt

eps0 = 8.854e-12
c = 3e8

# Frequency range (THz)
f = np.linspace(0.05e12, 3e12, 2000)
w = 2*np.pi*f

# ============================================================
# 1. DRUDE-LORENTZ MODEL FOR CARBON FIBERS
# ============================================================

def drude_lorentz(w, eps_inf, wp, gamma, w0=None, f0=None, gamma0=None):
    eps = eps_inf - wp**2/(w*(w+1j*gamma))
    if w0 is not None:
        eps += (f0*w0**2)/(w0**2 - w**2 - 1j*gamma0*w)
    return eps

# Example PAN and pitch fibers
eps_PAN = drude_lorentz(w, eps_inf=5, wp=2e13, gamma=5e12)
eps_pitch = drude_lorentz(w, eps_inf=5, wp=4e13, gamma=2e12)

# ============================================================
# 2. RAYLEIGH / MIE SCATTERING (simplified Rayleigh term)
# ============================================================

def rayleigh_scattering(a, wavelength, m):
    return (8*np.pi/3)*(a**6)/(wavelength**4)*np.abs((m**2-1)/(m**2+2))**2

a = 7e-6                     # fiber radius (7 microns)
wavelength = c/f
m = np.sqrt(eps_PAN)

sigma_Rayleigh = rayleigh_scattering(a, wavelength, m)
eps_scatt = -1j*sigma_Rayleigh/(w*eps0)

# ============================================================
# 3. COMPOSITE PERMITTIVITY (Maxwell Garnett + scattering)
# ============================================================

eps_matrix = 3.0 - 1j*0.05
Vf = 0.55

def maxwell_garnett(eps_m, eps_f, Vf):
    return eps_m * (eps_f + 2*eps_m + 2*Vf*(eps_f - eps_m)) / \
           (eps_f + 2*eps_m - Vf*(eps_f - eps_m))

eps_eff_PAN = maxwell_garnett(eps_matrix, eps_PAN, Vf) + eps_scatt
eps_eff_pitch = maxwell_garnett(eps_matrix, eps_pitch, Vf) + eps_scatt

# ============================================================
# 4. MULTILAYER TRANSFER MATRIX METHOD (TMM)
# ============================================================

def tmm_stack(eps_list, d_list, f):
    w = 2*np.pi*f
    k0 = w/c
    eta0 = 377.0

    M = np.array([[1+0j,0],[0,1+0j]])

    for eps, d in zip(eps_list, d_list):
        n = np.sqrt(eps)
        k = k0*n
        eta = eta0/n
        
        P = np.array([[np.exp(-1j*k*d), 0],
                      [0, np.exp(1j*k*d)]])
        
        I = 0.5/eta * np.array([[eta+eta0, eta-eta0],
                               [eta-eta0, eta+eta0]])
        
        M = I @ P @ M

    T = 1/np.abs(M[0,0])**2
    R = np.abs(M[1,0]/M[0,0])**2

    return T, R

# Example 3-layer laminate
eps_stack = [eps_matrix, eps_eff_PAN, eps_matrix]
d_stack = [200e-6, 150e-6, 200e-6]

T_stack, R_stack = tmm_stack(eps_stack, d_stack, f)

# ============================================================
# 5. THz-TDS FITTING (INVERSION)
# ============================================================

def extract_eps_from_T(E, Eref, d, f):
    H = E/Eref
    w = 2*np.pi*f
    phi = np.unwrap(np.angle(H))
    mag = np.abs(H)

    n = 1 + c/(w*d)*phi
    kappa = -c/(w*d)*np.log(mag)
    
    eps_real = n**2 - kappa**2
    eps_imag = 2*n*kappa

    return eps_real, eps_imag

# ============================================================
# 6. PLOTTING
# ============================================================

plt.figure(figsize=(10,6))
plt.plot(f*1e-12, np.real(eps_eff_PAN), label="PAN composite ε′")
plt.plot(f*1e-12, np.real(eps_eff_pitch), label="Pitch composite ε′")
plt.xlabel("Frequency (THz)"); plt.ylabel("ε′"); plt.legend(); plt.grid()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(f*1e-12, np.imag(eps_eff_PAN), label="PAN composite ε″")
plt.plot(f*1e-12, np.imag(eps_eff_pitch), label="Pitch composite ε″")
plt.xlabel("Frequency (THz)"); plt.ylabel("ε″"); plt.legend(); plt.grid()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(f*1e-12, R_stack, label="Multilayer Reflectivity")
plt.plot(f*1e-12, T_stack, label="Multilayer Transmission")
plt.xlabel("Frequency (THz)"); plt.ylabel("T / R"); plt.legend(); plt.grid()
plt.show()

