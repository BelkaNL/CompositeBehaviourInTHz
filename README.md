composite_validation_Bayesian.py extends with the folllowing functionalities: 

add frequency-dispersive Debye/Lorentz fillers
export CSV tables for reviewers
include Bayesian inverse fitting

This script composite_em_properties_ghz_thz.py generates:

ε′(f) and ε″(f) vs frequency
 reflectivity R(f)
comparison of E-glass, S-glass, PAN-based CF, Pitch-based CF
frequency range 30 GHz → 3 THz
Maxwell-Garnett effective medium theory
includes carbon fiber conductivity and THz dispersion
accounts for anisotropy through ⟂ and ∥ permittivity models (included below)


drude_lorentz_scattering_tmm_composites.py generates:

THz-TDS validated parameter sets (actual published values) 
Multi-layer laminate stack reflection/transmission model (transfer matrix method, 2×2 electromagnetic propagation matrices) 
Fit experimental THz-TDS data to extract ε′ and ε″ 
Add scattering (Rayleigh / Mie) terms for fiber roughness
Add Drude–Lorentz dispersion for carbon fibers
