import numpy as np

def maxwell_garnett(eps_m, eps_f, Vf):
    """
    Maxwell-Garnett model to calculate the effective permittivity of a composite material.
    
    Parameters:
    eps_m : complex or float
        The permittivity of the matrix material (can be complex for lossy materials).
    eps_f : complex or float
        The permittivity of the filler or fiber material (can be complex for lossy materials).
    Vf : float
        The volume fraction of the fiber material in the composite (0 <= Vf <= 1).
        
    Returns:
    eps_eff : complex
        The effective permittivity of the composite material.
    """
    
    # Implementing the Maxwell-Garnett equation
    numerator = eps_f + 2 * eps_m + 2 * Vf * (eps_f - eps_m)
    denominator = eps_f + 2 * eps_m - Vf * (eps_f - eps_m)
    
    eps_eff = eps_m * (numerator / denominator)
    
    return eps_eff
