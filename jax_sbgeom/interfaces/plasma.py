import jax.numpy as jnp 

def _hale_bosch(T):
    ## See: 
    # Bosch H.-S. and Hale G.M. 1992 Improved formulas for fusion cross-sections and thermal reactivities Nucl. Fusion 32 611
    C1 = 1.17302e-9
    C2 = 1.51361e-2
    C3 = 7.51886e-2
    C4 = 4.60643e-3
    C5 = 1.35e-2
    C6 = -1.06750e-4
    C7 = 1.366e-5
    Bg = 34.3827
    mc2 = 1124656    
    T = jnp.where(T < 1e-30, 1e-30, T)
    theta = T / ( 1 - (T* ( C2 + T* ( C4 + T * C6)))/ ( 1  + T* ( C3  + T * ( C5 + T * C7))))
    epsilon = (Bg**2 / (4 * theta)) ** ( 1 /3)
    sigmaV = C1 * theta * jnp.sqrt(epsilon / (mc2 * T**3)) * jnp.exp(- 3 * epsilon)
    return sigmaV * 1e-6 # Hale Bosch is in cm3 /s, we want m3/s

def _reaction_rate_profile(TD, TT, nD, nT):        
    sigmav = _hale_bosch(0.5 * (TD + TT))
    return nD * nT * sigmav

def _parametrised_profile(p0 : float, p1 : float, alpha : float, s : jnp.ndarray):
    return (p0 - p1) * (1 - s)**alpha + p1

def _flux_surface_reaction_rates(
        s_values : jnp.ndarray, 
        nd0 : float , nd1 : float , ndalpha : float,
        nt0 : float , nt1 : float , ntalpha : float,
        Ti0 : float , Ti1 : float , Tialpha : float,        
        ):
    '''
    Function to create a Flux_Surface_Source_14MeV using parametric profile.

    All profiles are defined as:
        (p0 - p1) * (1 - s)**alpha + p1

    Parameters:
    -----------
    s_values : jnp.ndarray
        Normalized flux surface label [0, 1]
    nd0 : float
        Central deuterium density in  m^-3
    nd1 : float
        Edge deuterium density in  m^-3
    ndalpha : float
        Deuterium density profile exponent
    nt0 : float
        Central tritium density in m^-3
    nt1 : float
        Edge tritium density in m^-3
    ntalpha : float
        Tritium density profile exponent
    Ti0 : float
        Central ion temperature in keV
    Ti1 : float
        Edge ion temperature in keV
    Tialpha : float
        Ion temperature profile exponent        

    Returns:
    --------
    
    reaction_rate : jnp.ndarray
        Reaction rate profile in m^-3 s^-1
    '''
    nd = _parametrised_profile(nd0, nd1, ndalpha, s_values)
    nt = _parametrised_profile(nt0, nt1, ntalpha, s_values)
    Ti = _parametrised_profile(Ti0, Ti1, Tialpha, s_values)
    reaction_rate  = _reaction_rate_profile(Ti, Ti, nd, nt)
    return reaction_rate

def flux_surface_reaction_rates_simple(s_values : jnp.ndarray, n0 : float, nalpha : float, Ti0 : float, Tialpha : float):
    '''
    Function to create a Flux_Surface_Source_14MeV using simple profile.
    All profiles are defined as:
        n(s) = n0 * (1 - s)**n_alpha
        T(s) = T0 * (1 - s)**Ti_alpha
    Tritium and deuterium densities are assumed equal (1/2 of n(s)). 
    Edge values are set to zero.

    Parameters:
    -----------
    s_values : jnp.ndarray
        Normalized flux surface label [0, 1]
    n0 : float
        Central ion density m^-3
    nalpha : float
        Ion density profile exponent
    Ti0 : float
        Central ion temperature in keV
    Tialpha : float
        Ion temperature profile exponent        
    n_samples : int
         
    '''
    return _flux_surface_reaction_rates(
        s_values, 
        n0 / 2, 0.0, nalpha,
        n0 / 2, 0.0, nalpha,
        Ti0, 0.0, Tialpha,
    )
