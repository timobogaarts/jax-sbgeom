import jax 
import jax.numpy as jnp
from functools import partial
from .flux_surfaces_base import _create_mpol_vector, _create_ntor_vector, FluxSurface, FluxSurfaceData, FluxSurfaceModes, FluxSurfaceSettings, _interpolate_s_grid_full_mod, _arc_length_theta_interpolating_s_grid_full_mod, _arc_length_theta_interpolating_s_grid_full_mod_finite_difference
from .flux_surfaces_base import _arc_length_theta_direct, _cylindrical_position_direct
from .flux_surfaces_extended import FluxSurfaceNormalExtended, FluxSurfaceNormalExtendedNoPhi, FluxSurfaceNormalExtendedConstantPhi, FluxSurfaceFourierExtended
from jax_sbgeom.jax_utils.utils import bilinear_interp, _resample_uniform_periodic_pchip, _resample_uniform_periodic_linear
from warnings import warn
from typing import Type, Tuple
import equinox as eqx
from dataclasses import dataclass

@jax.jit
def _dft_forward(points : jnp.ndarray) -> jnp.ndarray:
    '''
    Compute the scaled discrete fourier transform of a 2D grid of points

    Parameters
    ----------
    points : jnp.ndarray
        2D grid of points to compute the DFT of (n_theta, n_phi)
    Returns
    -------
    fft_values : jnp.ndarray
        The scaled DFT values
    n_theta : int
        Number of theta points
    n_phi : int
        Number of phi points
    '''    

    return jnp.fft.fft2(points, norm='forward')        

@jax.jit
def _cos_sin_from_dft_forward(dft_coefficients : jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    N, M = dft_coefficients.shape # static so can use control flow

    N_h = N // 2 + 1
    M_h = M // 2 + 1

    def divide_nyquist(arr, N, M):
        if N % 2 == 0:
            arr = arr.at[-1, :].divide(2.0)
        if M % 2 == 0:
            arr = arr.at[:, -1].divide(2.0)
        return arr

    # x^c_{kl}
    xckl = 2 * jnp.real(dft_coefficients[:N_h, :M_h])
    xckl = xckl.at[0, 0].divide(2.0)
    xckl = divide_nyquist(xckl, N, M)

    # x^{c-}_{kl}
    xcmkl = jnp.zeros_like(xckl)
    flipped = jnp.real(dft_coefficients[:, ::-1])
    xcmkl = xcmkl.at[1:, 1:].set(2 * flipped[1:N_h, :M_h - 1])
    xcmkl = divide_nyquist(xcmkl, N, M)

    # x^s_{kl}
    xskl = -2 * jnp.imag(dft_coefficients[:N_h, :M_h])
    xskl = divide_nyquist(xskl, N, M)

    # x^{s-}_{kl}
    xsmkl = jnp.zeros_like(xskl)
    flipped_imag = jnp.imag(dft_coefficients[:, ::-1])
    xsmkl = xsmkl.at[1:, 1:].set(-2 * flipped_imag[1:N_h, :M_h - 1])
    xsmkl = divide_nyquist(xsmkl, N, M)

    if N % 2 == 0:
        xsmkl = xsmkl.at[-1, :].multiply(-1.0)

    return xckl, xcmkl, xskl, xsmkl


@partial(jax.jit, static_argnums = 4)
def _convert_cos_sin_to_vmec(xckl : jnp.ndarray, xcmkl : jnp.ndarray, xskl : jnp.ndarray, xsmkl : jnp.ndarray, cosine : bool) -> jnp.ndarray:
    mpol = xckl.shape[0] - 1
    ntor = xckl.shape[1] - 1
    settings    = FluxSurfaceSettings(mpol=mpol, ntor=ntor, nfp=1)

    modes       = FluxSurfaceModes.from_settings(settings)
    mpol_vector = modes.mpol_vector
    ntor_vector = modes.ntor_vector
    ntor_vector_abs = jnp.abs(ntor_vector)

    if cosine:        
        mn_is0     = jnp.logical_or(mpol_vector == 0, ntor_vector == 0)
        n_isneg    = ntor_vector < 0
        v_mn0      =  xckl [mpol_vector, ntor_vector_abs] + xcmkl[mpol_vector, ntor_vector_abs]      # m = 0 or n = 0
        v_pos_npos =  xcmkl[mpol_vector, ntor_vector_abs]                                            # m > 0, n > 0
        v_pos_nneg =  xckl [mpol_vector, ntor_vector_abs]                                            # m > 0, n < 0
        return jnp.where( mn_is0, v_mn0, jnp.where( n_isneg, v_pos_nneg, v_pos_npos))
    else:
        mn_isboth0 = jnp.logical_and(mpol_vector == 0, ntor_vector == 0)
        m_is0      = mpol_vector == 0
        n_isneg    = ntor_vector < 0
        n_is0      = ntor_vector == 0

        v_mnboth0      = jnp.zeros_like(mpol_vector)                                                   # m = 0, n = 0  
        v_m0_npos      = - xskl [mpol_vector, ntor_vector_abs]   + xsmkl[mpol_vector, ntor_vector_abs] # m = 0, n > 0
        v_mpos_n0      =   xskl [mpol_vector, ntor_vector_abs]   + xsmkl[mpol_vector,ntor_vector_abs ] # m > 0, n = 0
        v_mpos_nneg    =   xskl [mpol_vector, ntor_vector_abs]                                         # m > 0, n < 0
        v_mpos_npos    =   xsmkl[mpol_vector, ntor_vector_abs]                                         # m > 0, n > 0        
        return jnp.where( mn_isboth0, v_mnboth0,                                        # m = 0, n = 0
                                      jnp.where( m_is0,      v_m0_npos,                 #  m_0, n! =0  
                                                jnp.where(n_is0,  v_mpos_n0,            # n = 0, m > 0
                                                        jnp.where(n_isneg, v_mpos_nneg, # m != 0 n < 0
                                                                    v_mpos_npos))))     # m!= 0 n > 0

@jax.jit
def _rz_to_vmec_representation(R_grid : jnp.ndarray, Z_grid : jnp.ndarray) -> FluxSurfaceData:
    assert R_grid.shape == Z_grid.shape, "R and Z grids must have the same shape but got {} and {}".format(R_grid.shape, Z_grid.shape)
    R_dft = _dft_forward(R_grid)
    Z_dft = _dft_forward(Z_grid)
    R_ckl, R_cmkl, R_skl, R_smkl = _cos_sin_from_dft_forward(R_dft)
    Z_ckl, Z_cmkl, Z_skl, Z_smkl = _cos_sin_from_dft_forward(Z_dft)
    R_vmec = _convert_cos_sin_to_vmec(R_ckl, R_cmkl, R_skl, R_smkl, cosine=True)
    Z_vmec = _convert_cos_sin_to_vmec(Z_ckl, Z_cmkl, Z_skl, Z_smkl, cosine=False)
    return FluxSurfaceData(R_vmec, Z_vmec)

def _index_mn(m,n, ntor):    
    return n + (m>0) * (2 * ntor + 1) * m  

def _size_mn(mpol, ntor):
    return (2 * ntor + 1) * mpol + ntor +1

@eqx.filter_jit
def _convert_array_to_different_settings(array : jnp.ndarray, new_settings : FluxSurfaceSettings, old_settings : FluxSurfaceSettings) -> jnp.ndarray:    
    '''
    Convert a Fourier representation from one (mpol, ntor) to another (mpol, ntor) by zero-padding or truncating.
    Does not take into account the field-period symmetry: this can thus also be used to convert to different nfp.

    Parameters:
    ----------
    array : jnp.ndarray
        Array of shape (..., N) where N is the number of Fourier modes in the old representation.   
    new_settings : FluxSurfaceSettings
        The new Fourier settings (mpol, ntor).
    old_settings : FluxSurfaceSettings
        The old Fourier settings (mpol, ntor).
    Returns:
    -------
    array_new : jnp.ndarray
        Array of shape (..., N_new) where N_new is the number of Fourier modes in the new representation.
    '''
    settings_1_nfp  = FluxSurfaceSettings(mpol=new_settings.mpol, ntor=new_settings.ntor, nfp=1) 
    mpol_vector_new = _create_mpol_vector(settings_1_nfp)
    ntor_vector_new = _create_ntor_vector(settings_1_nfp)     
    data_available  = jnp.logical_and(mpol_vector_new <= old_settings.mpol, jnp.abs(ntor_vector_new) <= old_settings.ntor)
    
    # we ensure we don't go out of bounds here by setting indices to 0 when data is not available
    # jnp.where *will* access both branches before selecting. We need to set the out-of-bounds indices to a safe value and then 
    # select 0.0 in the following where:
    index_mn_new    = jnp.where(data_available, _index_mn(mpol_vector_new, ntor_vector_new, old_settings.ntor), 0)
    array_new       = jnp.where(data_available, array[..., index_mn_new], 0.0)            
    return array_new

@eqx.filter_jit
def _convert_fluxsurfacedata_to_different_settings(data : FluxSurfaceData, new_settings : FluxSurfaceSettings, old_settings : FluxSurfaceSettings):
    Rmnc_new = _convert_array_to_different_settings(data.Rmnc, new_settings, old_settings)
    Zmns_new = _convert_array_to_different_settings(data.Zmns, new_settings, old_settings)
    return FluxSurfaceData(Rmnc_new, Zmns_new)

def convert_to_different_settings(fluxsurface : FluxSurface, settings_new : FluxSurfaceSettings) -> FluxSurface:
    '''
    Convert FluxSurface to a different (mpol, ntor) representation.

    Note that this returns the same type as the input fluxsurface. However, if it is e.g. a FluxSurfaceFourierExtended, the extension data 
    is not converted or used, so the return type will be only the base FluxSurface.

    Parameters:
    -----------
    fluxsurface : FluxSurface
        The flux surface to convert.
    settings_new : FluxSurfaceSettings
        The new Fourier settings (mpol, ntor).

    Returns:
    --------
    fluxsurface_new : FluxSurface
        New flux surface with Fourier coefficients in the new (mpol, ntor) representation. Same as type as input fluxsurface.
    
    '''
    return type(fluxsurface)(data = _convert_fluxsurfacedata_to_different_settings(fluxsurface.data, settings_new, fluxsurface.settings), modes = FluxSurfaceModes.from_settings(settings_new), settings = settings_new)

@eqx.filter_jit
def _convert_to_equal_arclength_single(flux_surface : FluxSurface, n_theta : int, n_phi : int, n_theta_s_arclength : int) -> Tuple[FluxSurfaceData, FluxSurfaceSettings]:    
    '''
    Convert a single flux surface to a Fourier representation sampled on an equal arclength poloidal grid.
    This requires a full FluxSurface instead of only settings and data, as the position function is needed. This makes batching easier as well:
    the function can be vmapped over FluxSurface objects. (see convert_to_equal_arclength)

    Parameters:
    -----------
    flux_surface : FluxSurface
        Flux surface to convert.
    n_theta : int
        Number of poloidal modes in the Fourier representation [= n_theta // 2]
    n_phi : int
        Number of toroidal modes in the Fourier representation [= n_phi // 2]
    n_theta_s_arclength : int
        Number of poloidal points to use for the arclength sampling grid.
    Returns:
    --------
    flux_surface_data : FluxSurfaceData
        Fourier representation of the sampled flux surface.
    settings : FluxSurfaceSettings
        Settings of the Fourier representation (mpol, ntor, nfp).
    '''
    assert flux_surface.data.Rmnc.ndim == 1, "convert_to_equal_arclength only supports single surface conversion"

    theta_s              = jnp.linspace(0, 2 * jnp.pi,                             n_theta_s_arclength, endpoint=False)
    phi_s                = jnp.linspace(0, 2 * jnp.pi / flux_surface.settings.nfp, n_phi,              endpoint=False) 
    theta_mg_s, phi_mg_s = jnp.meshgrid(theta_s, phi_s, indexing='ij')    
    arc_lengths          = _arc_length_theta_direct(flux_surface, theta_mg_s, phi_mg_s) #[n_theta_s_arclength, n_phi]
    
    new_theta_mg         = jax.vmap(_resample_uniform_periodic_pchip, in_axes=(1, None), out_axes=1)(arc_lengths, n_theta) * 2 * jnp.pi 
    
    _, phi_mg            = jnp.meshgrid(jnp.zeros(new_theta_mg.shape[0]), phi_s, indexing='ij')  # [n_theta_sample_arclength, n_phi]

    RZphi_sampled        = _cylindrical_position_direct(flux_surface, new_theta_mg, phi_mg)      # [n_theta_sample_arclength, n_phi, 3]
    
    flux_surface_data    = _rz_to_vmec_representation(RZphi_sampled[..., 0], RZphi_sampled[..., 1])
    
    return flux_surface_data, FluxSurfaceSettings(*mpol_ntor_from_ntheta_nphi(n_theta, n_phi), flux_surface.settings.nfp)

@eqx.filter_jit
def convert_to_equal_arclength(flux_surface : FluxSurface, n_theta : int, n_phi : int, n_theta_s_arclength : int) -> Tuple[FluxSurfaceData, FluxSurfaceSettings]:   
    if flux_surface.data.Rmnc.ndim == 1:
        return _convert_to_equal_arclength_single(flux_surface, n_theta, n_phi, n_theta_s_arclength)
    else:
        flux_surface_data, _  = jax.vmap(_convert_to_equal_arclength_single, in_axes=(FluxSurface(FluxSurfaceData(0,0), FluxSurfaceModes(None, None), FluxSurfaceSettings(None, None, None)), None, None, None ))(flux_surface, n_theta, n_phi, n_theta_s_arclength)
        return flux_surface_data, FluxSurfaceSettings(*mpol_ntor_from_ntheta_nphi(n_theta, n_phi), flux_surface.settings.nfp)
    
def mpol_ntor_from_ntheta_nphi(n_theta : int, n_phi : int) -> Tuple[int,int]:    
    mpol = n_theta // 2
    ntor = n_phi   // 2
    return mpol, ntor


@eqx.filter_jit
def create_fourier_representation(flux_surface : FluxSurface, s : jnp.ndarray, theta_grid : jnp.ndarray) -> Tuple[FluxSurfaceData, FluxSurfaceSettings]:
    '''
    Create a Fourier representation of a flux surface at given (s, theta) grid points.

    Parameters:
    -----------
    flux_surface : FluxSurface
        Flux_Surface to create the Fourier representation of.
    s : jnp.ndarray [n_theta, n_phi] or float
        Radial coordinate(s) at which to sample the flux surface. If an array, must have the same shape as theta_grid.
    theta_grid : jnp.ndarray [n_theta, n_phi]
        Grid of poloidal angles at which to sample the flux surface.    
    Returns:
    --------
    flux_surface_data : FluxSurfaceData
        Fourier representation of the sampled flux surface.
    settings : FluxSurfaceSettings
        Settings of the Fourier representation (mpol, ntor, nfp).    
    '''
    # Static Checks
    assert theta_grid.ndim == 2, "theta_grid must be a 2D grid (n_theta, n_phi) but got shape {}".format(theta_grid.shape)    
    
    if jnp.array(s).ndim != 0:       
        assert s.shape == theta_grid.shape, "If s is an array, it must have the same shape as theta_grid but got s shape {} and theta_grid shape {}".format(s.shape, theta_grid.shape)
    
    # Static warnings
    if isinstance(flux_surface, FluxSurfaceNormalExtended):
        warn("FluxSurfaceNormalExtended does not have phi_in = phi_out. This introduces errors when Fourier transforming", UserWarning)
    
    if type(flux_surface) == FluxSurface:
        warn("FluxSurface base class does not extend beyond the LCFS. Any conversion with s>0.0 will reproduce the LCFS", UserWarning)    

    n_theta                  = theta_grid.shape[0]
    n_phi                    = theta_grid.shape[1]

    phi_grid                 = jnp.linspace(0, 2*jnp.pi / flux_surface.settings.nfp, n_phi, endpoint=False)    
    _, phi_mg                = jnp.meshgrid(jnp.linspace(0, 2*jnp.pi, n_theta, endpoint=False), phi_grid, indexing='ij')

    RZphi_sampled            = flux_surface.cylindrical_position(s, theta_grid, phi_mg)
    
    flux_surface_data        = _rz_to_vmec_representation(RZphi_sampled[..., 0], RZphi_sampled[..., 1])
    
    return flux_surface_data, FluxSurfaceSettings(*mpol_ntor_from_ntheta_nphi(n_theta, n_phi), flux_surface.settings.nfp)

@eqx.filter_jit
def _create_fourier_representation_d_interp_single(flux_surfaces : FluxSurface, d : jnp.ndarray, n_theta : int, n_phi : int):
    '''
    Create a Fourier representation of an extended flux surface with an interpolated extension distance. 
    

    Parameters:
    -----------
    flux_surfaces : FluxSurface
        Flux_Surface to extend using the distance function. Flux surface must be of type FluxSurfaceNormalExtendedNoPhi or FluxSurfaceNormalExtendedConstantPhi to ensure valid results (phi_in must be phi_out for FFT)
    d : jnp.ndarray [n_theta_sampled, n_phi_sampled] or float
        Distance function to extend the flux surface with. Assumed to be full module: i.e. phi in [0, 2pi/nfp], theta in [0, 2pi] (included endpoints)
    n_theta : int
        Number of poloidal points in the output Fourier representation.
    n_phi : int
        Number of toroidal points in the output Fourier representation.
    Returns:
    --------
    
    '''
    theta, phi                  = jnp.linspace(0, 2*jnp.pi, n_theta, endpoint=False), jnp.linspace(0, 2*jnp.pi/flux_surfaces.settings.nfp, n_phi, endpoint=False)
    theta_mg, phi_mg            = jnp.meshgrid(theta, phi, indexing='ij')
    s_interp                    = _interpolate_s_grid_full_mod(theta_mg, phi_mg, flux_surfaces.settings.nfp, jnp.atleast_2d(d) + 1.0)
    flux_surface_data, settings =  create_fourier_representation(flux_surfaces, s_interp, theta_mg)
    return flux_surface_data, settings


# ===================================================================================================================================================================================
#                                                                           Convenience functions
# ===================================================================================================================================================================================


_create_fourier_representation_d_interp_vmap = jax.vmap(_create_fourier_representation_d_interp_single, in_axes=(None, 0, None, None))


@eqx.filter_jit
def create_fourier_representation_d_interp(flux_surface : FluxSurface, d : jnp.ndarray, n_theta : int, n_phi : int):
    '''
    Create a Fourier representation of an extended flux surface with an interpolated extension distance. 
    Can be batched over d: if d is a scalar or 2D array, a single flux surface is created.
    If d is a 1D or 3D array, multiple flux surfaces are created (batched).

    Parameters:
    -----------
    flux_surface : FluxSurface
        Flux_Surface to extend using the distance function. Flux surface must be of type FluxSurfaceNormalExtendedNoPhi or FluxSurfaceNormalExtendedConstantPhi to ensure valid results (phi_in must be phi out for FFT)
    d : jnp.ndarray 
        Distance function to extend the flux surface with. Assumed to be full module: i.e. phi in [0, 2pi/nfp], theta in [0, 2pi] (included endpoints)  
        If d is a scalar or 2D array, a single flux surface is created.
        If d is a 1D or 3D array, multiple flux surfaces are created (batched).
    n_theta : int
        Number of poloidal points in the output Fourier representation.
    n_phi : int
        Number of toroidal points in the output Fourier representation.
    Returns:
    --------
    flux_surface_data : FluxSurfaceData
        Fourier representation of the sampled flux surface.
    settings : FluxSurfaceSettings
        Settings of the Fourier representation (mpol, ntor, nfp).

    '''
    d = jnp.array(d)
    new_settings = FluxSurfaceSettings(*mpol_ntor_from_ntheta_nphi(n_theta, n_phi), flux_surface.settings.nfp)
    if d.ndim == 0 or d.ndim == 2:
        flux_surface_data, _ =  _create_fourier_representation_d_interp_single(flux_surface, d, n_theta, n_phi)
    elif d.ndim == 1 or d.ndim == 3:
        flux_surface_data, _ = _create_fourier_representation_d_interp_vmap(flux_surface, d, n_theta, n_phi)
    else:
        raise ValueError("d must be a scalar or 2D array but got shape {}".format(d.shape))
    return flux_surface_data, new_settings


def create_flux_surface_d_interp(flux_surface : FluxSurface, d : jnp.ndarray, n_theta : int, n_phi : int, type_c : Type =  FluxSurface) -> FluxSurface:         
    '''
    Convenience function of create_fourier_representation_d_interp + type_c.from_data_settings_full, returning a FluxSurface of given type.

    Parameters:
    -----------
    flux_surface : FluxSurface
        Flux_Surface to extend using the distance function. Flux surface must be of type FluxSurfaceNormal
    d : jnp.ndarray 
        Distance function to extend the flux surface with. Assumed to be full module: i.e. phi in [0, 2pi/nfp], theta in [0, 2pi] (included endpoints)  
    n_theta : int
        Number of poloidal points in the output Fourier representation.
    n_phi : int
        Number of toroidal points in the output Fourier representation.
    Returns:
    --------
    flux_surface : FluxSurface
        Flux surface with Fourier representation.   
    '''
    return type_c.from_data_settings_full(*create_fourier_representation_d_interp(flux_surface, d, n_theta, n_phi))

def create_extended_flux_surface_d_interp(flux_surface : FluxSurface, d : jnp.ndarray, n_theta : int, n_phi : int):
    '''
    Creates a FluxSurfaceFourierExtended by extending a given flux surface using a distance function d and interpolating the distance function.

    Convenience function of create_fourier_representation_d_interp + FluxSurface.from_data_settings_full + FluxSurfaceFourierExtended.from_flux_surface_and_extension, returning a FluxSurfaceFourierExtended.  

    Compared to create_flux_surface_d_interp, this function directly returns a FluxSurfaceFourierExtended.

    Parameters:
    -----------
    flux_surface : FluxSurface
        Flux_Surface to extend using the distance function. Flux surface must be of type FluxSurfaceNormal
    d : jnp.ndarray 
        Distance function to extend the flux surface with. Assumed to be full module: i.e. phi in [0, 2pi/nfp], theta in [0, 2pi] (included endpoints)  
    n_theta : int
        Number of poloidal points in the output Fourier representation.
    n_phi : int
        Number of toroidal points in the output Fourier representation.
    Returns:
    --------
    flux_surface_extended : FluxSurfaceFourierExtended
        Extended flux surface with Fourier representation.
    '''
    return FluxSurfaceFourierExtended.from_flux_surface_and_extension(FluxSurface(data = flux_surface.data, modes = flux_surface.modes, settings = flux_surface.settings), create_flux_surface_d_interp(flux_surface, d, n_theta, n_phi, type_c=FluxSurface))

@eqx.filter_jit
def create_fourier_representation_d_interp_equal_arclength(flux_surface : FluxSurface, d : jnp.ndarray, n_theta : int, n_phi : int, n_theta_s_arclength : int):    
    '''
    Convenience function of create_fourier_representation_d_interp + convert_to_equal_arclength

    Parameters:
    -----------
    flux_surface : FluxSurface
        Flux_Surface to extend using the distance function. Flux surface must be of type FluxSurfaceNormalExtendedNoPhi or FluxSurfaceNormalExtendedConstantPhi to ensure valid results (phi_in must be phi_out for FFT)
    d : jnp.ndarray 
        Distance function to extend the flux surface with. Assumed to be full module: i.e. phi in [0, 2pi/nfp], theta in [0, 2pi] (included endpoints)  
    n_theta : int
        Number of poloidal points in the output Fourier representation.
    n_phi : int
        Number of toroidal points in the output Fourier representation.
    n_theta_s_arclength : int
        Number of poloidal points to use for the arclength parametrization.
    Returns:
    --------
    flux_surface_data : FluxSurfaceData
        Fourier representation of the sampled flux surface.
    settings : FluxSurfaceSettings
        Settings of the Fourier representation (mpol, ntor, nfp).
    '''        
    return convert_to_equal_arclength(FluxSurface.from_data_settings(*create_fourier_representation_d_interp(flux_surface, d, n_theta, n_phi)), n_theta, n_phi, n_theta_s_arclength)

def create_flux_surface_d_interp_equal_arclength(flux_surface : FluxSurface, d : jnp.ndarray, n_theta : int, n_phi : int, n_theta_s_arclength : int, type_c : Type =  FluxSurface):        
    '''
    Convenience function of create_fourier_representation_d_interp + convert_to_equal_arclength + type_c.from_data_settings_full, returning a FluxSurface of given type.

    Parameters:
    -----------
    flux_surface : FluxSurface
        Flux_Surface to extend using the distance function. Flux surface must be of type FluxSurfaceNormal
    d : jnp.ndarray 
        Distance function to extend the flux surface with. Assumed to be full module: i.e. phi in [0, 2pi/nfp], theta in [0, 2pi] (included endpoints)  
    n_theta : int
        Number of poloidal points in the output Fourier representation.
    n_phi : int
        Number of toroidal points in the output Fourier representation.
    n_theta_s_arclength : int
        Number of poloidal points to use for the arclength parametrization.
    Returns:
    --------
    flux_surface : FluxSurface
        Flux surface with Fourier representation sampled on an equal arclength poloidal grid.
    '''
    return type_c.from_data_settings_full(*create_fourier_representation_d_interp_equal_arclength(flux_surface, d, n_theta, n_phi, n_theta_s_arclength))

def create_extended_flux_surface_d_interp_equal_arclength(flux_surface : FluxSurface, d : jnp.ndarray, n_theta : int, n_phi : int, n_theta_s_arclength : int):    
    '''
    Creates a FluxSurfaceFourierExtended by extending a given flux surface using a distance function d, interpolating the distance function, and sampling on an equal arclength poloidal grid.

    Convenience function of create_fourier_representation_d_interp + convert_to_equal_arclength + FluxSurface.from_data_settings_full + FluxSurfaceFourierExtended.from_flux_surface_and_extension, returning a FluxSurfaceFourierExtended.

    Compared to create_flux_surface_d_interp_equal_arclength, this function directly returns a FluxSurfaceFourierExtended.

    Parameters:
    -----------
    flux_surface : FluxSurface
        Flux_Surface to extend using the distance function. Flux surface must be of type FluxSurfaceNormalExtendedNoPhi or FluxSurfaceNormalExtendedConstantPhi to ensure valid results (phi_in must be phi_out for FFT)
    d : jnp.ndarray 
        Distance function to extend the flux surface with. Assumed to be full module: i.e. phi in [0, 2pi/nfp], theta in [0, 2pi] (included endpoints)  
        If d is a scalar or 2D array, a single flux surface is created.
        If d is a 1D or 3D array, multiple flux surfaces are created (batched).
    n_theta : int
        Number of poloidal points in the output Fourier representation.
    n_phi : int
        Number of toroidal points in the output Fourier representation.
    n_theta_s_arclength : int
        Number of poloidal points to use for the arclength parametrization.
    Returns:
    --------
    flux_surface_extended : FluxSurfaceFourierExtended
        Extended flux surface with Fourier representation sampled on an equal arclength poloidal grid.

    '''
    return FluxSurfaceFourierExtended.from_flux_surface_and_extension(FluxSurface(data = flux_surface.data, modes = flux_surface.modes, settings = flux_surface.settings), create_flux_surface_d_interp_equal_arclength(flux_surface, d, n_theta, n_phi, n_theta_s_arclength, type_c=FluxSurface))
