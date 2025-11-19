import jax 
import jax.numpy as jnp
from functools import partial
from .flux_surfaces_base import _create_mpol_vector, _create_ntor_vector, FluxSurface, FluxSurfaceData, FluxSurfaceSettings, _interpolate_s_grid_full_mod, _arc_length_theta_interpolating_s_grid_full_mod, _arc_length_theta_interpolating_s_grid_full_mod_finite_difference
from .flux_surfaces_base import _arc_length_theta_direct, _cylindrical_position_direct
from .flux_surfaces_extended import FluxSurfaceNormalExtended, FluxSurfaceNormalExtendedNoPhi, FluxSurfaceNormalExtendedConstantPhi, FluxSurfaceFourierExtended
from jax_sbgeom.jax_utils.utils import bilinear_interp, _resample_uniform_periodic_pchip, _resample_uniform_periodic_linear
from warnings import warn
from typing import Type
@jax.jit
def _dft_forward(points : jnp.ndarray):
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
def _cos_sin_from_dft_forward(dft_coefficients):
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
def _convert_cos_sin_to_vmec(xckl, xcmkl, xskl, xsmkl, cosine : bool):
    mpol = xckl.shape[0] - 1
    ntor = xckl.shape[1] - 1

    mpol_vector = _create_mpol_vector(mpol,ntor)
    ntor_vector = _create_ntor_vector(mpol,ntor, 1)  # symm not necessary here    

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
def _rz_to_vmec_representation(R_grid, Z_grid):
    assert R_grid.shape == Z_grid.shape, "R and Z grids must have the same shape but got {} and {}".format(R_grid.shape, Z_grid.shape)
    R_dft = _dft_forward(R_grid)
    Z_dft = _dft_forward(Z_grid)
    R_ckl, R_cmkl, R_skl, R_smkl = _cos_sin_from_dft_forward(R_dft)
    Z_ckl, Z_cmkl, Z_skl, Z_smkl = _cos_sin_from_dft_forward(Z_dft)
    R_vmec = _convert_cos_sin_to_vmec(R_ckl, R_cmkl, R_skl, R_smkl, cosine=True)
    Z_vmec = _convert_cos_sin_to_vmec(Z_ckl, Z_cmkl, Z_skl, Z_smkl, cosine=False)
    return R_vmec, Z_vmec

def _index_mn(m,n, ntor):    
    return n + (m>0) * (2 * ntor + 1) * m  

def _size_mn(mpol, ntor):
    return (2 * ntor + 1) * mpol + ntor +1

@partial(jax.jit, static_argnums = (1,2))
def _convert_to_different_mpol_ntor(array : jnp.ndarray, mpol_new : int, ntor_new : int, mpol_old : int, ntor_old : int):
    mpol_vector_new = _create_mpol_vector(mpol_new, ntor_new)
    ntor_vector_new = _create_ntor_vector(mpol_new, ntor_new, 1)  # symm not necessary here    
   
    data_available  = jnp.logical_and(mpol_vector_new <= mpol_old, jnp.abs(ntor_vector_new) <= ntor_old)
    
    # we ensure we don't go out of bounds here by setting indices to 0 when data is not available
    # jnp.where *will* access both branches before selecting. We need to set the out-of-bounds indices to a safe value and then 
    # select 0.0 in the following where:
    index_mn_new    = jnp.where(data_available, _index_mn(mpol_vector_new, ntor_vector_new, ntor_old), 0)
    array_new       = jnp.where(data_available, array[..., index_mn_new], 0.0)    
        
    return array_new

@partial(jax.jit, static_argnums = (4, 5, 6))
def convert_to_equal_arclength(Rmnc : jnp.ndarray, Zmns : jnp.ndarray, mpol_vector : jnp.ndarray, ntor_vector : jnp.ndarray, n_theta : int, n_phi : int, n_theta_s_arclength : int) -> FluxSurfaceData:    
    theta_s              = jnp.linspace(0, 2 * jnp.pi,       n_theta_s_arclength, endpoint=False)
    phi_s                = jnp.linspace(0, 2 * jnp.pi / ntor_vector[1], n_phi,   endpoint=False) # ntor_vector[1] is nfp
    theta_mg_s, phi_mg_s = jnp.meshgrid(theta_s, phi_s, indexing='ij')    
    arc_lengths          = _arc_length_theta_direct(Rmnc, Zmns, mpol_vector, ntor_vector, theta_mg_s, phi_mg_s) #[n_theta_s_arclength, n_phi]
    
    new_theta_mg         = jax.vmap(_resample_uniform_periodic_pchip, in_axes=(1, None), out_axes=1)(arc_lengths, n_theta) * 2 * jnp.pi # [n_theta_sample_arclength, n_phi]                   
    _, phi_mg            = jnp.meshgrid(jnp.zeros(new_theta_mg.shape[0]), phi_s, indexing='ij')  # [n_theta_sample_arclength, n_phi]

    RZphi_sampled        = _cylindrical_position_direct(Rmnc, Zmns, mpol_vector, ntor_vector, new_theta_mg, phi_mg)  # [n_theta_sample_arclength, n_phi, 3]
    R_grid               = RZphi_sampled[..., 0]
    Z_grid               = RZphi_sampled[..., 1]
    Rmnc, Zmns           = _rz_to_vmec_representation(R_grid, Z_grid)
    mpol_new, ntor_new   = mpol_ntor_from_ntheta_nphi(n_theta, n_phi)    
    return Rmnc, Zmns, mpol_new, ntor_new

def mpol_ntor_from_ntheta_nphi(n_theta : int, n_phi : int):    
    mpol = n_theta // 2
    ntor = n_phi   // 2
    return mpol, ntor

def create_fourier_representation(flux_surface : FluxSurface, s : jnp.ndarray, theta_grid : jnp.ndarray):
    # Static Checks
    assert theta_grid.ndim == 2, "theta_grid must be a 2D grid (n_theta, n_phi) but got shape {}".format(theta_grid.shape)    
    
    if jnp.array(s).ndim != 0:       
        assert s.shape == theta_grid.shape
    
    if isinstance(flux_surface, FluxSurfaceNormalExtended):
        warn("FluxSurfaceNormalExtended does not have phi_in = phi_out. This introduces errors when Fourier transforming", UserWarning)
    
    if type(flux_surface) == FluxSurface:
        warn("FluxSurface base class does not extend beyond the LCFS. Any conversion with s>0.0 will reproduce the LCFS", UserWarning)
    

    n_theta                  = theta_grid.shape[0]
    n_phi                    = theta_grid.shape[1]

    phi_grid                 = jnp.linspace(0, 2*jnp.pi / flux_surface.settings.nfp, n_phi, endpoint=False)    
    _, phi_mg                = jnp.meshgrid(jnp.linspace(0, 2*jnp.pi, n_theta, endpoint=False), phi_grid, indexing='ij')

    RZphi_sampled            = flux_surface.cylindrical_position(s, theta_grid, phi_mg)
    R_grid                   = RZphi_sampled[..., 0]
    Z_grid                   = RZphi_sampled[..., 1]

    Rmnc, Zmns               = _rz_to_vmec_representation(R_grid, Z_grid)

    mpol, ntor               = mpol_ntor_from_ntheta_nphi(n_theta, n_phi)
    return Rmnc, Zmns, mpol, ntor

@partial(jax.jit, static_argnums = (2,3))
def create_fourier_surface_extension_interp(flux_surfaces : FluxSurface, d : jnp.ndarray, n_theta : int, n_phi : int):
    '''
    Create a Fourier representation of a no-phi extended flux surface with an interpolated extension distance. 

    Parameters:
    -----------
    flux_surfaces : FluxSurface
        Flux_Surface to extend using the distance function. Flux surface must be of type FluxSurfaceNormalExtendedNoPhi or FluxSurfaceNormalExtendedConstantPhi to ensure valid results (phi_in must be phi_out for FFT)
    d : jnp.ndarray [n_theta_sampled, n_phi_sampled]
        Distance function to extend the flux surface with. Assumed to be full module: i.e. phi in [0, 2pi/nfp], theta in [0, 2pi] (included endpoints)
    n_theta : int
        Number of poloidal points in the output Fourier representation.
    n_phi : int
        Number of toroidal points in the output Fourier representation.
    Returns:
    --------
    Rmnc : jnp.ndarray
        Fourier coefficients of R in VMEC representation.
    Zmns : jnp.ndarray
        Fourier coefficients of Z in VMEC representation.
    mpol : int
        Number of poloidal modes in the Fourier representation [= n_theta // 2]
    ntor : int
        Number of toroidal modes in the Fourier representation [= n_phi // 2]
    nfp : int
        Number of field periods of the flux surface [same as input flux surface]
    '''
    theta, phi             = jnp.linspace(0, 2*jnp.pi, n_theta, endpoint=False), jnp.linspace(0, 2*jnp.pi/flux_surfaces.settings.nfp, n_phi, endpoint=False)
    theta_mg, phi_mg       = jnp.meshgrid(theta, phi, indexing='ij')

    s_interp               = _interpolate_s_grid_full_mod(theta_mg, phi_mg, flux_surfaces.settings.nfp, jnp.atleast_2d(d) + 1.0)
    Rmnc, Zmns, mpol, ntor = create_fourier_representation(flux_surfaces, s_interp, theta_mg)
    return Rmnc, Zmns, mpol, ntor, flux_surfaces.settings.nfp

@partial(jax.jit, static_argnums = (2,3,4))
def create_fourier_surface_extension_interp_equal_arclength(flux_surface : FluxSurface, d : jnp.ndarray, n_theta : int, n_phi : int, n_theta_s_arclength : int):    
    '''
    Create a Fourier representation of a no-phi extended flux surface with an interpolated extension distance and resampled to equal arclength.

    For a version which supports a batched d (which have the same shapes), use create_multiple_fourier_surface_extension_interp

    Parameters:
    -----------
    flux_surface : FluxSurface
        Flux_Surface to extend using the distance function. Flux surface must be of type FluxSurfaceNormalExtendedNoPhi or FluxSurfaceNormalExtendedConstantPhi to ensure valid results (phi_in must be phi out for FFT)
    d : jnp.ndarray [n_theta_sampled, n_phi_sampled]
        Distance function to extend the flux surface with. Assumed to be full module: i.e. phi in [0, 2pi/nfp], theta in [0, 2pi] (included endpoints)
    n_theta : int
        Number of poloidal points in the output Fourier representation.
    n_phi : int
        Number of toroidal points in the output Fourier representation.
    n_theta_s_arclength : int
        Number of poloidal points to use for the equal arclength resampling.
    Returns:
    --------
    Rmnc : jnp.ndarray
        Fourier coefficients of R in VMEC representation.
    Zmns : jnp.ndarray
        Fourier coefficients of Z in VMEC representation.
    mpol : int
        Number of poloidal modes in the Fourier representation [= n_theta // 2]
    ntor : int
        Number of toroidal modes in the Fourier representation [= n_phi // 2]
    nfp : int
        Number of field periods of the flux surface [same as input flux surface]
    '''
    mpol, ntor = mpol_ntor_from_ntheta_nphi(n_theta, n_phi)
    mpol_vector,  ntor_vector = _create_mpol_vector(mpol, ntor), _create_ntor_vector(mpol, ntor, 1) * flux_surface.settings.nfp
    Rmnc, Zmns, _, _, _ = create_fourier_surface_extension_interp(flux_surface, d, n_theta, n_phi)
    return *convert_to_equal_arclength(Rmnc, Zmns, mpol_vector, ntor_vector, n_theta, n_phi, n_theta_s_arclength), flux_surface.settings.nfp

@partial(jax.jit, static_argnums = (2,3,4))
def create_multiple_fourier_surface_extension_interp_equal_arclength(flux_surface : FluxSurface, d : jnp.ndarray, n_theta : int, n_phi : int, n_theta_s_arclength : int):
    '''
    Create a Fourier representation of multiple no-phi extended flux surfaces with an interpolated extension distance and resampled to equal arclength.
    Version which supports a batched d (which have the same shapes).

    Parameters:
    -----------
    flux_surface : FluxSurface
        Flux_Surface to extend using the distance function. Flux surface must be of type FluxSurfaceNormalExtendedNoPhi or FluxSurfaceNormalExtendedConstantPhi to ensure valid results (phi_in must be phi out for FFT)
    d : jnp.ndarray [n_theta_sampled, n_phi_sampled]
        Distance function to extend the flux surface with. Assumed to be full module: i.e. phi in [0, 2pi/nfp], theta in [0, 2pi] (included endpoints)
    n_theta : int
        Number of poloidal points in the output Fourier representation.
    n_phi : int
        Number of toroidal points in the output Fourier representation.
    n_theta_s_arclength : int
        Number of poloidal points to use for the equal arclength resampling.
    Returns:
    --------
    Rmnc : jnp.ndarray
        Fourier coefficients of R in VMEC representation.
    Zmns : jnp.ndarray
        Fourier coefficients of Z in VMEC representation.
    mpol : int
        Number of poloidal modes in the Fourier representation [= n_theta // 2]
    ntor : int
        Number of toroidal modes in the Fourier representation [= n_phi // 2]
    nfp : int
        Number of field periods of the flux surface [same as input flux surface]    
    '''
    Rmnc, Zmns, _, _, _ = jax.vmap(create_fourier_surface_extension_interp_equal_arclength, in_axes=(None, 0, None, None, None), out_axes = (0, 0, None, None, None))(flux_surface, d, n_theta, n_phi, n_theta_s_arclength)
    return Rmnc, Zmns, *mpol_ntor_from_ntheta_nphi(n_theta, n_phi), flux_surface.settings.nfp

@partial(jax.jit, static_argnums = (5,6))
def convert_to_different_mpol_ntor(Rmnc : jnp.ndarray, Zmns : jnp.ndarray, mpol_old : int, ntor_old : int, nfp : int, mpol_new : int, ntor_new : int):
    '''
    Convert Fourier coefficients to a different (mpol, ntor) representation.

    Parameters:
    -----------
    Rmnc : jnp.ndarray
        Fourier coefficients of R in VMEC representation.
    Zmns : jnp.ndarray
        Fourier coefficients of Z in VMEC representation.   
    mpol_old : int
        Original number of poloidal modes in the Fourier representation.
    ntor_old : int
        Original number of toroidal modes in the Fourier representation.
    nfp : int
        Number of field periods of the flux surface.
    mpol_new : int
        New number of poloidal modes in the Fourier representation.
    ntor_new : int
        New number of toroidal modes in the Fourier representation.
    Returns:
    --------
    Rmnc_new : jnp.ndarray
        Fourier coefficients of R in VMEC representation with new (mpol, ntor).
    Zmns_new : jnp.ndarray  
        Fourier coefficients of Z in VMEC representation with new (mpol, ntor).
    mpol_new : int
        New number of poloidal modes in the Fourier representation.
    ntor_new : int
        New number of toroidal modes in the Fourier representation.
    nfp : int
        Number of field periods of the flux surface.
    '''
    Rmnc_new = _convert_to_different_mpol_ntor(Rmnc, mpol_new, ntor_new, mpol_old, ntor_old)
    Zmns_new = _convert_to_different_mpol_ntor(Zmns, mpol_new, ntor_new, mpol_old, ntor_old)
    return Rmnc_new, Zmns_new, mpol_new, ntor_new, nfp

def _create_fluxsurface_from_rmnc_zmns(rmnc : jnp.ndarray, zmns : jnp.ndarray, mpol : int, ntor : int, nfp : int, type : Type =  FluxSurface):
    # we cannot jit this since it involves creation of objects.
    assert rmnc.shape == zmns.shape, "Rmnc and Zmns must have the same shape but got {} and {}".format(rmnc.shape, zmns.shape)
    if rmnc.ndim == 1 :
        Rmnc_ext = rmnc[None,:]
        Zmns_ext = zmns[None,:]
    else:
        Rmnc_ext = rmnc
        Zmns_ext = zmns
    
    settings = FluxSurfaceSettings(mpol, ntor, nfp, Rmnc_ext.shape[0])
    fs_jax   = type(FluxSurfaceData.from_rmnc_zmns_settings(Rmnc_ext, Zmns_ext, settings), settings)
    return fs_jax
