import jax 
import jax.numpy as jnp
from functools import partial
from .flux_surfaces_base import _create_mpol_vector, _create_ntor_vector, FluxSurface

from .flux_surfaces_extended import FluxSurfaceNormalExtended, FluxSurfaceNormalExtendedNoPhi, FluxSurfaceNormalExtendedConstantPhi

from warnings import warn

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

@partial(jax.jit, static_argnums = (1,2,3,4))
def _convert_to_different_ntor_mpol(array : jnp.ndarray, mpol_new : int, ntor_new : int, mpol_old : int, ntor_old : int):
    mpol_vector_new = _create_mpol_vector(mpol_new, ntor_new)
    ntor_vector_new = _create_ntor_vector(mpol_new, ntor_new, 1)  # symm not necessary here    
   
    data_available  = jnp.logical_and(mpol_vector_new <= mpol_old, jnp.abs(ntor_vector_new) <= ntor_old)
    
    # we ensure we don't go out of bounds here by setting indices to 0 when data is not available
    # jnp.where *will* access both branches before selecting. We need to set the out-of-bounds indices to a safe value and then 
    # select 0.0 in the following where:
    index_mn_new    = jnp.where(data_available, _index_mn(mpol_vector_new, ntor_vector_new, ntor_old), 0)
    array_new       = jnp.where(data_available, array[..., index_mn_new], 0.0)    
        
    return array_new

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
    

    n_theta = theta_grid.shape[0]
    n_phi   = theta_grid.shape[1]

    phi_grid                 = jnp.linspace(0, 2*jnp.pi / flux_surface.settings.nfp, n_phi, endpoint=False)    
    _, phi_mg                = jnp.meshgrid(jnp.linspace(0, 2*jnp.pi, n_theta, endpoint=False), phi_grid, indexing='ij')

    RZphi_sampled            = flux_surface.cylindrical_position(s, theta_grid, phi_mg)
    R_grid                   = RZphi_sampled[..., 0]
    Z_grid                   = RZphi_sampled[..., 1]

    Rmnc, Zmns               = _rz_to_vmec_representation(R_grid, Z_grid)

    mpol, ntor               = mpol_ntor_from_ntheta_nphi(n_theta, n_phi)
    return Rmnc, Zmns, mpol, ntor

