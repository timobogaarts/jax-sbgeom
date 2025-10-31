import jax 
import jax.numpy as jnp
from functools import partial
from .flux_surfaces_base import _create_mpol_vector, _create_ntor_vector, FluxSurface

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
    mpol = xckl.shape[0] - 1 # mpol needs to be 1 higher than maximum because vmec
    ntor = xckl.shape[1] - 1

    ntor_vector = _create_ntor_vector(ntor, mpol, 1)  # symm not necessary here
    mpol_vector = _create_mpol_vector(ntor, mpol)

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

@partial(jax.jit, static_argnums = (2,3))
def _create_sampling_rz(flux_surface, s, n_theta : int, n_phi : int):
    '''
    Create a grid of R,Z points on a flux surface at normalized radius s, compatible with Fourier transformation

    Parameters
    ----------
    flux_surface : FluxSurface
        The flux surface object
    s : float
        The normalized radius of the flux surface 
    n_theta : int
        The number of poloidal angles to sample
    n_phi : int
        The number of toroidal angles to sample
    Returns
    -------
    R : jnp.ndarray
        The R coordinates of the grid points
    Z : jnp.ndarray
        The Z coordinates of the grid points
    
    '''
    theta = jnp.linspace(0, 2*jnp.pi, n_theta, endpoint=False)
    phi   = jnp.linspace(0, 2*jnp.pi / flux_surface.settings.nfp, n_phi, endpoint=False)
    theta_mg, phi_mg = jnp.meshgrid(theta, phi, indexing='ij')

    positions_jax = flux_surface.cylindrical_position(s, theta_mg, phi_mg)

    return positions_jax[...,0], positions_jax[...,1]

def _index_mn(m,n, ntor):    
    return n + (m>0) * (2 * ntor + 1) * m  

def _size_mn(mpol, ntor):
    return (2 * ntor + 1) * mpol + ntor +1

@partial(jax.jit, static_argnums = (1,2,3,4))
def _convert_to_different_ntor_mpol(array : jnp.ndarray, mpol_new : int, ntor_new : int, mpol_old : int, ntor_old : int):
    ntor_vector_new = _create_ntor_vector(ntor_new, mpol_new, 1)  # symm not necessary here
    mpol_vector_new = _create_mpol_vector(ntor_new, mpol_new)
   
    data_available  = jnp.logical_and(mpol_vector_new <= mpol_old, jnp.abs(ntor_vector_new) <= ntor_old)
    
    # we ensure we don't go out of bounds here by setting indices to 0 when data is not available
    # jnp.where *will* access both branches before selecting. We need to set the out-of-bounds indices to a safe value and then 
    # select 0.0 in the following where:
    index_mn_new    = jnp.where(data_available, _index_mn(mpol_vector_new, ntor_vector_new, ntor_old), 0)
    array_new       = jnp.where(data_available, array[..., index_mn_new], 0.0)    
        
    return array_new


