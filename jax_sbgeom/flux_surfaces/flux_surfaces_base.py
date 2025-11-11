import jax.numpy as jnp
import h5py 
import jax
import numpy as onp
from dataclasses import dataclass
from jax_sbgeom.jax_utils.utils import stack_jacfwd, interpolate_array
from functools import partial

def _create_mpol_vector(mpol : int, ntor : int):    
    '''
    Create the poloidal mode number vector for VMEC representation.

    Uses [0] * ntor + 1, [1] * (2 * ntor + 1), [2] * (2 * ntor + 1), ..., [mpol] * (2 * ntor + 1) 
    First is because for zero poloidal mode, there is no difference between positive and negative toroidal modes
    They can be combined into a single coefficient. Since the zero mode also needs representation, there are ntor + 1 entries for m = 0.
    For m >0, there are 2 * ntor + 1 entries, since both positive and negative toroidal modes need representation and the zero mode.

    Parameters
    ----------
    mpol : int
        Maximum poloidal mode number.
    ntor : int
        Maximum toroidal mode number.    
    Returns
    -------
    jnp.ndarray
        The poloidal mode number vector.
    '''
    return jnp.array([0 for i in range(ntor + 1)] + sum([[i for j in range(2 * ntor + 1)]for i in range(1, mpol +1 )], []), dtype=int)
def _create_ntor_vector(mpol : int, ntor : int, symm : int):
    '''
    Create the toroidal mode number vector for VMEC representation.

    Uses [0, 1, 2, ..., ntor], [-ntor, ..., -1, 0, 1, ..., ntor], ..., [-ntor, ..., -1, 0, 1, ..., ntor] for m = 0, 1, ..., mpol
    Multiplied by symmetry factor symm.

    Parameters
    ----------
    mpol : int
        Maximum poloidal mode number.
    ntor : int
        Maximum toroidal mode number.    
    symm : int
        The symmetry factor (number of field periods)
    Returns
    -------
    jnp.ndarray
        The toroidal mode number vector.

    '''

    return jnp.array(list(range(0, (ntor + 1) * symm , symm)) + sum([list(range(-ntor * symm, (ntor + 1) * symm, symm)) for i in range(mpol)], []), dtype=int)


def _cylindrical_to_cartesian(RZphi : jnp.ndarray):
    R = RZphi[..., 0]
    Z = RZphi[..., 1]
    phi = RZphi[..., 2]
    x = R * jnp.cos(phi)
    y = R * jnp.sin(phi)
    return jnp.stack([x, y, Z], axis=-1)

def _cartesian_to_cylindrical(XYZ : jnp.ndarray):
    x = XYZ[..., 0]
    y = XYZ[..., 1]
    z = XYZ[..., 2]
    R = jnp.sqrt(x**2 + y**2)
    phi = jnp.arctan2(y, x)
    return jnp.stack([R, z, phi], axis=-1)


@jax.jit
def _check_whether_make_normals_point_outwards_required(Rmnc : jnp.ndarray, Zmns : jnp.ndarray, mpol_vector : jnp.ndarray):
    '''
    * Internal * 
    Check whether the Fourier coefficients need to be modified such that the normals point outwards.

    This corresponds to four cases:
    1. theta = 0 is outboard: 
        a. dZ_dtheta > 0 at theta = 0 -> normals point outwards -> no change
        b. dZ_dtheta < 0 at theta = 0 -> normals point inwards -> reverse theta
    2. theta = pi is outboard:
        a. dZ_dtheta > 0 at theta = 0 -> normals point inwards -> reverse theta
        b. dZ_dtheta < 0 at theta = 0 -> normals point outwards -> no change

    Parameters:
    -----------
    Rmnc : jnp.ndarray
        Array of radial Fourier coefficients. Shape (nsurf, nmodes)
    Zmns : jnp.ndarray
        Array of vertical Fourier coefficients. Shape (nsurf, nmodes)
    mpol_vector : jnp.ndarray
        Array of poloidal mode numbers. Shape (nmodes,)
    
    Returns:
    --------
    flip_theta : bool
        Whether to reverse theta to ensure normals point outwards.
    '''
    # This computes:
    # dZ_dtheta on the lcfs at  theta, phi = 0:
    # dZ_dtheta = jnp.sum(Zmns[-1,:] * mpol_vector * jnp.cos(mpol_vector * 0 - ntor_vector * 0)) = jnp.sum(Zmns[-1,:] * mpol_vector)        
    sum_Zmns = jnp.sum(Zmns[-1,:] * mpol_vector)
    
    # Rmnc at theta, phi = 0
    # R = jnp.sum(Rmnc[-1,:] * jnp.cos(mpol_vector * 0 - ntor_vector * 0)) = jnp.sum(Rmnc[-1,:])
    # Rmnc at theta, phi = pi, 0:
    # R = jnp.sum(Rmnc[-1,:] * jnp.cos(mpol_vector * jnp.pi - ntor_vector * 0)) = jnp.sum(Rmnc[-1,:] * (-1)**mpol_vector)
    # We want to determine whether dZ_dtheta points in the positive or negative Z direction at the outboard midplane.
    # This is accomplished by checking the sign of dZ_dtheta at the outboard midplane.
    # The outboard midplane is at theta = 0 if sum_Rmnc > 0 and at theta = pi if sum_Rmnc < 0
    cond_outboard = jnp.sum(Rmnc[-1,:]) > jnp.sum(Rmnc[-1,:] * (-1)**mpol_vector)

    
    original_u = jnp.where(cond_outboard, 
                       jnp.where(sum_Zmns > 0, True, False),   # cond_outboard == True branch
                       jnp.where(sum_Zmns > 0, False, True)    # cond_outboard == False branch
    )
    return jnp.logical_not(original_u)

def _reverse_theta_single(m_vec, n_vec, coeff_vec, cosine_sign : bool):
    '''
    * Internal * 

    Changes the Fourier coefficients such that theta is replaced by -theta.

    Parameters:
    -----------
    m_vec : jnp.ndarray
        Array of poloidal mode numbers.
    n_vec : jnp.ndarray
        Array of toroidal mode numbers.
    coeff_vec : jnp.ndarray
        Array of Fourier coefficients. (Rmnc or Zmns)
    cosine_sign : bool
        If True, the coefficients correspond to cosine terms. If False, they correspond to sine terms.
    Returns:
    --------
    new_coeff_vec : jnp.ndarray
        The modified Fourier coefficients after reversing theta.
    '''
    assert coeff_vec.ndim == 1
    assert m_vec.shape == n_vec.shape == coeff_vec.shape
    
    keys = jnp.stack([m_vec, n_vec], axis=1)
    
    reversed_keys = jnp.stack([m_vec, -n_vec], axis=1)  # target keys
    
    reversed_keys_mod = jnp.where(keys[:,0:1] > 0, reversed_keys, keys)
        
    matches = jnp.all(keys[:, None, :] == reversed_keys_mod[None, :, :], axis=-1)

#    assert jnp.all(jnp.any(matches, axis=0))

    idx_map = jnp.argmax(matches, axis=0)
    
    # Build new coefficient map
    # For cosine, we just swap the numbers
    # For sine, we swap and change sign, except for the m=0 terms. These stay the same
    new_coeff_vec = jax.lax.cond(cosine_sign, lambda _ : coeff_vec[idx_map], lambda _ :  jnp.where(keys[:,0] > 0, -coeff_vec[idx_map], coeff_vec[idx_map]), operand=None)
    
    return new_coeff_vec

reverse_theta_total  = jax.jit(jax.vmap(_reverse_theta_single, in_axes=(None, None, 0, None), out_axes=0))

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FluxSurfaceSettings:
    mpol : int     # maximum poloidal mode number [inclusive]
    ntor : int     # maximum toroidal mode number [inclusive]
    nfp  : int     # number of field periods
    nsurf : int    # number of flux surfaces

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FluxSurfaceData:
    Rmnc : jnp.ndarray
    Zmns : jnp.ndarray
    mpol_vector : jnp.ndarray
    ntor_vector : jnp.ndarray    

    @classmethod
    def from_rmnc_zmns_settings(cls, Rmnc : jnp.ndarray, Zmns : jnp.ndarray, settings : FluxSurfaceSettings, make_normals_point_outwards : bool = True):
        mpol_vector = _create_mpol_vector(settings.mpol, settings.ntor)
        ntor_vector = _create_ntor_vector(settings.mpol, settings.ntor, settings.nfp)

        if make_normals_point_outwards:
            flip_theta = _check_whether_make_normals_point_outwards_required(Rmnc, Zmns, mpol_vector)
            Rmnc_mod = jnp.where(
                flip_theta, 
                reverse_theta_total(mpol_vector, ntor_vector, Rmnc, True),
                Rmnc

            )
            
            Zmns_mod = jnp.where(
                flip_theta,
                reverse_theta_total(mpol_vector, ntor_vector, Zmns, False),
                Zmns
            )        
        else:
            Rmnc_mod = Rmnc 
            Zmns_mod = Zmns

        assert(Rmnc.shape == Zmns.shape)
        assert(Rmnc.shape[0] == settings.nsurf)
        assert(Rmnc.shape[1] == len(mpol_vector))
        return cls(Rmnc=Rmnc_mod, Zmns=Zmns_mod, mpol_vector=mpol_vector, ntor_vector=ntor_vector)

    def __iter__(self):
        return iter((self.Rmnc, self.Zmns, self.mpol_vector, self.ntor_vector))
    
def _data_settings_from_hdf5(filename : str, make_normals_point_outwards : bool = True):
    """Load a FluxSurface from an VMEC-type HDF5 file.

        Parameters:
        ----------
        filename : str
            Path to the HDF5 file.
        Returns:
        -------
        FluxSurface
            The loaded FluxSurface object.

    """

    with h5py.File(filename) as f:
        Rmnc = jnp.array(f['rmnc'])            
        Zmns = jnp.array(f['zmns'])            

        mpol = int(f['mpol'][()]) - 1 # vmec uses mpol 1 larger than maximum poloidal mode number
        ntor = int(f['ntor'][()])                        
        nfp  = int(f['nfp'][()])

        assert( jnp.all( _create_mpol_vector(mpol, ntor) == jnp.array(f['xm'])))      # sanity check
        assert( jnp.all( _create_ntor_vector(mpol, ntor, nfp) == jnp.array(f['xn']))) # sanity check
        
        nsurf = int(Zmns.shape[0])
        settings = FluxSurfaceSettings(
            mpol=mpol,
            ntor=ntor,
            nfp=nfp,                
            nsurf=nsurf
        )

        data = FluxSurfaceData.from_rmnc_zmns_settings(Rmnc, Zmns, settings)

        return data, settings
    

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FluxSurface:
    ''' 
    Class representing a set of flux surfaces using a VMEC-like representation.

    Attributes:
    -----------
    Rmnc : (nsurf, nmodes) jnp.ndarray
        Radial Fourier coefficients for the R coordinate.
    Zmns : (nsurf, nmodes) jnp.ndarray
        Vertical Fourier coefficients for the Z coordinate.
    settings : FluxSurfaceSettings
        Settings object containing parameters like mpol, ntor, nfp, and the mode vectors
    '''
    data            : FluxSurfaceData = None
    settings        : FluxSurfaceSettings = None
    

    @classmethod
    def from_hdf5(cls, filename : str):        
        data, settings = _data_settings_from_hdf5(filename)
        return cls(data=data, settings=settings)
    
    @classmethod
    def from_flux_surface(cls, flux_surface_base : "FluxSurface"):
        return cls(data = flux_surface_base.data, settings = flux_surface_base.settings)
    
    @classmethod
    def from_rmnc_zmns_mpol_ntor(cls, Rmnc : jnp.ndarray, Zmns : jnp.ndarray, mpol : int, ntor : int, nfp : int, make_normals_point_outwards : bool = True):
        nsurf = Rmnc.shape[0]
        settings = FluxSurfaceSettings(
            mpol=mpol,
            ntor=ntor,
            nfp=nfp,                
            nsurf=nsurf
        )
        data = FluxSurfaceData.from_rmnc_zmns_settings(Rmnc, Zmns, settings, make_normals_point_outwards)
        return cls(data=data, settings=settings)
    

    def cylindrical_position(self, s, theta, phi):
        return _cylindrical_position_interpolated(self.data, self.settings, s, theta, phi)
    
    def cartesian_position(self, s, theta, phi):
        return _cartesian_position_interpolated(self.data, self.settings, s, theta, phi)
    
    def normal(self, s, theta, phi):
        return _normal_interpolated(self.data, self.settings, s, theta, phi)
    
    def principal_curvatures(self, s, theta, phi):
        return _principal_curvatures_interpolated(self.data, self.settings, s, theta, phi)
    
    
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ToroidalExtent:
    start : float
    end   : float

    @classmethod
    def half_module(self, flux_surface : FluxSurface, dphi = 0.0):
        return self(dphi, 2 * jnp.pi / flux_surface.settings.nfp / 2.0 + dphi)
    
    @classmethod
    def full_module(self, flux_surface : FluxSurface, dphi = 0.0):
        return self(dphi, 2 * jnp.pi / flux_surface.settings.nfp + dphi)
    
    @classmethod 
    def full(self):
        return self(0.0, 2 * jnp.pi)

    def full_angle(self):
        return bool(jnp.allclose(self.end - self.start, 2 * jnp.pi))
    
    def __iter__(self):
        return iter((self.start, self.end, self.full_angle()))
    
    


# ===================================================================================================================================================================================
#                                                                           Positions
# ===================================================================================================================================================================================
@partial(jax.jit)
def _cylindrical_position_interpolated(data : FluxSurfaceData, settings  : FluxSurfaceSettings,  s,  theta, phi):
    
    
    # This in essence computes:
    # R   = jnp.sum(Rmnc_interp[..., None] * jnp.cos(mpol_vector[..., None] * theta[None, ...] - ntor_vector[..., None] * phi[None, ...]), axis=-1)
    # However, although the above can be more efficient, it creates large intermediate arrays and is thus undesirable.
    # Also, we call interpolate_array once per mode and per point in this setup
    # Instead, we could have vectorized this calculation over all points, but that would also create large intermediate arrays.
    
    # Now, no n_modes x n_points arrays are created.

    # This function is valid for both s,theta,phi all scalars and broadcastable arrays. 
    def fourier_sum(vals, i):
        R, Z = vals
        R = R + interpolate_array(data.Rmnc[..., i], s) * jnp.cos(data.mpol_vector[i] * theta - data.ntor_vector[i] * phi)
        Z = Z + interpolate_array(data.Zmns[..., i], s) * jnp.sin(data.mpol_vector[i] * theta - data.ntor_vector[i] * phi)
        return (R,Z), None
    
    # The fourier_sum function automatically broadcast arrays. However, we need to ensure that 
    # we start the scan with a zero object that has the correct final shape. Thus,
    # we create dummy arrays that have the correct shape.
    # The phi_bc  is required to ensure the final array phi is stackable with R, Z.
    s_bc, theta_bc, phi_bc = jnp.broadcast_arrays(s, theta, phi)
    
    n_modes = data.Rmnc.shape[1]
    
    R,Z = jax.lax.scan(fourier_sum, (jnp.zeros_like(theta_bc), jnp.zeros_like(theta_bc)), jnp.arange(n_modes))[0]
    
    return jnp.stack([R, Z, phi_bc],axis=-1)
    

@partial(jax.jit)
def _cartesian_position_interpolated(data : FluxSurfaceData, settings : FluxSurfaceSettings, s, theta, phi):
    RZphi = _cylindrical_position_interpolated(data, settings, s, theta, phi)
    return _cylindrical_to_cartesian(RZphi)

_dx_dtheta = jax.jit(jnp.vectorize(jax.jacfwd(_cartesian_position_interpolated, argnums=3), excluded=(0,1), signature='(),(),()->(3)'))

@partial(jax.jit)
def _arc_length_theta(data : FluxSurfaceData, settings : FluxSurfaceSettings, s, theta, phi):
    dx_dtheta = _dx_dtheta(data, settings, s, theta, phi)
    dx_dtheta_norm = jnp.linalg.norm(dx_dtheta, axis=-1)
    return dx_dtheta_norm
# ===================================================================================================================================================================================
#                                                                           Normals
# ===================================================================================================================================================================================


# this function requires scalars to work since it needs to return a (3,2) array
# vmapping works, but loses the flexibility of either of the inputs being arrays, scalars or multidimensional arrays
# furthermore, the jacobians are stacked to ensure jnp.vectorize can be used (it does not support multiple outputs like given by jacfwd)
_cartesian_position_interpolated_grad = jax.jit(jnp.vectorize(stack_jacfwd(_cartesian_position_interpolated, argnums=(3,4)), excluded=(0,1), signature='(),(),()->(3,2)'))

@partial(jax.jit)
def _dx_dphi_cross_dx_dtheta(data : FluxSurfaceData,  settings : FluxSurfaceSettings, s, theta, phi):
    dX_dtheta_and_dX_dphi = _cartesian_position_interpolated_grad(data, settings, s, theta, phi)
    # We use dr/dphi x dr/dtheta 
    # Then, we want to have the normal vector outwards to the LCFS and not point into the plasma
    # This is accomplised by using the dphi_x_dtheta member. 
    n = jnp.cross(dX_dtheta_and_dX_dphi[..., 1], dX_dtheta_and_dX_dphi[..., 0])    
    return n

@partial(jax.jit)
def _normal_interpolated(data : FluxSurfaceData,  settings : FluxSurfaceSettings, s, theta, phi):    
    # We use dr/dphi x dr/dtheta 
    # Then, we want to have the normal vector outwards to the LCFS and not point into the plasma
    # This is accomplised by using the dphi_x_dtheta member. 
    n = _dx_dphi_cross_dx_dtheta(data, settings, s, theta, phi)
    n = n / jnp.linalg.norm(n, axis=-1, keepdims=True)
    return n

# ===================================================================================================================================================================================
#                                                                           Principal curvatures
# ===================================================================================================================================================================================
_cartesian_position_interpolated_grad_grad = jax.jit(jnp.vectorize(stack_jacfwd(stack_jacfwd(_cartesian_position_interpolated, argnums=(3,4)), argnums = (3,4)), excluded=(0,1), signature='(),(),()->(3,2,2)'))

@partial(jax.jit)
def _principal_curvatures_interpolated(data : FluxSurfaceData,  settings : FluxSurfaceSettings, s, theta, phi):
    dX_dtheta_and_dX_dphi                        = _cartesian_position_interpolated_grad(data, settings, s, theta, phi)
    d2X_dtheta2_and_d2X_dthetadphi_and_d2X_dphi2 = _cartesian_position_interpolated_grad_grad(data, settings, s, theta, phi)

    # dx_dtheta_and_dX_dphi has shape (..., 3, 2), last index 0 is d/dtheta, last index 1 is d/dphi
    # d2X_dtheta2_and_d2X_dthetadphi_and_d2X_dphi2 has shape (..., 3, 2, 2) # 0,0 is d2/dtheta2, 0,1 and 1,0 is d2/dthetadphi, 1,1 is d2/dphi2
    
    E = jnp.einsum("...i, ...i->...", dX_dtheta_and_dX_dphi[..., 0], dX_dtheta_and_dX_dphi[..., 0])
    F = jnp.einsum("...i, ...i->...", dX_dtheta_and_dX_dphi[..., 0], dX_dtheta_and_dX_dphi[..., 1])
    G = jnp.einsum("...i, ...i->...", dX_dtheta_and_dX_dphi[..., 1], dX_dtheta_and_dX_dphi[..., 1])
    
    normal_vector = jnp.cross(dX_dtheta_and_dX_dphi[..., 1], dX_dtheta_and_dX_dphi[..., 0])
    normal_vector = normal_vector / jnp.linalg.norm(normal_vector, axis=-1, keepdims=True)

    L = jnp.einsum("...i, ...i->...", normal_vector, d2X_dtheta2_and_d2X_dthetadphi_and_d2X_dphi2[..., 0, 0])
    M = jnp.einsum("...i, ...i->...", normal_vector, d2X_dtheta2_and_d2X_dthetadphi_and_d2X_dphi2[..., 0, 1])
    N = jnp.einsum("...i, ...i->...", normal_vector, d2X_dtheta2_and_d2X_dthetadphi_and_d2X_dphi2[..., 1, 1])

    H = (E * N - 2 * F * M + G * L) / (2 * (E * G - F**2))
    K = (L * N - M**2) / (E * G - F**2)
    
    sqrt_discriminant = jnp.sqrt(H**2 - K) 
    k1 = - (H + sqrt_discriminant)
    k2 = - (H - sqrt_discriminant)

    return jnp.stack([k1, k2], axis=-1)



# ===================================================================================================================================================================================
#                                                                           Volume and Surface
# ===================================================================================================================================================================================

@partial(jax.jit, static_argnums=(1))
def _volume_from_fourier(data : FluxSurfaceData, settings : FluxSurfaceSettings, s : float):
    ''' 
    Compute the volume enclosed by the flux surface at s using a Fourier representation.

    A full module is used for the integration. Using the divergence theorem, the volume is computed as:
        V = (1/3) * ∫∫ (r · n) dA
    where r is the position vector, n is the outward normal vector, and dA is the differential area element on the surface.
    Note that dA = |dx/dtheta x dx/dphi| dtheta dphi and thus r · n dA = r · (dx/dphi x dx/dtheta) dtheta dphi.
    The trapezoidal rule is then used to arrive at the final value.


    Parameters:
    -----------
    data : FluxSurfaceData
        The flux surface data containing Fourier coefficients.
    settings : FluxSurfaceSettings
        The flux surface settings.
    s : float
        The normalized flux surface label (0 <= s <= 1).
    Returns:
    --------
    volume : float
        The volume enclosed by the flux surface at s.
    '''
    # x:          m,n Fourier modes
    # dx_dtheta:  m,n fourier modes
    # dx_dphi:    m,n fourier modes
    # normal: m + m, m+ fourier modes
    # x.normal -> 3 * m,n fourier modes
    # Nyquist -> 6 times the mode number
    nyquist_sampling = 6

    n_theta = settings.mpol * nyquist_sampling +1
    n_phi   = settings.ntor *  nyquist_sampling  +1
    
    theta = jnp.linspace(0, 2 * jnp.pi, n_theta, endpoint=False)
    phi   = jnp.linspace(0, 2 * jnp.pi / settings.nfp, n_phi, endpoint=False)
    
    dtheta = 2 * jnp.pi / n_theta
    dphi   = 2 * jnp.pi / settings.nfp  /  n_phi 

    tt, pp = jnp.meshgrid(theta, phi, indexing='ij')
    
    surface_normals      = _dx_dphi_cross_dx_dtheta(data, settings, s, tt, pp)

    r                     = _cartesian_position_interpolated(data, settings, s, tt, pp)    
    f_ij = jnp.einsum('...i,...i->...', r, surface_normals)

    volume = jnp.sum(f_ij) * dtheta * dphi / 3.0 * settings.nfp 

    return volume

@partial(jax.jit, static_argnums=(1))
def _volume_from_fourier_half_mod(data : FluxSurfaceData, settings : FluxSurfaceSettings, s : float):
    ''' 
    Compute the volume enclosed by the flux surface at s using a Fourier representation.

    A half module is used for the integration. Using the divergence theorem, the volume is computed as:
        V = (1/3) * ∫∫ (r · n) dA
    where r is the position vector, n is the outward normal vector, and dA is the differential area element on the surface.
    Note that dA = |dx/dtheta x dx/dphi| dtheta dphi and thus r · n dA = r · (dx/dphi x dx/dtheta) dtheta dphi.
    The trapezoidal rule is then used to arrive at the final value.


    Parameters:
    -----------
    data : FluxSurfaceData
        The flux surface data containing Fourier coefficients.
    settings : FluxSurfaceSettings
        The flux surface settings.
    s : float
        The normalized flux surface label (0 <= s <= 1).
    Returns:
    --------
    volume : float
        The volume enclosed by the flux surface at s.
    '''

    nyquist_sampling = 6
    n_theta = settings.mpol * nyquist_sampling + 1

    # We add one to always satisfy nyquist.
    n_phi   = int((settings.ntor *  nyquist_sampling + 1) / 2) 
    
    # Now, given that we want to sample half of a module, we have two choices depending on the full module n_phi:

    # - Include the half module boundary

    # - Exclude the half module boundary

    # If we include the half-module boundary, we have to use 
    # phi = jnp.linspace(0, 2 * jnp.pi / settings.nfp , n_phi, endpoint=True)
    # and then subtract half of the initial phi=0 and half of the phi = pi / nfp boundary contributions from the volume integral

    # If we exclude the half-module boundary, we have to use
    # phi = jnp.linspace(0, 2 * jnp.pi / settings.nfp, 2 * n_phi, endpoint=True)[:n_phi]
    # and then double the volume integral and subtracth only half of the initial phi=0 boundary.
    
    # We chose the latter option since it is one less computation. Numerically, they are exactly the same.

    theta = jnp.linspace(0, 2 * jnp.pi, n_theta, endpoint=False)
    phi   = jnp.linspace(0, 2 * jnp.pi / settings.nfp , 2* n_phi, endpoint=True)[:n_phi]
    
    dtheta = 2 * jnp.pi / n_theta
    dphi   = phi[1]- phi[0]
        
    tt, pp = jnp.meshgrid(theta, phi, indexing='ij')
    
    surface_normals        = _dx_dphi_cross_dx_dtheta(data, settings, s, tt, pp)

    r                      = _cartesian_position_interpolated(data, settings, s, tt, pp)    
    f_ij                   = jnp.einsum('...i,...i->...', r, surface_normals)

    base_half_mod          = jnp.sum(f_ij) * dtheta * dphi / 3.0 
    boundary_correction_b1 = jnp.sum(f_ij[:,0]) * dtheta * dphi / 3.0 

    return (base_half_mod * 2.0 -  boundary_correction_b1) * settings.nfp
    


