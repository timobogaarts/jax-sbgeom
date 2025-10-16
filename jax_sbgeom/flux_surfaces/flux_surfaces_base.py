import jax.numpy as jnp
import h5py 
import jax
import numpy as onp
from dataclasses import dataclass
from jax_sbgeom.jax_utils.utils import stack_jacfwd
from functools import partial

def _create_mpol_vector(ntor : int, mpol : int):    
    return jnp.array([0 for i in range(ntor + 1)] + sum([[i for j in range(2 * ntor + 1)]for i in range(1, mpol )], []), dtype=int)

def _create_ntor_vector(ntor : int, mpol : int, symm : int):
    return jnp.array(list(range(0, (ntor + 1) * symm , symm)) + sum([list(range(-ntor * symm, (ntor + 1) * symm, symm)) for i in range(mpol - 1)], []), dtype=int)


def _cylindrical_to_cartesian(RZphi : jnp.ndarray):
    R = RZphi[..., 0]
    Z = RZphi[..., 1]
    phi = RZphi[..., 2]
    x = R * jnp.cos(phi)
    y = R * jnp.sin(phi)
    return jnp.stack([x, y, Z], axis=-1)



@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FluxSurfaceSettings:
    mpol : int
    ntor : int
    nfp  : int
    nsurf : int    

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FluxSurfaceData:
    Rmnc : jnp.ndarray
    Zmns : jnp.ndarray
    mpol_vector : jnp.ndarray
    ntor_vector : jnp.ndarray
    dphi_x_dtheta : float

    @classmethod
    def from_rmnc_zmns_settings(cls, Rmnc : jnp.ndarray, Zmns : jnp.ndarray, settings : FluxSurfaceSettings):
        mpol_vector = _create_mpol_vector(settings.ntor, settings.mpol)
        ntor_vector = _create_ntor_vector(settings.ntor, settings.mpol, settings.nfp)

        # This computes:
        # dZ_dtheta on the lcfs at  theta, phi = 0:
        # dZ_dtheta = jnp.sum(Zmns[-1,:] * mpol_vector * jnp.cos(mpol_vector * 0 - ntor_vector * 0)) = jnp.sum(Zmns[-1,:] * mpol_vector)        
        sum_Zmns = jnp.sum(Zmns[-1,:] * mpol_vector)
        
        # Rmnc at theta, phi = 0
        # R = jnp.sum(Rmnc[-1,:] * jnp.cos(mpol_vector * 0 - ntor_vector * 0)) = jnp.sum(Rmnc[-1,:])
        # Rmnc at theta, phi = pi, 0:
        # R = jnp.sum(Rmnc[-1,:] * jnp.cos(mpol_vector * jnp.pi - ntor_vector * 0)) = jnp.sum(Rmnc[-1,:] * (-1)**mpol_vector)
        # We want to determine whether dZ_dtheta points in the positive or negative Z direction at the outboard midplane (R maximum)
        # This is accomplished by checking the sign of dZ_dtheta at the outboard midplane.
        # The outboard midplane is at theta = 0 if sum_Rmnc > 0 and at theta = pi if sum_Rmnc < 0
        cond_outboard = jnp.sum(Rmnc[-1,:]) > jnp.sum(Rmnc[-1,:] * (-1)**mpol_vector)

        # If cond_positive is true, then dZ_dtheta points in the positive Z direction at the outboard midplane.
        # Thus, we want to use dphi_x_dtheta = 1.0
        # If cond_positive is false, then dZ_dtheta points in the negative Z direction at the outboard midplane.
        # Thus, we want to use dphi_x_dtheta = -1.0
        
        dphi_x_dtheta = jnp.where(
            cond_outboard,
            jnp.where(sum_Zmns > 0, 1.0, -1.0),   # cond_outboard == True branch
            jnp.where(sum_Zmns > 0, -1.0, 1.0)    # cond_outboard == False branch
        )        

        assert(Rmnc.shape == Zmns.shape)
        assert(Rmnc.shape[0] == settings.nsurf)
        assert(Rmnc.shape[1] == len(mpol_vector))
        return cls(Rmnc=Rmnc, Zmns=Zmns, mpol_vector=mpol_vector, ntor_vector=ntor_vector, dphi_x_dtheta=dphi_x_dtheta)

    def __iter__(self):
        return iter((self.Rmnc, self.Zmns, self.mpol_vector, self.ntor_vector, self.dphi_x_dtheta))
    

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

            mpol = int(f['mpol'][()])
            ntor = int(f['ntor'][()])                        
            nfp  = int(f['nfp'][()])

            assert( jnp.all( _create_mpol_vector(ntor, mpol) == jnp.array(f['xm'])))      # sanity check
            assert( jnp.all( _create_ntor_vector(ntor, mpol, nfp) == jnp.array(f['xn']))) # sanity check
            
            nsurf = Zmns.shape[0]
            settings = FluxSurfaceSettings(
                mpol=mpol,
                ntor=ntor,
                nfp=nfp,                
                nsurf=nsurf
            )

            data = FluxSurfaceData.from_rmnc_zmns_settings(Rmnc, Zmns, settings)
        return cls(data=data, settings=settings)
    

    def cylindrical_position(self, s, theta, phi):
        return _cylindrical_position_interpolated_jit(self.data, self.settings, s, theta, phi)
    
    def cartesian_position(self, s, theta, phi):
        return _cartesian_position_interpolated_jit(self.data, self.settings, s, theta, phi)
    
    def normal(self, s, theta, phi):
        return _normal_interpolated_jit(self.data, self.settings, s, theta, phi)
    
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
#                                                                           Interpolation of arrays
# ===================================================================================================================================================================================
def _interpolate_fractions(s, nsurf):    
    s_start =  s * (nsurf-1)
    i0      = jnp.floor(s_start).astype(int)
    i1      = jnp.minimum(i0 + 1, nsurf - 1)    
    ds      = s_start - i0    
    return i0, i1, ds

def _interpolate_array(x_interp, s):
    nsurf = x_interp.shape[0]
    i0, i1, ds   = _interpolate_fractions(s, nsurf)
    x0 = x_interp[i0]
    x1 = x_interp[i1]
    return (1 - ds) * x0 + ds * x1

# ===================================================================================================================================================================================
#                                                                           Positions
# ===================================================================================================================================================================================

def _cylindrical_position_interpolated(data : FluxSurfaceData, settings  : FluxSurfaceSettings,  s,  theta, phi):
    
    
    # This in essence computes:
    # R   = jnp.sum(Rmnc_interp[..., None] * jnp.cos(mpol_vector[..., None] * theta[None, ...] - ntor_vector[..., None] * phi[None, ...]), axis=-1)
    # However, although the above can be more efficient, it creates large intermediate arrays and is thus undesirable.
    # Also, we call _interpolate_array once per mode and per point in this setup
    # Instead, we could have vectorized this calculation over all points, but that would also create large intermediate arrays.
    
    # Now, no n_modes x n_points arrays are created.

    # This function is valid for both s,theta,phi all scalars and broadcastable arrays. 
    def fourier_sum(vals, i):
        R, Z = vals
        R = R + _interpolate_array(data.Rmnc[..., i], s) * jnp.cos(data.mpol_vector[i] * theta - data.ntor_vector[i] * phi)
        Z = Z + _interpolate_array(data.Zmns[..., i], s) * jnp.sin(data.mpol_vector[i] * theta - data.ntor_vector[i] * phi)
        return (R,Z), None
    
    # The fourier_sum function automatically broadcast arrays. However, we need to ensure that 
    # we start the scan with a zero object that has the correct final shape. Thus,
    # we create dummy arrays that have the correct shape.
    # The phi_bc  is required to ensure the final array phi is stackable with R, Z.
    s_bc, theta_bc, phi_bc = jnp.broadcast_arrays(s, theta, phi)
    
    n_modes = data.Rmnc.shape[1]
    
    R,Z = jax.lax.scan(fourier_sum, (jnp.zeros_like(theta_bc), jnp.zeros_like(theta_bc)), jnp.arange(n_modes))[0]
    
    return jnp.stack([R, Z, phi_bc],axis=-1)
    

_cylindrical_position_interpolated_jit = jax.jit(_cylindrical_position_interpolated)

def _cartesian_position_interpolated(data : FluxSurfaceData, settings : FluxSurfaceSettings, s, theta, phi):
    RZphi = _cylindrical_position_interpolated(data, settings, s, theta, phi)
    return _cylindrical_to_cartesian(RZphi)

_cartesian_position_interpolated_jit = jax.jit(_cartesian_position_interpolated)

# ===================================================================================================================================================================================
#                                                                           Normals
# ===================================================================================================================================================================================


# this function requires scalars to work since it needs to return a (3,2) array
# vmapping works, but loses the flexibility of either of the inputs being arrays, scalars or multidimensional arrays
# furthermore, the jacobians are stacked to ensure jnp.vectorize can be used (it does not support multiple outputs like given by jacfwd)
_cartesian_position_interpolated_grad = jax.jit(jnp.vectorize(stack_jacfwd(_cartesian_position_interpolated, argnums=(3,4)), excluded=(0,1), signature='(),(),()->(3,2)'))


def _normal_interpolated(data : FluxSurfaceData,  settings : FluxSurfaceSettings, s, theta, phi):
    dX_dtheta_and_dX_dphi = _cartesian_position_interpolated_grad(data, settings, s, theta, phi)
    # We use dr/dphi x dr/dtheta 
    # Then, we want to have the normal vector outwards to the LCFS and not point into the plasma
    # This is accomplised by using the dphi_x_dtheta member. 
    n = jnp.cross(dX_dtheta_and_dX_dphi[..., 1], dX_dtheta_and_dX_dphi[..., 0]) * data.dphi_x_dtheta
    n = n / jnp.linalg.norm(n, axis=-1, keepdims=True)
    return n

_normal_interpolated_jit = jax.jit(_normal_interpolated)


# ===================================================================================================================================================================================
#                                                                           Principal curvatures
# ===================================================================================================================================================================================
_cartesian_position_interpolated_grad_grad = jax.jit(jnp.vectorize(stack_jacfwd(stack_jacfwd(_cartesian_position_interpolated, argnums=(3,4)), argnums = (3,4)), excluded=(0,1), signature='(),(),()->(3,2,2)'))

@jax.jit
def _principal_curvatures_interpolated(data : FluxSurfaceData,  settings : FluxSurfaceSettings, s, theta, phi):
    dX_dtheta_and_dX_dphi                        = _cartesian_position_interpolated_grad(data, settings, s, theta, phi)
    d2X_dtheta2_and_d2X_dthetadphi_and_d2X_dphi2 = _cartesian_position_interpolated_grad_grad(data, settings, s, theta, phi)

    # dx_dtheta_and_dX_dphi has shape (..., 3, 2), last index 0 is d/dtheta, last index 1 is d/dphi
    # d2X_dtheta2_and_d2X_dthetadphi_and_d2X_dphi2 has shape (..., 3, 2, 2) # 0,0 is d2/dtheta2, 0,1 and 1,0 is d2/dthetadphi, 1,1 is d2/dphi2
    
    E = jnp.einsum("...i, ...i->...", dX_dtheta_and_dX_dphi[..., 0], dX_dtheta_and_dX_dphi[..., 0])
    F = jnp.einsum("...i, ...i->...", dX_dtheta_and_dX_dphi[..., 0], dX_dtheta_and_dX_dphi[..., 1])
    G = jnp.einsum("...i, ...i->...", dX_dtheta_and_dX_dphi[..., 1], dX_dtheta_and_dX_dphi[..., 1])
    
    normal_vector = jnp.cross(dX_dtheta_and_dX_dphi[..., 1], dX_dtheta_and_dX_dphi[..., 0]) * data.dphi_x_dtheta
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