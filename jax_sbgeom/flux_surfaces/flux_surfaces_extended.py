import jax.numpy as jnp
import h5py 
import jax
import numpy as onp
from dataclasses import dataclass
from jax_sbgeom.jax_utils.utils import stack_jacfwd
from functools import partial
from .flux_surfaces_base import FluxSurface, ToroidalExtent, FluxSurfaceSettings, FluxSurfaceData, _data_settings_from_hdf5, _cartesian_to_cylindrical, _normal_interpolated_jit, _principal_curvatures_interpolated

from .flux_surfaces_base import _cartesian_position_interpolated, _normal_interpolated

@jax.tree_util.register_dataclass
@dataclass(frozen =  True)
class FluxSurfaceNormalExtended(FluxSurface):

    def cartesian_position(self, s, theta, phi):
        return _normal_extended_cartesian_position_jit(self.data, self.settings, s, theta, phi)
    
    def cylindrical_position(self, s, theta, phi):
        return _normal_extended_cylindrical_position_jit(self.data, self.settings, s, theta, phi)
    
      # For a normal extended flux surface, the normal *remains the same* in the extended region
    def normal(self, s, theta, phi):        
        return _normal_extended_normal_jit(self.data, self.settings, s, theta, phi)
    
    
    def principal_curvatures(self, s, theta, phi):        
        return _normal_extended_principal_curvatures_jit(self.data, self.settings, s, theta, phi)


@jax.tree_util.register_dataclass
@dataclass(frozen =  True)
class FluxSurfaceNormalExtendedNoPhi(FluxSurface):

    def cartesian_position(self, s, theta, phi):
        return _normal_extended_no_phi_cartesian_position_jit(self.data, self.settings, s, theta, phi)
    
    def cylindrical_position(self, s, theta, phi):
        return _normal_extended_no_phi_cylindrical_position_jit(self.data, self.settings, s, theta, phi)
    
    # For a normal extended flux surface, the normal *remains the same* in the extended region
    def normal(self, s, theta, phi):
        return _normal_extended_no_phi_normal_jit(self.data, self.settings, s, theta, phi)
    
    # Principal curvatures could be implemented in the extension region using 
    def principal_curvatures(self, s, theta, phi):
        return _normal_extended_no_phi_principal_curvatures_jit(self.data, self.settings, s, theta, phi)

@jax.tree_util.register_dataclass
@dataclass(frozen =  True)
class FluxSurfaceNormalExtendedConstantPhi(FluxSurface):

    def cartesian_position(self, s, theta, phi):
        return _normal_extended_constant_phi_cartesian_position_jit(self.data, self.settings, s, theta, phi)
    
    def cylindrical_position(self, s, theta, phi):        
        return _normal_extended_constant_phi_cylindrical_position_jit(self.data, self.settings, s, theta, phi)
    
    # For a normal extended flux surface, the normal *remains the same* in the extended region
    def normal(self, s, theta, phi):        
        return _normal_extended_constant_phi_normal_jit(self.data, self.settings, s, theta, phi)
    
    # Principal curvatures could be implemented in the extension region using 
    def principal_curvatures(self, s, theta, phi):        
        return _normal_extended_constant_phi_principal_curvatures_jit(self.data, self.settings, s, theta, phi)


# ===================================================================================================================================================================================
#                                                                           Normal Extended
# ===================================================================================================================================================================================

def _normal_extended_cartesian_position(data : FluxSurfaceData, settings  : FluxSurfaceSettings,  s,  theta, phi):    
    positions = _cartesian_position_interpolated(data, settings, jnp.minimum(s, 1.0), theta, phi) 
    normals   = _normal_interpolated(data, settings, 1.0, theta, phi) # this will not give NaNs, as s=1.0 is always on the surface (non axis)    
    distance_1d = jnp.maximum(s - 1.0, 0.0)
    # We have to ensure that both do not produce nan values. 
    # This is the case, as the positions are evaluated at s <= 1.0 and normals at s = 1.0    
    return positions + normals * distance_1d[..., None]

def _normal_extended_cylindrical_position(data : FluxSurfaceData, settings  : FluxSurfaceSettings,  s,  theta, phi):
    return _cartesian_to_cylindrical(_normal_extended_cartesian_position(data, settings, s, theta, phi))

def _normal_extended_normal(data : FluxSurfaceData, settings  : FluxSurfaceSettings,  s,  theta, phi):
    return _normal_interpolated(data, settings, jnp.minimum(s, 1.0), theta, phi)

def _normal_extended_principal_curvatures(data : FluxSurfaceData, settings  : FluxSurfaceSettings,  s,  theta, phi):
    curvatures = _principal_curvatures_interpolated(data, settings, jnp.minimum(s, 1.0), theta, phi)
    d = jnp.maximum(s - 1.0, 0.0)        
    gamma_0 = jnp.where(1 + curvatures[...,0] * d >= 0.0, jnp.ones_like(d), jnp.ones_like(d) * -1.0) # the >= or > is arbitrary
    gamma_1 = jnp.where(1 + curvatures[...,1] * d >= 0.0, jnp.ones_like(d), jnp.ones_like(d) * -1.0) # the >= or > is arbitrary        
    kappa_0 = curvatures[...,0] / jnp.abs(1.0 + curvatures[...,0] * d) * gamma_0
    kappa_1 = curvatures[...,1] / jnp.abs(1.0 + curvatures[...,1] * d) * gamma_1
    return jnp.stack([kappa_0, kappa_1], axis=-1)


_normal_extended_cartesian_position_jit   = jax.jit(_normal_extended_cartesian_position)
_normal_extended_cylindrical_position_jit = jax.jit(_normal_extended_cylindrical_position)
_normal_extended_normal_jit               = jax.jit(_normal_extended_normal)
_normal_extended_principal_curvatures_jit = jax.jit(_normal_extended_principal_curvatures)


# ===================================================================================================================================================================================
#                                                                           No Phi
# ===================================================================================================================================================================================
def _hat_phi(positions):
    x = positions[...,0]
    y = positions[...,1]
    z = positions[...,2]
    r = jnp.sqrt(x**2 + y**2)        
    safe_r  = jnp.clip(r, min = 1e-12)
    hat_phi = jnp.stack([-y / safe_r, x / safe_r, jnp.zeros_like(z)], axis=-1)
    return hat_phi

def _normal_extended_no_phi_cartesian_position(data : FluxSurfaceData, settings  : FluxSurfaceSettings,  s,  theta, phi):
    positions                = _cartesian_position_interpolated(data, settings, jnp.minimum(s, 1.0), theta, phi)
    normals                  = _normal_interpolated(data, settings, 1.0, theta, phi)
    hat_phi                  = _hat_phi(positions)
    
    phi_component            = jnp.einsum("...i,...i->...", normals, hat_phi)
    
    normal_no_phi            = normals - phi_component[..., None] * hat_phi
    normal_no_phi_normalised = normal_no_phi / jnp.linalg.norm(normal_no_phi, axis=-1, keepdims=True)    
    
    distance_1d              = jnp.maximum(s - 1.0, 0.0)

    return positions + normal_no_phi_normalised * distance_1d[..., None]

def _normal_extended_no_phi_cylindrical_position(data : FluxSurfaceData, settings  : FluxSurfaceSettings,  s,  theta, phi):
    return _cartesian_to_cylindrical(_normal_extended_no_phi_cartesian_position(data, settings, s, theta, phi))

def _normal_extended_no_phi_normal(data : FluxSurfaceData, settings  : FluxSurfaceSettings,  s,  theta, phi):
    s_bc, theta_bc, phi_bc = jnp.broadcast_arrays(s, theta, phi)
    return jnp.full(s_bc.shape + (3,), jnp.nan)

def _normal_extended_no_phi_principal_curvatures(data : FluxSurfaceData, settings  : FluxSurfaceSettings,  s,  theta, phi):
    s_bc, theta_bc, phi_bc = jnp.broadcast_arrays(s, theta, phi)
    return jnp.full(s_bc.shape + (2,), jnp.nan)

_normal_extended_no_phi_cartesian_position_jit   = jax.jit(_normal_extended_no_phi_cartesian_position)
_normal_extended_no_phi_cylindrical_position_jit = jax.jit(_normal_extended_no_phi_cylindrical_position)
_normal_extended_no_phi_normal_jit               = jax.jit(_normal_extended_no_phi_normal)
_normal_extended_no_phi_principal_curvatures_jit = jax.jit(_normal_extended_no_phi_principal_curvatures)

# ===================================================================================================================================================================================
#                                                                          Constant Phi
# ===================================================================================================================================================================================
def _distance_between_angles(angle1, angle2):
    return jnp.arctan2(jnp.sin(angle1 - angle2), jnp.cos(angle1 - angle2))

def _distance_between_phi_phi_desired(data, settings, s, theta, phi, x):
    positions =  _normal_extended_cartesian_position(data, settings, s, theta, x)
    return _distance_between_angles(jnp.arctan2(positions[...,1], positions[...,0]), phi)

@partial(jax.jit, static_argnums=(5)) # since we re-use this function multiple times, we jit it here
def _normal_extended_constant_phi_find_phi(data, settings, s, theta, phi, n_iter : int = 5):
    assert n_iter >= 1, "n_iter must be at least 1"

    _, _, phi_bc = jnp.broadcast_arrays(s, theta, phi)
        
    x_minus_two = phi_bc + 1e-3
    x_minus_one = phi_bc
    
    f_minus_two = _distance_between_phi_phi_desired(data, settings, s, theta, phi, x_minus_two)

    def secant_iteration(i, vals):
        x_minus_two, x_minus_one, f_minus_two = vals
        f_minus_one = _distance_between_phi_phi_desired(data, settings, s, theta, phi, x_minus_one)

        x_new = x_minus_one - f_minus_one * (x_minus_one - x_minus_two) / (f_minus_one - f_minus_two + 1e-16)
        return (x_minus_one, x_new, f_minus_one)

    x_final = jax.lax.fori_loop(0, n_iter, secant_iteration, (x_minus_two, x_minus_one, f_minus_two))[1]
    return x_final

def _normal_extended_constant_phi_cartesian_position(data : FluxSurfaceData, settings  : FluxSurfaceSettings,  s,  theta, phi, n_iter : int = 5):
    phi_c = _normal_extended_constant_phi_find_phi(data, settings, s, theta, phi, n_iter)    
    return _normal_extended_cartesian_position(data, settings, s, theta, phi_c)

def _normal_extended_constant_phi_cylindrical_position(data : FluxSurfaceData, settings  : FluxSurfaceSettings,  s,  theta, phi, n_iter : int = 5):
    return _cartesian_to_cylindrical(_normal_extended_constant_phi_cartesian_position(data, settings, s, theta, phi, n_iter))

def _normal_extended_constant_phi_normal(data : FluxSurfaceData, settings  : FluxSurfaceSettings,  s,  theta, phi, n_iter : int = 5):
    phi_c = _normal_extended_constant_phi_find_phi(data, settings, s, theta, phi, n_iter)    
    return _normal_extended_normal(data, settings, s, theta, phi_c)

def _normal_extended_constant_phi_principal_curvatures(data : FluxSurfaceData, settings  : FluxSurfaceSettings,  s,  theta, phi, n_iter : int = 5):
    phi_c = _normal_extended_constant_phi_find_phi(data, settings, s, theta, phi, n_iter)    
    return _normal_extended_principal_curvatures(data, settings, s, theta, phi_c)

_normal_extended_constant_phi_cartesian_position_jit   = jax.jit(_normal_extended_constant_phi_cartesian_position, static_argnums=(5))
_normal_extended_constant_phi_cylindrical_position_jit = jax.jit(_normal_extended_constant_phi_cylindrical_position, static_argnums=(5))
_normal_extended_constant_phi_normal_jit               = jax.jit(_normal_extended_constant_phi_normal, static_argnums=(5))
_normal_extended_constant_phi_principal_curvatures_jit = jax.jit(_normal_extended_constant_phi_principal_curvatures, static_argnums=(5))






