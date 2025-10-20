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

    @classmethod
    def from_hdf5(cls, filename : str):
        data, settings = _data_settings_from_hdf5(filename)
        return cls(data = data, settings = settings)
    
    @classmethod
    def from_flux_surface(cls, flux_surface_base : FluxSurface):
        return cls(data = flux_surface_base.data, settings = flux_surface_base.settings)
    

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

    @classmethod
    def from_hdf5(cls, filename : str):
        data, settings = _data_settings_from_hdf5(filename)
        return cls(data = data, settings = settings)
    
    @classmethod
    def from_flux_surface(cls, flux_surface_base : FluxSurface):
        return cls(data = flux_surface_base.data, settings = flux_surface_base.settings)
    

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


# ===================================================================================================================================================================================
#                                                                           Positions
# ===================================================================================================================================================================================

def _normal_extended_cartesian_position(data : FluxSurfaceData, settings  : FluxSurfaceSettings,  s,  theta, phi):    
    positions = _cartesian_position_interpolated(data, settings, jnp.minimum(s, 1.0), theta, phi) 
    normals   = _normal_interpolated(data, settings, 1.0, theta, phi)
    s_1d = jnp.atleast_1d(s)
    distance_1d = jnp.maximum(s_1d - 1.0, 0.0)

    # We have to ensure that both do not produce nan values. 
    # This is the case, as the positions are evaluated at s <= 1.0 and normals at s = 1.0    
    return jnp.where(s_1d[..., None] <= 1.0,  positions, positions + normals * distance_1d[..., None])

def _normal_extended_cylindrical_position(data : FluxSurfaceData, settings  : FluxSurfaceSettings,  s,  theta, phi):
    return _cartesian_to_cylindrical(_normal_extended_cartesian_position(data, settings, s, theta, phi))

def _normal_extended_normal(data : FluxSurfaceData, settings  : FluxSurfaceSettings,  s,  theta, phi):
    return _normal_interpolated(data, settings, jnp.minimum(s, 1.0), theta, phi)

_normal_extended_cartesian_position_jit   = jax.jit(_normal_extended_cartesian_position)
_normal_extended_cylindrical_position_jit = jax.jit(_normal_extended_cylindrical_position)
_normal_extended_normal_jit               = jax.jit(_normal_extended_normal)

def _normal_extended_no_phi_normal(data : FluxSurfaceData, settings  : FluxSurfaceSettings,  s,  theta, phi):
    pass

def _normal_extended_no_phi_cartesian_position(data : FluxSurfaceData, settings  : FluxSurfaceSettings,  s,  theta, phi):
    pass

def _normal_extended_no_phi_cylindrical_position(data : FluxSurfaceData, settings  : FluxSurfaceSettings,  s,  theta, phi):
    pass

_normal_extended_no_phi_cartesian_position_jit   = jax.jit(_normal_extended_no_phi_cartesian_position)
_normal_extended_no_phi_cylindrical_position_jit = jax.jit(_normal_extended_no_phi_cylindrical_position)
_normal_extended_no_phi_normal_jit               = jax.jit(_normal_extended_no_phi_normal)

# ===================================================================================================================================================================================
#                                                                          Curvature
# ===================================================================================================================================================================================
def _normal_extended_principal_curvatures(data : FluxSurfaceData, settings  : FluxSurfaceSettings,  s,  theta, phi):

    def compute_beyond_lcfs(s, theta, phi):
        curvatures = _principal_curvatures_interpolated(data, settings, jnp.ones_like(s), theta, phi)
        d = jnp.maximum(s - 1.0, 0.0)        
        gamma_0 = jnp.where(1 + curvatures[...,0] * d >= 0.0, jnp.ones_like(d), jnp.ones_like(d) * -1.0) # the >= or > is arbitrary
        gamma_1 = jnp.where(1 + curvatures[...,1] * d >= 0.0, jnp.ones_like(d), jnp.ones_like(d) * -1.0) # the >= or > is arbitrary        
        kappa_0 = curvatures[...,0] / jnp.abs(1.0 + curvatures[...,0] * d) * gamma_0
        kappa_1 = curvatures[...,1] / jnp.abs(1.0 + curvatures[...,1] * d) * gamma_1
        return jnp.stack([kappa_0, kappa_1], axis=-1)
    
    def compute_within_lcfs(s, theta, phi):
        return _principal_curvatures_interpolated(data, settings, jnp.minimum(s, 1.0), theta, phi)
    
    s_1d = jnp.atleast_1d(s)    
    # Both functions are well-defined everywhere due to the use of jnp.minimum and jnp.maximum
    return jnp.where(s_1d[..., None] <= 1.0, compute_within_lcfs(s, theta, phi), compute_beyond_lcfs(s, theta, phi))

_normal_extended_principal_curvatures_jit = jax.jit(_normal_extended_principal_curvatures)

def _normal_extended_no_phi_principal_curvatures(data : FluxSurfaceData, settings  : FluxSurfaceSettings,  s,  theta, phi):
    s_bc, theta_bc, phi_bc = jnp.broadcast_arrays(s, theta, phi)
    return jnp.full(s_bc.shape + (2,), jnp.nan)
    
_normal_extended_no_phi_principal_curvatures_jit = jax.jit(_normal_extended_no_phi_principal_curvatures)