import jax.numpy as jnp
import h5py 
import jax
import numpy as onp
from dataclasses import dataclass
from jax_sbgeom.jax_utils.utils import stack_jacfwd
from functools import partial
from .flux_surfaces_base import FluxSurface, ToroidalExtent, FluxSurfaceSettings, FluxSurfaceData, _data_modes_settings_from_hdf5, _cartesian_to_cylindrical, _principal_curvatures_interpolated, _cylindrical_position_interpolated, _cylindrical_to_cartesian

from .flux_surfaces_base import _cartesian_position_interpolated, _normal_interpolated
import equinox as eqx

@jax.tree_util.register_dataclass
@dataclass(frozen =  True)
class FluxSurfaceNormalExtended(FluxSurface):

    def cartesian_position(self, s, theta, phi):
        return _normal_extended_cartesian_position(self, s, theta, phi)
    
    def cylindrical_position(self, s, theta, phi):
        return _normal_extended_cylindrical_position(self, s, theta, phi)
    
      # For a normal extended flux surface, the normal *remains the same* in the extended region
    def normal(self, s, theta, phi):        
        return _normal_extended_normal(self, s, theta, phi)
    
    
    def principal_curvatures(self, s, theta, phi):        
        return _normal_extended_principal_curvatures(self, s, theta, phi)


@jax.tree_util.register_dataclass
@dataclass(frozen =  True)
class FluxSurfaceNormalExtendedNoPhi(FluxSurface):

    def cartesian_position(self, s, theta, phi):
        return _normal_extended_no_phi_cartesian_position(self, s, theta, phi)
    
    def cylindrical_position(self, s, theta, phi):
        return _normal_extended_no_phi_cylindrical_position(self, s, theta, phi)
    
    # For a normal extended flux surface, the normal *remains the same* in the extended region
    def normal(self, s, theta, phi):
        return _normal_extended_no_phi_normal(self, s, theta, phi)
    
    # Principal curvatures could be implemented in the extension region using 
    def principal_curvatures(self, s, theta, phi):
        return _normal_extended_no_phi_principal_curvatures(self, s, theta, phi)

@jax.tree_util.register_dataclass
@dataclass(frozen =  True)
class FluxSurfaceNormalExtendedConstantPhi(FluxSurface):

    def cartesian_position(self, s, theta, phi):
        return _normal_extended_constant_phi_cartesian_position(self, s, theta, phi)
    
    def cylindrical_position(self, s, theta, phi):        
        return _normal_extended_constant_phi_cylindrical_position(self, s, theta, phi)
    
    # For a normal extended flux surface, the normal *remains the same* in the extended region
    def normal(self, s, theta, phi):        
        return _normal_extended_constant_phi_normal(self, s, theta, phi)
    
    # Principal curvatures could be implemented in the extension region using 
    def principal_curvatures(self, s, theta, phi):        
        return _normal_extended_constant_phi_principal_curvatures(self, s, theta, phi)

@jax.tree_util.register_dataclass
@dataclass(frozen =  True)
class FluxSurfaceFourierExtended(FluxSurface):
    '''
    A flux surface that is extended using another flux surface defined in Fourier space.
    This does not necessarily have to have the same mpol & ntor as the inner flux surface.
    
    The inner flux surface is used for s <= 1.0, and the extension flux surface is used for s > 1.0.

    s = 1.0 corresponds to the LCFS of the inner surface
    s = 2.0 corresponds to the first surface of the extension surface
    etc.. 
    s = n_extension + 1.0 corresponds to the last surface of the extension surface
    '''
    extension_flux_surface : FluxSurface = None

    @classmethod
    def from_flux_surface_and_extension(cls, flux_surface : FluxSurface, extension_flux_surface : FluxSurface):
        return cls(data = flux_surface.data, modes = flux_surface.modes, settings = flux_surface.settings, extension_flux_surface = extension_flux_surface)
    
    def cartesian_position(self, s, theta, phi):
        return _fourier_extended_cartesian_position(self, self.extension_flux_surface, s, theta, phi)
    
    def cylindrical_position(self, s, theta, phi):
        return _fourier_extended_cylindrical_position(self, self.extension_flux_surface, s, theta, phi)
    
    # def normal(self, s, theta, phi):
    #     return _fourier_extended_normal(self, self.extension_flux_surface.data, self.extension_flux_surface.settings, s, theta, phi)
    
    # def principal_curvatures(self, s, theta, phi):
    #     return _fourier_extended_principal_curvatures(self, self.extension_flux_surface.data, self.extension_flux_surface.settings, s, theta, phi)

# ===================================================================================================================================================================================
#                                                                           Normal Extended
# ===================================================================================================================================================================================

@eqx.filter_jit
def _normal_extended_cartesian_position(flux_surface : FluxSurface,  s,  theta, phi):    
    positions = _cartesian_position_interpolated(flux_surface, jnp.minimum(s, 1.0), theta, phi) 
    normals   = _normal_interpolated(flux_surface, 1.0, theta, phi) # this will not give NaNs, as s=1.0 is always on the surface (non axis)    
    distance_1d = jnp.maximum(s - 1.0, 0.0)
    # We have to ensure that both do not produce nan values. 
    # This is the case, as the positions are evaluated at s <= 1.0 and normals at s = 1.0    
    return positions + normals * distance_1d[..., None]

@eqx.filter_jit
def _normal_extended_cylindrical_position(flux_surface : FluxSurface,  s,  theta, phi):
    return _cartesian_to_cylindrical(_normal_extended_cartesian_position(flux_surface, s, theta, phi))

@eqx.filter_jit
def _normal_extended_normal(flux_surface : FluxSurface,  s,  theta, phi):
    return _normal_interpolated(flux_surface, jnp.minimum(s, 1.0), theta, phi)

@eqx.filter_jit
def _normal_extended_principal_curvatures(flux_surface : FluxSurface,  s,  theta, phi):
    curvatures = _principal_curvatures_interpolated(flux_surface, jnp.minimum(s, 1.0), theta, phi)
    d = jnp.maximum(s - 1.0, 0.0)        
    gamma_0 = jnp.where(1 + curvatures[...,0] * d >= 0.0, jnp.ones_like(d), jnp.ones_like(d) * -1.0) # the >= or > is arbitrary
    gamma_1 = jnp.where(1 + curvatures[...,1] * d >= 0.0, jnp.ones_like(d), jnp.ones_like(d) * -1.0) # the >= or > is arbitrary        
    kappa_0 = curvatures[...,0] / jnp.abs(1.0 + curvatures[...,0] * d) * gamma_0
    kappa_1 = curvatures[...,1] / jnp.abs(1.0 + curvatures[...,1] * d) * gamma_1
    return jnp.stack([kappa_0, kappa_1], axis=-1)


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

@eqx.filter_jit
def _normal_extended_no_phi_cartesian_position(flux_surface : FluxSurface,  s,  theta, phi):
    positions                = _cartesian_position_interpolated(flux_surface, jnp.minimum(s, 1.0), theta, phi)
    normals                  = _normal_interpolated(flux_surface, 1.0, theta, phi)
    hat_phi                  = _hat_phi(positions)
    
    phi_component            = jnp.einsum("...i,...i->...", normals, hat_phi)
    
    normal_no_phi            = normals - phi_component[..., None] * hat_phi
    normal_no_phi_normalised = normal_no_phi / jnp.linalg.norm(normal_no_phi, axis=-1, keepdims=True)    
    
    distance_1d              = jnp.maximum(s - 1.0, 0.0)

    return positions + normal_no_phi_normalised * distance_1d[..., None]

@eqx.filter_jit
def _normal_extended_no_phi_cylindrical_position(flux_surface : FluxSurface,  s,  theta, phi):
    return _cartesian_to_cylindrical(_normal_extended_no_phi_cartesian_position(flux_surface, s, theta, phi))

@eqx.filter_jit
def _normal_extended_no_phi_normal(flux_surface : FluxSurface,  s,  theta, phi):
    s_bc, theta_bc, phi_bc = jnp.broadcast_arrays(s, theta, phi)
    return jnp.full(s_bc.shape + (3,), jnp.nan)

@eqx.filter_jit
def _normal_extended_no_phi_principal_curvatures(flux_surface : FluxSurface,  s,  theta, phi):
    s_bc, theta_bc, phi_bc = jnp.broadcast_arrays(s, theta, phi)
    return jnp.full(s_bc.shape + (2,), jnp.nan)

_normal_extended_no_phi_dx_dtheta = jax.jit(jnp.vectorize(jax.jacfwd(_normal_extended_no_phi_cartesian_position, argnums=3), excluded=(0,1), signature='(),(),()->(3)'))

@eqx.filter_jit
def __normal_extended_no_phi_arc_length_theta(flux_surface : FluxSurface, s, theta, phi):
    return jnp.linalg.norm(_normal_extended_no_phi_dx_dtheta(flux_surface, s, theta, phi), axis=-1)




# ===================================================================================================================================================================================
#                                                                          Constant Phi
# ===================================================================================================================================================================================
def _distance_between_angles(angle1, angle2):
    return jnp.arctan2(jnp.sin(angle1 - angle2), jnp.cos(angle1 - angle2))

def _distance_between_phi_phi_desired(flux_surface, s, theta, phi, x):
    positions =  _normal_extended_cartesian_position(flux_surface, s, theta, x)
    return _distance_between_angles(jnp.arctan2(positions[...,1], positions[...,0]), phi)

@eqx.filter_jit
def _normal_extended_constant_phi_find_phi(flux_surface, s, theta, phi, n_iter : int = 5):
    assert n_iter >= 1, "n_iter must be at least 1"

    _, _, phi_bc = jnp.broadcast_arrays(s, theta, phi)
        
    x_minus_two = phi_bc + 1e-3
    x_minus_one = phi_bc
    
    f_minus_two = _distance_between_phi_phi_desired(flux_surface, s, theta, phi, x_minus_two)

    def secant_iteration(i, vals):
        x_minus_two, x_minus_one, f_minus_two = vals
        f_minus_one = _distance_between_phi_phi_desired(flux_surface, s, theta, phi, x_minus_one)

        x_new = x_minus_one - f_minus_one * (x_minus_one - x_minus_two) / (f_minus_one - f_minus_two + 1e-16)
        return (x_minus_one, x_new, f_minus_one)

    x_final = jax.lax.fori_loop(0, n_iter, secant_iteration, (x_minus_two, x_minus_one, f_minus_two))[1]
    return x_final

@eqx.filter_jit
def _normal_extended_constant_phi_cartesian_position(flux_surface : FluxSurface,  s,  theta, phi, n_iter : int = 5):
    phi_c = _normal_extended_constant_phi_find_phi(flux_surface, s, theta, phi, n_iter)    
    return _normal_extended_cartesian_position(flux_surface, s, theta, phi_c)

@eqx.filter_jit
def _normal_extended_constant_phi_cylindrical_position(flux_surface : FluxSurface,  s,  theta, phi, n_iter : int = 5):
    return _cartesian_to_cylindrical(_normal_extended_constant_phi_cartesian_position(flux_surface, s, theta, phi, n_iter))

@eqx.filter_jit
def _normal_extended_constant_phi_normal(flux_surface : FluxSurface,  s,  theta, phi, n_iter : int = 5):
    phi_c = _normal_extended_constant_phi_find_phi(flux_surface, s, theta, phi, n_iter)    
    return _normal_extended_normal(flux_surface, s, theta, phi_c)

@eqx.filter_jit
def _normal_extended_constant_phi_principal_curvatures(flux_surface : FluxSurface,  s,  theta, phi, n_iter : int = 5):
    phi_c = _normal_extended_constant_phi_find_phi(flux_surface, s, theta, phi, n_iter)    
    return _normal_extended_principal_curvatures(flux_surface, s, theta, phi_c)


# ===================================================================================================================================================================================
#                                                                          Fourier Extended
# ===================================================================================================================================================================================


@eqx.filter_jit
def _fourier_extended_cylindrical_position(flux_surface : FluxSurface, extension : FluxSurface, s, theta, phi):
    # This is not necessarily completely efficient: but we cannot avoid evaluating both positions in batched operations. 
    n_surf_extension    = jnp.maximum(extension.data.Rmnc.shape[0], 2) # if there's only one extension surface this ensures we get a valid result (s=0.5 interpolation on the extension is just the surface itself anyway)

    inner_positions     = _cylindrical_position_interpolated(flux_surface, jnp.minimum(s, 1.0), theta, phi) 
    d_value             = jnp.maximum(s - 1.0, 0.0)
    d_value_extension   = jnp.maximum(d_value - 1.0, 0.0)
    normalized_d_value  = d_value_extension / (n_surf_extension - 1.0)

    extension_positions    = _cylindrical_position_interpolated(extension, normalized_d_value , theta, phi)
    extension_positions_d0 = _cylindrical_position_interpolated(extension, jnp.zeros_like(s) , theta, phi)    
    only_extension         = jnp.array(s >=2.0)
    
    return jnp.where(only_extension[..., None], extension_positions,
                     inner_positions + (extension_positions_d0- inner_positions) * d_value[..., None])

@eqx.filter_jit
def _fourier_extended_cartesian_position(flux_surface : FluxSurface, extension : FluxSurface, s, theta, phi):    
    return _cylindrical_to_cartesian(_fourier_extended_cylindrical_position(flux_surface, extension, s, theta, phi))
    


    





