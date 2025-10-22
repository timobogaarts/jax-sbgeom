from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax 
import jax.numpy as jnp
import numpy as onp
from typing import Literal
from .base_coil import Coil, _finite_size_from_data,_radial_vector_centroid_from_data, _frame_from_radial_vector
from jax_sbgeom.jax_utils.utils import interpolate_array_modulo_broadcasted, interpolate_fractions_modulo
import warnings
from functools import partial
from .base_coil import _rmf_radial_vector_from_data

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DiscreteCoil(Coil):
    positions    : jnp.ndarray  # [..., 3] Cartesian positions of discrete coil points
    _centre_i    : jnp.ndarray  # Centre of the coil: is simply the mean of the positions. This could be a cached property, but this does not play well with JAX.
        
    @classmethod 
    def from_positions(cls, positions: jnp.ndarray):
        '''
        Create a DiscreteCoil from discrete positions

        Parameters
        ----------
        positions : jnp.ndarray
            Cartesian positions of discrete coil points [..., 3]
        Returns
        -------
        DiscreteCoil
            DiscreteCoil object
        '''
        return cls(positions=positions, _centre_i = jnp.mean(positions, axis=0))

    def position(self, s):
        '''
        Position along the coil as a function of arc length

        Parameters
        ----------
        s : jnp.ndarray
            Arc length(s) along the coil
        Returns
        -------
        jnp.ndarray
            Cartesian position(s) along the coil
        '''
        return _discrete_coil_position(self, s)

    
    def tangent(self, s):
        '''
        Tangent vector along the coil as a function of arc length

        Parameters
        ----------
        s : jnp.ndarray
            Arc length(s) along the coil
        Returns
        -------
        jnp.ndarray
            Tangent vector(s) along the coil
        '''
        return _discrete_coil_tangent(self, s)
    
    def centre(self):
        '''
        Centre of the coil

        Returns
        -------
        jnp.ndarray
            Centre of the coil
        '''
        return self._centre_i
    
        
    def radial_vector(self, s, method : Literal['frenet_serret', 'centroid', 'rmf','rotated_from_centroid'], **kwargs):
        '''
        Radial vector along the coil as a function of arc length

        Parameters
        ----------
        s : jnp.ndarray
            Arc length(s) along the coil
        method : Literal['frenet_serret', 'centroid', 'rmf']
            Method to compute the radial vector
        **kwargs :
            Additional arguments for the radial vector computation.
                For 'rmf' method: number_of_rmf_samples used for interpolation
        Returns
        -------
        jnp.ndarray [..., 3]
            Radial vector(s) along the coil
        '''
        if method == "centroid":    
            return _discrete_coil_radial_vector_centroid(self, s)
        elif method == "frenet_serret":
            return _discrete_coil_radial_vector_frenet_serret(self, s)
        elif method == "rmf":
            warnings.warn("A rotation-minimized-frame is only defined over the entire coil. This routine prcomputes it for the entire coil and interpolates it. This may be inefficient if only a few s values are requested.", RuntimeWarning)
            return _discrete_coil_radial_vector_rmf(self, s, number_of_rmf_samples=kwargs.get("number_of_rmf_samples", 1000))
        else:
            raise NotImplementedError(f"Radial vector method '{method}' not implemented for DiscreteCoil.")
    
    def finite_size_frame(self, s, method : Literal['frenet_serret', 'centroid', 'rotated_from_centroid'], **kwargs):
        '''
        Finite size frame along the coil as a function of arc length

        Parameters
        ----------
        s : jnp.ndarray
            Arc length(s) along the coil
        method : Literal['frenet_serret', 'centroid', 'rmf']
            Method to compute the finite size frame
        **kwargs :
            Additional arguments for the finite size frame computation.
                For 'rmf' method: number_of_rmf_samples used for interpolation
        Returns
        -------
        jnp.ndarray [..., 2, 3]
            Finite size frame vector(s) along the coil
        '''
        return _discrete_coil_finite_size_frame(self, s, method, **kwargs)
         
    def finite_size(self, s, width_radial, width_tangent, method, **kwargs):
        '''
        Finite size along the coil as a function of arc length

        Parameters
        ----------
        s : jnp.ndarray
            Arc length(s) along the coil
        width_radial : float
            Width in the radial direction
        width_tangent : float
            Width in the tangent direction
        method : Literal['frenet_serret', 'centroid', 'rotated_from_centroid']
            Method to compute the finite size frame
        **kwargs :
            Additional arguments for the finite size frame computation. Used for the number of rmf samples for the full coil when the method is rmf. Default is 1000.
        Returns
        -------
        jnp.ndarray [..., 4, 3]
            Finite size vector(s) along the coil
        '''
        return _discrete_coil_finite_size(self, s, width_radial, width_tangent, method, **kwargs)
    
# ===================================================================================================================================================================================
#                                                                           Implementation
# ===================================================================================================================================================================================
@jax.jit
def _discrete_coil_discrete_position(discrete_coil : DiscreteCoil,  index):
    return discrete_coil.positions[index % discrete_coil.positions.shape[0]]

@jax.jit
def _discrete_coil_discrete_tangent(discrete_coil : DiscreteCoil, index):    
    i1      = index + 1
    pos_i0  = discrete_coil.positions[index % discrete_coil.positions.shape[0]]
    pos_i1  = discrete_coil.positions[i1 %    discrete_coil.positions.shape[0]]
    tangent = pos_i1 - pos_i0
    tangent = tangent / jnp.linalg.norm(tangent, axis=-1, keepdims=True)
    return tangent

@jax.jit
def _discrete_coil_position(discrete_coil : DiscreteCoil,  s):
    return interpolate_array_modulo_broadcasted(discrete_coil.positions, s)

@jax.jit
def _discrete_coil_tangent(discrete_coil : DiscreteCoil, s):
    i0, i1, ds = interpolate_fractions_modulo(s, discrete_coil.positions.shape[0])
    pos_i0 = _discrete_coil_discrete_position(discrete_coil, i0)
    pos_i1 = _discrete_coil_discrete_position(discrete_coil, i1)
    tangent = pos_i1 - pos_i0
    tangent = tangent / jnp.linalg.norm(tangent, axis=-1, keepdims=True)
    return tangent
   
# ===================================================================================================================================================================================
#                                                                           Finite Sizes
# ===================================================================================================================================================================================

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                           Centroids
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


@jax.jit 
def _discrete_coil_radial_vector_centroid_index(discrete_coil : DiscreteCoil,  index):    
    '''
    
    Internal function to find the centroid radial vector at a discrete coil index

    Parameters
    ----------
    discrete_coil : DiscreteCoil
        Discrete coil object
    index : jnp.ndarray
        Discrete coil indexs
    Returns
    -------
    jnp.ndarray [..., 3]
        Radial vector(s) at the discrete coil indexes
    '''
    pos_i      = _discrete_coil_discrete_position(discrete_coil, index)
    tangent_i  = _discrete_coil_discrete_tangent(discrete_coil, index)
    return _radial_vector_centroid_from_data(discrete_coil._centre_i, pos_i, tangent_i)

@jax.jit
def _discrete_coil_radial_vector_centroid(discrete_coil : DiscreteCoil, s):
    ''' 
    Internal function to find the centroid radial vector at arc length s

    Interpolates radial vectors at surrounding data points

    Parameters
    ----------
    discrete_coil : DiscreteCoil
        Discrete coil object
    s : jnp.ndarray
        Arc length(s) along the coil
    Returns
    -------
    jnp.ndarray [..., 3]
        Radial vector(s) along the coil
    '''
    i0, i1, ds = interpolate_fractions_modulo(s,  discrete_coil.positions.shape[0])      
    radial_i0   = _discrete_coil_radial_vector_centroid_index(discrete_coil, i0)
    radial_i1   = _discrete_coil_radial_vector_centroid_index(discrete_coil, i1)
    return radial_i0 * (1.0 - ds)[..., jnp.newaxis] + radial_i1 * ds[..., jnp.newaxis]

@jax.jit
def _discrete_coil_finite_size_frame_centroid(discrete_coil : DiscreteCoil, s):
    '''
    Compute finite size frame at a location s along the discrete coil, using centroid method

    Centroid frames at the discrete points are interpolated.

    Parameters
    ----------
    discrete_coil : DiscreteCoil
        Discrete coil object
    s : jnp.ndarray 
        Arc length(s) along the coil
    
    Returns
    -------
    jnp.ndarray [..., 2, 3]
        Finite size frame(s) along the coil
    '''
    i0, i1, ds      = interpolate_fractions_modulo(s, discrete_coil.positions.shape[0])  
    radial_vector_i0 = _discrete_coil_radial_vector_centroid_index(discrete_coil, i0)
    radial_vector_i1 = _discrete_coil_radial_vector_centroid_index(discrete_coil, i1)

    tangent_i0      = _discrete_coil_discrete_tangent(discrete_coil, i0)
    tangent_i1      = _discrete_coil_discrete_tangent(discrete_coil, i1)    

    frame_0         = _frame_from_radial_vector(tangent_i0, radial_vector_i0)
    frame_1         = _frame_from_radial_vector(tangent_i1, radial_vector_i1)    
    
    return (1 - ds)[..., jnp.newaxis, jnp.newaxis] * frame_0 + ds[..., jnp.newaxis, jnp.newaxis] * frame_1


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                           Frenet-Serret
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@jax.jit
def _discrete_coil_radial_vector_frenet_serret(discrete_coil : DiscreteCoil, s):
    '''
    Internal function to find the frenet-serret radial vector at arc length s

    Not valid for discrete coils due to vanishing curvature between the data points

    Parameters
    ----------
    discrete_coil : DiscreteCoil
        Discrete coil object
    s : jnp.ndarray
        Arc length(s) along the coil
    Returns
    -------
    jnp.ndarray [..., 3]
        Radial vector(s) along the coil (jnp.nan)


    '''
    warnings.warn("Frenet-Serret frame is ill-defined for DiscreteCoil due to zero curvature. Returning NaN. ", RuntimeWarning)            
    return jnp.full(s.shape + (3,), jnp.nan)


@jax.jit
def _discrete_coil_finite_size_frame_frenet_serret(discrete_coil : DiscreteCoil, s):
    '''
    Internal function to find the frenet-serret finite size frame at arc length s
    Not valid for discrete coils due to vanishing curvature between the data points

    Parameters
    ----------
    discrete_coil : DiscreteCoil
        Discrete coil object
    s : jnp.ndarray
        Arc length(s) along the coil    

    Returns
    -------
    jnp.ndarray [..., 2, 3]
        Finite size frame(s) along the coil (jnp.nan)
    '''
    return jnp.full(s.shape + (2,3), jnp.nan)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                           RMF
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@partial(jax.jit, static_argnums = (1,))
def _discrete_coil_compute_full_coil_rmf(discrete_coil : DiscreteCoil, number_of_samples : int):
    '''
    Internal function to compute the full rotation minimized frame along the discrete coil

    Parameters
    ----------
    discrete_coil : DiscreteCoil
        Discrete coil object
    number_of_samples : int
        Number of samples to use for precomputing the RMF frame
    Returns
    -------
    jnp.ndarray (number_of_samples, 3)
        Rotation minimized frame along the coil
    '''
    s_samples = jnp.linspace(0.0, 1.0, number_of_samples, endpoint=False)  # (n_s,)
    positions    = _discrete_coil_position(discrete_coil, s_samples)  # (n_s, 3)
    tangents     = _discrete_coil_tangent(discrete_coil, s_samples)  # (n_s, 3)
    coil_centre = discrete_coil._centre_i # (3,)    
    return _rmf_radial_vector_from_data(coil_centre, positions, tangents) # (n_s, 3)


@partial(jax.jit, static_argnums = (2,))
def _discrete_coil_radial_vector_rmf(discrete_coil : DiscreteCoil, s, number_of_rmf_samples : int):    
    '''
    Internal function to find the rmf radial vector at arc length s
    Note that a rotation minimized frame is only defined over the entire coil. This routine prcomputes it for the entire coil and interpolates it. 
    This may be inefficient if only a few s values are requested. 

    This supports vectorized s in any dimension.

    Parameters
    ----------
    discrete_coil : DiscreteCoil
        Discrete coil object
    s : jnp.ndarray 
        Arc length(s) along the coil    
    number_of_rmf_samples : int
        Number of samples to use for precomputing the RMF frame
    Returns
    -------
    jnp.ndarray [s.shape, 3]
        Radial vector(s) along the coil
    '''
    rmf_frame = _discrete_coil_compute_full_coil_rmf(discrete_coil, number_of_rmf_samples)  # (n_s, 3, 3)    
    return interpolate_array_modulo_broadcasted(rmf_frame, s)  # (..., 3, 3)


@partial(jax.jit, static_argnums = (2,))
def _discrete_coil_finite_size_frame_rmf(discrete_coil : DiscreteCoil, s, number_of_rmf_samples : int):
    '''
    Internal function to find the rmf finite size frame at arc length s 

    Note that a rotation minimized frame is only defined over the entire coil. This routine prcomputes it for the entire coil and interpolates it. 
    This may be inefficient if only a few s values are requested.

    Parameters
    ----------
    discrete_coil : DiscreteCoil
        Discrete coil object
    s : jnp.ndarray
        Arc length(s) along the coil    
    number_of_rmf_samples : int
        Number of samples to use for precomputing the RMF frame
    Returns
    -------
    jnp.ndarray [..., 2, 3]
        Finite size frame(s) along the coil
    '''
    radial_vectors = _discrete_coil_radial_vector_rmf(discrete_coil, s, number_of_rmf_samples)
    tangents       = _discrete_coil_tangent(discrete_coil, s)    
    return _frame_from_radial_vector(tangents, radial_vectors)



# ===================================================================================================================================================================================
#                                                                           Convenience functions
# ===================================================================================================================================================================================
def _discrete_coil_finite_size_frame(discrete_coil : DiscreteCoil, s, method : str, **kwargs):        
    '''
    Convenience function to compute finite size frame at a location s along the discrete coil, using specified method
    Parameters
    ----------
    discrete_coil : DiscreteCoil    
        Discrete coil object
    s : jnp.ndarray 
        Arc length(s) along the coil
    method : str
        Method to compute the finite size frame. One of 'frenet_serret', 'centroid', 'rmf'
    **kwargs : dict
        Additional arguments for the finite size frame computation.
        Used for the number of rmf samples for the full coil when the method is rmf. Default is 1000.
    Returns
    -------
    jnp.ndarray [..., 2, 3]
        Finite size frame(s) along the coil
    '''
    
    # This *cannot* be jitted because of the **kwargs
    if method == "centroid":    
        return _discrete_coil_finite_size_frame_centroid(discrete_coil, s)
    elif method == "frenet_serret":
        return _discrete_coil_finite_size_frame_frenet_serret(discrete_coil, s)
    elif method == "rmf":
        return _discrete_coil_finite_size_frame_rmf(discrete_coil, s, kwargs.get("number_of_rmf_samples", 1000))
    else:
        raise NotImplementedError(f"Finite size method '{method}' not implemented for _discrete_coil_finite_size_frame")

def _discrete_coil_finite_size(discrete_coil : DiscreteCoil, s, width_radial, width_tangent, method : str, **kwargs):
    '''
    Convenience function to compute finite size at a location s along the discrete coil, using specified method 

    Parameters
    ----------
    discrete_coil : DiscreteCoil    
        Discrete coil object
    s : jnp.ndarray 
        Arc length(s) along the coil
    width_radial : float
        Width in the radial direction
    width_tangent : float
        Width in the tangent direction
    method : str
        Method to compute the finite size frame. One of 'frenet_serret', 'centroid', 'rmf'
    **kwargs : dict 
        Additional arguments for the finite size frame computation.
        Used for the number of rmf samples for the full coil when the method is rmf. Default is 1000.
    Returns
    -------
    jnp.ndarray [..., 4, 3]
        Finite size vector(s) along the coil
    '''
    finite_size_frame = _discrete_coil_finite_size_frame(discrete_coil, s, method, **kwargs)    
    location          = _discrete_coil_position(discrete_coil, s)                  # ..., 3
    return _finite_size_from_data(location, finite_size_frame, width_radial, width_tangent) # ..., 4, 3


    
