from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax 
import jax.numpy as jnp
import numpy as onp
from typing import Literal
from .base_coil import Coil
from .base_coil import _radial_vector_centroid_from_data, _frame_from_radial_vector

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FourierCoil(Coil):
    fourier_cos : jnp.ndarray 
    fourier_sin : jnp.ndarray
    centre_i    : jnp.ndarray  

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
        return _fourier_position(self, s)
    
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
        return _fourier_tangent(self, s)
    
    def normal(self, s):
        '''
        Normal vector along the coil as a function of arc length

        Parameters
        ----------
        s : jnp.ndarray
            Arc length(s) along the coil
        Returns
        -------
        jnp.ndarray
            Normal vector(s) along the coil
        '''
        return _fourier_normal(self, s)

    def centre(self):
        return self.centre_i
    
    def _radial_vector_centroid(self, s):
        return _fourier_coil_radial_vector_centroid(self, s)
    
    def _finite_size_frame_centroid(self, s):
        return _fourier_coil_finite_size_frame_centroid(self, s)
    
    def _radial_vector_frenet_serret(self, s):
        return _fourier_coil_radial_vector_frenet_serret(self, s)

    def _finite_size_frame_frenet_serret(self, s):
        return _fourier_coil_finite_size_frame_frenet_serret(self, s)

@jax.jit
def _fourier_position(coil : FourierCoil, s):
    '''
    Position along the coil as a function of arc length

    Parameters
    ----------
    coil : FourierCoil
        Coil object
    s : jnp.ndarray
        Arc length(s) along the coil
    Returns
    -------
    jnp.ndarray
        Cartesian position(s) along the coil
    '''
    n_modes = coil.fourier_cos.shape[-2]
    n    = jnp.arange(1.0, n_modes + 1.0) * 2 * jnp.pi # shape (N_modes,)
    
    # The final shape should be (fourier_coil_batch_dimensions, s_shape, 3)
    final_shape = jnp.array(s).shape + (3,)
    initial_sum = jnp.zeros(final_shape)

    def fourier_sum(vals, i):
        xyz = vals # shape (..., s_shape, 3)
        
        angle_cos = jnp.cos(n[i] * s) # shape (s_shape,)
        angle_sin = jnp.sin(n[i] * s) # shape (s_shape,)
        xyz = xyz + coil.fourier_cos[i, :] * angle_cos[..., None] + coil.fourier_sin[ i, :] * angle_sin[..., None]
        return xyz, None
    
    # Fourier_cos is shape (..., N_modes, 3) where ... are batch dimensions
    # so we need to create an output shape of (..., s_shape, 3)
    xyz = jax.lax.scan(fourier_sum, initial_sum, jnp.arange(n_modes))[0]
    return xyz + coil.centre_i 

@jax.jit
def _fourier_tangent(coil : FourierCoil, s):
    '''
    Tangent vector along the coil as a function of arc length

    Parameters
    ----------
    coil : FourierCoil
        Coil object
    s : jnp.ndarray
        Arc length(s) along the coil
    Returns
    -------
    jnp.ndarray
        Tangent vector(s) along the coil
    '''
    n_modes = coil.fourier_cos.shape[-2]
    n    = jnp.arange(1.0, n_modes + 1.0) * 2 * jnp.pi    

    final_shape = jnp.array(s).shape + (3,)
    initial_sum = jnp.zeros(final_shape)

    def fourier_sum(vals, i):
        xyz = vals
        angle_cos = jnp.cos(n[i] * s)
        angle_sin = jnp.sin(n[i] * s)
        xyz = xyz + (- coil.fourier_cos[i, :] * n[i] * angle_sin[..., None] + coil.fourier_sin[i, :] * n[i] * angle_cos[..., None])
        return xyz, None
    
    xyz = jax.lax.scan(fourier_sum, initial_sum, jnp.arange(n_modes))[0]

    return xyz / jnp.linalg.norm(xyz, axis=-1, keepdims=True)

@jax.jit
def _fourier_normal(coil : FourierCoil, s):
    '''
    Normal vector along the coil as a function of arc length

    Parameters
    ----------
    coil : FourierCoil
        Coil object
    s : jnp.ndarray
        Arc length(s) along the coil
    Returns
    -------
    jnp.ndarray
        Normal vector(s) along the coil
    '''
    n_modes = coil.fourier_cos.shape[-2]
    n    = jnp.arange(1.0, n_modes + 1.0) * 2 * jnp.pi    

    final_shape = jnp.array(s).shape + (3,)
    initial_sum = jnp.zeros(final_shape)

    def fourier_sum(vals, i):
        xyz = vals
        angle_cos = jnp.cos(n[i] * s)
        angle_sin = jnp.sin(n[i] * s)
        xyz = xyz + (- coil.fourier_cos[i, :] * (n[i]**2) * angle_cos[..., None] - coil.fourier_sin[i, :] * (n[i]**2) * angle_sin[..., None])
        return xyz, None
    
    xyz = jax.lax.scan(fourier_sum, initial_sum,  jnp.arange(n_modes))[0]

    return xyz / jnp.linalg.norm(xyz, axis=-1, keepdims=True)
            

   
# ===================================================================================================================================================================================
#                                                                           Finite Sizes
# ===================================================================================================================================================================================

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                           Centroids
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@jax.jit 
def _fourier_coil_radial_vector_centroid(coil : FourierCoil, s):
    '''
    
    Internal function to find the centroid radial vector at arc length s

    Parameters
    ----------
    coil : FourierCoil
        Coil object
    s : jnp.ndarray
        Arc length(s) along the coil
    Returns
    -------
    jnp.ndarray [..., 3]
        Radial vector(s) along the coil
    '''
    pos_i      = _fourier_position(coil, s)
    tangent_i  = _fourier_tangent(coil, s)
    return _radial_vector_centroid_from_data(coil.centre_i, pos_i, tangent_i)

@jax.jit
def _fourier_coil_finite_size_frame_centroid(coil : FourierCoil, s):
    '''
    Internal function to find the centroid finite size frame at arc length s

    Parameters
    ----------
    coil : FourierCoil
        Coil object
    s : jnp.ndarray
        Arc length(s) along the coil
    Returns
    -------
    jnp.ndarray [..., 3, 3]
        Finite size frame(s) along the coil
    '''
    radial_vector = _fourier_coil_radial_vector_centroid(coil, s)
    tangent       = _fourier_tangent(coil, s)
    return _frame_from_radial_vector(tangent, radial_vector)


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                           Frenet-Serret
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@jax.jit 
def _fourier_coil_radial_vector_frenet_serret(coil : FourierCoil, s):
    '''
    
    Internal function to find the Frenet-Serret radial vector at arc length s

    Parameters
    ----------
    coil : FourierCoil
        Coil object
    s : jnp.ndarray
        Arc length(s) along the coil
    Returns
    -------
    jnp.ndarray [..., 3]
        Radial vector(s) along the coil
    '''            
    return _fourier_normal(coil, s)

@jax.jit
def _fourier_coil_finite_size_frame_frenet_serret(coil : FourierCoil, s):
    '''
    Internal function to find the Frenet-Serret finite size frame at arc length s

    Parameters
    ----------
    coil : FourierCoil
        Coil object
    s : jnp.ndarray
        Arc length(s) along the coil
    Returns
    -------
    jnp.ndarray [..., 3, 3]
        Finite size frame(s) along the coil
    '''
    radial_vector = _fourier_coil_radial_vector_frenet_serret(coil, s)
    tangent       = _fourier_tangent(coil, s)
    return _frame_from_radial_vector(tangent, radial_vector)