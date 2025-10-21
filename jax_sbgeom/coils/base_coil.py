from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax 
import jax.numpy as jnp
import numpy as onp
from typing import Literal

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Coil(ABC):


    @abstractmethod 
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def centre(self):
        '''
        Centre of the coil

        Returns
        -------
        jnp.ndarray (3,)
            Centre of the coil
        '''
        pass


    
    def finite_size_frame(self, s, method : Literal['frenet_serret', 'centroid', 'rmf', 'rotated_from_centroid'], **kwargs):
        '''
        Finite size frame along the coil as a function of arc length

        Parameters
        ----------
        s : jnp.ndarray
            Arc length(s) along the coil
        Returns
        -------
        jnp.ndarray [..., 2, 3]
            Finite size vector(s) along the coil
        '''
        pass
    
    
    def finite_size(self, s, width_0, width_1, method : Literal['frenet_serret', 'centroid', 'rmf', 'rotated_from_centroid'], **kwargs):
        '''
        Finite size along the coil as a function of arc length
        Parameters
        ----------
        s : jnp.ndarray
            Arc length(s) along the coil
        Returns
        -------
        jnp.ndarray [..., 2, 3]
            Finite size vector(s) along the coil
        '''
    
        pass


    def finite_size_lines(self, width_0, width_1, n_per_line : int, method : Literal['frenet_serret', 'centroid', 'rmf', 'rotated_from_centroid'], **kwargs):
        '''
        Generate lines along the finite size of the coil

        '''
        pass

    def mesh_triangles(self, width_0, width_1, n_per_line : int, method : Literal['frenet_serret', 'centroid', 'rmf', 'rotated_from_centroid'], **kwargs):
        '''
        Generate triangular mesh of the finite size coil surface

        '''
        pass


# ===================================================================================================================================================================================
#                                                                           Finite Size
# ===================================================================================================================================================================================
def _finite_size_from_data(location, frame, width_0 : float, width_1 : float):    
    v_0  = location + width_0 * frame[..., 0, :] + width_1 * frame[..., 1, :] # ..., 3
    v_1  = location - width_0 * frame[..., 0, :] + width_1 * frame[..., 1, :] # ..., 3
    v_2  = location + width_0 * frame[..., 0, :] - width_1 * frame[..., 1, :] # ..., 3 
    v_3  = location - width_0 * frame[..., 0, :] - width_1 * frame[..., 1, :] # ..., 3
    return jnp.stack([v_0, v_1, v_2, v_3], axis=-2) # ..., 4, 3

# =================================================================================================================================================================================
#                                                                           Centroids
# =================================================================================================================================================================================
@jax.jit
def _finite_size_frame_centroid_from_data(coil_centre, positions, tangents):
    d_i       =  positions - coil_centre # ..., 3 - 3 = ..., 3
    
    e_R       = d_i - jnp.einsum("...j,...i,...i->...j", tangents, d_i, tangents) # ..., 3
    e_phi     = jnp.cross(tangents, e_R) # ..., 3
    
    e_R_n     = e_R / jnp.linalg.norm(e_R, axis=-1, keepdims=True) # ..., 3
    e_phi_n   = e_phi / jnp.linalg.norm(e_phi, axis=-1, keepdims=True) # ..., 3
    return jnp.stack([e_R_n, e_phi_n], axis=-2) # ..., 2, 3





# ===================================================================================================================================================================================
#                                                                           Convenience functions
# ===================================================================================================================================================================================
def _finite_size_from_frame(coil, s, width_0 : float, width_1 : float, frame):    
    location = coil.position(s)                  # ..., 3    
    return _finite_size_from_data(location, frame, width_0, width_1)



@jax.jit
def _finite_size_frame_centroid(coil, s):
    position = coil.position(s) # ..., 3
    tangent  = coil.tangent(s)
    return _finite_size_frame_centroid_from_data(coil.centre(), position, tangent)



