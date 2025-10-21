from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax 
import jax.numpy as jnp
import numpy as onp
from typing import Literal
from .base_coil import Coil, _finite_size_frame_centroid_from_data, _finite_size_from_data

from jax_sbgeom.jax_utils.utils import interpolate_array_modulo_broadcasted, interpolate_fractions_modulo

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
        return _discrete_coil_position(self.positions, s)

    
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
        return _discrete_coil_tangent(self.positions, s)
    
    def centre(self):
        '''
        Centre of the coil

        Returns
        -------
        jnp.ndarray
            Centre of the coil
        '''
        return self._centre_i
    
# ===================================================================================================================================================================================
#                                                                           Implementation
# ===================================================================================================================================================================================
@jax.jit
def _discrete_coil_discrete_position(positions,  index):
    return positions[index % positions.shape[0]]

@jax.jit
def _discrete_coil_discrete_tangent(positions, index):    
    i1      = index + 1
    pos_i0  = positions[index % positions.shape[0]]
    pos_i1  = positions[i1 % positions.shape[0]]
    tangent = pos_i1 - pos_i0
    tangent = tangent / jnp.linalg.norm(tangent, axis=-1, keepdims=True)
    return tangent

@jax.jit
def _discrete_coil_position(positions,  s):
    return interpolate_array_modulo_broadcasted(positions, s)

@jax.jit
def _discrete_coil_tangent(positions, s):
    i0, i1, ds = interpolate_fractions_modulo(s, positions.shape[0])
    pos_i0 = _discrete_coil_discrete_position(positions, i0)
    pos_i1 = _discrete_coil_discrete_position(positions, i1)
    tangent = pos_i1 - pos_i0
    tangent = tangent / jnp.linalg.norm(tangent, axis=-1, keepdims=True)

    return tangent


def _discrete_coil_finite_size_frame_centroid(positions, centre, s):
    i0, i1, ds = interpolate_fractions_modulo(s, positions.shape[0])
    
    pos_i0 = _discrete_coil_discrete_position(positions, i0)
    pos_i1 = _discrete_coil_discrete_position(positions, i1)

    tangent_i0 = _discrete_coil_discrete_tangent(positions, i0)
    tangent_i1 = _discrete_coil_discrete_tangent(positions, i1)    

    frame_i0   = _finite_size_frame_centroid_from_data(centre, pos_i0, tangent_i0)
    frame_i1   = _finite_size_frame_centroid_from_data(centre, pos_i1, tangent_i1)        

    return frame_i0 * (1.0 - ds)[..., jnp.newaxis, jnp.newaxis] + frame_i1 * ds[..., jnp.newaxis, jnp.newaxis]