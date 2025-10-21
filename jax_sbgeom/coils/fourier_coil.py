from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax 
import jax.numpy as jnp
import numpy as onp
from typing import Literal
from .base_coil import Coil

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FourierCoil(Coil):
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

    
    def FiniteSizeFrame(self, s, method : Literal['frenet_serret', 'centroid', 'rmf', 'rotated_from_centroid'], **kwargs):
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