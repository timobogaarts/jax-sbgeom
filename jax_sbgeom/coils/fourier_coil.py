from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax 
import jax.numpy as jnp
import numpy as onp
from typing import Literal
from .base_coil import Coil
from .base_coil import _radial_vector_centroid_from_data, _frame_from_radial_vector
from .discrete_coil import DiscreteCoil
from jax_sbgeom.jax_utils.utils import stack_jacfwd
from functools import partial
from .coilset import CoilSet

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


_grad_fourier_position = jax.jit(jnp.vectorize(stack_jacfwd(_fourier_position, argnums=1), excluded=(0,), signature='()->(3)'))

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
    grad_pos = _grad_fourier_position(coil, s) # shape (..., 3)    
    tangent  = grad_pos / jnp.linalg.norm(grad_pos, axis=-1, keepdims=True)
    return tangent

_grad_grad_fourier_position = jax.jit(jnp.vectorize(stack_jacfwd(_fourier_tangent, argnums=1), excluded=(0,), signature='()->(3)'))
            

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
    tangent_deriv = _grad_grad_fourier_position(coil, s) # shape (..., 3)
    
    normal = tangent_deriv / jnp.linalg.norm(tangent_deriv, axis=-1, keepdims=True)
    return normal

   
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


#=====================================================================================================================================================================================
#                                                                           Converting curve to Fourier coefficients
#=====================================================================================================================================================================================

@partial(jax.jit, static_argnums = 1)
def _xyz_to_fourier_coefficients(positions : jnp.ndarray, n_modes : int):
    # positions is a 1D array.
    N = positions.shape[0]

    loc_fourier = jnp.fft.rfft(positions)
    loc_fourier_cos  =   jnp.real(loc_fourier[1:]) / N * 2.0
    loc_fourier_sin  = - jnp.imag(loc_fourier[1:]) / N * 2.0    
    
    centre = jnp.real(loc_fourier[0]) / N
    if N%2==0:
        # Nyquist mode only has a cosine component
        loc_fourier_cos = loc_fourier_cos.at[-1].set(loc_fourier_cos[-1] * 0.5)
    
    
    loc_fourier_cos = loc_fourier_cos[:n_modes]
    loc_fourier_sin = loc_fourier_sin[:n_modes]
    return loc_fourier_cos, loc_fourier_sin, centre
    

xyz_fourier_batched = jax.jit(jnp.vectorize(_xyz_to_fourier_coefficients, signature='(N)->(M),(M),()', excluded=(1,)))

    
@partial(jax.jit, static_argnums = 1)
def curve_to_fourier_coefficients(positions : jnp.ndarray, n_modes : int = None):

    positions_first_batch = jnp.moveaxis(positions, -1, 0)
    fourier_cos, fourier_sin, centre = xyz_fourier_batched(positions_first_batch, n_modes)

    return jnp.moveaxis(fourier_cos,0 , -1), jnp.moveaxis(fourier_sin,0, -1),  jnp.moveaxis(centre, 0, -1)

@partial(jax.jit, static_argnums = 1)
def convert_to_fourier_coil(coil : DiscreteCoil, n_modes : int = None):
    '''
    Convert a DiscreteCoil to a FourierCoil by computing Fourier coefficients from the discrete positions

    Parameters
    ----------
    coil : DiscreteCoil
        Discrete coil object
    n_modes : int
        Number of Fourier modes to use. If None, uses N/2 modes where N is the number of discrete points in the coil.

    Returns
    -------
    FourierCoil
        Fourier coil object
    '''
    fourier_cos, fourier_sin, centre = curve_to_fourier_coefficients(coil.positions, n_modes)
    return FourierCoil(fourier_cos=fourier_cos, fourier_sin=fourier_sin, centre_i=centre)

@partial(jax.jit, static_argnums = 1)
def convert_to_fourier_coilset(coilset : CoilSet, n_modes : int = None):
    '''
    Convert a DiscreteCoil to a FourierCoil by computing Fourier coefficients from the discrete positions

    Parameters
    ----------
    coil : DiscreteCoil
        Discrete coil object
    n_modes : int
        Number of Fourier modes to use. If None, uses N/2 modes where N is the number of discrete points in the coil.

    Returns
    -------
    FourierCoil
        Fourier coil object
    '''
    fourier_cos, fourier_sin, centre = curve_to_fourier_coefficients(coilset.coils.positions, n_modes)
    return CoilSet(FourierCoil(fourier_cos=fourier_cos, fourier_sin=fourier_sin, centre_i=centre), fourier_cos.shape[0])





