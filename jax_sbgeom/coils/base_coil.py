from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax 
import jax.numpy as jnp
import numpy as onp
from typing import Literal
from jax_sbgeom.jax_utils.utils import interpolate_array_modulo_broadcasted
from functools import partial
from typing import Type

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Coil(ABC):
    @abstractmethod 
    def position(self, s):     
        ...

    @abstractmethod
    def tangent(self, s):       
        ...

    @abstractmethod
    def centre(self):      
        ...

    @abstractmethod
    def normal(self, s):
        ...

# Functional versions for vmapping
def coil_position(coil: Coil, s):
    return coil.position(s)

def coil_tangent(coil: Coil, s):
    return coil.tangent(s)

def coil_centre(coil: Coil):
    return coil.centre()
# ===================================================================================================================================================================================
#                                                                           Finite Size Utility Methods
# ===================================================================================================================================================================================
    
# The reason this is not just a function is to allow for finite size methods that need to precompute data from the coil
# such as a RMF method
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FiniteSizeMethod(ABC):

    @classmethod
    @abstractmethod
    def from_coil(cls, coil : Coil, **kwargs):
        ...
            
    @abstractmethod
    def compute_radial_vector(self, coil : Coil, s : jnp.ndarray):        
        ...

    def compute_finite_size_frame(self, coil : Coil, s : jnp.ndarray):
        return _compute_finite_size_frame(coil, self, s)

@jax.jit
def _compute_radial_vector(coil : Coil, finitesizemethod : FiniteSizeMethod, s : jnp.ndarray):
    return finitesizemethod.compute_radial_vector(coil, s)

@jax.jit
def _compute_finite_size_frame(coil : Coil, finitesizemethod : FiniteSizeMethod, s : jnp.ndarray):    
    radial_vectors = _compute_radial_vector(coil, finitesizemethod, s)
    return _frame_from_radial_vector(coil.tangent(s), radial_vectors)


@jax.jit
def _frame_from_radial_vector(tangents, radial_vectors):
    '''
    Compute finite size frame from tangent and radial vector.

    First finite size direction is radial direction, second is phi direction (perpendicular to both tangent and radial)
    
    Parameters
    ----------
    tangents : jnp.ndarray [..., 3]
        Tangent vector(s) along the coil
    radial_vectors : jnp.ndarray [..., 3]
        Radial vector(s) along the coil 

    Returns
    -------
    jnp.ndarray [..., 2, 3]
        Finite size frame(s) along the coil
    '''
    e_R_n   = radial_vectors / jnp.linalg.norm(radial_vectors, axis=-1, keepdims=True) # ..., 3
    e_phi_n = jnp.cross(tangents, e_R_n) # ..., 3
    e_phi_n = e_phi_n / jnp.linalg.norm(e_phi_n, axis=-1, keepdims=True) # ..., 3
    return jnp.stack([e_R_n, e_phi_n], axis=-2) # ..., 2, 3

#===================================================================================================================================================================================
#                                                                           Finite Size Coil
# ===================================================================================================================================================================================
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FiniteSizeCoil():
    coil : Coil
    finite_size_method : FiniteSizeMethod

    @classmethod    
    def from_coil(cls, coil : Coil, finite_size_method : Type[FiniteSizeMethod], **kwargs):
        return cls(coil = coil, finite_size_method = finite_size_method.from_coil(coil, **kwargs))

    def position(self, s):       
        return self.coil.position(s)
    
    def tangent(self, s):        
        return self.coil.tangent(s)
    
    def centre(self):        
        return self.coil.centre()    

    def radial_vector(self, s):
        '''
        Radial vector along the coil as a function of arc length

        Uses the FiniteSizeMethod to compute the radial vector
        Uses the finite size method to compute the radial vector

        Parameters
        ----------
        s : jnp.ndarray
            Arc length(s) along the coil
        Returns
        -------
        jnp.ndarray [..., 3]
            Radial vector(s) along the coil
        '''
        return _compute_radial_vector(self.coil, self.finite_size_method, s)

    def finite_size_frame(self, s):
        '''
        Finite size frame along the coil as a function of arc length
        Uses the finite size method to compute the finite size frame.

        Parameters
        ----------
        s : jnp.ndarray
            Arc length(s) along the coil
        Returns
        -------
        jnp.ndarray [..., 2, 3]
            Finite size vector(s) along the coil
        '''
        return _compute_finite_size_frame(self.coil, self.finite_size_method, s)
    
    
    def finite_size(self, s, width_radial : float, width_phi : float):
        '''
        Finite size along the coil as a function of arc length
        Uses the finite size method to compute the finite size.

        Parameters
        ----------
        s : jnp.ndarray
            Arc length(s) along the coil
        width_radial : float
            Width in the first finite size direction
        width_phi : float
            Width in the second finite size direction
            
        Returns
        -------
        jnp.ndarray [..., 4, 3]
            Finite size vector(s) along the coil
        '''
        return _compute_finite_size(self.coil, self.finite_size_method, s, width_radial, width_phi)


@jax.jit
def _compute_finite_size(coil : Coil, finitesizemethod : FiniteSizeMethod, s : jnp.ndarray, width_radial : float, width_phi : float):
    '''
    Compute finite size along the coil as a function of arc length
    Uses the finite size method to compute the finite size.

    Parameters
    ----------
    coil : Coil
        Coil object
    finitesizemethod : FiniteSizeMethod
        Finite size method object
    s : jnp.ndarray
        Arc length(s) along the coil
    width_radial : float
        Width in the first finite size direction
    width_phi : float
        Width in the second finite size direction   
    Returns
    -------
    jnp.ndarray [..., 4, 3]
        Finite size vector(s) along the coil
    '''
    location = coil.position(s)
    frame    = finitesizemethod.compute_finite_size_frame(coil, s)
    return _finite_size_from_data(location, frame, width_radial, width_phi)


@jax.jit
def _finite_size_from_data(location, frame, width_radial : float, width_phi : float):    
    '''
    Compute the finite size vertices from location, frame and widths. 
    The frame is assumed to be orthonormal and its first index corresponds to the radial direction, second to the phi direction.

    The finite size is in the following order:
    v_0 : + radial, + phi
    v_1 : - radial, + phi
    v_2 : - radial, - phi
    v_3 : + radial, - phi
    

    Parameters
    ----------
    location : jnp.ndarray [..., 3]
        Location(s) along the coil
    frame : jnp.ndarray [..., 2, 3]
        Finite size frame(s) along the coil
    width_radial : float
        Width in the first finite size direction
    width_phi : float
        Width in the second finite size direction
    Returns
    -------
    jnp.ndarray [..., 4, 3]
        Finite size vertex locations along the coil
    '''
    v_0  = location + width_radial * frame[..., 0, :] + width_phi * frame[..., 1, :] # ..., 3
    v_1  = location - width_radial * frame[..., 0, :] + width_phi * frame[..., 1, :] # ..., 3
    v_2  = location - width_radial * frame[..., 0, :] - width_phi * frame[..., 1, :] # ..., 3
    v_3  = location + width_radial * frame[..., 0, :] - width_phi * frame[..., 1, :] # ..., 3 
    return jnp.stack([v_0, v_1, v_2, v_3], axis=-2) # ..., 4, 3
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                           Centroid
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CentroidFrame(FiniteSizeMethod):
    '''
    Finite size method using centroid frame

    Centroid frame is defined by the radial vector being the vector that is pointing from the coil centre to the coil position, projected onto the plane normal to the tangent        
    '''
    @classmethod
    def from_coil(cls, coil : Coil, **kwargs):
        return CentroidFrame()

    def compute_radial_vector(self, coil : Coil, s : jnp.ndarray):
        return _compute_radial_vector_centroid(coil, s)

@jax.jit
def _compute_radial_vector_centroid(coil : Coil, s : jnp.ndarray):
    '''
    Compute radial vector along the coil as a function of arc length using centroid method
    Uses the coils internal method to compute the radial vector (as it may be coil-type specific)
    '''
    coil_position = coil.position(s)
    coil_tangent  = coil.tangent(s)
    coil_centre   = coil.centre()
    return _radial_vector_centroid_from_data(coil_centre, coil_position, coil_tangent)

@jax.jit
def _radial_vector_centroid_from_data(coil_centre, positions, tangents):
    '''
    Compute the centroid radial vector from coil centre, position and tangent data
    This is not always desired: discrete coils have discontinuous tangents and thus discontinuous radial vectors
    Therefore, such a discrete coild should interpolate the radial vectors to have a smooth coil.
    '''
    d_i       =  positions - coil_centre # ..., 3 - 3 = ..., 3
    e_R       = d_i - jnp.einsum("...j,...i,...i->...j", tangents, d_i, tangents) # ..., 3
    return e_R / jnp.linalg.norm(e_R, axis=-1, keepdims=True) # ..., 3

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                           Frenet-Serret
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FrenetSerretFrame(FiniteSizeMethod):    
    '''
    Finite size method using Frenet-Serret frame    
    '''
    @classmethod
    def from_coil(cls, coil : Coil, **kwargs):
        return FrenetSerretFrame()
        
    def compute_radial_vector(self, coil : Coil, s : jnp.ndarray):
        return _compute_radial_vector_frenet_serret(coil, s)
 
    
@jax.jit
def _compute_radial_vector_frenet_serret(coil : Coil, s : jnp.ndarray):
    '''
    Compute radial vector along the coil as a function of arc length using Frenet-Serret method
    Uses the coils internal method to compute the radial vector (as it may be coil-type specific or not even exist)
    '''
    return coil.normal(s)
    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                           Radial Vector Frame
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RadialVectorFrame(FiniteSizeMethod):
    '''
    Finite size method using precomputed radial vectors
    It interpolates in between the radial vectors to compute the radial vector and finite size frame at arbitrary locations along the coil.
    The radial vectors are assumed to be given at uniform arc length intervals (endpoint not included).
    '''
    radial_vectors_i : jnp.ndarray

    @classmethod
    def from_coil(cls, coil : Coil, **kwargs):
        return RadialVectorFrame.from_radial_vectors(kwargs.get("radial_vectors"))

    @classmethod 
    def from_radial_vectors(cls, radial_vectors : jnp.ndarray):        
        return cls(radial_vectors_i = radial_vectors)
        
    def compute_radial_vector(self, coil : Coil, s : jnp.ndarray):
        # Coil was already used to compute radial_vectors_i
        # This assumes that this class was instantiated using from_coil with the same coil
        return _interpolate_radial_vectors(self.radial_vectors_i, s)
    
@jax.jit
def _interpolate_radial_vectors(radial_vectors_rmf, s):
    '''
    Interpolate radial vectors at arc length s.
    Uses modulo arithmic to define s > 1.0.
    '''
    return interpolate_array_modulo_broadcasted(radial_vectors_rmf, s)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                           Rotation Minimized Frame
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RotationMinimizedFrame(RadialVectorFrame):
    '''
    Finite size method using rotation minimized frame. This is a subclass of RadialVectorFrame.
    The radial vectors are computed using the rotation minimizing frame algorithm.
    '''
    @classmethod 
    def from_coil(cls, coil : Coil, number_of_rmf_samples : int):
        radial_vector = _compute_full_rmf(coil, number_of_rmf_samples)
        return cls(radial_vectors_i=radial_vector)
 

@partial(jax.jit, static_argnums=(1,))
def _compute_full_rmf(coil : Coil, number_of_rmf_samples : int):
    '''
    Compute the full rotation minimizing frame along the coil from the coil object.
    The radial vectors are computed at uniform arc length intervals (endpoint not included).
    See [1] for computation details.

    Parameters:
    ----------
    coil : Coil
        Coil object
    number_of_rmf_samples : int
        Number of samples to compute the RMF at (uniform arc length intervals, endpoint not included
    Returns
    -------
    jnp.ndarray (number_of_rmf_samples, 3)
        Rotation minimizing frame along the coil
    
    References
    ----------
    [1] Wang, Wenping, et al. "Computation of rotation minimizing frames." ACM Transactions on Graphics (TOG) 27.1 (2008): 1-18.


    '''
    s_rmf        = jnp.linspace(0.0,1.0, number_of_rmf_samples, endpoint=False)
    positions_rmf = coil.position(s_rmf)
    tangents_rmf  = coil.tangent(s_rmf)
    coil_centre   = coil.centre()
    return _rmf_radial_vector_from_data(coil_centre, positions_rmf, tangents_rmf)


@jax.jit
def _rmf_radial_vector_from_data(coil_centre, positions, tangents):
    '''
    Compute the rotation minimizing frame along the coil from position and tangent data
    
    We assume that the positions and tangents are given at uniform arc length intervals (endpoint not included).
    In other words, sampled at jnp.linspace(0.0, 1.0, n_samples, endpoint=False).

    See [1] for computation details.

    References
    ----------
    [1] Wang, Wenping, et al. "Computation of rotation minimizing frames." ACM Transactions on Graphics (TOG) 27.1 (2008): 1-18.

    Parameters
    ----------
    coil_centre : jnp.ndarray (3,)
        Centre of the coil
    positions : jnp.ndarray (n_samples, 3)
        Positions along the coil
    tangents : jnp.ndarray (n_samples, 3)
        Tangents along the coil

    Returns
    -------
    jnp.ndarray (n_samples, 2, 3)
        Rotation minimizing frame along the coil
    '''
    n_samples = positions.shape[0]    
    
    initial_vec = jnp.cross(tangents[0], positions[0] - coil_centre)

    initial_result = initial_vec / jnp.linalg.norm(initial_vec)

    def rmf_step(carry, x):
        result_prev   = carry # need this at the end
        pos_i, pos_i_p1, tan_i, tan_i_p1 = x

        v1 = pos_i_p1 - pos_i
        c1 = jnp.dot(v1, v1)

        rL_i = result_prev - 2.0 / c1 * jnp.dot(result_prev, v1)  * v1
        tL_i = tan_i       - 2.0 / c1 * jnp.dot(tan_i, v1) * v1

        v2 = tan_i_p1 - tL_i
        c2 = jnp.dot(v2, v2)
        result_i = rL_i - 2.0 / c2 * jnp.dot(rL_i, v2) * v2
        return result_i, result_i

    final_result, result_arr = jax.lax.scan(
        rmf_step, initial_result,
        (positions[:-1], positions[1:], tangents[:-1], tangents[1:]),
    )

    total_ri = jnp.concatenate([initial_result[None, :], result_arr], axis=0)  # (n_samples, 3)

    # Add a simple periodic correction to ensure that the start and end vectors are aligned    
    angle = jnp.arccos(jnp.dot(final_result, initial_result))

    tanv0 = _angle_axis_to_matrix(tangents[-1], angle)

    angle_corr       = jax.lax.cond(jnp.arccos(jnp.dot(initial_result, tanv0 @ final_result)) > angle, lambda _ : -angle, lambda _ : angle, operand = None)

    uniform_rotation = jnp.linspace(0.0, angle_corr, n_samples)
    
    angle_axis_matrices = _angle_axis_to_matrix_vmap(tangents, uniform_rotation)

    result_new = jnp.einsum("...ij,...j->...i", angle_axis_matrices, total_ri)

    return result_new


def _angle_axis_to_matrix(axis, angle):
    """
    Convert an angle-axis representation to a 3x3 rotation matrix.

    axis: shape (3,)
    angle: scalar
    """
    axis = axis / jnp.linalg.norm(axis)
    ux, uy, uz = axis

    K = jnp.array([
        [0,    -uz,   uy],
        [uz,    0,   -ux],
        [-uy,  ux,    0]
    ])

    I = jnp.eye(3)
    R = I + jnp.sin(angle) * K + (1 - jnp.cos(angle)) * (K @ K)
    return R

_angle_axis_to_matrix_vmap = jax.vmap(_angle_axis_to_matrix, in_axes=(0, 0))

# ===================================================================================================================================================================================
#                                                                           Convenience functions
# ===================================================================================================================================================================================

