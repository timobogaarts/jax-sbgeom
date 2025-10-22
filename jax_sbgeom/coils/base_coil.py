from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax 
import jax.numpy as jnp
import numpy as onp
from typing import Literal
from jax_sbgeom.jax_utils.utils import interpolate_array_modulo_broadcasted

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

    @abstractmethod
    def radial_vector(self, s, method : Literal['frenet_serret', 'centroid', 'rotated_from_centroid'], **kwargs):
        '''
        Radial vector along the coil as a function of arc length

        Parameters
        ----------
        s : jnp.ndarray
            Arc length(s) along the coil
        Returns
        -------
        jnp.ndarray [..., 3]
            Radial vector(s) along the coil
        '''
        raise NotImplementedError("Radial vector method not implemented for base Coil class.")

    @abstractmethod
    def finite_size_frame(self, s, method : Literal['frenet_serret', 'centroid', 'rotated_from_centroid'], **kwargs):
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
        raise NotImplementedError("Finite size frame method not implemented for base Coil class.")
    
    @abstractmethod
    def finite_size(self, s, width_radial, width_tangent, method : Literal['frenet_serret', 'centroid', 'rotated_from_centroid'], **kwargs):
        '''
        Finite size along the coil as a function of arc length
        Parameters
        ----------
        s : jnp.ndarray
            Arc length(s) along the coil
        width_0 : float
            Width in the first finite size direction
        width_1 : float
            Width in the second finite size direction
        method : Literal['frenet_serret', 'centroid', 'rmf', 'rotated_from_centroid']
            Method to compute the finite size frame
        **kwargs : dict
            Additional arguments for the finite size frame method
            
        Returns
        -------
        jnp.ndarray [..., 4, 3]
            Finite size vector(s) along the coil
        '''
        raise NotImplementedError("Finite size frame method not implemented for base Coil class.")



    # def finite_size_lines(self, width_0, width_1, n_per_line : int, method : Literal['frenet_serret', 'centroid', 'rmf', 'rotated_from_centroid'], **kwargs):
    #     '''
    #     Generate lines along the finite size of the coil

    #     Note that the rotation-minimized frame is not defined for a single s

    #     '''
    #     if method == "rmf":
        
        

    def mesh_triangles(self, width_0, width_1, n_per_line : int, method : Literal['frenet_serret', 'centroid', 'rmf', 'rotated_from_centroid'], **kwargs):
        '''
        Generate triangular mesh of the finite size coil surface

        '''
        pass


# ===================================================================================================================================================================================
#                                                                           Finite Size
# ===================================================================================================================================================================================
def _finite_size_from_data(location, frame, width_radial : float, width_phi : float):    
    v_0  = location + width_radial * frame[..., 0, :] + width_phi * frame[..., 1, :] # ..., 3
    v_1  = location - width_radial * frame[..., 0, :] + width_phi * frame[..., 1, :] # ..., 3
    v_2  = location + width_radial * frame[..., 0, :] - width_phi * frame[..., 1, :] # ..., 3 
    v_3  = location - width_radial * frame[..., 0, :] - width_phi * frame[..., 1, :] # ..., 3
    return jnp.stack([v_0, v_1, v_2, v_3], axis=-2) # ..., 4, 3


def _frame_from_radial_vector(tangents, radial_vectors):
    e_R_n   = radial_vectors / jnp.linalg.norm(radial_vectors, axis=-1, keepdims=True) # ..., 3
    e_phi_n = jnp.cross(tangents, e_R_n) # ..., 3
    e_phi_n = e_phi_n / jnp.linalg.norm(e_phi_n, axis=-1, keepdims=True) # ..., 3
    return jnp.stack([e_R_n, e_phi_n], axis=-2) # ..., 2, 3

# =================================================================================================================================================================================
#                                                                           Centroids
# =================================================================================================================================================================================

@jax.jit
def _radial_vector_centroid_from_data(coil_centre, positions, tangents):
    d_i       =  positions - coil_centre # ..., 3 - 3 = ..., 3
    e_R       = d_i - jnp.einsum("...j,...i,...i->...j", tangents, d_i, tangents) # ..., 3
    return e_R

# =================================================================================================================================================================================
#                                                                           RMF
# =================================================================================================================================================================================

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


# ===================================================================================================================================================================================
#                                                                           Convenience functions
# ===================================================================================================================================================================================

