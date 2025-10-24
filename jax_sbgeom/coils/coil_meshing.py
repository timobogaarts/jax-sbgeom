from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax 
import jax.numpy as jnp
import numpy as onp
from typing import Literal
from jax_sbgeom.jax_utils.utils import interpolate_array_modulo_broadcasted
from jax_sbgeom.flux_surfaces.flux_surface_meshing import _build_triangles_surface, _build_closed_strips
from .base_coil import FiniteSizeCoil
from functools import partial

def _mesh_finite_sized_lines_connectivity(n_samples : int, n_lines_per_coil : int, normal_orientation : bool):
    '''
    Given a set of finite sized lines, mesh the surface of the coil

    Parameters
    ----------
    n_samples : int
        Number of samples along the coil
    n_lines_per_coil : int
        Number of finite sized lines per coil
    normal_orientation : bool
        Whether to orient the normals outwards (True) or inwards (False)
    Returns
    -------
    jnp.ndarray
        Connectivity array of the meshed coil surface
    '''

    return _build_triangles_surface(n_samples, n_samples, n_lines_per_coil, n_lines_per_coil, normal_orientation)

def _mesh_rectangular_finite_sized_coils_connectivity(n_samples : int, normal_orientation : bool):
    return _mesh_finite_sized_lines_connectivity(n_samples, 4, normal_orientation)

@partial(jax.jit, static_argnums = 1)
def mesh_coil_surface(coil : FiniteSizeCoil, n_s : int, width_radial : float, width_phi : float):
    '''
    Mesh the surface of a coil

    Parameters
    ----------
    coil : Coil
        Coil to mesh
    n_s : int
        Number of samples along the coil
    method : str
        Method to use for meshing. Options are 'centroid' and 'rmf'
    kwargs : dict
        Additional arguments for the meshing method
    Returns
    -------
    jnp.ndarray
        Vertices of the meshed coil surface
    jnp.ndarray
        Connectivity array of the meshed coil surface
    '''
    finite_size_lines = coil.finite_size(jnp.linspace(0, 1.0, n_s, endpoint=False), width_radial, width_phi)    
    connectivity = _mesh_rectangular_finite_sized_coils_connectivity(n_s, normal_orientation=True)
    return finite_size_lines.reshape(-1, 3), connectivity