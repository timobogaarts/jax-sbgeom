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
from .coilset import FiniteSizeCoilSet

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

@partial(jax.jit, static_argnums = (0,1,2,3))
def _mesh_rectangular_finite_sized_coilset_connectivity(n_coils, n_samples : int, n_lines_per_coil : int, normal_orientation : bool):
    connectivity_base = _mesh_finite_sized_lines_connectivity(n_samples, n_lines_per_coil, normal_orientation)
    offsets = jnp.arange(n_coils) * (n_samples * n_lines_per_coil)    
    return (connectivity_base[None, :, :] + offsets[:, None, None]).reshape(-1, connectivity_base.shape[1])


def mesh_coilset_surface(coils : FiniteSizeCoilSet, n_s : int, width_radial : float, width_phi : float):
    '''
    Mesh the surface of a coilset

    The coils vertices are originally:
    [n_coils, n_s, 4, 3] (4 lines per coil)
    
    The coils connectivity is originally:
    [n_coils, n_s, 4, 2, 3] (4 lines per coil, 2 triangles per quad)
    Both are reshaped to (-1,3) to facilate easier post processing.
    

    Parameters
    ----------
    coil : Coil
        Coil to mesh
    n_s : int
        Number of samples along the coil
    width_radial : float
        Radial width of the finite sized coil
    width_phi : float
        Toroidal width of the finite sized coil
    Returns
    -------
    jnp.ndarray
        Vertices of the meshed coil surface
    jnp.ndarray
        Connectivity array of the meshed coil surface
    '''
    finite_size_lines = coils.finite_size(jnp.linspace(0, 1.0, n_s, endpoint=False), width_radial, width_phi)            
    connectivity = _mesh_rectangular_finite_sized_coilset_connectivity(coils.n_coils, n_s, 4, True)
    return finite_size_lines.reshape(-1, 3), connectivity