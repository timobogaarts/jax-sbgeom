from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax 
import jax.numpy as jnp
import numpy as onp
from typing import Literal
from jax_sbgeom.jax_utils.utils import interpolate_array_modulo_broadcasted
from .base_coil import Coil


# def _mesh_finite_sized_lines_connectivity(n_samples, n_lines_per_coil):
#     '''
#     Given a set of finite sized lines, mesh the surface of the coil

#     Parameters
#     ----------
#     finite_lines : jnp.ndarray (n_samples, n_lines_per_section, 3, 3)
#         Finite sized lines along the coil
#     '''
    

# def mesh_coil_surface(coil : Coil, n_s : int, )