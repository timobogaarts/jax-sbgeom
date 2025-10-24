from .base_coil import Coil
from typing import List
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Type
from .base_coil import coil_position, coil_tangent, coil_normal, coil_centre

from .base_coil import FiniteSizeMethod, FiniteSizeCoil,  _compute_radial_vector, _compute_finite_size_frame, _compute_finite_size, _setup_finitesizemethod_from_coil

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CoilSet:
    coils : Coil
    
    @classmethod
    def from_list(cls, coils : List[Coil]):
        coils_v = jax.tree.map(lambda *xs : jnp.stack(xs), *coils)
        return cls(coils = coils_v)
    
    def centre(self):
        return _coil_centre_vmap(self.coils)
    
    def position(self, s):
        return _coil_position_vmap_same_s(self.coils, s)
    def tangent(self, s):
        return _coil_tangent_vmap_same_s(self.coils, s)
    def normal(self, s):
        return _coil_normal_vmap_same_s(self.coils, s)
    
    def position_different_s(self, s):
        return _coil_position_vmap_different_s(self.coils, s)
    def tangent_different_s(self, s):
        return _coil_tangent_vmap_different_s(self.coils, s)
    def normal_different_s(self, s):
        return _coil_normal_vmap_different_s(self.coils, s)
    

_coil_centre_vmap          = jax.jit(jax.vmap(coil_centre, in_axes=(0,)))
_coil_position_vmap_same_s = jax.jit(jax.vmap(coil_position, in_axes=(0, None)))
_coil_tangent_vmap_same_s  = jax.jit(jax.vmap(coil_tangent, in_axes=(0, None)))
_coil_normal_vmap_same_s   = jax.jit(jax.vmap(coil_normal, in_axes=(0, None)))

_coil_position_vmap_different_s = jax.jit(jax.vmap(coil_position, in_axes=(0, 0)))
_coil_tangent_vmap_different_s  = jax.jit(jax.vmap(coil_tangent, in_axes=(0, 0)))
_coil_normal_vmap_different_s   = jax.jit(jax.vmap(coil_normal, in_axes=(0, 0)))

_radial_vector_same_s  = jax.jit(jax.vmap(_compute_radial_vector, in_axes=(0, 0, None)))
_finite_size_frame_same_s = jax.jit(jax.vmap(_compute_finite_size_frame, in_axes=(0, 0, None)))
_finite_size_same_s = jax.jit(jax.vmap(_compute_finite_size, in_axes=(0, 0, None, None, None)))

_radial_vector_different_s  = jax.jit(jax.vmap(_compute_radial_vector, in_axes=(0, 0, 0)))
_finite_size_frame_different_s = jax.jit(jax.vmap(_compute_finite_size_frame, in_axes=(0, 0, 0)))
_finite_size_different_s = jax.jit(jax.vmap(_compute_finite_size, in_axes=(0, 0, 0, None, None)))

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FiniteSizeCoilSet(CoilSet):
    
    @classmethod    
    def from_list(cls, coils : List[FiniteSizeCoil]):
        coils_v = jax.tree.map(lambda *xs : jnp.stack(xs), *coils)        
        return cls(coils = coils_v)
    
    @classmethod
    def from_coils(cls, coils : List[Coil], method : Type[FiniteSizeMethod], *args):
        coils_v = jax.tree.map(lambda *xs : jnp.stack(xs), *coils)
        finitesizemethod = method.setup_from_coils(coils_v, *args)
        print(finitesizemethod)
        return cls(FiniteSizeCoil(coils_v, method(*finitesizemethod)))
        

    def radial_vector(self, s):
        return _radial_vector_same_s(self.coils.coil, self.coils.finite_size_method, s)
    
    def finite_size_frame(self, s):
        return _finite_size_frame_same_s(self.coils.coil, self.coils.finite_size_method, s)
    
    def finite_size(self, s, width_radial : float, width_phi : float):
        return _finite_size_same_s(self.coils.coil, self.coils.finite_size_method, s, width_radial, width_phi)
    
    def radial_vector_different_s(self, s):
        return _radial_vector_different_s(self.coils.coil, self.coils.finite_size_method, s)
    
    def finite_size_frame_different_s(self, s):
        return _finite_size_frame_different_s(self.coils.coil, self.coils.finite_size_method, s)
    
    def finite_size_different_s(self, s, width_radial : float, width_phi : float):
        return _finite_size_different_s(self.coils.coil, self.coils.finite_size_method, s, width_radial, width_phi)   