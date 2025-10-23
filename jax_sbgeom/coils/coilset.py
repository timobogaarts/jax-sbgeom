from .base_coil import Coil
from typing import List
import jax
import jax.numpy as jnp
from dataclasses import dataclass

from .base_coil import coil_position, coil_tangent

from .base_coil import FiniteSizeMethod

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CoilSet:
    coils : Coil
    
    @classmethod
    def from_list(cls, coils : List[Coil]):
        coils_v = jax.tree.map(lambda *xs : jnp.stack(xs), *coils)
        return cls(coils = coils_v)
    
    def position(self, s):
        return _coilset_position(self, s)
    def tangent(self, s):
        return _coilset_tangent(self, s)
    
    def position_different_s(self, s):
        return coilset_position_different_s(self, s)    
    def tangent_different_s(self, s):
        return coilset_tangent_different_s(self, s)
    

_coil_position_vmap_same_s = jax.vmap(coil_position, in_axes=(0, None))
_coil_tangent_vmap_same_s  = jax.vmap(coil_tangent, in_axes=(0, None))

@jax.jit
def _coilset_position(coilset : CoilSet, s):
    return _coil_position_vmap_same_s(coilset.coils, s)

@jax.jit
def _coilset_tangent(coilset : CoilSet, s):
    return _coil_tangent_vmap_same_s(coilset.coils, s)  

_coil_position_vmap_different_s = jax.vmap(coil_position, in_axes=(0, 0))
_coil_tangent_vmap_different_s  = jax.vmap(coil_tangent, in_axes=(0, 0))

@jax.jit
def coilset_position_different_s(coilset : CoilSet, s):
    return _coil_position_vmap_different_s(coilset.coils, s)    
@jax.jit
def coilset_tangent_different_s(coilset : CoilSet, s):
    return _coil_tangent_vmap_different_s(coilset.coils, s)

### TODO FINITE SIZE VECTORIZATION!

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CoilSetFiniteSize:
    coil : Coil
    finitesizemethod : FiniteSizeMethod

    @classmethod    
    def from_lists(cls, coils : List[Coil], finitesizemethods : List[FiniteSizeMethod]):
        coils_v = jax.tree.map(lambda *xs : jnp.stack(xs), *coils)
        finitesizemethods_v = jax.tree.map(lambda *xs : jnp.stack(xs), *finitesizemethods)
        return cls(coil = coils_v, finitesizemethod = finitesizemethods_v)
    
    def position(self, s):
        return _coilset_position(self, s)
    
    def tangent(self, s):
        return _coilset_tangent(self, s)
    
    def position_different_s(self, s):
        return coilset_position_different_s(self, s)    
    
    def tangent_different_s(self, s):
        return coilset_tangent_different_s(self, s)
    