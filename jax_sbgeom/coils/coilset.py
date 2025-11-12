from .base_coil import Coil
from typing import List
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Type
from .base_coil import coil_position, coil_tangent, coil_normal, coil_centre, ensure_coil_rotation

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
    
    def __getitem__(self, idx):
        return jax.tree.map(lambda x: x[idx], self.coils)
    
    @property
    def n_coils(self):
        return jax.tree_util.tree_flatten(self.coils)[0][0].shape[0]

                

_coil_centre_vmap               = jax.jit(jax.vmap(coil_centre, in_axes=(0,)))
_coil_position_vmap_same_s      = jax.jit(jax.vmap(coil_position, in_axes=(0, None)))
_coil_tangent_vmap_same_s       = jax.jit(jax.vmap(coil_tangent, in_axes=(0, None)))
_coil_normal_vmap_same_s        = jax.jit(jax.vmap(coil_normal, in_axes=(0, None)))

_coil_position_vmap_different_s = jax.jit(jax.vmap(coil_position, in_axes=(0, 0)))
_coil_tangent_vmap_different_s  = jax.jit(jax.vmap(coil_tangent, in_axes=(0, 0)))
_coil_normal_vmap_different_s   = jax.jit(jax.vmap(coil_normal, in_axes=(0, 0)))

_ensure_coilset_rotation_vmap   = jax.jit(jax.vmap(ensure_coil_rotation, in_axes=(0, None)))

_radial_vector_same_s           = jax.jit(jax.vmap(_compute_radial_vector, in_axes=(0, 0, None)))
_finite_size_frame_same_s       = jax.jit(jax.vmap(_compute_finite_size_frame, in_axes=(0, 0, None)))
_finite_size_same_s             = jax.jit(jax.vmap(_compute_finite_size, in_axes=(0, 0, None, None, None)))

_radial_vector_different_s      = jax.jit(jax.vmap(_compute_radial_vector, in_axes=(0, 0, 0)))
_finite_size_frame_different_s  = jax.jit(jax.vmap(_compute_finite_size_frame, in_axes=(0, 0, 0)))
_finite_size_different_s        = jax.jit(jax.vmap(_compute_finite_size, in_axes=(0, 0, 0, None, None)))

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
        return cls(FiniteSizeCoil(coils_v, method(*finitesizemethod)))
    
    @classmethod
    def from_coilset(cls, coilset : CoilSet, method : Type[FiniteSizeMethod], *args):
        finitesizemethod = method.setup_from_coils(coilset.coils, *args)
        return cls(FiniteSizeCoil(coilset.coils, method(*finitesizemethod)))

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
    



@jax.jit
def order_coilset_phi(coilset : CoilSet):
    '''
    Orders a CoilSet in increasing toroidal angle (phi). Works with both CoilSet and FiniteSizeCoilSet.

    Parameters:
        coilset (CoilSet) : CoilSet to order
    Returns:
        CoilSet           : ordered Coil_Set       
    '''

    phis = jnp.arctan2(coilset.centre()[:,1], coilset.centre()[:,0])
    
    permutation = jnp.argsort(phis)
    new_coils = jax.tree.map(lambda x : jnp.take(x,permutation, axis=0), coilset.coils)

    return type(coilset)(coils=new_coils)
    

def ensure_coilset_rotation(coilset : CoilSet, positive_rotation : bool):
    '''
    Ensures that all coils in a CoilSet are defined in the same direction.

    Parameters:
        coilset (CoilSet) : CoilSet to ensure rotation
    Returns:
        CoilSet           : CoilSet with all coils rotation
    '''
    return type(coilset)(_ensure_coilset_rotation_vmap(coilset.coils, positive_rotation))

def filter_coilset(coilset : CoilSet, mask):
    return type(coilset)(jax.tree.map(lambda x : x[mask], coilset.coils))

def filter_coilset_phi(coilset : CoilSet, phi_min : float, phi_max : float):
    '''
    Filters a CoilSet to only include coils with centre phi between phi_min and phi_max.

    Parameters:
        coilset (CoilSet) : CoilSet to filter
        phi_min (float)   : minimum phi
        phi_max (float)   : maximum phi
    Returns:
        CoilSet           : filtered CoilSet       
    '''
    phis = jnp.arctan2(coilset.centre()[:,1], coilset.centre()[:,0])
    mask = (phis >= phi_min) & (phis <= phi_max)
    return filter_coilset(coilset, mask)