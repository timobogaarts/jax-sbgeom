import jax 
import jax.numpy as jnp
from dataclasses import dataclass
from jax_sbgeom.flux_surfaces import ToroidalExtent, mesh_tetrahedra, FluxSurface, FluxSurfaceFourierExtended, ParametrisedSurface
from jax_sbgeom.flux_surfaces.flux_surface_meshing import _mesh_tetrahedra, mesh_watertight_layers

from functools import cached_property

import equinox as eqx
from abc import abstractmethod

# This provides a interface for creating blanket geometries around flux surfaces

class LayeredBlanket(eqx.Module):
    '''
    Class representing a layered blanket structure around a flux surface.

    Attributes
    ----------
    d_layers : tuple
        Tuple of radial distances defining the layers of the blanket. These can be shaped differently.
    n_layers : int
        Number of layers in the blanket
    '''
    d_layers : tuple    

    @property
    def n_layers(self):
        return len(self.d_layers)
    

class BlanketMeshStructure(eqx.Module):
    '''
    Class representing the structure of a volume blanket mesh.

    Has several convenience functions to slice the blanket and functions 
    defined on the blanket.
    '''

    n_theta : int
    '''
    Number of poloidal points in the blanket mesh. 
    '''
    n_phi   : int
    '''
    Number of toroidal points in the blanket mesh. 
    '''
    n_s     : int
    '''
    Number of discrete layers in the blanket mesh. 
    '''
    include_axis : bool
    '''
    Whether the axis is included in the blanket mesh. 
    '''
    full_angle   : bool
    '''
    Whether the blanket mesh covers a full torus. 
    '''

    @property 
    def n_phi_blocks(self):
        return jnp.where(self.full_angle, self.n_phi, self.n_phi - 1)
    @property 
    def n_theta_blocks(self):
        return self.n_theta     
    
    @property
    def n_layered_blocks(self):
        return self.n_s - 1
    
    def _norm_negative(self, layer_i : int):
        return jnp.where(layer_i < 0, self.n_layered_blocks + layer_i , layer_i)
    
    def _n_tet_per_block(self, layer_i : int):
        layer_i_mod = self._norm_negative(layer_i)
        return jnp.where(jnp.logical_and(layer_i_mod == 0, self.include_axis), 3, 6 )
    
    def n_blocks_in_layer(self, layer_i : int):
        layer_i_mod = self._norm_negative(layer_i)
        return jnp.where(jnp.logical_and(layer_i_mod == 0, self.include_axis), 3 * self.n_theta_blocks * self.n_phi_blocks, 6 * self.n_theta_blocks * self.n_phi_blocks)                    
    
    def layer_start(self, layer_i : int):
        layer_i_mod = self._norm_negative(layer_i)
        
        return jnp.where(self.include_axis, 3 * self.n_theta_blocks * self.n_phi_blocks * (layer_i_mod > 0) + 6 * self.n_theta_blocks * self.n_phi_blocks * (layer_i_mod - 1) * (layer_i_mod > 0), 0)
    
    def layer_slice(self, layer_i : int):    
        return slice(self.layer_start(layer_i), self.layer_start(layer_i) + self.n_blocks_in_layer(layer_i))
        
    def reshape_to_layer(self, layer_i : int, arr : jnp.ndarray):
        '''
        Reshapes a flat array of shape (n_elements,) to the shape of the blocks in the given layer, which is (n_phi_blocks, n_theta_blocks, n_tet_per_block).
        This n_tet_per_block is 3 for the first layer if the axis is included, and 6 otherwise. For all other layers, it is 6.
        '''
        return arr[self.layer_slice(layer_i)].reshape(self.n_phi_blocks, self.n_theta_blocks, self._n_tet_per_block(layer_i))
    
    def reshape_without_axis(self, arr : jnp.ndarray):
        '''
        Reshapes a flat array of shape (n_elements,) to the shape of the blocks in the blanket, which is (n_layered_blocks, n_phi_blocks, n_theta_blocks, n_tet_per_block).
    
        Discards the first layer if an axis is present and then reshapes the last.        
        '''  
        if self.include_axis:
            return arr[self.layer_start(1):].reshape(self.n_layered_blocks - 1, self.n_phi_blocks, self.n_theta_blocks, 6)      
        else:
            return arr.reshape(self.n_layered_blocks, self.n_phi_blocks, self.n_theta_blocks, 6)        
    
    def map_radial_array_to_layers(self, arr : jnp.ndarray):
        '''
        Maps a flat array of shape (n_s,) to the shape of the layers in the blanket, which is (n_layered_blocks,).
        This is useful for mapping a radial function defined on the layers to the blocks in the blanket (e.g. materials)
        '''
        assert arr.shape == (self.n_layered_blocks,), f"Input array has shape {arr.shape}, but expected shape is {(self.n_layered_blocks,)}"

        return jnp.repeat(arr, self.n_blocks_in_layer(jnp.arange(self.n_layered_blocks)), axis=0)
    
    @property 
    def n_elements(self):        
        return jnp.where(self.include_axis, 6 * self.n_theta_blocks * self.n_phi_blocks * (self.n_s - 2) + 3 * self.n_theta_blocks * self.n_phi_blocks, 6 * self.n_theta_blocks * self.n_phi_blocks * (self.n_s - 1))
    
    @property 
    def n_points(self):
        return jnp.where(self.include_axis, self.n_theta * self.n_phi * (self.n_s - 1) + self.n_phi, self.n_theta * self.n_phi * self.n_s)

class LayeredDiscreteBlanket(LayeredBlanket):
    '''
    Class representing a layered, structured, discrete blanket structure around a flux surface.
    '''

    n_theta            : int
    '''
    Number of poloidal points in the blanket mesh. 
    '''
    
    n_phi              : int
    '''
    Number of toroidal points in the blanket mesh. 
    '''

    resolution_layers  : tuple
    '''
    Tuple of number of discrete layers in each layer of the blanket. The total number of discrete layers is given by the sum of the entries in this tuple. The length of this tuple should be equal to n_layers.
    '''

    toroidal_extent    : ToroidalExtent
    '''
    Toroidal extent of the blanket. This can be a full torus, a half module, etc. depending on the application.
    '''
    
    def __check_init__(self) :
        if len(self.resolution_layers) != self.n_layers:
            raise ValueError(f"Length of resolution_layers {len(self.resolution_layers)} does not match number of layers {self.n_layers} defined by self.d_layers")
    
    @property 
    def n_discrete_layers(self):
        return jnp.sum(jnp.array(self.resolution_layers))
    
    @abstractmethod
    def volume_mesh(parametrised_surface : ParametrisedSurface):
        ...

    @abstractmethod
    def surface_mesh(parametrised_surface : ParametrisedSurface):
        ...
    
    @property
    @abstractmethod
    def volume_mesh_structure(self) -> BlanketMeshStructure:
        ...

    @property
    def layer_array(self):
        '''
        Maps a flat array of shape (n_elements,) to the shape of the layers in the blanket, which is (n_layered_blocks,).
        This is useful for mapping a radial function defined on the layers to the blocks in the blanket (e.g. materials)
        '''
        return self.volume_mesh_structure.map_radial_array_to_layers(jnp.repeat(jnp.arange(self.n_layers), jnp.array(self.resolution_layers)))


class LayeredDiscreteBlanketTransformed(LayeredDiscreteBlanket):
    '''
    Class representing a layered discrete blanket structure around a flux surface, where the flux surface is transformed such that s = 1.0 corresponds to the innermost layer of the blanket, s = 2.0 to the first external layer, etc.

    No volume mesh is assumed; a choice has to be made whether to include the plasma
    '''
        
    def surface_mesh(self, parametrised_surface : ParametrisedSurface):
        return mesh_watertight_layers(parametrised_surface, 2.0 + jnp.arange(len(self.d_layers)), self.toroidal_extent, self.n_theta, self.n_phi)
    

class LayeredDiscreteBlanketPlasmaTransformed(LayeredDiscreteBlanketTransformed):
    '''
    Class representing a layered discrete blanket structure around a flux surface, where the flux surface is transformed such that s = 1.0 corresponds to the innermost layer of the blanket, s = 2.0 to the first external layer, etc.

    Until s=1, it is meshed with a power-law spacing. This is done with the number of points specified in the first entry of resolution_layers (i.e., resolution_layers[0] - 1 layers).
    Then, the first external layer is placed immediately afterwards. Therefore, the total number of element layers until the first external layer is resolution_layers[0].
    The number of elements in the first external layer is resolution_layers[1], and so on.

    .. warning::
        It does not use the blanket.d_layers data: it assumes the flux surface is already transformed such that s = 2.0 is equal to the first layer, s=3.0 to the second, etc.
    '''
    s_power_sampling : int = 2

    def volume_mesh(self, parametrised_surface : ParametrisedSurface):
        return mesh_tetrahedral_blanket_transformed_axis(parametrised_surface, self, self.s_power_sampling)

    @property
    def volume_mesh_structure(self) -> BlanketMeshStructure:
        n_s = self.n_discrete_layers + 1 
        return BlanketMeshStructure(n_theta=self.n_theta, n_phi=self.n_phi, n_s = n_s, include_axis=True, full_angle=self.toroidal_extent.full_angle())

    @property
    def s_spacing(self):
        return _compute_s_spacing_transformed_axis(self, self.s_power_sampling)
    
    
    
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DiscreteFiniteSizeCoilSet:
    '''
    Class representing a set of discrete finite size coils forming a coilset for blanket creation.

    Attributes
    -------     
    n_points_per_coil : int
        Number of discrete points per coil
    toroidal_extent : ToroidalExtent
        Toroidal extent of the coilset
    width_R : float
        Radial width of the finite sized coils
    width_phi : float
        Toroidal width of the finite sized coils
    '''
    n_points_per_coil : int
    toroidal_extent   : ToroidalExtent
    width_R : float
    width_phi : float     

def _compute_layer_slice(n_discrete_layers : int, n_theta : int, n_phi : int, layer_i : int):
    if layer_i < 0:
        layer_i = n_discrete_layers + layer_i
    if layer_i >= n_discrete_layers:
        raise ValueError(f"Layer {layer_i} is out of bounds for {n_discrete_layers} layers.")
    layer_wedge = 3 * n_theta * ( n_phi - 1)
    layer_else = 6 *  n_theta  * (n_phi - 1)
    dedge = 0
    if layer_i == 0:  
        layer_blocks = slice(0, layer_wedge)
    else:
        layer_blocks = slice(layer_wedge + (layer_i - 1) * layer_else + dedge, layer_wedge +  layer_i * layer_else - dedge)
    return layer_blocks

def _compute_s_spacing_transformed_axis(blanket : LayeredDiscreteBlanket, s_power_sampling : int):
    '''
    Computes the s spacing for the layered discrete blanket in s for in [0, 1 + n_layers].

    Parameters
    ----------
    blanket : LayeredDiscreteBlanket
        The layered discrete blanket to compute the spacing for.
    s_power_sampling : int
        The power to which the radial coordinate s is raised when sampling.
        Higher values lead to more points near the inner layers.
    Returns
    -------
    jnp.ndarray
        The s spacing for the layered discrete blanket. 

    '''
    inner_blanket_spacing = jnp.linspace(0.0, 1.0, blanket.resolution_layers[0]) ** s_power_sampling
    s_layers              = jnp.concatenate([inner_blanket_spacing, jnp.concatenate([jnp.linspace(2.0 + i, 3.0 + i, blanket.resolution_layers[i  + 1], endpoint=False) for i in range(blanket.n_layers - 1)], axis=0), jnp.array([1.0 + blanket.n_layers])])                 
    return s_layers

def compute_d_spacing(blanket  : LayeredDiscreteBlanket, s_power_sampling : int):
    '''
    Computes the d spacing for the layered discrete blanket in d for in [0, d_layers[-1]].

    Parameters
    ----------
    blanket : LayeredDiscreteBlanket
        The layered discrete blanket to compute the spacing for.
    s_power_sampling : int
        The power to which the radial coordinate s is raised when sampling.
        Higher values lead to more points near the inner layers.
    Returns
    -------
    jnp.ndarray
        The d spacing for the layered discrete blanket.
    '''
    inner_blanket_spacing = jnp.linspace(0.0, 1.0, blanket.resolution_layers[0]) ** s_power_sampling    
    d_layers              = jnp.concatenate([inner_blanket_spacing, jnp.concatenate([jnp.linspace(1.0 + blanket.d_layers[i], 1.0 + blanket.d_layers[i+1], blanket.resolution_layers[i  + 1], endpoint=False) for i in range(blanket.n_layers - 1)], axis=0), jnp.array([1.0 + blanket.d_layers[-1]])])                 
    return d_layers

def mesh_tetrahedral_blanket_transformed_axis(flux_surface : FluxSurface, blanket : LayeredDiscreteBlanket, s_power_sampling : int):
    '''
    Create a blanket mesh using structured tetrahedral meshing. It assumes that the flux surface is already transformed such that s = 1.0 corresponds to the innermost layer of the blanket, s = 2.0 to the first external layer, etc.
    The axis is included.

    Until s=1, it is meshed with a power-law spacing. This is done with the number of points specified in the first entry of resolution_layers (i.e., resolution_layers[0] - 1 layers).
    Then, the first external layer is placed immediately afterwards. Therefore, the total number of element layers until the first external layer is resolution_layers[0].
    The number of elements in the first external layer is resolution_layers[1], and so on.

    .. warning::
        It does not use the blanket.d_layers data: it assumes the flux surface is already transformed such that s = 2.0 is equal to the first layer, s=3.0 to the second, etc.

    Parameters
    ----------
    blanket : LayeredDiscreteBlanket
        The layered discrete blanket to mesh.
    s_power_sampling : int
        The power to which the radial coordinate s is raised when sampling.
        Higher values lead to more points near the inner layers.
    Returns
    -------
    nodes : jax.numpy.ndarray
        The nodes of the tetrahedral mesh.
    connectivity : jax.numpy.ndarray
        The connectivity of the tetrahedral mesh.
    '''
    
    s_layers = _compute_s_spacing_transformed_axis(blanket, s_power_sampling)
    return _mesh_tetrahedra(flux_surface, s_layers, True, blanket.toroidal_extent.start, blanket.toroidal_extent.end, bool(blanket.toroidal_extent.full_angle()), blanket.n_theta, blanket.n_phi)

    

