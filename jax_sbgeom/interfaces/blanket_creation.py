import jax 
import jax.numpy as jnp
from dataclasses import dataclass
from jax_sbgeom.flux_surfaces import ToroidalExtent, mesh_tetrahedra, FluxSurface, FluxSurfaceFourierExtended
from jax_sbgeom.flux_surfaces.flux_surface_meshing import _mesh_tetrahedra

from functools import cached_property
# This provides a interface for creating blanket geometries around flux surfaces

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LayeredBlanket:
    d_layers : tuple    

    @property
    def n_layers(self):
        return len(self.d_layers)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LayeredDiscreteBlanket(LayeredBlanket):
    n_theta            : int
    n_phi              : int
    resolution_layers  : tuple
    toroidal_extent    : ToroidalExtent
    
    def __post_init__(self) :
        if len(self.resolution_layers) != self.n_layers:
            raise ValueError(f"Length of resolution_layers {len(self.resolution_layers)} does not match number of layers {self.n_layers}.")
    
    @property 
    def n_discrete_layers(self):
        return jnp.sum(jnp.array(self.resolution_layers))
    
    def layer_slice(self, layer_index : int):
        return _compute_layer_slice(self.n_discrete_layers, self.n_theta, self.n_phi, layer_index)
    

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


def mesh_tetrahedral_blanket(flux_surface : FluxSurface, blanket : LayeredDiscreteBlanket, s_power_sampling : int):
    '''
    Create a blanket mesh using structured tetrahedral meshing.

    Until s=1, it is meshed with a power-law spacing. This is done with the number of points specified in the first entry of resolution_layers (i.e., resolution_layers[0] - 1 layers).
    Then, the first external layer is placed immediately afterwards. Therefore, the total number of element layers until the first external layer is resolution_layers[0].
    The number of elements in the first external layer is resolution_layers[1], and so on.

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
    elements : jax.numpy.ndarray
        The elements of the tetrahedral mesh.=
    '''

    inner_blanket_spacing = jnp.linspace(0.0, 1.0, blanket.resolution_layers[0]) ** s_power_sampling
    s_layers              = jnp.concatenate([inner_blanket_spacing, jnp.concatenate([jnp.linspace(2.0 + i, 3.0 + i, blanket.resolution_layers[i  + 1], endpoint=False) for i in range(blanket.n_layers - 1)], axis=0), jnp.array([1.0 + blanket.n_layers])])                 
    return _mesh_tetrahedra(flux_surface, s_layers, True, blanket.toroidal_extent.start, blanket.toroidal_extent.end, bool(blanket.toroidal_extent.full_angle()), blanket.n_theta, blanket.n_phi)

    

