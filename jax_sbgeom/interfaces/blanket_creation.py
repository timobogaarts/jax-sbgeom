import jax 
import jax.numpy as jnp
from dataclasses import dataclass
from jax_sbgeom.flux_surfaces import ToroidalExtent, mesh_tetrahedra, FluxSurface, FluxSurfaceFourierExtended
from jax_sbgeom.flux_surfaces.flux_surface_meshing import _mesh_tetrahedra
from jax_sbgeom.flux_surfaces.convert_to_vmec import create_fourier_surface_extension_interp_equal_arclength, _create_fluxsurface_from_rmnc_zmns


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



def mesh_tetrahedral_blanket(flux_surface : FluxSurface, blanket : LayeredDiscreteBlanket, s_power_sampling : int):
    '''
    Create a blanket mesh using structured tetrahedral meshing.

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
                
    # for i in range(blanket.n_layers):

    return _mesh_tetrahedra(flux_surface, s_layers, True, blanket.toroidal_extent.start, blanket.toroidal_extent.end, bool(blanket.toroidal_extent.full_angle()), blanket.n_theta, blanket.n_phi)

    

