from . import FluxSurfaceNormalExtendedNoPhi

import jax.numpy as jnp
from warnings import warn
from jax_sbgeom.jax_utils.raytracing import find_minimum_distance_to_mesh
import equinox as eqx

@eqx.filter_jit
def generate_thickness_matrix(flux_surface : FluxSurfaceNormalExtendedNoPhi, mesh, n_theta : int, n_phi : int):
    '''
    Generate thickness matrix of an external mesh with respect to a no-phi extended flux surface.

    Uses the internal raytracing utilities to compute the minimum distance from the flux surface to the mesh along the normal directions.

    Parameters:
    ----------
    flux_surface : FluxSurfaceNormalExtendedNoPhi
        Flux surface to compute thickness from.
    mesh : Tuple[jnp.ndarray, jnp.ndarray]
        Mesh of the external object (vertices, connectivity).
    n_theta : int
        Number of poloidal points.
    n_phi : int
        Number of toroidal points.  
    Returns:
    -------
    theta : jnp.ndarray [n_theta, n_phi]
        Poloidal angles of the thickness matrix.
    phi : jnp.ndarray   [n_theta, n_phi]
        Toroidal angles of the thickness matrix.
    dmesh : jnp.ndarray [n_theta, n_phi]
        Thickness matrix values.

    '''
    if not isinstance(flux_surface, FluxSurfaceNormalExtendedNoPhi):
        warn("in generate_thickness_matrix, expected as type FluxSurfaceNormalExtendedNoPhi, but got type: " + str(type(flux_surface)) + ". Results may be incorrect as this does not "
        "guarantee a straight line as extension", RuntimeWarning)
    theta = jnp.linspace(0, 2 * jnp.pi, n_theta)
    phi   = jnp.linspace(0, 2 * jnp.pi / flux_surface.settings.nfp, n_phi)
    theta, phi = jnp.meshgrid(theta, phi, indexing='ij')
    positions_lcfs_mg  = flux_surface.cartesian_position(1.0,  theta, phi)
    directions_lcfs_mg = flux_surface.cartesian_position(2.0, theta, phi) - positions_lcfs_mg
    dmesh = find_minimum_distance_to_mesh(positions_lcfs_mg, directions_lcfs_mg, mesh) 
    return theta, phi, dmesh