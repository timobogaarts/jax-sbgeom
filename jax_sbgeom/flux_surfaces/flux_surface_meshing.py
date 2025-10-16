from .flux_surfaces_base import FluxSurface, FluxSurfaceSettings, ToroidalExtent

import jax
import jax.numpy as jnp
from functools import partial

from typing import List
# ===================================================================================================================================================================================
#                                                                           Triangle Meshing
# ===================================================================================================================================================================================

@partial(jax.jit, static_argnums = (0,1,2,3))
def build_triangles(n_theta : int, theta_blocks : int, n_phi : int, phi_blocks : int, normals_facing_outwards=True):
    '''
    Build triangle connectivity for a mesh on a toroidal surface with n_theta poloidal and n_phi toroidal points.
    The mesh is assumed to be structured, with quads between each set of 4 neighboring points.
    The mesh is assumed to wrap around in poloidal (theta) direction. In toroidal_direction (phi) it only wraps around if phi_blocks = n_phi, i.e. full toroidal coverage.

    The triangles can be oriented either inwards or outwards (required for consistency if using watertight surfaces)

    Parameters
    ----------
    n_theta : int
        The number of poloidal points.
    theta_blocks : int
        The number of poloidal blocks (quads). Equal to n_theta.
    n_phi : int
        The number of toroidal points.
    phi_blocks : int
        The number of toroidal blocks (quads). Equal to n_phi if full toroidal coverage, else n_phi - 1.
    normals_facing_outwards : bool, optional
        Whether the normals should face outwards (right-hand rule). Default is True.

    Returns
    -------
    triangles : jnp.ndarray
        An array of shape (n_triangles, 3) containing the indices of the vertices for each triangle.
    '''
    
    u_i_block, v_i_block = jnp.meshgrid(jnp.arange(theta_blocks), jnp.arange(phi_blocks), indexing='ij')
    

    # Compute neighboring indices with wrapping
    bottom_left_u_i,  bottom_left_v_i  = u_i_block,                 v_i_block
    bottom_right_u_i, bottom_right_v_i = u_i_block,                 (v_i_block + 1) % n_phi
    top_left_u_i,     top_left_v_i     = (u_i_block + 1) % n_theta, v_i_block
    top_right_u_i,    top_right_v_i    = (u_i_block + 1) % n_theta, (v_i_block + 1) % n_phi

    # In the array each vertex is indexed as u * Nv + v
    uv_index = lambda u, v: u * n_phi + v

    def outfacing_normals(x):
        return jnp.stack([
            uv_index(bottom_left_u_i, bottom_left_v_i),
            uv_index(bottom_right_u_i, bottom_right_v_i),
            uv_index(top_left_u_i, top_left_v_i),
        ], axis=-1), jnp.stack([
            uv_index(bottom_right_u_i, bottom_right_v_i),
            uv_index(top_right_u_i, top_right_v_i),
            uv_index(top_left_u_i, top_left_v_i),
        ], axis=-1)
    def infacing_normals(x):
        return jnp.stack([
            uv_index(bottom_right_u_i, bottom_right_v_i),
            uv_index(bottom_left_u_i, bottom_left_v_i),
            uv_index(top_left_u_i, top_left_v_i),
        ], axis=-1), jnp.stack([
            uv_index(top_right_u_i, top_right_v_i),
            uv_index(bottom_right_u_i, bottom_right_v_i),
            uv_index(top_left_u_i, top_left_v_i),
        ], axis=-1)
        
    tri1 , tri2 = jax.lax.cond(normals_facing_outwards, outfacing_normals, infacing_normals, operand=None)

    # Combine and flatten
    triangles = jnp.concatenate([tri1, tri2], axis=-1)
    triangles = triangles.reshape(-1, 3)

    return triangles

@partial(jax.jit, static_argnums = (4,5,6))
def _mesh_surface(flux_surfaces : FluxSurface, s : float, phi_start : float, phi_end : bool, full_angle : bool,  n_theta : int, n_phi : int, normals_facing_outwards  : bool):
    '''
    Mesh a flux surface at normalized radius s with n_theta poloidal and n_phi toroidal points.

    full_angle defines not the extent, only the connectivity. It only makes sense to set it to true if phi_start = 0 and phi_end = 2π.
    This is separate here because the number of trianlges depends on full_angle:

    2 * n_phi * n_theta if full_angle else 2 * (n_phi - 1) * n_theta

    Therefore, to allow jitting, it is a separate, static, argument. Use the convenience function mesh_surface below to ensure consistency.

    The number of points is always n_phi * n_theta, regardless of full_angle. 

    Parameters
    ----------
    flux_surfaces : FluxSurface
        The flux surface object containing the parameterization.
    s : float
        The normalized radius of the flux surface to mesh.
    phi_start : float
        The starting toroidal angle (in radians) if not full_angle.
    phi_end : float
        The ending toroidal angle (in radians) if not full_angle.
    full_angle : bool
        Whether to mesh the full 0 to 2π toroidal angle or a segment.
    n_theta : int
        The number of poloidal points.
    n_phi : int
        The number of toroidal points.
    normals_facing_outwards : bool, optional
        Whether the normals should face outwards (right-hand rule). Default is True.
    Returns
    -------
    positions : jnp.ndarray
        An array of shape (n_points, 3) containing the Cartesian coordinates of the mesh points.
    triangles : jnp.ndarray
        An array of shape (n_triangles, 3) containing the indices of the vertices for each triangle.

    '''

    phi_blocks = n_phi if full_angle else n_phi - 1 # full_angle is static so this is possible
    theta_blocks = n_theta
    
    total_points = n_theta * n_phi
    total_blocks = phi_blocks * theta_blocks
    
    theta = jnp.linspace(0, 2 * jnp.pi, n_theta, endpoint=False)
    phi   = jax.lax.cond(full_angle, lambda _ : jnp.linspace(phi_start, phi_end, n_phi, endpoint=False), 
                                     lambda _ :jnp.linspace(phi_start,  phi_end, n_phi, endpoint=True), 
                                     operand = None)    
    
    t, p  = jnp.meshgrid(theta, phi, indexing='ij')

    positions = flux_surfaces.cartesian_position(s,t,p).reshape(-1,3)        
    
    triangles = build_triangles(n_theta, theta_blocks, n_phi,  phi_blocks, normals_facing_outwards)

    return positions, triangles

_mesh_multiple_surfaces = jax.jit(jax.vmap(_mesh_surface, in_axes=(None, 0, None, None, None, None, None, 0)), static_argnums=(4,5,6))


def _concatenate_meshes(positions_list : List[jnp.ndarray], connectivity_list : List[jnp.ndarray]):
    '''
    Internal function to concatenate multiple meshes into a single mesh.

    Parameters
    ----------
    positions_array : List[jnp.ndarray]
        A list of arrays of shape (n_points, 3) containing the Cartesian coordinates of the mesh points for each surface.
    connectivity_array : List[jnp.ndarray]
        A list of arrays of shape (n_points_per_element, 3) containing the indices of the vertices for each triangle for each surface.        
    '''
    n_meshes          = len(positions_list)
    n_points_per_mesh = [pos.shape[0] for pos in positions_list]
    




def _mesh_surfaces_closed(flux_surfaces: FluxSurface, s_values_start : float, s_value_end : float, phi_start : float, phi_end : float, full_angle : bool, n_theta : int, n_phi : int):
    
    s_values_outer = jnp.array([s_values_start, s_value_end])
    normals_facing_outwards = jnp.array([False, True])

    multiple_surface_mesh =  _mesh_multiple_surfaces(flux_surfaces, s_values_outer, phi_start, phi_end, full_angle, n_theta, n_phi,  normals_facing_outwards)

    return multiple_surface_mesh


def mesh_surface(flux_surfaces: FluxSurface, s : float, toroidal_extent : ToroidalExtent, n_theta : int, n_phi : int, normals_facing_outwards : bool = True):
    """
    Create a mesh of points on a flux surface at normalized radius s, with n_theta poloidal and n_phi toroidal points.

    This cannot be jitted because the toroidal extent determines whether it is closed, which determines the number of triangles. Therefore, the 
    size of the arrays is unknown at compile time (which cannot be jitted unless toroidal_extent is static, but this is inconvenient because then the function would recompile for every different extent).

    This is therefore a convenience function only. Internal functions should not build on this function.

    Parameters
    ----------
    flux_surfaces : FluxSurface
        The flux surface object containing the parameterization.
    s : float
        The normalized radius of the flux surface to mesh.
    toroidal_extent : ToroidalExtent
        The toroidal extent of the mesh. 
    n_theta : int
        The number of poloidal points.
    n_phi : int
        The number of toroidal points.
    normals_facing_outwards : bool, optional
        Whether the normals should face outwards. Default is True.    
    Returns
    -------
    positions : jnp.ndarray
        An array of shape (n_points, 3) containing the Cartesian coordinates of the mesh points.
    triangles : jnp.ndarray
        An array of shape (n_triangles, 3) containing the indices of the vertices for each triangle.

    """

    return _mesh_surface(flux_surfaces, s, *toroidal_extent, n_theta, n_phi,  normals_facing_outwards)




    