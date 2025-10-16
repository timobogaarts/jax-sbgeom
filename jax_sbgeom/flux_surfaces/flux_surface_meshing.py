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

@partial(jax.jit, static_argnums = (0,3))
def build_closed_strip(n_strip : int, offset_index_0 : int, offset_index_1 : int, normals_orientation : bool, multiplier_1 : int = 1, multiplier_2 : int = 1):
    '''
    This functions connects two closed strips together. Each strip has n_strip vertices, and the first vertex of each strip is at offset_index_0 and offset_index_1 respectively.
    Normals_orientation determines whether the normals should face outwards (True) or inwards (False).

    '''    
    u_i_block = jnp.arange(n_strip)
    
    bottom_left  =  u_i_block * multiplier_1  + offset_index_0
    bottom_right = ( (u_i_block + 1) % n_strip ) * multiplier_1 + offset_index_0 

    top_left     = u_i_block * multiplier_2 + offset_index_1 
    top_right    = ((u_i_block + 1) % n_strip ) * multiplier_2 + offset_index_1
    

    def outfacing_normals(x):
        return jnp.stack([
            bottom_left,
            bottom_right,
            top_left,
        ], axis=-1), jnp.stack([
            bottom_right,
            top_right,
            top_left,
        ], axis=-1)
    def infacing_normals(x):
        return jnp.stack([
            bottom_right,
            bottom_left,
            top_left,
        ], axis=-1), jnp.stack([
            top_right,
            bottom_right,
            top_left,
        ], axis=-1) 
    
    tri1, tri2 = jax.lax.cond(normals_orientation, outfacing_normals, infacing_normals, operand=None)
    triangles = jnp.concatenate([tri1, tri2], axis=-1)
    triangles = triangles.reshape(-1, 3)
    return triangles

_build_closed_strips = jax.vmap(build_closed_strip, in_axes=(None,0,0,None))



def build_triangles_plane(n_radial : int, n_theta : int, normals_orientation : bool):
    # Similar to build_triangles, but for a planar grid in (r, theta) space
    # The connectivity is the same as for a toroidal surface, so we exploit that by 
    # calling the above function with n_phi = n_radial and phi_blocks = n_radial - 1
    return build_triangles(n_theta, n_theta, n_radial, n_radial - 1, normals_orientation)



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

    n_points_per_mesh = jnp.array([pos.shape[0] for pos in positions_list])
    
    positions = jnp.concatenate(positions_list, axis=0)

    added_connectivity = jnp.cumulative_sum(n_points_per_mesh, include_initial=True)
    connectivity = jnp.concatenate([conn + added_connectivity[i] for i, conn in enumerate(connectivity_list)], axis=0)

    return positions, connectivity





def _mesh_surfaces_closed(flux_surfaces: FluxSurface,
                          s_values_start : float, s_value_end : float, include_axis : bool,
                          phi_start : float, phi_end : float, full_angle : bool,
                          n_theta : int, n_phi : int, n_cap : int):
    
    

    # We have the following possibilities:
    # 1. include_axis = True & full_angle = True: We have one surface only. This is a closed surface, so we can just return it.
    # 2. include_axis = False & full_angle = True: We have two surfaces. These are already closed so we can just return them.
    # 3. include_axis = False & full_angle = False: We have two surfaces. These are not closed surfaces, so we need to add caps at the edges from s_0 to s_end
    # 4. include_axis = True & full_angle = False: We have one surface only. This is not a closed surface, so we need to add caps at the edges from 0 ( = s_0 but wedge) to s_end

    # They are all compile-time constants (and in fact need to be so as output shape depends on their value)

    if include_axis and full_angle:        
        # 1. include_axis = True & full_angle = True: We have one surface only. This is a closed surface, so we can just return it.
        return _mesh_surface(flux_surfaces, s = s_value_end, phi_start = phi_start, phi_end = phi_end, full_angle = full_angle, n_theta = n_theta, n_phi = n_phi, normals_facing_outwards =  True)
    elif (not include_axis) and full_angle:
        # 2. include_axis = False & full_angle = True: We have two surfaces. These are already closed so we can just return them.
        s_values_outer          = jnp.array([s_values_start, s_value_end])        
        normals_facing_outwards = jnp.array([False, True])        
        multiple_surface_mesh   =  _mesh_multiple_surfaces(flux_surfaces, s_values_outer, phi_start, phi_end,  full_angle, n_theta,  n_phi, normals_facing_outwards)
        return _concatenate_meshes(*multiple_surface_mesh)
    elif (not include_axis) and (not full_angle):
        

        s_values_outer          = jnp.array([s_values_start, s_value_end])        
        normals_facing_outwards = jnp.array([False, True])        
        multiple_surface_mesh   =  _mesh_multiple_surfaces(flux_surfaces, s_values_outer, phi_start, phi_end,  full_angle, n_theta,  n_phi, normals_facing_outwards)
        outer_positions, outer_connectivity = _concatenate_meshes(*multiple_surface_mesh)

        s_cap = jnp.linspace(s_values_start, s_value_end, n_cap + 2)[1:-1] # Exclude first and last
        t_cap = jnp.linspace(0, 2 * jnp.pi, n_theta, endpoint=False)

        t_cap_mg, s_cap_mg = jnp.meshgrid(t_cap, s_cap, indexing='ij')

        positions_cap_start = flux_surfaces.cartesian_position(s_cap_mg, t_cap_mg, phi_start).reshape(-1,3)
        positions_cap_end   = flux_surfaces.cartesian_position(s_cap_mg, t_cap_mg, phi_end  ).reshape(-1,3) 

        # However, we have to modify the first and last rings of the outer mesh to connect to the outer surfaces
        connectivity_start_f  = build_closed_strip(n_theta, 0, outer_positions.shape[0], normals_orientation=False, multiplier_1 = n_phi, multiplier_2 = 1)        
        
        print(connectivity_start_f)
        connectivity_start_m  = build_triangles_plane(n_cap, n_theta, normals_orientation=True) + outer_positions.shape[0]
        
        #connectivity_start_e  = build_closed_strip(n_theta, outer_positions.shape[0] - n_theta, outer_positions.shape[0] + (n_cap - 1) * n_theta, normals_orientation=True)
        connectivity_start    = jnp.concatenate([connectivity_start_f, connectivity_start_m], axis=0) 



        #connectivity_end    = build_triangles_plane(n_cap, n_theta, normals_orientation=False) 

        positions_total = jnp.concatenate([outer_positions, positions_cap_start, positions_cap_end], axis=0)
        connectivity_total = jnp.concatenate([outer_connectivity, connectivity_start], axis=0)

        
        
        
        return positions_total, connectivity_total



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




    