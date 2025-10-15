from .flux_surfaces_base import FluxSurface, FluxSurfaceSettings, ToroidalExtent

import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums = (0,1,2,3,4))
def build_triangles(n_theta : int, theta_blocks : int, n_phi : int, phi_blocks : int, normals_facing_outwards=True):
    
    u_i_block, v_i_block = jnp.meshgrid(jnp.arange(theta_blocks), jnp.arange(phi_blocks), indexing='ij')
    

    # Compute neighboring indices with wrapping
    bottom_left_u_i,  bottom_left_v_i  = u_i_block,                 v_i_block
    bottom_right_u_i, bottom_right_v_i = u_i_block,                 (v_i_block + 1) % n_phi
    top_left_u_i,     top_left_v_i     = (u_i_block + 1) % n_theta, v_i_block
    top_right_u_i,    top_right_v_i    = (u_i_block + 1) % n_theta, (v_i_block + 1) % n_phi

    # In the array each vertex is indexed as u * Nv + v
    uv_index = lambda u, v: u * n_phi + v

    
    if normals_facing_outwards:
        tri1 = jnp.stack([
            uv_index(bottom_left_u_i, bottom_left_v_i),
            uv_index(bottom_right_u_i, bottom_right_v_i),
            uv_index(top_left_u_i, top_left_v_i),
        ], axis=-1)

        tri2 = jnp.stack([
            uv_index(bottom_right_u_i, bottom_right_v_i),
            uv_index(top_right_u_i, top_right_v_i),
            uv_index(top_left_u_i, top_left_v_i),
        ], axis=-1)
    else:
        tri1 = jnp.stack([
            uv_index(bottom_right_u_i, bottom_right_v_i),
            uv_index(bottom_left_u_i, bottom_left_v_i),
            uv_index(top_left_u_i, top_left_v_i),
        ], axis=-1)

        tri2 = jnp.stack([
            uv_index(top_right_u_i, top_right_v_i),
            uv_index(bottom_right_u_i, bottom_right_v_i),
            uv_index(top_left_u_i, top_left_v_i),
        ], axis=-1)

    # Combine and flatten
    triangles = jnp.concatenate([tri1, tri2], axis=-1)
    triangles = triangles.reshape(-1, 3)

    return triangles

@partial(jax.jit, static_argnums = (2,3,4,7))
def _Mesh_Surface(flux_surfaces, s, n_theta, n_phi, full_angle, phi_start, phi_end ,normals_facing_outwards  : bool ):

    phi_blocks   = n_phi if full_angle else n_phi - 1
    theta_blocks = n_theta
    total_points = n_theta * n_phi
    total_blocks = phi_blocks * theta_blocks
    
    theta = jnp.linspace(0, 2 * jnp.pi, n_theta, endpoint=False)
    phi   = jax.lax.cond(full_angle, lambda _ : jnp.linspace(phi_start, phi_end, n_phi, endpoint=False), 
                                     lambda _ :jnp.linspace( phi_start, phi_end, n_phi, endpoint=True), 
                                     operand = None)    
    
    t, p  = jnp.meshgrid(theta, phi, indexing='ij')

    positions = flux_surfaces.cartesian_position(s,t,p).reshape(-1,3)
    
    triangles = build_triangles(n_theta, theta_blocks, n_phi,  phi_blocks, normals_facing_outwards)

    return positions, triangles

def Mesh_Surface(flux_surfaces: FluxSurface, s : float, n_theta : int, n_phi : int, toroidal_extent : ToroidalExtent, normals_facing_outwards : bool = True):
    """
    Create a mesh of points on a flux surface at normalized radius s, with n_theta poloidal and n_phi toroidal points.
    """
    # static sizes
#    full_angle = bool(toroidal_extent.full_angle())
    

    return _Mesh_Surface(flux_surfaces, s, n_theta, n_phi, bool(toroidal_extent.full_angle()), toroidal_extent.start, toroidal_extent.end, normals_facing_outwards)
    


    


    