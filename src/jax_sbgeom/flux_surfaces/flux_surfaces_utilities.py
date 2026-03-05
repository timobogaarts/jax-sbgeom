from . import FluxSurfaceNormalExtendedNoPhi

import jax.numpy as jnp
from warnings import warn
from jax_sbgeom.jax_utils.raytracing import find_minimum_distance_to_mesh
import equinox as eqx
from . import ToroidalExtent

@eqx.filter_jit
def generate_thickness_matrix(flux_surface : FluxSurfaceNormalExtendedNoPhi, mesh, n_theta : int, n_phi : int):
    '''
    Generate thickness matrix of an external mesh with respect to a no-phi extended flux surface.

    Uses the internal raytracing utilities to compute the minimum distance from the flux surface to the mesh along the normal directions.

    Parameters
    ----------
    flux_surface : FluxSurfaceNormalExtendedNoPhi
        Flux surface to compute thickness from.
    mesh : Tuple[jnp.ndarray, jnp.ndarray]
        Mesh of the external object (vertices, connectivity).
    n_theta : int
        Number of poloidal points.
    n_phi : int
        Number of toroidal points.  
    Returns
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
    phi   = jnp.linspace(0, 2 * jnp.pi / flux_surface.nfp, n_phi)
    theta, phi = jnp.meshgrid(theta, phi, indexing='ij')
    positions_lcfs_mg  = flux_surface.cartesian_position(1.0,  theta, phi)
    directions_lcfs_mg = flux_surface.cartesian_position(2.0, theta, phi) - positions_lcfs_mg
    dmesh = find_minimum_distance_to_mesh(positions_lcfs_mg, directions_lcfs_mg, mesh) 
    return theta, phi, dmesh

#========================================================
# Utilities for converting between half and full modules
#=========================================================

@eqx.filter_jit
def convert_half_module_points_to_full_module(points_half : jnp.ndarray):
    '''
    Convert points defined in a half module to points in a full module.     

    points_half is assumed to be in a shape 

    [...., n_theta_points, n_phi_points, 3]

    The first point in the half module is used to determine the module start.

    Then, all points are transformed as 

    R_new = R 
    Z_new = -Z
    phi_new = 2*phi_0 - phi

    Finally, the points are flipped across the theta dimension to create the full module.


    Parameters
    ----------
    points_half : jnp.ndarray
        An array of shape (..., n_theta_points, n_phi_points, 3) containing the Cartesian coordinates of the points in the half module.
    Returns
    -------
    points_full : jnp.ndarray
        An array of shape (..., 2*n_theta_points -  1, n_phi_points, 3) containing the Cartesian coordinates of the points in the full module.
    


    '''
    last_point = points_half[(-1,) * (points_half.ndim - 1)]    
    phi_0 = jnp.arctan2(last_point[1], last_point[0])
    
    half_module_slices_to_duplicate = points_half[..., :-1, :][..., ::-1, :]

    r_dup = jnp.sqrt(half_module_slices_to_duplicate[..., 0]**2 + half_module_slices_to_duplicate[..., 1]**2)
    z_dup = half_module_slices_to_duplicate[..., 2]
    phi_dup = jnp.arctan2(half_module_slices_to_duplicate[..., 1], half_module_slices_to_duplicate[..., 0])

    r_new = r_dup 
    z_new = -z_dup
    phi_new = 2*phi_0 - phi_dup

    x_new = r_new * jnp.cos(phi_new)
    y_new = r_new * jnp.sin(phi_new)

    new_points = jnp.stack([x_new, y_new, z_new], axis=-1)
    # now we flip across the theta dimension:

    new_points_theta_flip  = jnp.concatenate([new_points[..., 0, :, :][..., jnp.newaxis, :, :], new_points[..., 1:, :,:][..., ::-1, :, :]], axis=-3)

    return jnp.concatenate([points_half, new_points_theta_flip], axis=-2)

    
@eqx.filter_jit
def convert_full_module_points_multiple_full_module(points_full_module : jnp.ndarray, toroidal_extent_full_module : ToroidalExtent, n_before : int, n_after : int):
    '''
    Convert points from a full module to multiple full modules. 
        
    Parameters
    -----------
    points_full_module : jnp.ndarray
        The points of the full module, shape [..., n_phi_points, 3]
    toroidal_extent_full_module : ToroidalExtent
        The toroidal extent of the full module. (Although this could be derived by the points themselves, we keep it separate to allow for full modules that do not directly correspond to 
        an exact phi plane on both ends, e.g. when using a non-constant phi flux surface extension)
    n_before : int
        The number of full modules to add before the original full module
    n_after : int
        The number of full modules to add after the original full module
    '''
    assert points_full_module.shape[-1] == 3, f"points_full_module should have shape [..., n_phi_points, 3] but got shape {points_full_module.shape}"
    assert points_full_module.ndim >= 2, f"points_full_module should have at least two dimensions but got shape {points_full_module.shape}"
    assert points_full_module.shape[-2] > 0, f"points_full_module should have at least one phi point but got shape {points_full_module.shape}"

    assert n_before >= 0, f"n_before should be non-negative but got {n_before}"
    assert n_after >= 0, f"n_after should be non-negative but got {n_after}"

    r_points_full_module = jnp.sqrt(points_full_module[...,0]**2 + points_full_module[...,1]**2)
    z_points_full_module = points_full_module[...,2]
    phi_points_full_module = jnp.arctan2(points_full_module[...,1], points_full_module[...,0])

    d_phi_full_module = toroidal_extent_full_module.end - toroidal_extent_full_module.start

    if n_before > 0:
        points_before = jnp.array([jnp.stack([r_points_full_module, z_points_full_module, phi_points_full_module - (i+1)*d_phi_full_module], axis=-1) for i in range(n_before)])[..., :-1,:]
        points_before_cartesian = jnp.stack([points_before[...,0] * jnp.cos(points_before[...,2]), points_before[...,0] * jnp.sin(points_before[...,2]), points_before[...,1]], axis=-1)
        points_before_cartesian = jnp.moveaxis(points_before_cartesian, 0, -3) # we move the phi axis to the end to make concatenation easier later: now shape [..., n_before, n_phi, 3]
        shape_before_rs = tuple(points_before_cartesian.shape[:-3]) + (n_before * (points_before_cartesian.shape[-2] ), 3)
        points_before_cartesian = points_before_cartesian.reshape(shape_before_rs)
    
    if n_after > 0:        
        points_after = jnp.array([jnp.stack([r_points_full_module, z_points_full_module, phi_points_full_module + (i+1)*d_phi_full_module], axis=-1) for i in range(n_after)])[..., 1:,:]
        points_after_cartesian = jnp.stack([points_after[...,0] * jnp.cos(points_after[...,2]), points_after[...,0] * jnp.sin(points_after[...,2]), points_after[...,1]], axis=-1)
        points_after_cartesian = jnp.moveaxis(points_after_cartesian, 0, -3) # we move the phi axis to the end to make concatenation easier later: now shape [..., n_after, n_phi, 3]    
        shape_after_rs = tuple(points_after_cartesian.shape[:-3]) + (n_after * (points_after_cartesian.shape[-2] ), 3)
        points_after_cartesian = points_after_cartesian.reshape(shape_after_rs)

    if n_before == 0 and n_after == 0:
        return points_full_module
    
    elif n_before == 0 and n_after > 0:
        return jnp.concatenate([points_full_module, points_after_cartesian], axis=-2)
    elif n_before > 0 and n_after == 0:
        return jnp.concatenate([points_before_cartesian, points_full_module], axis=-2)
    else:   
        return jnp.concatenate([points_before_cartesian, points_full_module, points_after_cartesian], axis=-2)


def convert_half_module_points_to_multiple_full_modules_mesh(points_half_module, toroidal_extent_full_module : ToroidalExtent, n_before : int, n_after : int, normals_orientation : bool ):
    from .flux_surface_meshing import _build_triangles_surface
    '''
    Convert a mesh defined in a half module to multiple full modules. This is a utility function that combines the convert_half_module_points_to_full_module and convert_full_module_points_multiple_full_module functions for convenience.

    Parameters
    -----------
    points_half_module : jnp.ndarray
        The points of the half module, defined as an array of shape [ n_theta_points, n_phi_points, 3].
    toroidal_extent_full_module : ToroidalExtent
        The toroidal extent of the full module. This is used to determine the toroidal extent of the full module for the conversion.
    n_before : int
        The number of full modules to add before the original full module
    n_after : int
        The number of full modules to add after the original full module
    '''
    assert points_half_module.ndim == 3, f"points_half_module should have shape [n_theta_points, n_phi_points, 3] but got shape {points_half_module.shape}"
    assert points_half_module.shape[-1] == 3, f"points_half_module should have shape [n_theta_points, n_phi_points, 3] but got shape {points_half_module.shape}"

    
    points_full_module = convert_half_module_points_to_full_module(points_half_module)
    points_multiple_full_module = convert_full_module_points_multiple_full_module(points_full_module, toroidal_extent_full_module, n_before, n_after)    
    ntheta = points_multiple_full_module.shape[0]
    nphi = points_multiple_full_module.shape[1]

    triangles = _build_triangles_surface(ntheta, ntheta, nphi, nphi - 1, normals_orientation)
    return points_multiple_full_module.reshape(-1,3), triangles





