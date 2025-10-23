import jax 
import jax.numpy as jnp

import numpy as onp
def stack_jacfwd(fun, argnums):
    jacfwd_internal = jax.jacfwd(fun, argnums = argnums)
    def jac_stack_wrap(*args):                
        return jnp.stack(jacfwd_internal(*args), axis=-1)    
    return jac_stack_wrap

# ===================================================================================================================================================================================
#                                                                           Interpolation of arrays
# ===================================================================================================================================================================================
def interpolate_fractions(s, nsurf):    
    s_start =  s * (nsurf-1)
    i0      = jnp.floor(s_start).astype(int)
    i1      = jnp.minimum(i0 + 1, nsurf - 1)    
    ds      = s_start - i0    
    return i0, i1, ds

def interpolate_array(x_interp, s):
    nsurf = x_interp.shape[0]
    i0, i1, ds   = interpolate_fractions(s, nsurf)
    x0 = x_interp[i0]
    x1 = x_interp[i1]
    return (1 - ds) * x0 + ds * x1

def interpolate_fractions_modulo(s, nsurf):    
    s_start =  s * nsurf
    i0      = jnp.floor(s_start).astype(int)
    i1      = i0 + 1
    ds      = s_start - i0    
    return i0, i1, ds

def interpolate_array_modulo(x_interp, s):
    nsurf = x_interp.shape[0]
    i0, i1, ds   = interpolate_fractions_modulo(s, nsurf)
    x0 = x_interp[i0 % nsurf]
    x1 = x_interp[i1 % nsurf]
    return (1 - ds) * x0 + ds * x1


def interpolate_array_modulo_broadcasted(x_interp, s):
    nsurf = x_interp.shape[-2]
    i0, i1, ds   = interpolate_fractions_modulo(s, nsurf)
    x0 = x_interp[..., i0 % nsurf, :] # s_shape, something
    x1 = x_interp[..., i1 % nsurf, :] # s_shape, something

    return (1 - ds[..., None]) * x0 + ds[..., None] * x1



def _mesh_to_pyvista_mesh(pts, conn):    
    import pyvista as pv
    if conn.shape[-1] == 3:
        
        points_onp = onp.array(pts)
        conn_onp   = onp.array(conn)
        faces_pv = onp.hstack([onp.full((conn_onp.shape[0], 1), 3), conn_onp]).astype(onp.int64)
        faces_pv = faces_pv.flatten()
        mesh = pv.PolyData(points_onp, faces_pv)
        return mesh
    elif conn.shape[-1] == 4:
        points_onp = onp.array(pts)
        conn_onp   = onp.array(conn)
        cells = onp.hstack([onp.full((conn_onp.shape[0], 1), 4), conn_onp]).astype(onp.int64)
        cells = cells.flatten()
        mesh = pv.UnstructuredGrid(cells, onp.full(conn_onp.shape[0], 10), points_onp)
        return mesh
    else:
        raise ValueError("Connectivity must be triangles or tetrahedra")