import jax 
import jax.numpy as jnp

import numpy as onp
def stack_jacfwd(fun, argnums):
    jacfwd_internal = jax.jacfwd(fun, argnums = argnums)
    def jac_stack_wrap(*args):                
        return jnp.stack(jacfwd_internal(*args), axis=-1)    
    return jac_stack_wrap



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