import jax_sbgeom as jsb

import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import numpy as onp
import sys 
import os


from functools import partial
import jax_sbgeom.coils as jsc
import jax_sbgeom.flux_surfaces as jsf

from jax_sbgeom.jax_utils import mesh_to_pyvista_mesh
import pyvista as pv
import jax_sbgeom.coils.coil_winding_surface as cws
import h5py

import pyvista as pv


def run_closed_surfaces_test(vmec_i):
        
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    sys.path.append(project_root)
    vmec_files = ["/home/tbogaarts/stellarator_paper/base_data/vmecs/helias3_vmec.nc4", "/home/tbogaarts/stellarator_paper/base_data/vmecs/helias5_vmec.nc4", "/home/tbogaarts/stellarator_paper/base_data/vmecs/squid_vmec.nc4"]


    from tests.flux_surfaces.test_flux_surface_base import  _get_flux_surfaces, _get_all_closed_surfaces
    def _get_flux_surfaces(vmec_file):
        fs_jax    = jsb.flux_surfaces.FluxSurface.from_hdf5(vmec_file)    
        return fs_jax

    vmec_file = vmec_files[vmec_i]
    fs_jax = _get_flux_surfaces(vmec_file)  # just to compile
    surfaces =  _get_all_closed_surfaces(fs_jax)


    def plot_all_surfaces(all_surfaces):
        plotter = pv.Plotter(shape=(2,2))

        for i, surface in enumerate(all_surfaces):
            pts     = onp.asarray(surface[0])
            conn    = onp.asarray(surface[1].reshape(-1,3), dtype=onp.int64)

            faces = onp.hstack([onp.full((conn.shape[0],1),3), conn]).flatten()
            mesh = pv.PolyData(pts, faces)
            
            mesh.compute_normals(cell_normals=True, inplace=True)

            plotter.subplot(i // 2, i % 2)
            mesh_colors = onp.array(onp.arange(mesh.n_cells), dtype=float)
            plotter.add_mesh(mesh, scalars = mesh_colors, show_edges=True, opacity = 0.9, cmap = 'rainbow')
            plotter.add_arrows(mesh.cell_centers().points, mesh['Normals'], mag=0.2, color='red')

        plotter.show()
    plot_all_surfaces(surfaces)

def optimize_cws_and_plot(coil_i : int = 2, convert_to_fourier : bool = True):
   
    coil_files = ["/home/tbogaarts/stellarator_paper/base_data/vmecs/HELIAS3_coils_all.h5", "/home/tbogaarts/stellarator_paper/base_data/vmecs/HELIAS5_coils_all.h5", "/home/tbogaarts/stellarator_paper/base_data/vmecs/squid_coilset.h5"]

    coil_file = coil_files[coil_i]

    with h5py.File(coil_file) as f:
        positions = jnp.array(f['Dataset1'])

    jax_coils   = [jsc.DiscreteCoil.from_positions(positions[i]) for i in range(positions.shape[0])]    
    jax_coilset = jsc.CoilSet.from_list(jax_coils)    
    if convert_to_fourier:
        coilset           = jsc.fourier_coil.convert_to_fourier_coilset(jax_coilset)
    else:
        coilset = jax_coilset
        
    u_penalty = 1.0
    r_penalty = 0.1
    (xarr, optimizer_state), coilset_ordered = cws.optimize_coil_surface(coilset, uniformity_penalty=u_penalty, repulsive_penalty=r_penalty, optimization_settings=jsb.jax_utils.optimize.OptimizationSettings(max_iterations=500, tolerance=1e-4))
    
    

    x_base = jnp.ones(xarr.shape[0])

    def create_coil_splines_figure(xarr, xarr_opt, coilset):

        finitesize_coilset = jsc.FiniteSizeCoilSet.from_coilset(coilset, jsc.RotationMinimizedFrame, 100)

        

        base_mesh = mesh_to_pyvista_mesh(*jsc.mesh_coilset_surface(finitesize_coilset, 100, 0.2, 0.2))

        sarr_base = jsc.coil_winding_surface._create_total_s(xarr, coilset.n_coils)
        sarr_opt  = jsc.coil_winding_surface._create_total_s(xarr_opt, coilset.n_coils)

        positions     = coilset.position_different_s(sarr_base)
        positions_opt = coilset.position_different_s(sarr_opt)
    

        coil_surface_non_opt = mesh_to_pyvista_mesh(jnp.moveaxis(positions, 1,0).reshape(-1,3), jsb.flux_surfaces.flux_surface_meshing._mesh_surface_connectivity(positions.shape[1], positions.shape[0],True, True))
        coil_surface_opt     = mesh_to_pyvista_mesh(jnp.moveaxis(positions_opt, 1,0).reshape(-1,3), jsb.flux_surfaces.flux_surface_meshing._mesh_surface_connectivity(positions.shape[1], positions.shape[0],True, True))
        



        plotter = pv.Plotter(shape=(1,2))
        plotter.subplot(0,0)    
        
        plotter.add_mesh(base_mesh, opacity = 1.0)
        plotter.add_mesh(coil_surface_non_opt, opacity = 1.0, show_edges = True)

        plotter.subplot(0,1)            
        plotter.add_mesh(base_mesh,opacity = 1.0)
        plotter.add_mesh(coil_surface_opt, opacity = 1.0, show_edges = True)
        plotter.link_views()
        plotter.show()


    create_coil_splines_figure(x_base, xarr, coilset_ordered)

def _get_discrete_coils(coil_file):
    with h5py.File(coil_file, 'r') as f:
        coil_data = jnp.array(f['Dataset1'])

    return jsb.coils.CoilSet.from_list([jsb.coils.DiscreteCoil.from_positions(coil_data[i]) for i in range(coil_data.shape[0])])


def aabb_to_lines(aabbs):
    """
    Convert (N,2,3) AABBs to a single PolyData with all box edges.
    Vectorized and fast.
    """
    aabbs = onp.asarray(aabbs)
    N = aabbs.shape[0]
    mn = aabbs[:,0,:]  # shape (N,3)
    mx = aabbs[:,1,:]  # shape (N,3)

    # 8 corners per box in normalized [0,1] space
    corners = onp.array([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1]
    ])  # (8,3)

    # Compute all points: (N,8,3)
    points = mn[:,None,:] + (mx - mn)[:,None,:] * corners[None,:,:]
    points = points.reshape(-1,3)  # (8*N,3)

    # Edges of one box
    edges = onp.array([
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ])  # (12,2)

    # Repeat edges for all boxes with proper offset
    offsets = onp.arange(N) * 8  # (N,)
    all_edges = edges[None,:,:] + offsets[:,None,None]  # (N,12,2)
    all_edges = all_edges.reshape(-1,2)  # (12*N,2)

    # Build connectivity array for PolyData
    connectivity = onp.hstack([onp.full((all_edges.shape[0],1),2), all_edges])  # (12*N,3)
    connectivity = connectivity.flatten()

    return pv.PolyData(points, lines=connectivity)


def plot_bvh(coil_i = 2, idx_child = 0):
    import jax_sbgeom.jax_utils.raytracing as RT
    coil_files = ["/home/tbogaarts/stellarator_paper/base_data/vmecs/HELIAS3_coils_all.h5", "/home/tbogaarts/stellarator_paper/base_data/vmecs/HELIAS5_coils_all.h5", "/home/tbogaarts/stellarator_paper/base_data/vmecs/squid_coilset.h5"]

    coil_file = coil_files[coil_i]

    coilset_jax       = _get_discrete_coils(coil_file)

    sarr, coilset = jsb.coils.coil_winding_surface.optimize_coil_surface(coilset_jax, n_samples_per_coil=  300)

    positions_coilset = coilset.position_different_s(jsb.coils.coil_winding_surface._create_total_s(sarr[0], coilset.n_coils))
    vertices = jsb.flux_surfaces.flux_surface_meshing._mesh_surface_connectivity(positions_coilset.shape[1], positions_coilset.shape[0], True, True)
    positions_standard_ordering = jnp.moveaxis(positions_coilset, 0, 1) # ntheta, nphi [number of coils], 3
    mesh_pv_cws = mesh_to_pyvista_mesh(positions_standard_ordering.reshape(-1,3), vertices)
    mesh_cws_def = (positions_standard_ordering.reshape(-1,3), vertices)


    bvh = RT.build_lbvh(mesh_cws_def[0], mesh_cws_def[1])


    probe_idx = 2500
    probe = bvh.inverse_order[probe_idx]


    aabbs = []
    def get_parent(idx):
        
        parent = jnp.where((bvh.left_idx == idx) | (bvh.right_idx == idx))
    
        if parent[0].shape[0] == 0:
            return 
        if (idx < bvh.order.shape[0]) & (parent[0].shape[0] == 2):
            return parent[0][1]
        elif (parent[0].shape[0] == 1) & (idx >= bvh.order.shape[0]):
            return parent[0][0]
        else:
            print(parent[0].shape, idx, bvh.order.shape)
            print("WJTKDSJF", idx< bvh.order.shape[0], parent[0].shape[0]==2)
            raise ValueError("Something went wrong when getting parent in BVH:", parent, idx)
        
    pi = probe
    n_levels_approx = jnp.log2(bvh.order.shape[0]).astype(int) + 5
    for levels in range(n_levels_approx):
        aabbs.append(get_parent(pi))
        if aabbs[-1] is None:
            break
        pi = aabbs[-1]

    aabb_polydata = []
    for aabb_i in aabbs:
        if aabb_i is not None:
            aabb_polydata.append(aabb_to_lines(bvh.aabb[aabb_i:aabb_i+1]))

    colors = ["red", "green", "blue", "yellow", "cyan"]
    plotter = pv.Plotter()
    plotter.add_mesh(mesh_to_pyvista_mesh(*mesh_cws_def))
    plotter.add_mesh(pv.PolyData(onp.array(mesh_cws_def[0][mesh_cws_def[1][probe_idx]])), 'red')
    for i, aabb_i in enumerate(aabb_polydata):
        plotter.add_mesh(aabb_polydata[i], color = colors[i%len(colors)])

    plotter.show()
    
    
def plot_different_cws_methods(vmec_i = 2):
    coil_files = ["/home/tbogaarts/stellarator_paper/base_data/vmecs/HELIAS3_coils_all.h5", "/home/tbogaarts/stellarator_paper/base_data/vmecs/HELIAS5_coils_all.h5", "/home/tbogaarts/stellarator_paper/base_data/vmecs/squid_coilset.h5"]
    vmec_files = ["/home/tbogaarts/stellarator_paper/base_data/vmecs/helias3_vmec.nc4", "/home/tbogaarts/stellarator_paper/base_data/vmecs/helias5_vmec.nc4", "/home/tbogaarts/stellarator_paper/base_data/vmecs/squid_vmec.nc4"]

    fs_jax    = jsb.flux_surfaces.FluxSurfaceNormalExtendedNoPhi.from_hdf5(vmec_files[vmec_i])
    

    def _get_discrete_coils(coil_file):
        with h5py.File(coil_file, 'r') as f:
            coil_data = jnp.array(f['Dataset1'])
        return jsb.coils.CoilSet.from_list([jsb.coils.DiscreteCoil.from_positions(coil_data[i]) for i in range(coil_data.shape[0])])
    
    coilset_jax       = _get_discrete_coils(coil_files[vmec_i])
    optimized_params, ordered_coilset = jsb.coils.coil_winding_surface.optimize_coil_surface(coilset_jax, n_samples_per_coil=100)
    positions_cws_opt = jsb.coils.coil_winding_surface._create_cws_interpolated(ordered_coilset, 200, optimized_params[0])
    cws_direct  = jsb.coils.coil_winding_surface._cws_direct(positions_cws_opt, None)
    cws_fourier = jsb.coils.coil_winding_surface._cws_fourier(positions_cws_opt, 200)
    cws_spline  = jsb.coils.coil_winding_surface._cws_spline(positions_cws_opt, 200)
    plotter = pv.Plotter(shape=(1,3))

    plotter.subplot(0,0)
    plotter.add_mesh(mesh_to_pyvista_mesh(*cws_direct), color='red', opacity=1.0, show_edges=True)

    plotter.subplot(0,1)
    plotter.add_mesh(mesh_to_pyvista_mesh(*cws_fourier), color='green', opacity=1.0, show_edges=True)
    plotter.subplot(0,2)
    plotter.add_mesh(mesh_to_pyvista_mesh(*cws_spline), color='lightblue', opacity=1.0, show_edges=True)
    plotter.link_views()
    plotter.show()




if __name__ == "__main__":
    run_closed_surfaces_test(vmec_i=1)
