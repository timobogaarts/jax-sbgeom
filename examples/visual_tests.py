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

from jax_sbgeom.jax_utils.utils import _mesh_to_pyvista_mesh
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

        

        base_mesh = _mesh_to_pyvista_mesh(*jsc.mesh_coilset_surface(finitesize_coilset, 100, 0.2, 0.2))

        sarr_base = jsc.coil_winding_surface._create_total_s(xarr, coilset.n_coils)
        sarr_opt  = jsc.coil_winding_surface._create_total_s(xarr_opt, coilset.n_coils)

        positions     = coilset.position_different_s(sarr_base)
        positions_opt = coilset.position_different_s(sarr_opt)
    

        coil_surface_non_opt = _mesh_to_pyvista_mesh(jnp.moveaxis(positions, 1,0).reshape(-1,3), jsb.flux_surfaces.flux_surface_meshing._mesh_surface_connectivity(positions.shape[1], positions.shape[0],True, True))
        coil_surface_opt     = _mesh_to_pyvista_mesh(jnp.moveaxis(positions_opt, 1,0).reshape(-1,3), jsb.flux_surfaces.flux_surface_meshing._mesh_surface_connectivity(positions.shape[1], positions.shape[0],True, True))
        



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

if __name__ == "__main__":
    run_closed_surfaces_test(vmec_i=1)
