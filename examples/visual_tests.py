import jax_sbgeom as jsb

import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import numpy as onp
import sys 
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)
vmec_files = ["/home/tbogaarts/stellarator_paper/base_data/vmecs/helias3_vmec.nc4", "/home/tbogaarts/stellarator_paper/base_data/vmecs/helias5_vmec.nc4", "/home/tbogaarts/stellarator_paper/base_data/vmecs/squid_vmec.nc4"]

from jax_sbgeom.flux_surfaces.flux_surfaces_base import _cartesian_position_interpolated_jit, _cylindrical_position_interpolated, _cartesian_position_interpolated_grad, ToroidalExtent
from tests.flux_surfaces.flux_surface_base import test_position, _get_flux_surfaces, _sampling_grid, _1d_sampling_grid, test_normals, test_meshing_surface, test_principal_curvatures, _get_all_closed_surfaces, test_all_closed_surfaces
#
import pyvista as pv
pv.set_jupyter_backend('client')

def run_closed_surfaces_test(vmec_i):
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
            plotter.add_arrows(mesh.cell_centers().points, mesh['Normals'], mag=0.5, color='red')

        plotter.show()
    plot_all_surfaces(surfaces)

if __name__ == "__main__":
    run_closed_surfaces_test(vmec_i=1)