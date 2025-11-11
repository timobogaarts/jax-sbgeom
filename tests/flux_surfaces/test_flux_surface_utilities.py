import jax_sbgeom as jsb
import pytest
import jax
import jax.numpy as jnp 
import numpy as onp
from jax_sbgeom.jax_utils import raytracing as RT
from jax_sbgeom.jax_utils.utils import _mesh_to_pyvista_mesh
jax.config.update("jax_enable_x64", True)


def _get_coil_and_fs_files():
#    return [()"/home/tbogaarts/stellarator_paper/base_data/vmecs/HELIAS3_coils_all.h5", "/home/tbogaarts/stellarator_paper/base_data/vmecs/HELIAS5_coils_all.h5", "/home/tbogaarts/stellarator_paper/base_data/vmecs/squid_coilset.h5"]
    return [("/home/tbogaarts/stellarator_paper/base_data/vmecs/helias3_vmec.nc4", "/home/tbogaarts/stellarator_paper/base_data/vmecs/HELIAS3_coils_all.h5"), ("/home/tbogaarts/stellarator_paper/base_data/vmecs/helias5_vmec.nc4", "/home/tbogaarts/stellarator_paper/base_data/vmecs/HELIAS5_coils_all.h5"), ("/home/tbogaarts/stellarator_paper/base_data/vmecs/squid_vmec.nc4", "/home/tbogaarts/stellarator_paper/base_data/vmecs/squid_coilset.h5")]

def _get_discrete_coils(coil_file):
    import h5py
    with h5py.File(coil_file, 'r') as f:
        coil_data = jnp.array(f['Dataset1'])
    return jsb.coils.CoilSet.from_list([jsb.coils.DiscreteCoil.from_positions(coil_data[i]) for i in range(coil_data.shape[0])])

@pytest.fixture(scope="session", params = _get_coil_and_fs_files())
def _get_cws_and_flux_surface(request):
    fs_jax = jsb.flux_surfaces.FluxSurfaceNormalExtendedNoPhi.from_hdf5(request.param[0])
    coilset_jax = _get_discrete_coils(request.param[1])
    cws_mesh = jsb.coils.coil_winding_surface.create_optimized_coil_winding_surface(coilset_jax, 100, 200, 'spline')
    return cws_mesh, fs_jax

def check_cws_ray_tracing(cws_mesh, fs_jax):
    n_theta            = 210
    n_phi              = 100
    theta              = jnp.linspace(0, 2 * jnp.pi, n_theta)
    phi                = jnp.linspace(0, 2 * jnp.pi / fs_jax.settings.nfp, n_phi)
    theta, phi         = jnp.meshgrid(theta, phi, indexing='ij')
    positions_lcfs_mg  = fs_jax.cartesian_position(1.0,  theta, phi)
    directions_lcfs_mg = fs_jax.cartesian_position(2.0, theta, phi) - positions_lcfs_mg 
    mesh_rt            = _mesh_to_pyvista_mesh(*cws_mesh)
    trimesh_mesh       = mesh_rt.extract_surface()
    
    final_points, final_rays, final_cells = trimesh_mesh.multi_ray_trace(onp.array(positions_lcfs_mg).reshape(-1,3), onp.array(directions_lcfs_mg).reshape(-1,3), first_point = True,retry = True)
    final_points                          = jnp.array(final_points).reshape(positions_lcfs_mg.shape)
    dmesh_trimesh                         = jnp.linalg.norm(final_points - positions_lcfs_mg, axis=-1).reshape(theta.shape)
    dmesh_jax                             = RT.find_minimum_distance_to_mesh(positions_lcfs_mg, directions_lcfs_mg, cws_mesh)
    dmesh_utility                         = jsb.flux_surfaces.flux_surfaces_utilities.generate_thickness_matrix(fs_jax, cws_mesh, n_theta, n_phi)
    reinterpolated_positions              = fs_jax.cartesian_position(1.0 + dmesh_jax, theta, phi)
    onp.testing.assert_allclose(onp.array(dmesh_utility[2]), onp.array(dmesh_jax), atol=1e-10)
    onp.testing.assert_allclose(onp.array(dmesh_trimesh), onp.array(dmesh_jax), atol=1e-10)
    onp.testing.assert_allclose(onp.array(reinterpolated_positions), onp.array(final_points), atol=1e-10)

@pytest.mark.slow
def test_cws_ray_tracing(_get_cws_and_flux_surface):
    cws_mesh, fs_jax = _get_cws_and_flux_surface
    check_cws_ray_tracing(cws_mesh, fs_jax)
    