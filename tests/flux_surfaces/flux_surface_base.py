import jax_sbgeom as jsb
import jax.numpy as jnp
import numpy as onp
import StellBlanket.SBGeom as SBGeom 
import jax
import time


from jax_sbgeom.flux_surfaces.flux_surfaces_base import _check_whether_make_normals_point_outwards_required, ToroidalExtent

jax.config.update("jax_enable_x64", True)

def _get_flux_surfaces(vmec_file):
    fs_jax    = jsb.flux_surfaces.FluxSurface.from_hdf5(vmec_file)
    fs_sbgeom = SBGeom.VMEC.Flux_Surfaces_From_HDF5(vmec_file)
    return fs_jax, fs_sbgeom

def _sampling_grid(fs_jax : jsb.flux_surfaces.FluxSurface, n_s : int = 50, n_theta : int = 55, n_phi : int = 45, include_axis = True):

    ss = jax.lax.cond(include_axis, lambda x : jnp.linspace(0,1,n_s), lambda x : jnp.linspace(0,1, n_s + 1)[1:], None)
    
    tt = jnp.linspace(0, 2 * jnp.pi, n_theta, endpoint=False)
    pp = jnp.linspace(0, 2 * jnp.pi / fs_jax.settings.nfp, n_phi, endpoint=True)
    return jnp.meshgrid(ss, tt, pp, indexing='ij')


def _1d_sampling_grid(fs_jax : jsb.flux_surfaces.FluxSurface, n_s : int = 50, n_theta : int = 55, n_phi : int = 45, include_axis = True, reverse_theta : bool = False):
    ss, tt, pp = _sampling_grid(fs_jax, n_s, n_theta, n_phi, include_axis)
    if reverse_theta:
        tt = - onp.array(tt)
    return onp.array(ss).ravel(), onp.zeros(ss.shape).ravel(), onp.array(tt).ravel(), onp.array(pp).ravel()


def time_jsb_function(fun, *args, n_repetitions = 10, jsb = True):
    timings = []    
    for n in range(n_repetitions):
        start = time.time()
        if jsb:
            result = fun(*args).block_until_ready()
        else:
            result = fun(*args)
        end = time.time()
        timings.append(end - start)
    return result, onp.mean(timings), onp.std(timings)

def time_jsb_function_mult(fun, *args, n_repetitions = 10, jsb = True):
    timings = []
    for n in range(n_repetitions):
        start = time.time()
        if jsb:
            result = fun(*args)
            result[0].block_until_ready()            
        else:
            result = fun(*args)
        end = time.time()
        timings.append(end - start)
    return result, onp.mean(timings), onp.std(timings)

def time_jsb_function_nested(fun, *args, n_repetitions = 10, jsb = True):
    timings = []
    for n in range(n_repetitions):
        start = time.time()
        if jsb:
            result = fun(*args)
            result[0][0].block_until_ready()            
        else:
            result = fun(*args)
        end = time.time()
        timings.append(end - start)
    return result, onp.mean(timings), onp.std(timings)

def print_timings(name : str, time_jax : float, std_jax : float, time_sbgeom : float, std_sbgeom : float):
    print(f"{name:20s} passed | jax-sbgeom: {time_jax*1e3:8.3f} ms ± {std_jax*1e3:8.3f} ms | SBGeom: {time_sbgeom*1e3:8.3f} ms ± {std_sbgeom*1e3:8.3f} ms")

# ===================================================================================================================================================================================
#                                                                          Base Flux Surface functions
# ===================================================================================================================================================================================

def test_position(vmec_file = "/home/tbogaarts/data/helias5_vmec.nc4", n_repetitions = 1):
    fs_jax, fs_sbgeom = _get_flux_surfaces(vmec_file)

    pos_jax, time_jax, std_jax          = time_jsb_function(fs_jax.cartesian_position, *_sampling_grid(fs_jax),    n_repetitions= n_repetitions )

    reverse_theta = fs_sbgeom.du_x_dv_sign() == 1.0
    pos_sbgeom, time_sbgeom, std_sbgeom = time_jsb_function(fs_sbgeom.Return_Position, *_1d_sampling_grid(fs_jax, reverse_theta=reverse_theta), n_repetitions=n_repetitions, jsb=False)
    
    assert jnp.allclose(pos_jax, pos_sbgeom.reshape(pos_jax.shape), atol=1e-13)

    print_timings("Position", time_jax, std_jax, time_sbgeom, std_sbgeom)


def test_normals(vmec_file = "/home/tbogaarts/data/helias5_vmec.nc4", n_repetitions = 1):
    fs_jax, fs_sbgeom = _get_flux_surfaces(vmec_file)

    reverse_theta = fs_sbgeom.du_x_dv_sign() == 1.0

    norm_jax, time_jax, std_jax          = time_jsb_function(fs_jax.normal, *_sampling_grid(fs_jax, include_axis=False),              n_repetitions= n_repetitions)
    norm_sbgeom, time_sbgeom, std_sbgeom = time_jsb_function(fs_sbgeom.Return_Normal, *_1d_sampling_grid(fs_jax, include_axis=False, reverse_theta=reverse_theta), n_repetitions= n_repetitions, jsb=False)
    
    assert jnp.allclose(norm_jax, norm_sbgeom.reshape(norm_jax.shape), atol=1e-13)

    print_timings("Normal", time_jax, std_jax, time_sbgeom, std_sbgeom)

def test_principal_curvatures(vmec_file = "/home/tbogaarts/data/helias5_vmec.nc4", n_repetitions = 1):
    fs_jax, fs_sbgeom = _get_flux_surfaces(vmec_file)

    curv_jax, time_jax, std_jax          = time_jsb_function(fs_jax.principal_curvatures, *_sampling_grid(fs_jax, include_axis=False),      n_repetitions= n_repetitions)

    def return_all_principal_curvatures(s,d,  theta, phi):
        # SBGeom returns does not have a vectorized function. we do the inefficient thing here.
        assert s.ndim == 1 and theta.ndim == 1 and phi.ndim == 1 and s.shape == theta.shape and s.shape == phi.shape
        k1 = onp.zeros(s.shape)
        k2 = onp.zeros(s.shape)
        for i in range(s.shape[0]):
            p_curv = fs_sbgeom.Return_Principal_Curvatures(s[i], d[i], theta[i], phi[i])
            k1[i] = p_curv[0]
            k2[i] = p_curv[1]
        return onp.stack([k1, k2], axis=-1)
    
    reverse_theta = fs_sbgeom.du_x_dv_sign() == 1.0

    curv_sbgeom, time_sbgeom, std_sbgeom = time_jsb_function(return_all_principal_curvatures, *_1d_sampling_grid(fs_jax, include_axis=False, reverse_theta=reverse_theta), n_repetitions= n_repetitions, jsb=False)
    
    assert jnp.allclose(curv_jax, curv_sbgeom.reshape(curv_jax.shape), atol=1e-13)

    print_timings("Principal Curvatures", time_jax, std_jax, time_sbgeom, std_sbgeom)


# ===================================================================================================================================================================================
#                                                                           Meshing
# ===================================================================================================================================================================================
def _flip_vertices_theta(positions, n_theta, n_phi):
    positions_rs = positions.reshape(n_theta, n_phi, 3)

    first = jnp.take(positions_rs, indices = 0, axis = 0)
    rest = jnp.flip(jnp.take(positions_rs, indices = jnp.arange(1, n_theta), axis = 0), axis = 0)
    positions_rs_flipped = jnp.concatenate([first[None, :, :], rest], axis = 0)
    return positions_rs_flipped.reshape(-1, 3)


def test_meshing_surface(vmec_file = "/home/tbogaarts/data/helias5_vmec.nc4", tor_extent : str = 'half_module', n_repetitions = 1):
    fs_jax, fs_sbgeom = _get_flux_surfaces(vmec_file)

    if tor_extent == 'half_module':
        tor_extent= jsb.flux_surfaces.flux_surfaces_base.ToroidalExtent.half_module(fs_jax)
    elif tor_extent == 'full':
        tor_extent= jsb.flux_surfaces.flux_surfaces_base.ToroidalExtent.full()
    else:
        raise ValueError(f"Unknown toroidal extent: {tor_extent}")
    s = 0.356622756

    n_theta = 500 
    n_phi  = 60


    (pos_jax, tri_jax), time_jax, std_jax          = time_jsb_function_mult(jsb.flux_surfaces.flux_surface_meshing.mesh_surface, fs_jax, s, tor_extent,  n_theta, n_phi, True, n_repetitions=n_repetitions)
    mesh_sbgeom, time_sbgeom, std_sbgeom           = time_jsb_function(     fs_sbgeom.Mesh_Surface, s, 0.0, n_phi, n_theta, tor_extent.start, tor_extent.end,               True,   n_repetitions=n_repetitions, jsb=False)

    # The sampling cannot be directly influenced. 
    # Instead, we just reverse the theta direction by flipping all vertices if required 
    # This also takes care of the fact that SBGeom does not have normals facing outwards: they get flipped as well so will be equal again.
    reverse_theta = fs_sbgeom.du_x_dv_sign() == 1.0
    if reverse_theta:
        pos_jax_mod = _flip_vertices_theta(pos_jax, n_theta, n_phi)
    else:
        pos_jax_mod = pos_jax

    assert jnp.allclose(pos_jax_mod, mesh_sbgeom.vertices, atol=1e-13)
    assert jnp.allclose(tri_jax, mesh_sbgeom.connectivity, atol=1e-13)

    print_timings("mesh_surface", time_jax, std_jax, time_sbgeom, std_sbgeom)


def _get_mesh_surfaces_closed(flux_surfaces: jsb.flux_surfaces.FluxSurface,
                          s_values_start : float, s_value_end : float,
                          phi_start : float, phi_end : float,
                          n_theta : int, n_phi : int, n_cap : int):
    
    tor_extent = ToroidalExtent(phi_start, phi_end)
    meshes =  jsb.flux_surfaces.flux_surface_meshing.mesh_surfaces_closed(flux_surfaces,
                                                                        s_values_start,
                                                                        s_value_end,
                                                                        tor_extent,                                                                        
                                                                        n_theta,
                                                                        n_phi,
                                                                        n_cap)
    
    pts     = meshes[0]
    conn    = meshes[1]

    assert conn.min() >= 0 and conn.max() < pts.shape[0]
    
    return pts, conn

def _get_all_closed_surfaces(fs_jax):
    
    single_surface  = _get_mesh_surfaces_closed(fs_jax, 0.0, 1.0,  0.0, 2.0 * jnp.pi,  50, 60, 10)    
    two_surfaces    = _get_mesh_surfaces_closed(fs_jax, 0.2, 1.0,  0.0, 2.0 * jnp.pi,  50, 60, 10)
    closed_no_axis  = _get_mesh_surfaces_closed(fs_jax, 0.2, 1.0,  0.0, 0.3 * jnp.pi,  50, 60, 10)    
    closed_axis     = _get_mesh_surfaces_closed(fs_jax, 0.0, 1.0,  0.0, 0.3 * jnp.pi,  50, 60, 10)

    return [single_surface, two_surfaces, closed_no_axis, closed_axis]
    

def _mesh_to_pyvista_mesh(pts, conn):
    import pyvista as pv
    points_onp = onp.array(pts)
    conn_onp   = onp.array(conn)
    faces_pv = onp.hstack([onp.full((conn_onp.shape[0], 1), 3), conn_onp]).astype(onp.int64)
    faces_pv = faces_pv.flatten()
    mesh = pv.PolyData(points_onp, faces_pv)
    return mesh

def test_all_closed_surfaces(vmec_file, n_repetitions = 1):
    fs_jax, fs_sbgeom = _get_flux_surfaces(vmec_file)

    surfaces, time_jax, std_jax = time_jsb_function_nested(_get_all_closed_surfaces, fs_jax, n_repetitions=n_repetitions)

    for surf in surfaces:
        points, connectivity = surf
        mesh = _mesh_to_pyvista_mesh(points, connectivity)  
        assert jnp.allclose(mesh.volume, jsb.flux_surfaces.flux_surface_meshing._volume_of_mesh(points, connectivity), atol=1e-10)
    
    print_timings("all closed surfaces", time_jax, std_jax, 0.0, 0.0)
    print("\t (sbgeom has different closed surfaces)")

def test_volumes(vmec_file, n_repetitions = 1):
    from jax_sbgeom.flux_surfaces.flux_surfaces_base import _volume_from_fourier_half_mod, _volume_from_fourier
    fs_jax, fs_sbgeom = _get_flux_surfaces(vmec_file)

    s = 0.5

    vol_jax, time_jax, std_jax          = time_jsb_function(_volume_from_fourier_half_mod, fs_jax.data, fs_jax.settings, s, n_repetitions=n_repetitions)
    vol_jax2, time_jax2, std_jax2       = time_jsb_function(_volume_from_fourier, fs_jax.data, fs_jax.settings, s, n_repetitions=n_repetitions)

    assert jnp.allclose(vol_jax, vol_jax2, atol=1e-13)

    print_timings("Volume", time_jax, std_jax, time_jax2, std_jax2)
    print("\t (sbgeom has no volume function)")