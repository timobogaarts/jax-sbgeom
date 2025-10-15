import jax_sbgeom as jsb
import jax.numpy as jnp
import numpy as onp
import StellBlanket.SBGeom as SBGeom 
import jax
import time

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


def _1d_sampling_grid(fs_jax : jsb.flux_surfaces.FluxSurface, n_s : int = 50, n_theta : int = 55, n_phi : int = 45, include_axis = True):
    ss, tt, pp = _sampling_grid(fs_jax, n_s, n_theta, n_phi, include_axis)
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

def print_timings(name : str, time_jax : float, std_jax : float, time_sbgeom : float, std_sbgeom : float):
    print(f"{name:20s} passed | jax-sbgeom: {time_jax*1e3:8.3f} ms ± {std_jax*1e3:8.3f} ms | SBGeom: {time_sbgeom*1e3:8.3f} ms ± {std_sbgeom*1e3:8.3f} ms")

# ===================================================================================================================================================================================
#                                                                           Positions
# ===================================================================================================================================================================================

def test_position(vmec_file = "/home/tbogaarts/data/helias5_vmec.nc4", n_repetitions = 1):
    fs_jax, fs_sbgeom = _get_flux_surfaces(vmec_file)

    pos_jax, time_jax, std_jax          = time_jsb_function(fs_jax.cartesian_position, *_sampling_grid(fs_jax))
    pos_sbgeom, time_sbgeom, std_sbgeom = time_jsb_function(fs_sbgeom.Return_Position, *_1d_sampling_grid(fs_jax), jsb=False)
    
    assert jnp.allclose(pos_jax, pos_sbgeom.reshape(pos_jax.shape), atol=1e-13)

    print_timings("Position", time_jax, std_jax, time_sbgeom, std_sbgeom)


def test_normals(vmec_file = "/home/tbogaarts/data/helias5_vmec.nc4", n_repetitions = 1):
    fs_jax, fs_sbgeom = _get_flux_surfaces(vmec_file)

    norm_jax, time_jax, std_jax          = time_jsb_function(fs_jax.normal, *_sampling_grid(fs_jax, include_axis=False))
    norm_sbgeom, time_sbgeom, std_sbgeom = time_jsb_function(fs_sbgeom.Return_Normal, *_1d_sampling_grid(fs_jax, include_axis=False), jsb=False)
    
    assert jnp.allclose(norm_jax, norm_sbgeom.reshape(norm_jax.shape), atol=1e-13)

    print_timings("Normal", time_jax, std_jax, time_sbgeom, std_sbgeom)


# ===================================================================================================================================================================================
#                                                                           Meshing
# ===================================================================================================================================================================================

def test_meshing_surface(vmec_file = "/home/tbogaarts/data/helias5_vmec.nc4", n_repetitions = 1):
    fs_jax, fs_sbgeom = _get_flux_surfaces(vmec_file)

    tor_extent= jsb.flux_surfaces.flux_surfaces_base.ToroidalExtent.half_module(fs_jax)
    s = 0.356622756

    n_theta = 500 
    n_phi  = 60

    (pos_jax, tri_jax), time_jax, std_jax          = time_jsb_function_mult(jsb.flux_surfaces.flux_surface_meshing.Mesh_Surface, fs_jax, s, n_theta, n_phi, tor_extent, True)
    mesh_sbgeom, time_sbgeom, std_sbgeom         = time_jsb_function(fs_sbgeom.Mesh_Surface, s, 0.0, n_phi, n_theta, tor_extent.start, tor_extent.end, jsb=False)

    assert jnp.allclose(pos_jax, mesh_sbgeom.vertices, atol=1e-13)
    assert jnp.allclose(tri_jax, mesh_sbgeom.connectivity, atol=1e-13)

    print_timings("Mesh_Surface", time_jax, std_jax, time_sbgeom, std_sbgeom)