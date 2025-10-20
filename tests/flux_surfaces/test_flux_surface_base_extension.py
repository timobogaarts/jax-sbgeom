import jax_sbgeom as jsb
import jax.numpy as jnp
import numpy as onp
import StellBlanket.SBGeom as SBGeom 
import jax
import time

from functools import partial
from jax_sbgeom.flux_surfaces.flux_surfaces_base import _check_whether_make_normals_point_outwards_required, ToroidalExtent
import pytest
from tests.flux_surfaces.test_flux_surface_base import _get_flux_surfaces, _get_files, time_jsb_function, time_jsb_function_mult, time_jsb_function_nested, print_timings, _sampling_grid, _1d_sampling_grid
jax.config.update("jax_enable_x64", True)

cached= False
if cached:
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


def _get_files():
    vmec_files = ["/home/tbogaarts/stellarator_paper/base_data/vmecs/helias3_vmec.nc4", "/home/tbogaarts/stellarator_paper/base_data/vmecs/helias5_vmec.nc4", "/home/tbogaarts/stellarator_paper/base_data/vmecs/squid_vmec.nc4"]
    return vmec_files

def _get_extended_flux_surfaces(vmec_file):    
    fs_jax    = jsb.flux_surfaces.FluxSurfaceNormalExtended.from_hdf5(vmec_file)
    fs_sbgeom = SBGeom.Flux_Surfaces_Normal_Extended(SBGeom.VMEC.Flux_Surfaces_From_HDF5(vmec_file))
    return fs_jax, fs_sbgeom

smax = 2.0
def _extended_sampling_grid(fs_jax : jsb.flux_surfaces.FluxSurface, n_s : int = 6, n_theta : int = 7, n_phi : int = 5, include_axis = True):

    ss = jax.lax.cond(include_axis, lambda x : jnp.linspace(0,smax,n_s), lambda x : jnp.linspace(0,smax, n_s + 1)[1:], None)
    
    tt = jnp.linspace(0, 2 * jnp.pi, n_theta, endpoint=False)
    pp = jnp.linspace(0, 2 * jnp.pi / fs_jax.settings.nfp, n_phi, endpoint=True)
    return jnp.meshgrid(ss, tt, pp, indexing='ij')

def _extended_1d_sampling_grid(fs_jax : jsb.flux_surfaces.FluxSurface, n_s : int = 6, n_theta : int = 7, n_phi : int = 5, include_axis = True, reverse_theta : bool = False):
    ss, tt, pp = _extended_sampling_grid(fs_jax, n_s, n_theta, n_phi, include_axis)
    if reverse_theta:
        tt = - onp.array(tt)
    s = onp.minimum(onp.array(ss).ravel(), 1.0)
    d = onp.maximum(onp.array(ss).ravel() - 1.0, 0.0)
    t = onp.array(tt).ravel()
    p = onp.array(pp).ravel()
    return s, d, t, p

# ===================================================================================================================================================================================
#                                                                          
# ===================================================================================================================================================================================

@pytest.mark.parametrize("vmec_file", _get_files())
def test_normal_extension_position(vmec_file, n_repetitions=1):
    fs_jax, fs_sbgeom = _get_extended_flux_surfaces(vmec_file)
    
    pos_jax, time_jax, std_jax          = time_jsb_function(fs_jax.cartesian_position, *_extended_sampling_grid(fs_jax),    n_repetitions= n_repetitions )

    reverse_theta = fs_sbgeom.du_x_dv_sign() == 1.0
    pos_sbgeom, time_sbgeom, std_sbgeom = time_jsb_function(fs_sbgeom.Return_Position, *_extended_1d_sampling_grid(fs_jax, reverse_theta=reverse_theta), n_repetitions=n_repetitions, jsb=False)
    
    assert jnp.allclose(pos_jax, pos_sbgeom.reshape(pos_jax.shape), atol=1e-13)

    print_timings("Position", time_jax, std_jax, time_sbgeom, std_sbgeom)

@pytest.mark.parametrize("vmec_file", _get_files())
def test_normal_extension_principal_curvatures(vmec_file, n_repetitions = 1):
    
    fs_jax, fs_sbgeom = _get_extended_flux_surfaces(vmec_file)

    curv_jax, time_jax, std_jax          = time_jsb_function(fs_jax.principal_curvatures, *_extended_sampling_grid(fs_jax, include_axis=False),      n_repetitions= n_repetitions)

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

    curv_sbgeom, time_sbgeom, std_sbgeom = time_jsb_function(return_all_principal_curvatures, *_extended_1d_sampling_grid(fs_jax, include_axis=False, reverse_theta=reverse_theta), n_repetitions= n_repetitions, jsb=False)
    
    assert jnp.allclose(curv_jax, curv_sbgeom.reshape(curv_jax.shape), atol=1e-13)

    print_timings("Principal Curvatures", time_jax, std_jax, time_sbgeom, std_sbgeom)

# we do not need to test the normal vector: it is by definition the same as the base flux surface and SBGeom does not return it.
