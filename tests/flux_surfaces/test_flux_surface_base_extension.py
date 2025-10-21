import jax_sbgeom as jsb
import jax.numpy as jnp
import numpy as onp
import StellBlanket.SBGeom as SBGeom 
import jax
import time

from functools import partial

import pytest
from tests.flux_surfaces.test_flux_surface_base import _get_files, _check_vectorized



jax.config.update("jax_enable_x64", True)

cached= True
if cached:
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


@pytest.fixture(scope="session", params = _get_files())
def _get_normal_extended_flux_surfaces(request):    
    fs_jax    = jsb.flux_surfaces.FluxSurfaceNormalExtended.from_hdf5(request.param)
    fs_sbgeom = SBGeom.Flux_Surfaces_Normal_Extended(SBGeom.VMEC.Flux_Surfaces_From_HDF5(request.param))
    return fs_jax, fs_sbgeom

def _extension_s_max():
    return 2.0

def _extended_sampling_grid(fs_jax : jsb.flux_surfaces.FluxSurface, n_s : int = 6, n_theta : int = 7, n_phi : int = 5, include_axis = True):

    ss = jax.lax.cond(include_axis, lambda x : jnp.linspace(0,_extension_s_max(),n_s), lambda x : jnp.linspace(0, _extension_s_max(), n_s + 1)[1:], None)
    
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

def _extended_1d_sampling_grid_stop_1(fs_jax : jsb.flux_surfaces.FluxSurface, n_s : int = 6, n_theta : int = 7, n_phi : int = 5, include_axis = True, reverse_theta : bool = False):
    ss, tt, pp = _extended_sampling_grid(fs_jax, n_s, n_theta, n_phi, include_axis)
    if reverse_theta:
        tt = - onp.array(tt)
    s = onp.minimum(onp.array(ss).ravel(), 1.0)
    d = 0.0 * onp.array(ss).ravel()
    t = onp.array(tt).ravel()
    p = onp.array(pp).ravel()
    return s, d, t, p

# ===================================================================================================================================================================================
#                                                                          Positions 
# ===================================================================================================================================================================================
from tests.flux_surfaces.test_flux_surface_base import _check_position_both, _check_normals_both, _check_principal_curvatures_both
def test_normal_extension_position(_get_normal_extended_flux_surfaces, n_repetitions=1):
    fs_jax, fs_sbgeom = _get_normal_extended_flux_surfaces
    _check_position_both(fs_jax, fs_sbgeom, _extended_sampling_grid, _extended_1d_sampling_grid, n_repetitions=n_repetitions)

def test_normal_extension_normals(_get_normal_extended_flux_surfaces, n_repetitions=1):
    fs_jax, fs_sbgeom = _get_normal_extended_flux_surfaces
    # SBGeom doesn't return normals in the extended region, so only test up to s=1 (that should be the same)
    _check_normals_both(fs_jax, fs_sbgeom, _extended_sampling_grid, _extended_1d_sampling_grid_stop_1, n_repetitions=n_repetitions)
   

def test_normal_extension_principal_curvatures(_get_normal_extended_flux_surfaces, n_repetitions=1):
    fs_jax, fs_sbgeom = _get_normal_extended_flux_surfaces
    
    _check_principal_curvatures_both(fs_jax, fs_sbgeom, _extended_sampling_grid, _extended_1d_sampling_grid, n_repetitions=n_repetitions)

def test_vectorization(_get_normal_extended_flux_surfaces):
    fs_jax, fs_sbgeom = _get_normal_extended_flux_surfaces
    _check_vectorized(fs_jax.cartesian_position)
    _check_vectorized(fs_jax.normal)
    _check_vectorized(fs_jax.principal_curvatures)
# we do not need to test the normal vector: it is by definition the same as the base flux surface and SBGeom does not return it.
