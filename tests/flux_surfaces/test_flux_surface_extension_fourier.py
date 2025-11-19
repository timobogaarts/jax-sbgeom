import jax_sbgeom as jsb
import jax.numpy as jnp
import numpy as onp
import StellBlanket.SBGeom as SBGeom 
import jax
jax.config.update("jax_enable_x64", True)

import time
from functools import partial
import pytest
from tests.flux_surfaces.test_flux_surface_base import _get_files, _check_vectorized



@pytest.fixture(scope="session", params = _get_files())
def _get_normal_extended_flux_surfaces(request):    
    fs_jax    = jsb.flux_surfaces.FluxSurfaceNormalExtended.from_hdf5(request.param)
    fs_sbgeom = SBGeom.Flux_Surfaces_Normal_Extended(SBGeom.VMEC.Flux_Surfaces_From_HDF5(request.param))
    return fs_jax, fs_sbgeom

@pytest.fixture(scope="session", params = _get_files())
def _get_normal_extended_no_phi_flux_surfaces(request):    
    fs_jax    = jsb.flux_surfaces.FluxSurfaceNormalExtendedNoPhi.from_hdf5(request.param)
    fs_sbgeom = SBGeom.Flux_Surfaces_Normal_Extended_No_Phi(SBGeom.VMEC.Flux_Surfaces_From_HDF5(request.param))
    return fs_jax, fs_sbgeom

@pytest.fixture(scope="session", params = _get_files())
def _get_normal_extended_constant_phi_flux_surfaces(request):    
    fs_jax    = jsb.flux_surfaces.FluxSurfaceNormalExtendedConstantPhi.from_hdf5(request.param)
    fs_sbgeom = SBGeom.Flux_Surfaces_Normal_Extended_Constant_Phi(SBGeom.VMEC.Flux_Surfaces_From_HDF5(request.param))
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

# ======================================================================================================================================================================================
#                                                                   Tests for Conversion to Fourier Representation
# ======================================================================================================================================================================================
def test_conversion_to_fourier_representation(_get_normal_extended_no_phi_flux_surfaces):

    fs_jax, fs_sbgeom = _get_normal_extended_no_phi_flux_surfaces
    
    n_theta = fs_jax.settings.mpol * 2 + 3 # nyquist
    n_phi   = fs_jax.settings.ntor * 2 + 51 # nyquist
    symm    = fs_jax.settings.nfp

    
    theta   = jnp.linspace(0, 2*jnp.pi, n_theta, endpoint=False)
    phi     = jnp.linspace(0, 2*jnp.pi / symm, n_phi, endpoint=False)

    theta_mg, phi_mg = jnp.meshgrid(theta, phi, indexing='ij')    

    d_edge           = 1.12

    fs_data, new_settings = jsb.flux_surfaces.convert_to_vmec.create_fourier_representation(fs_jax, 1.0 + d_edge, theta_mg)
    sbgeom_version          = SBGeom.VMEC.Convert_to_Fourier_Extended(fs_sbgeom, [d_edge], n_theta, n_phi)

    Rmnc_s = sbgeom_version.Rmnc_Extension()
    Zmns_s = sbgeom_version.Zmns_Extension()

    mpol_new = jsb.flux_surfaces.flux_surfaces_base._create_mpol_vector(new_settings)
    ntor_new = jsb.flux_surfaces.flux_surfaces_base._create_ntor_vector(new_settings)
    if fs_sbgeom.du_x_dv_sign() == 1.0:
        Rmnc_flip = jsb.flux_surfaces.flux_surfaces_base._reverse_theta_single(mpol_new, ntor_new, fs_data.Rmnc, True)
        Zmns_flip = jsb.flux_surfaces.flux_surfaces_base._reverse_theta_single(mpol_new, ntor_new, fs_data.Zmns, False)
        
        onp.testing.assert_allclose(onp.array(Rmnc_flip), onp.array(Rmnc_s[0]), rtol=1e-5, atol=1e-8)        
        onp.testing.assert_allclose(onp.array(Zmns_flip), onp.array(Zmns_s[0]), rtol=1e-5, atol=1e-8)
    else:  
        onp.testing.assert_allclose(onp.array(fs_data.Rmnc), onp.array(Rmnc_s[0]), rtol=1e-5, atol=1e-8)        
        onp.testing.assert_allclose(onp.array(fs_data.Zmns), onp.array(Zmns_s[0]), rtol=1e-5, atol=1e-8)

    
    fs_jax_converted = jsb.flux_surfaces.FluxSurface.from_rmnc_zmns_settings(fs_data.Rmnc[None,:], fs_data.Zmns[None, :], new_settings, False) 

    # It should match the original flux surface in the extended region on the sampled grid
    original_sampled_grid = fs_jax.cartesian_position(1.0 + d_edge, theta_mg, phi_mg)
    resampled_grid        = fs_jax_converted.cartesian_position(1.0, theta_mg, phi_mg)

    onp.testing.assert_allclose(onp.array(original_sampled_grid), onp.array(resampled_grid), rtol=1e-12, atol=1e-12)

    

def test_fourier_extension(_get_normal_extended_no_phi_flux_surfaces):
    # We test both the version for one extension surface and multiple extension surfaces

    fs_jax, fs_sbgeom = _get_normal_extended_no_phi_flux_surfaces
    
    n_theta = fs_jax.settings.mpol * 2 + 3 # nyquist
    n_phi   = fs_jax.settings.ntor * 2 + 51 # nyquist
    symm    = fs_jax.settings.nfp

    
    theta   = jnp.linspace(0, 2*jnp.pi, n_theta, endpoint=False)
    phi     = jnp.linspace(0, 2*jnp.pi / symm, n_phi, endpoint=False)

    theta_mg, phi_mg = jnp.meshgrid(theta, phi, indexing='ij')    

    d_edge           = [1.12, 2.25, 3.5]

    sbgeom_version          = SBGeom.VMEC.Convert_to_Fourier_Extended(fs_sbgeom, [1.0, 2.0, 3.0], n_theta, n_phi, d_edge)
    
    Rmnc_s = sbgeom_version.Rmnc_Extension()
    Zmns_s = sbgeom_version.Zmns_Extension()

    Rmnc_jax_list = []
    Zmns_jax_list = []
    mpol_list     = []
    ntor_list     = []
    for idx, i in enumerate(d_edge):
        FS_data, new_settings = jsb.flux_surfaces.convert_to_vmec.create_fourier_representation(fs_jax, 1.0 + i, theta_mg)
        Rmnc_jax_list.append(FS_data.Rmnc)
        Zmns_jax_list.append(FS_data.Zmns)
        mpol_list.append(new_settings.mpol)
        ntor_list.append(new_settings.ntor)
        Rmnc = FS_data.Rmnc
        Zmns = FS_data.Zmns
        
        mpol_new = jsb.flux_surfaces.flux_surfaces_base._create_mpol_vector(new_settings)
        ntor_new = jsb.flux_surfaces.flux_surfaces_base._create_ntor_vector(new_settings)
        if fs_sbgeom.du_x_dv_sign() == 1.0:
            Rmnc_flip = jsb.flux_surfaces.flux_surfaces_base._reverse_theta_single(mpol_new, ntor_new, Rmnc, True)
            Zmns_flip = jsb.flux_surfaces.flux_surfaces_base._reverse_theta_single(mpol_new, ntor_new, Zmns, False)        
            onp.testing.assert_allclose(onp.array(Rmnc_flip), onp.array(Rmnc_s[idx]), rtol=1e-5, atol=1e-8)        
            onp.testing.assert_allclose(onp.array(Zmns_flip), onp.array(Zmns_s[idx]), rtol=1e-5, atol=1e-8)
        else:  
            onp.testing.assert_allclose(onp.array(Rmnc), onp.array(Rmnc_s[idx]), rtol=1e-5, atol=1e-8)        
            onp.testing.assert_allclose(onp.array(Zmns), onp.array(Zmns_s[idx]), rtol=1e-5, atol=1e-8)   

    new_settings = jsb.flux_surfaces.FluxSurfaceSettings(
        mpol = mpol_list[0],
        ntor = ntor_list[0],
        nfp = fs_jax.settings.nfp
    )
    flux_surface_fourier = jsb.flux_surfaces.FluxSurface.from_data_settings_full(jsb.flux_surfaces.FluxSurfaceData(jnp.stack(Rmnc_jax_list, axis=0), jnp.stack(Zmns_jax_list, axis=0)),  new_settings)    
    flux_surface_fourier_extended = jsb.flux_surfaces.flux_surfaces_extended.FluxSurfaceFourierExtended.from_flux_surface_and_extension(fs_jax, flux_surface_fourier)
    s_grid = jnp.linspace(1.0, 4.0, 50)    
    d_grid_sbgeom = jnp.linspace(0.0, 3.0, 50)
    positions_sbgeom = sbgeom_version.Return_Position(onp.ones_like(s_grid), onp.array(d_grid_sbgeom), 0.0, 0.2)    
    positions_jax = flux_surface_fourier_extended.cartesian_position(s_grid, 0.0, 0.2)    
    onp.testing.assert_allclose(onp.array(positions_jax), onp.array(positions_sbgeom), rtol=1e-5, atol=1e-8)

    d_edge           = [1.12]

    sbgeom_version          = SBGeom.VMEC.Convert_to_Fourier_Extended(fs_sbgeom, [1.0], n_theta, n_phi, d_edge)
    
    Rmnc_s = sbgeom_version.Rmnc_Extension()
    Zmns_s = sbgeom_version.Zmns_Extension()

    Rmnc_jax_list = []
    Zmns_jax_list = []
    mpol_list     = []
    ntor_list     = []
    for idx, i in enumerate(d_edge):
        FS_data, new_settings = jsb.flux_surfaces.convert_to_vmec.create_fourier_representation(fs_jax, 1.0 + i, theta_mg)
        Rmnc_jax_list.append(FS_data.Rmnc)
        Zmns_jax_list.append(FS_data.Zmns)
        mpol_list.append(new_settings.mpol)
        ntor_list.append(new_settings.ntor)
        mpol_new = jsb.flux_surfaces.flux_surfaces_base._create_mpol_vector(new_settings)
        ntor_new = jsb.flux_surfaces.flux_surfaces_base._create_ntor_vector(new_settings)

        Rmnc = FS_data.Rmnc
        Zmns = FS_data.Zmns
        if fs_sbgeom.du_x_dv_sign() == 1.0:
            Rmnc_flip = jsb.flux_surfaces.flux_surfaces_base._reverse_theta_single(mpol_new, ntor_new, Rmnc, True)
            Zmns_flip = jsb.flux_surfaces.flux_surfaces_base._reverse_theta_single(mpol_new, ntor_new, Zmns, False)        
            onp.testing.assert_allclose(onp.array(Rmnc_flip), onp.array(Rmnc_s[idx]), rtol=1e-5, atol=1e-8)        
            onp.testing.assert_allclose(onp.array(Zmns_flip), onp.array(Zmns_s[idx]), rtol=1e-5, atol=1e-8)
        else:  
            onp.testing.assert_allclose(onp.array(Rmnc), onp.array(Rmnc_s[idx]), rtol=1e-5, atol=1e-8)        
            onp.testing.assert_allclose(onp.array(Zmns), onp.array(Zmns_s[idx]), rtol=1e-5, atol=1e-8)   
    new_settings = jsb.flux_surfaces.FluxSurfaceSettings(
        mpol = mpol_list[0],
        ntor = ntor_list[0],
        nfp = fs_jax.settings.nfp
    )
    flux_surface_fourier = jsb.flux_surfaces.FluxSurface.from_data_settings_full(jsb.flux_surfaces.FluxSurfaceData(jnp.stack(Rmnc_jax_list, axis=0), jnp.stack(Zmns_jax_list, axis=0)),  new_settings)    

    flux_surface_fourier_extended = jsb.flux_surfaces.flux_surfaces_extended.FluxSurfaceFourierExtended.from_flux_surface_and_extension(fs_jax, flux_surface_fourier)
    s_grid = jnp.linspace(1.0, 2.0, 50)    
    d_grid_sbgeom = jnp.linspace(0.0, 1.0, 50)
    positions_sbgeom = sbgeom_version.Return_Position(onp.ones_like(s_grid), onp.array(d_grid_sbgeom), 0.0, 0.2)    
    positions_jax = flux_surface_fourier_extended.cartesian_position(s_grid, 0.0, 0.2)    
    onp.testing.assert_allclose(onp.array(positions_jax), onp.array(positions_sbgeom), rtol=1e-5, atol=1e-8)