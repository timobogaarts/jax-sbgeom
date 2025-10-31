import os

import jax_sbgeom as jsb
import jax.numpy as jnp
import numpy as onp
import StellBlanket.SBGeom as SBGeom 
import jax
jax.config.update("jax_enable_x64", True)


import time
from functools import partial
from jax_sbgeom.flux_surfaces.flux_surfaces_base import _check_whether_make_normals_point_outwards_required, ToroidalExtent
import pytest
from functools import lru_cache

from jax_sbgeom.jax_utils.utils import _mesh_to_pyvista_mesh


def _check_vectorized(fun):
    s_1 = jnp.array([0.5264, 0.567837])
    fun(1.0, 0.2 ,0.3)
    
    fun(s_1, 0.2, 0.3)
    fun(0.2, s_1, 0.2)
    fun(0.2, 0.2, s_1)

    fun(0.2, s_1, s_1)
    fun(s_1, 0.2, s_1)
    fun(s_1, s_1, 0.3)
    
    fun(s_1, s_1, s_1)
        
    b_0, b_1 = jnp.meshgrid(s_1, s_1, indexing='ij')

    fun(1.0, b_0, b_1)
    fun(b_0, 1.0, b_1)
    fun(b_0, b_1, 1.0)
    fun(b_0, b_1, b_1)


def _get_files():
    vmec_files = ["/home/tbogaarts/stellarator_paper/base_data/vmecs/helias3_vmec.nc4", "/home/tbogaarts/stellarator_paper/base_data/vmecs/helias5_vmec.nc4", "/home/tbogaarts/stellarator_paper/base_data/vmecs/squid_vmec.nc4"]
    return vmec_files

@pytest.fixture(scope="session", params = _get_files())
def _get_flux_surfaces(request):    
    fs_jax    = jsb.flux_surfaces.FluxSurface.from_hdf5(request.param)
    fs_sbgeom = SBGeom.VMEC.Flux_Surfaces_From_HDF5(request.param)
    return fs_jax, fs_sbgeom

def _sampling_grid(fs_jax : jsb.flux_surfaces.FluxSurface, n_s : int = 6, n_theta : int = 7, n_phi : int = 5, include_axis = True):

    ss = jax.lax.cond(include_axis, lambda x : jnp.linspace(0,1,n_s), lambda x : jnp.linspace(0,1, n_s + 1)[1:], None)
    
    tt = jnp.linspace(0, 2 * jnp.pi, n_theta, endpoint=False)
    pp = jnp.linspace(0, 2 * jnp.pi / fs_jax.settings.nfp, n_phi, endpoint=True)
    return jnp.meshgrid(ss, tt, pp, indexing='ij')


def _1d_sampling_grid(fs_jax : jsb.flux_surfaces.FluxSurface, n_s : int = 6, n_theta : int = 7, n_phi : int = 5, include_axis = True, reverse_theta : bool = False):
    ss, tt, pp = _sampling_grid(fs_jax, n_s, n_theta, n_phi, include_axis)
    if reverse_theta:
        tt = - onp.array(tt)
    return onp.array(ss).ravel(), onp.zeros(ss.shape).ravel(), onp.array(tt).ravel(), onp.array(pp).ravel()

def _get_single_flux_surface(vmec_file):
    fs_jax    = jsb.flux_surfaces.FluxSurface.from_hdf5(vmec_file)
    return fs_jax


# ===================================================================================================================================================================================
#                                                                          Base Flux Surface functions
# ===================================================================================================================================================================================
   
def _check_position_both(fs_jax, fs_sbgeom, sampling_func, sampling_func_1d, atol =1e-13):
    pos_jax     = fs_jax.cartesian_position(*sampling_func(fs_jax) )
    pos_sbgeom  = fs_sbgeom.Return_Position(*sampling_func_1d(fs_jax, reverse_theta = fs_sbgeom.du_x_dv_sign() == 1.0))
    onp.testing.assert_allclose(pos_jax, pos_sbgeom.reshape(pos_jax.shape), atol = atol)    

   
def _check_normals_both(fs_jax, fs_sbgeom, sampling_func, sampling_func_1d, atol =1e-13):
    norm_jax    = fs_jax.normal(*sampling_func(fs_jax, include_axis=False))
    norm_sbgeom = fs_sbgeom.Return_Normal(*sampling_func_1d(fs_jax, include_axis=False, reverse_theta=fs_sbgeom.du_x_dv_sign() == 1.0))      
    onp.testing.assert_allclose(norm_jax, norm_sbgeom.reshape(norm_jax.shape), atol= atol)
    

def _check_principal_curvatures_both(fs_jax, fs_sbgeom, sampling_func, sampling_func_1d, atol = 1e-13):
    curv_jax         = fs_jax.principal_curvatures(*sampling_func(fs_jax, include_axis=False))
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
    curv_sbgeom = return_all_principal_curvatures(*sampling_func_1d(fs_jax, include_axis=False, reverse_theta=fs_sbgeom.du_x_dv_sign() == 1.0)) 
    onp.testing.assert_allclose(curv_jax, curv_sbgeom.reshape(curv_jax.shape), atol= atol)    
    
def test_position(_get_flux_surfaces):    
    fs_jax, fs_sbgeom = _get_flux_surfaces
    _check_position_both(fs_jax, fs_sbgeom, _sampling_grid, _1d_sampling_grid)

def test_normals(_get_flux_surfaces):
    fs_jax, fs_sbgeom = _get_flux_surfaces
    _check_normals_both(fs_jax, fs_sbgeom, _sampling_grid, _1d_sampling_grid)

def test_principal_curvatures(_get_flux_surfaces):
   fs_jax, fs_sbgeom = _get_flux_surfaces
   _check_principal_curvatures_both(fs_jax, fs_sbgeom, _sampling_grid, _1d_sampling_grid) 


@pytest.mark.slow
def test_vectorization(_get_flux_surfaces):
    fs_jax, fs_sbgeom = _get_flux_surfaces
    _check_vectorized(fs_jax.cartesian_position)
    _check_vectorized(fs_jax.normal)
    _check_vectorized(fs_jax.principal_curvatures)
# ===================================================================================================================================================================================
#                                                                           Meshing
# ===================================================================================================================================================================================
def _flip_vertices_theta(positions, n_theta, n_phi):
    positions_rs = positions.reshape(n_theta, n_phi, 3)

    first = jnp.take(positions_rs, indices = 0, axis = 0)
    rest = jnp.flip(jnp.take(positions_rs, indices = jnp.arange(1, n_theta), axis = 0), axis = 0)
    positions_rs_flipped = jnp.concatenate([first[None, :, :], rest], axis = 0)
    return positions_rs_flipped.reshape(-1, 3)

def _flip_vertices_theta_tetrahedral(positions, n_theta, n_phi, axis_included : bool):
    vmap_flip = jax.vmap(partial(_flip_vertices_theta, n_theta = n_theta, n_phi=n_phi), in_axes=0, out_axes=0)

    if axis_included:        
        pos_surfs      = positions[n_phi:].reshape(-1, n_theta, n_phi, 3)        
        pos_theta_flip = vmap_flip( pos_surfs ).reshape(-1,3)
        total_flip     = jnp.vstack([positions[:n_phi], pos_theta_flip])
    else:
        total_flip     = vmap_flip(positions.reshape(-1, n_theta, n_phi, 3)).reshape(-1,3)
    return total_flip

def _check_meshing_surface_both(fs_jax, fs_sbgeom,  smax, tor_extent : str = 'half_module', atol = 1e-13):    
    if tor_extent == 'half_module':
        tor_extent= jsb.flux_surfaces.flux_surfaces_base.ToroidalExtent.half_module(fs_jax)
    elif tor_extent == 'full':
        tor_extent= jsb.flux_surfaces.flux_surfaces_base.ToroidalExtent.full()
    else:
        raise ValueError(f"Unknown toroidal extent: {tor_extent}")
    s = 0.356622756

    n_theta = 50
    n_phi  = 60

    pos_jax, tri_jax        = jsb.flux_surfaces.flux_surface_meshing.mesh_surface(fs_jax, smax, tor_extent,  n_theta, n_phi, True)
    mesh_sbgeom     = fs_sbgeom.Mesh_Surface(min(smax, 1), max(smax- 1, 0.0), n_phi, n_theta, tor_extent.start, tor_extent.end, True)

    # The sampling cannot be directly influenced. 
    # Instead, we just reverse the theta direction by flipping all vertices if required 
    # This also takes care of the fact that SBGeom does not have normals facing outwards: they get flipped as well so will be equal again.
    reverse_theta = fs_sbgeom.du_x_dv_sign() == 1.0
    if reverse_theta:
        pos_jax_mod = _flip_vertices_theta(pos_jax, n_theta, n_phi)
    else:
        pos_jax_mod = pos_jax

    onp.testing.assert_allclose(pos_jax_mod, mesh_sbgeom.vertices, atol = atol)
    onp.testing.assert_allclose(tri_jax, mesh_sbgeom.connectivity, atol = atol)    



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


def _check_all_closed_surfaces(fs_jax, fs_sbgeom, atol = 1e-10):
    surfaces = _get_all_closed_surfaces(fs_jax )

    for surf in surfaces:
        points, connectivity = surf
        mesh = _mesh_to_pyvista_mesh(points, connectivity)  
        onp.testing.assert_allclose(mesh.volume, jsb.flux_surfaces.flux_surface_meshing._volume_of_mesh(points, connectivity), atol= atol)

    
def _check_all_tetrahedral_meshes(fs_jax, fs_sbgeom, atol = 1e-13):
    n_phi   = 31
    n_theta = 20
    s_disc = 5

    s_values = [
        onp.linspace(0.0, 1.0, s_disc),
        onp.linspace(0.0, 1.0, s_disc),
        onp.linspace(0.1, 1.0, s_disc),
        onp.linspace(0.1, 1.0, s_disc)
    ]
    phi_ends = [
        onp.pi,
        2 * onp.pi,
        onp.pi,
        2 * onp.pi
    ]
    def create_jax_mesh(s_values, phi_end):
                    
        mesh_j =  jsb.flux_surfaces.flux_surface_meshing.mesh_tetrahedra(fs_jax, s_values, ToroidalExtent(0, phi_end), n_theta, n_phi)
        
        return mesh_j[0], mesh_j[1]
    
    def create_sbgeom_mesh(s_values, phi_end):            
        mesh = fs_sbgeom.Mesh_Tetrahedrons(onp.array(s_values), onp.zeros(len(s_values)), n_phi, n_theta, 0.0, float(phi_end))
    
        return mesh.vertices, mesh.connectivity
        
    for s_i, phi_i in zip(s_values, phi_ends):
        mesh_jax    = create_jax_mesh(s_i, phi_i)
        mesh_sbgeom = create_sbgeom_mesh(s_i, phi_i)

        onp.testing.assert_allclose(mesh_jax[1], mesh_sbgeom[1]), "Tetrahedral mesh connectivity does not match SBGeom"

        if fs_sbgeom.du_x_dv_sign() == 1.0:
            total_flip = _flip_vertices_theta_tetrahedral(mesh_jax[0], n_theta, n_phi, axis_included = s_i[0]==0)
            onp.testing.assert_allclose(total_flip, mesh_sbgeom[0], atol = atol), "Tetrahedral mesh points do not match SBGeom"                    
        else:
            onp.testing.assert_allclose(mesh_jax[0], mesh_sbgeom[0], atol = atol), "Tetrahedral mesh points do not match SBGeom"
    
        
def _check_watertight_surfaces(fs_jax, fs_sbgeom, atol = 1e-13):
    n_phi   = 31
    n_theta = 20
    s_disc = 5

    s_values = jnp.linspace(0.1, 1.0, s_disc)
    phi_end  =  0.2 * jnp.pi # sbgeom doesn't handle non-closed surfaces, so we only test for equality with that.

    def create_jax_mesh(s_values, phi_end):                    
        mesh_j =  jsb.flux_surfaces.flux_surface_meshing.mesh_watertight_layers(fs_jax, s_values, ToroidalExtent(0, phi_end), n_theta, n_phi)        
        return mesh_j[0], mesh_j[1]
    
    mesh_jax = create_jax_mesh(s_values, phi_end)

    
    sdiff = onp.diff(s_values)
    mesh_sbgeom = fs_sbgeom.Mesh_Watertight_Flux_Surfaces(float(s_values[0]), 0.0, sdiff, onp.zeros_like(sdiff), n_phi, n_theta, 0.0, 0.2 * onp.pi)

    def check_points():
        reverse_theta = fs_sbgeom.du_x_dv_sign() == 1.0        
        if reverse_theta:
            total_flip = _flip_vertices_theta_tetrahedral(mesh_jax[0], n_theta, n_phi, axis_included = False)
            
            onp.testing.assert_allclose(total_flip, mesh_sbgeom[0], atol = atol), "Watertight mesh points do not match SBGeom"
        else:
            onp.testing.assert_allclose(mesh_jax[0], mesh_sbgeom[0], atol = atol), "Watertight mesh points do not match SBGeom"

    def check_connectivity():
        for i in range(len(mesh_jax[1])):
            conn_jax = mesh_jax[1][i]
            conn_sbgeom = mesh_sbgeom[1][i]
            onp.testing.assert_allclose(conn_jax, conn_sbgeom), "Watertight mesh connectivity does not match SBGeom"
    check_points()
    check_connectivity()    


def test_meshing_surface_half_mod(_get_flux_surfaces):    
    fs_jax, fs_sbgeom = _get_flux_surfaces
    _check_meshing_surface_both(fs_jax, fs_sbgeom, 1.0, tor_extent= "half_module")

def test_meshing_surface_full(_get_flux_surfaces):    
    fs_jax, fs_sbgeom = _get_flux_surfaces
    _check_meshing_surface_both(fs_jax, fs_sbgeom, 1.0, tor_extent= "full")

def test_all_closed_surfaces(_get_flux_surfaces):    
    fs_jax, fs_sbgeom = _get_flux_surfaces
    _check_all_closed_surfaces(fs_jax, fs_sbgeom)

def test_all_tetrahedral_meshes(_get_flux_surfaces):    
    fs_jax, fs_sbgeom = _get_flux_surfaces
    _check_all_tetrahedral_meshes(fs_jax, fs_sbgeom)
  
def test_watertight_surfaces(_get_flux_surfaces):    
    fs_jax, fs_sbgeom = _get_flux_surfaces
    _check_watertight_surfaces(fs_jax, fs_sbgeom)


# ===================================================================================================================================================================================
#                                                                           FluxSurface base functions
# ===================================================================================================================================================================================

def _check_volumes(fs_jax, fs_sbgeom, atol = 1e-13):

    s = 0.5357

    vol_jax  = jsb.flux_surfaces.flux_surfaces_base._volume_from_fourier_half_mod(fs_jax.data, fs_jax.settings, s)
    vol_jax2 = jsb.flux_surfaces.flux_surfaces_base._volume_from_fourier(fs_jax.data, fs_jax.settings, s)

    onp.testing.assert_allclose(vol_jax, vol_jax2, atol=atol)
    

def test_volumes(_get_flux_surfaces):
        
    fs_jax, fs_sbgeom = _get_flux_surfaces
    _check_volumes(fs_jax, fs_sbgeom)

# ===================================================================================================================================================================================
#                                                                          Conversion to Fourier
# ===================================================================================================================================================================================
def check_RZ_to_VMEC(Rgrid, Zgrid):
    from jax_sbgeom.flux_surfaces.convert_to_VMEC import _convert_cos_sin_to_vmec, _dft_forward, _cos_sin_from_dft_forward
    # Reimplementation
    R_dft = _dft_forward(Rgrid)
    Z_dft = _dft_forward(Zgrid)

    R_ckl, R_cmkl, R_skl, R_smkl = _cos_sin_from_dft_forward(R_dft)
    Z_ckl, Z_cmkl, Z_skl, Z_smkl = _cos_sin_from_dft_forward(Z_dft)

    # SBGeom implementation
    R_dft_sbg, N, M = SBGeom.VMEC._Scaled_DFT(Rgrid)
    Z_dft_sbg, N, M = SBGeom.VMEC._Scaled_DFT(Zgrid)

    R_ckl_sbg, R_cmkl_sbg, R_skl_sbg, R_smkl_sbg = SBGeom.VMEC._Calculate_CosSin_From_DFT(R_dft_sbg , N, M)
    Z_ckl_sbg, Z_cmkl_sbg, Z_skl_sbg, Z_smkl_sbg = SBGeom.VMEC._Calculate_CosSin_From_DFT(Z_dft_sbg , N, M)

    onp.testing.assert_allclose(R_dft, R_dft_sbg, rtol=1e-12, atol=1e-12)
    onp.testing.assert_allclose(Z_dft, Z_dft_sbg, rtol=1e-12, atol=1e-12)

    onp.testing.assert_allclose(R_ckl, R_ckl_sbg, rtol=1e-12, atol=1e-12)
    onp.testing.assert_allclose(R_cmkl, R_cmkl_sbg, rtol=1e-12, atol=1e-12)
    onp.testing.assert_allclose(R_skl, R_skl_sbg, rtol=1e-12, atol=1e-12)
    onp.testing.assert_allclose(R_smkl, R_smkl_sbg, rtol=1e-12, atol=1e-12)

    onp.testing.assert_allclose(Z_ckl, Z_ckl_sbg, rtol=1e-12, atol=1e-12)
    onp.testing.assert_allclose(Z_cmkl, Z_cmkl_sbg, rtol=1e-12, atol=1e-12)
    onp.testing.assert_allclose(Z_skl, Z_skl_sbg, rtol=1e-12, atol=1e-12)
    onp.testing.assert_allclose(Z_smkl, Z_smkl_sbg, rtol=1e-12, atol=1e-12)

    Rmnc = _convert_cos_sin_to_vmec(R_ckl, R_cmkl, R_skl, R_smkl,True)
    Zmns = _convert_cos_sin_to_vmec(Z_ckl, Z_cmkl, Z_skl, Z_smkl,False)
    

    Rmnc_sbg = SBGeom.VMEC._Convert_CosSin_to_VMEC_R(R_ckl_sbg, R_cmkl_sbg, R_skl_sbg, R_smkl_sbg)
    Zmns_sbg = SBGeom.VMEC._Convert_CosSin_to_VMEC_Z(Z_ckl_sbg, Z_cmkl_sbg, Z_skl_sbg, Z_smkl_sbg)
    

    onp.testing.assert_allclose(Rmnc, Rmnc_sbg[0], rtol=1e-12, atol=1e-12)
    onp.testing.assert_allclose(Zmns, Zmns_sbg[0], rtol=1e-12, atol=1e-12)

def create_RZ_grid(fs_jax, s : float, n_theta : int, n_phi : int):
    theta = jnp.linspace(0, 2 * jnp.pi, n_theta, endpoint=False)
    phi   = jnp.linspace(0, 2 * jnp.pi / fs_jax.settings.nfp, n_phi, endpoint=True)
    theta_grid, phi_grid = jnp.meshgrid(theta, phi, indexing='ij')
    positions_jax = fs_jax.cylindrical_position(s, theta_grid, phi_grid)
    
    return positions_jax[...,0], positions_jax[...,1]

def _variations():
    n_theta_list = [51, 52, 51, 52]
    n_phi_list   = [51, 51, 52, 52]
    return n_theta_list, n_phi_list

@pytest.mark.slow
def test_RZ_to_VMEC():
    fs_jax = _get_single_flux_surface(_get_files()[0])

    for n_theta, n_phi in zip(*_variations()):
        check_RZ_to_VMEC(*create_RZ_grid(fs_jax, 0.55, n_theta, n_phi))

def test_RZ_to_VMEC_lcfs(_get_flux_surfaces):
    fs_jax, fs_sbgeom = _get_flux_surfaces

    n_theta = fs_jax.settings.mpol * 2 + 1 # just above nyquist
    n_phi   = fs_jax.settings.ntor * 2 + 1 # just above nyquist

    sampling_r, sampling_z = jsb.flux_surfaces.convert_to_VMEC._create_sampling_rz(fs_jax, 1.0, n_theta, n_phi)
    print(sampling_r.shape, sampling_z.shape)

    Rmnc, Zmns             = jsb.flux_surfaces.convert_to_VMEC._rz_to_vmec_representation(sampling_r, sampling_z)

    onp.testing.assert_allclose(Rmnc, fs_jax.data.Rmnc[-1,:], rtol=1e-12, atol=1e-12)
    onp.testing.assert_allclose(Zmns, fs_jax.data.Zmns[-1,:], rtol=1e-12, atol=1e-12)


def test_extension_VMEC(_get_flux_surfaces):
    fs_jax, fs_sbgeom = _get_flux_surfaces

    # Create a new flux surface with higher mpol and ntor
    mpol_new = fs_jax.settings.mpol + 11
    ntor_new = fs_jax.settings.ntor + 13
    sampling_grid  = _sampling_grid(fs_jax)

    Rmnc_Zmns_new = jsb.flux_surfaces.convert_to_VMEC._convert_to_different_ntor_mpol(jnp.stack([fs_jax.data.Rmnc, fs_jax.data.Zmns], axis=0), mpol_new, ntor_new, fs_jax.settings.mpol, fs_jax.settings.ntor)

    new_settings = jsb.flux_surfaces.flux_surfaces_base.FluxSurfaceSettings(
        mpol = mpol_new,
        ntor = ntor_new,
        nfp  = fs_jax.settings.nfp,
        nsurf= fs_jax.settings.nsurf
    )
    fs_new = jsb.flux_surfaces.FluxSurface(jsb.flux_surfaces.flux_surfaces_base.FluxSurfaceData.from_rmnc_zmns_settings(Rmnc_Zmns_new[0], Rmnc_Zmns_new[1], new_settings                                                       
                                                          ), settings=new_settings)
    onp.testing.assert_allclose(fs_new.cartesian_position(*sampling_grid), fs_jax.cartesian_position(*sampling_grid), rtol=1e-12, atol=1e-12)

    