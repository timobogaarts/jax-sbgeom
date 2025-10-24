import os
import jax_sbgeom as jsb
import jax.numpy as jnp
import numpy as onp
import StellBlanket.SBGeom as SBGeom 
import jax
import time

from functools import partial

import pytest
from functools import lru_cache

jax.config.update("jax_enable_x64", True)


def test_backend():
    import jax
    print("Backend:", jax.default_backend())

def _check_single_vectorized(fun):
    s_0 = 0.1
    s_1 = jnp.linspace(0.0, 1.0, 24)
    s_2 = s_1.reshape((4,6))
    s_3 = s_1.reshape((2,3,4))


    fun(s_0)
    r_1 = fun(s_1)
    r_2 = fun(s_2)
    r_3 = fun(s_3)

    onp.testing.assert_allclose(r_1, r_2.reshape(r_1.shape))
    onp.testing.assert_allclose(r_1, r_3.reshape(r_1.shape))

def _get_coil_files():
    return ["/home/tbogaarts/stellarator_paper/base_data/vmecs/HELIAS3_coils_all.h5", "/home/tbogaarts/stellarator_paper/base_data/vmecs/HELIAS5_coils_all.h5", "/home/tbogaarts/stellarator_paper/base_data/vmecs/squid_coilset.h5"]

@pytest.fixture(scope="session", params = _get_coil_files())
def _get_all_discrete_coils(request):    
    coilset_sbgeom = SBGeom.Coils.Discrete_Coil_Set_From_HDF5(request.param)    
    coilset_jaxsbgeom = [jsb.coils.DiscreteCoil.from_positions(coilset_sbgeom[i].Get_Vertices()) for i in range(coilset_sbgeom.Number_of_Coils())]
    
    return coilset_jaxsbgeom, coilset_sbgeom


@pytest.fixture(scope="session", params = _get_coil_files())
def _get_all_fourier_coils(request):
    coilset_sbgeom = SBGeom.Coils.Discrete_Coil_Set_From_HDF5(request.param)
    coilset_fourier = SBGeom.Coils.Convert_to_Fourier_Coils(coilset_sbgeom)
    
    coilset_jax = [jsb.coils.FourierCoil(jnp.array(i.Get_Fourier_Cos()), jnp.array(i.Get_Fourier_Sin()), jnp.array(i.Get_Centre())) for i in coilset_fourier]
    return coilset_jax, coilset_fourier

@pytest.fixture(scope="session", params = _get_coil_files())
def _get_all_fourier_coils_truncated(request):
    # This is just to ensure that we have the shaping correct
    coilset_sbgeom = SBGeom.Coils.Discrete_Coil_Set_From_HDF5(request.param)
    coilset_fourier = SBGeom.Coils.Convert_to_Fourier_Coils(coilset_sbgeom, Nftrunc = 5)
    
    coilset_jax = [jsb.coils.FourierCoil(jnp.array(i.Get_Fourier_Cos()), jnp.array(i.Get_Fourier_Sin()), jnp.array(i.Get_Centre())) for i in coilset_fourier]
    return coilset_jax, coilset_fourier

#=================================================================================================================================================
#                                                   Position & Tangent tests
#=================================================================================================================================================

def _sampling_s(n_s : int = 111):
    return jnp.linspace(0.0, 1.0, n_s)

def _check_positions(coilset_jsb, coilset_sbgeom, atol = 1e-12):
    for i in range(coilset_sbgeom.Number_of_Coils()):
        coil_jsb = coilset_jsb[i]
        coil_sbgeom = coilset_sbgeom[i]
        s_samples = _sampling_s()
        pos_jsb    = coil_jsb.position(s_samples)
        pos_sbgeom = coil_sbgeom.Position(onp.array(s_samples))

        onp.testing.assert_allclose(pos_jsb, pos_sbgeom, atol=atol)

def _check_tangent(coilset_jsb, coilset_sbgeom, atol = 1e-12):
    for i in range(coilset_sbgeom.Number_of_Coils()):
        coil_jsb = coilset_jsb[i]
        coil_sbgeom = coilset_sbgeom[i]
        s_samples = _sampling_s()
        pos_jsb    = coil_jsb.tangent(s_samples)
        pos_sbgeom = coil_sbgeom.Tangent(onp.array(s_samples))

        onp.testing.assert_allclose(pos_jsb, pos_sbgeom, atol=atol)

def test_discrete_coil_position(_get_all_discrete_coils):
    coilset_jaxsbgeom, coilset_sbgeom = _get_all_discrete_coils
    _check_positions(coilset_jaxsbgeom, coilset_sbgeom, atol=1e-12)

def test_discrete_coil_tangent(_get_all_discrete_coils):
    coilset_jaxsbgeom, coilset_sbgeom = _get_all_discrete_coils
    _check_tangent(coilset_jaxsbgeom, coilset_sbgeom, atol=1e-12)

def test_fourier_coil_position(_get_all_fourier_coils):
    coilset_jax, coilset_sbgeom = _get_all_fourier_coils
    _check_positions(coilset_jax, coilset_sbgeom, atol=1e-12)
def test_fourier_coil_tangent(_get_all_fourier_coils):
    coilset_jax, coilset_sbgeom = _get_all_fourier_coils
    _check_tangent(coilset_jax, coilset_sbgeom, atol=1e-12)

#=================================================================================================================================================
#                                                   Finite Size Tests
#=================================================================================================================================================
def _sampling_s_finite_size(n_s : int = 111):
    return jnp.linspace(0.0, 1.0, n_s, endpoint=False)

def _switch_finite_size(coil_sbgeom, width_0, width_1, method, ns, **kwargs):
    if method == "centroid":
        return coil_sbgeom.Finite_Size_Lines_Centroid(width_1, width_0, ns)
    elif method == "frenet_serret":
        return coil_sbgeom.Finite_Size_Lines_Frenet(width_1, width_0, ns)
    elif method == "rmf":
        return coil_sbgeom.Finite_Size_Lines_RMF(width_1, width_0, ns)
    else:
        raise NotImplementedError(f"Finite size method '{method}' not implemented for SBGeom discrete coils.")

def _switch_finite_size_cjax(coil_jax, width_0, width_1, method, ns, **kwargs):
    if method == "centroid":
        finitesize_coil = jsb.coils.base_coil.FiniteSizeCoil(coil_jax, jsb.coils.base_coil.CentroidFrame())
        return finitesize_coil
    elif method == "frenet_serret":
        finitesize_coil = jsb.coils.base_coil.FiniteSizeCoil(coil_jax, jsb.coils.base_coil.FrenetSerretFrame())
        return finitesize_coil
    elif method == "rmf":
        number_of_rmf_samples = kwargs.get("number_of_rmf_samples", 1000)
        kwargdict = {"number_of_rmf_samples" : number_of_rmf_samples}
        finitesize_coil = jsb.coils.base_coil.FiniteSizeCoil(coil_jax, jsb.coils.base_coil.RotationMinimizedFrame.from_coil(coil_jax, number_of_rmf_samples))
        return finitesize_coil
    else:
        raise NotImplementedError(f"Finite size method '{method}' not implemented for SBGeom discrete coils.")

def _switch_finite_size_cjax_lines(coil_jax, width_0, width_1, method, ns, **kwargs):
    fscoil = _switch_finite_size_cjax(coil_jax, width_0, width_1, method, ns, **kwargs)    
    return fscoil.finite_size(_sampling_s_finite_size(ns), width_0, width_1)
    
def _check_finite_size(coilset_jsb, coilset_sbgeom, method, rtol = 1e-12, atol = 1e-12, **kwargs):
    for i in range(coilset_sbgeom.Number_of_Coils()):
        coil_jsb = coilset_jsb[i]
        coil_sbgeom = coilset_sbgeom[i]
        s_samples = _sampling_s_finite_size()
        width_0 = 0.3
        width_1 = 0.5
        jsb_lines    = _switch_finite_size_cjax_lines(coil_jsb, width_0, width_1, method, ns = s_samples.shape[0], **kwargs)
        
        sbgeom_lines = _switch_finite_size(coil_sbgeom, width_0, width_1, method, ns = s_samples.shape[0])

        jsb_comparison = jnp.moveaxis(jsb_lines, 1,0).reshape(-1,3)
        onp.testing.assert_allclose(sbgeom_lines, jsb_comparison, rtol = rtol, atol=atol)

# No centroid finite size for discrete coils: it is defined differently in SBGeom and jax-sbgeom
def test_discrete_coil_finite_size_rmf(_get_all_discrete_coils):
    coilset_jaxsbgeom, coilset_sbgeom = _get_all_discrete_coils
    nrmf = _sampling_s_finite_size().shape[0]
    _check_finite_size(coilset_jaxsbgeom, coilset_sbgeom, method="rmf", rtol =1e-7,  atol=1e-7, number_of_rmf_samples = nrmf)

def test_fourier_coil_finite_size_centroid(_get_all_fourier_coils):
    coilset_jax, coilset_sbgeom = _get_all_fourier_coils
    _check_finite_size(coilset_jax, coilset_sbgeom, method="centroid", rtol =1e-12,  atol=1e-12)

def test_fourier_coil_finite_size_rmf(_get_all_fourier_coils):
    coilset_jax, coilset_sbgeom = _get_all_fourier_coils
    nrmf = _sampling_s_finite_size().shape[0]
    _check_finite_size(coilset_jax, coilset_sbgeom, method="rmf", rtol =1e-12,  atol=1e-12, number_of_rmf_samples = nrmf)

def test_fourier_coil_finite_size_frenet(_get_all_fourier_coils):
    coilset_jax, coilset_sbgeom = _get_all_fourier_coils
    nrmf = _sampling_s_finite_size().shape[0]
    _check_finite_size(coilset_jax, coilset_sbgeom, method="frenet_serret", rtol =1e-12,  atol=1e-12, number_of_rmf_samples = nrmf)

#=================================================================================================================================================
#                                                  Meshing
#=================================================================================================================================================
def _mesh_switch_finite_size(coil_sbgeom, width_0, width_1, method, ns, **kwargs):
    if method == "centroid":
        return coil_sbgeom.Mesh_Triangles_Centroid(width_1, width_0, ns)
    elif method == "frenet_serret":
        return coil_sbgeom.Mesh_Triangles_Frenet(width_1, width_0, ns)
    elif method == "rmf":
        return coil_sbgeom.Mesh_Triangles_RMF(width_1, width_0, ns)
    else:
        raise NotImplementedError(f"Finite size method '{method}' not implemented for SBGeom discrete coils.")

def _check_coils(coilset_jsb, coilset_sbgeom, method, atol = 1e-12, **kwargs):
    for i in range(coilset_sbgeom.Number_of_Coils()):
        coil_jsb = coilset_jsb[i]
        coil_sbgeom = coilset_sbgeom[i]
        sampling_s = _sampling_s_finite_size()
        width_0 = 0.3
        width_1 = 0.5
        
        fs_coil     = _switch_finite_size_cjax(coil_jsb, width_0, width_1, method, ns = sampling_s.shape[0], **kwargs)
        jsb_mesh    = jsb.coils.coil_meshing.mesh_coil_surface(fs_coil, sampling_s.shape[0], width_0, width_1)
        sbgeom_mesh = _mesh_switch_finite_size(coil_sbgeom, width_0, width_1, method, ns = sampling_s.shape[0], **kwargs)
        onp.testing.assert_allclose(jsb_mesh[0], sbgeom_mesh.vertices, atol=atol)
        onp.testing.assert_array_equal(jsb_mesh[1], sbgeom_mesh.connectivity)

# No centroid meshing for discrete coils: it is defined differently in SBGeom and jax-sbgeom
def test_discrete_coil_rmf_meshing(_get_all_discrete_coils):
    coilset_jaxsbgeom, coilset_sbgeom = _get_all_discrete_coils
    _check_coils(coilset_jaxsbgeom, coilset_sbgeom, method="rmf", atol=1e-7, number_of_rmf_samples = _sampling_s_finite_size().shape[0])

def test_fourier_coil_centroid_meshing(_get_all_fourier_coils):
    coilset_jax, coilset_sbgeom = _get_all_fourier_coils
    _check_coils(coilset_jax, coilset_sbgeom, method="centroid", atol=1e-12)
def test_fourier_coil_rmf_meshing(_get_all_fourier_coils):
    coilset_jax, coilset_sbgeom = _get_all_fourier_coils
    _check_coils(coilset_jax, coilset_sbgeom, method="rmf", atol=1e-12, number_of_rmf_samples = _sampling_s_finite_size().shape[0])
def test_fourier_coil_frenet_meshing(_get_all_fourier_coils):
    coilset_jax, coilset_sbgeom = _get_all_fourier_coils
    _check_coils(coilset_jax, coilset_sbgeom, method="frenet_serret", atol=1e-12, number_of_rmf_samples = _sampling_s_finite_size().shape[0])

#=================================================================================================================================================
#                                                  Vectorization Tests
#=================================================================================================================================================

@pytest.mark.slow
def test_discrete_coil_vectorized_position(_get_all_discrete_coils):
    coilset_jaxsbgeom, coilset_sbgeom = _get_all_discrete_coils
    
    _check_single_vectorized(coilset_jaxsbgeom[0].position)
    _check_single_vectorized(coilset_jaxsbgeom[0].tangent)

    def centroid(s):
        finitesize_coil = jsb.coils.base_coil.FiniteSizeCoil(coilset_jaxsbgeom[0], jsb.coils.base_coil.CentroidFrame())
        return finitesize_coil.finite_size(s, 0.3, 0.5)
    _check_single_vectorized(centroid)
    def frenet_serret(s):
        finitesize_coil = jsb.coils.base_coil.FiniteSizeCoil(coilset_jaxsbgeom[0], jsb.coils.base_coil.FrenetSerretFrame())
        return finitesize_coil.finite_size(s, 0.3, 0.5)
    _check_single_vectorized(frenet_serret)
    def rmf(s):
        finitesize_coil = jsb.coils.base_coil.FiniteSizeCoil(coilset_jaxsbgeom[0], jsb.coils.base_coil.RotationMinimizedFrame.from_coil(coilset_jaxsbgeom[0], 111))
        return finitesize_coil.finite_size(s, 0.3, 0.5)
    _check_single_vectorized(rmf)
    

@pytest.mark.slow
def test_fourier_coil_vectorized_position(_get_all_fourier_coils):
    coilset_jax, coilset_sbgeom = _get_all_fourier_coils
    
    _check_single_vectorized(coilset_jax[0].position)
    _check_single_vectorized(coilset_jax[0].tangent)

    def centroid(s):
        finitesize_coil = jsb.coils.base_coil.FiniteSizeCoil(coilset_jax[0], jsb.coils.base_coil.CentroidFrame())
        return finitesize_coil.finite_size(s, 0.3, 0.5)
    _check_single_vectorized(centroid)
    def frenet_serret(s):
        finitesize_coil = jsb.coils.base_coil.FiniteSizeCoil(coilset_jax[0], jsb.coils.base_coil.FrenetSerretFrame())
        return finitesize_coil.finite_size(s, 0.3, 0.5)
    _check_single_vectorized(frenet_serret)
    def rmf(s):
        finitesize_coil = jsb.coils.base_coil.FiniteSizeCoil(coilset_jax[0], jsb.coils.base_coil.RotationMinimizedFrame.from_coil(coilset_jax[0],111))
        return finitesize_coil.finite_size(s, 0.3, 0.5)
    _check_single_vectorized(rmf)

#=================================================================================================================================================
#                                                  CoilSet Tests
#=================================================================================================================================================
def check_vector_coilset(coils_jax):
    s =  jnp.linspace(0, 1, 1000)

    pos_base = []
    tan_base = []

    for i in coils_jax:
        pos_base.append(i.position(s))
        tan_base.append(i.tangent(s))

    pos_base = jnp.array(pos_base)
    tan_base = jnp.array(tan_base)

    coilsetv = jsb.coils.CoilSet.from_list(coils_jax)


    pos_vec = coilsetv.position(s)
    tan_vec = coilsetv.tangent(s)

    onp.testing.assert_allclose(pos_base, pos_vec)
    onp.testing.assert_allclose(tan_base, tan_vec)


def test_vector_coilset_discrete(_get_all_discrete_coils):
    coils_jaxsbgeom, coilset_sbgeom = _get_all_discrete_coils
    check_vector_coilset(coils_jaxsbgeom)   

def test_vector_coilset_fourier(_get_all_fourier_coils_truncated):
    coils_jax, coilset_sbgeom = _get_all_fourier_coils_truncated
    check_vector_coilset(coils_jax)