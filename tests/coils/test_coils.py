import os
import jax_sbgeom as jsb
import jax.numpy as jnp
import numpy as onp
import StellBlanket.SBGeom as SBGeom 
import jax
import time

from functools import partial
from typing import Type, List

import pytest
from functools import lru_cache

jax.config.update("jax_enable_x64", True)


def test_backend():
    import jax
    

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
#                                                  Converting Tests
#=================================================================================================================================================

@pytest.mark.parametrize("n_ftrunc", [None, 11])
def test_converting_fourier_coils(_get_all_discrete_coils, n_ftrunc):
    coilset_jaxsbgeom, coilset_sbgeom = _get_all_discrete_coils
    fourier_coilset_sbgeom = SBGeom.Coils.Convert_to_Fourier_Coils(coilset_sbgeom, Nftrunc = n_ftrunc)
    
    fourier_coilset_jax= jsb.coils.fourier_coil.convert_to_fourier_coilset(jsb.coils.CoilSet.from_list(coilset_jaxsbgeom), n_modes = n_ftrunc)

    for i in range(coilset_sbgeom.Number_of_Coils()):
        coil_jsb = coilset_jaxsbgeom[i]
        fourier_coil_sbgeom = fourier_coilset_sbgeom[i]
                
        fourier_coil_jaxsbgeom = jsb.coils.fourier_coil.convert_to_fourier_coil(coil_jsb, n_modes = n_ftrunc)

        onp.testing.assert_allclose(fourier_coil_jaxsbgeom.fourier_cos, fourier_coil_sbgeom.Get_Fourier_Cos(), rtol=1e-12, atol=1e-12)
        onp.testing.assert_allclose(fourier_coil_jaxsbgeom.fourier_sin, fourier_coil_sbgeom.Get_Fourier_Sin(), rtol=1e-12, atol=1e-12)
        onp.testing.assert_allclose(fourier_coil_jaxsbgeom.centre_i, fourier_coil_sbgeom.Get_Centre(), rtol=1e-12, atol=1e-12)
        onp.testing.assert_allclose(fourier_coilset_jax.coils.fourier_cos[i], fourier_coilset_sbgeom[i].Get_Fourier_Cos(), rtol=1e-12, atol=1e-12)
        onp.testing.assert_allclose(fourier_coilset_jax.coils.fourier_sin[i], fourier_coilset_sbgeom[i].Get_Fourier_Sin(), rtol=1e-12, atol=1e-12)
        onp.testing.assert_allclose(fourier_coilset_jax.coils.centre_i[i], fourier_coilset_sbgeom[i].Get_Centre(), rtol=1e-12, atol=1e-12)

def test_equal_arclength(_get_all_fourier_coils):
    coilset_jax, coilset_sbgeom = _get_all_fourier_coils

    n_points_sample  = 2531
    n_points_desired = 1513
    n_points_ea      = 511

    arclength_f = jax.vmap(jsb.coils.fourier_coil._arc_length_fourier, in_axes=(0,None))

    fourier_coilset = jsb.coils.CoilSet.from_list(coilset_jax)

    coilset_new = jsb.coils.fourier_coil.convert_fourier_coilset_to_equal_arclength(fourier_coilset, n_points_sample, n_points_desired, method='pchip')
    coilset_new_lin = jsb.coils.fourier_coil.convert_fourier_coilset_to_equal_arclength(fourier_coilset, n_points_sample, n_points_desired, method='linear')

    def check_arclength_uniformity(coilset, n_points_ea, threshold):
        arclength_f_ea = arclength_f(coilset, jnp.linspace(0.0, 1.0, n_points_ea))
        arclength_avg =  jnp.mean(arclength_f_ea, axis=1) # average over the sampling points
        uniformity_max = jnp.max(jnp.abs((arclength_f_ea - arclength_avg[:, None]) / arclength_avg[:, None]))        
        assert uniformity_max < threshold
        
    check_arclength_uniformity(coilset_new.coils, n_points_ea, threshold=1e-3)
    check_arclength_uniformity(coilset_new_lin.coils, n_points_ea, threshold=5e-3)


#=================================================================================================================================================
#                                                  Reversal
#=================================================================================================================================================    
classes          = [jsb.coils.CentroidFrame, jsb.coils.RotationMinimizedFrame, jsb.coils.RadialVectorFrame, jsb.coils.FrenetSerretFrame]
classes_discrete = classes[:-1]  # Frenet-Serret frame not implemented for discrete coils

def _radial_vector(coil_i, n_coils):
    x = jnp.cos(jnp.linspace(0, 2*jnp.pi, 100) + coil_i  / n_coils * 2 * jnp.pi)
    y = jnp.sin(jnp.linspace(0, 2*jnp.pi, 100) + coil_i  / n_coils * 2 * jnp.pi)
    z = jnp.zeros_like(x)
    return jnp.stack([x, y, z], axis=-1)

_radial_vectors = jax.vmap(_radial_vector, in_axes=(0, None))

def additional_arguments_per_coil(frame_class : Type[jsb.coils.base_coil.FiniteSizeMethod], coil_i : int, ncoils : int):
    if frame_class == jsb.coils.RotationMinimizedFrame:
        return (10,)
    elif frame_class == jsb.coils.RadialVectorFrame:
        return (_radial_vector(coil_i, ncoils), )
    else:
        return ()
def additional_arguments(frame_class : Type[jsb.coils.base_coil.FiniteSizeMethod], coils_list : List):
    if frame_class == jsb.coils.RotationMinimizedFrame:
        return (10,)
    elif frame_class == jsb.coils.RadialVectorFrame:
        ncoils = len(coils_list)
        radial_vectors = _radial_vectors(jnp.arange(ncoils), ncoils)
        return (radial_vectors, )
    else:
        return ()        

def check_reverse_finite_size(finitesize_coil):
    if isinstance(finitesize_coil.coil, jsb.coils.DiscreteCoil):
        # discretecoils do not have the same tangent on *exactly* the discrete points of the coil..
        #  the forward derivative is different on those points.
        # so we use the parametrisation without being and endpoint
        s         = jnp.linspace(0, 1.0,211, endpoint=False)[1:]
        s_reverse = jnp.linspace(0,-1.0,211, endpoint=False)[1:]
    else:
        s = jnp.linspace(0,1,100)
        s_reverse = jnp.linspace(0,-1,100)
    

    reversed_finitesize_coil = finitesize_coil.reverse_parametrisation()
    position_original = finitesize_coil.position(s)
    position_reversed = reversed_finitesize_coil.position(s_reverse)
    
    radial_vector_original = finitesize_coil.radial_vector(s)
    radial_vector_reversed = reversed_finitesize_coil.radial_vector(s_reverse)

    tangent_original = finitesize_coil.tangent(s)
    tangent_reversed = reversed_finitesize_coil.tangent(s_reverse)
    
    
    onp.testing.assert_allclose(position_original,      position_reversed)        
    onp.testing.assert_allclose(tangent_original,      -tangent_reversed)    
    onp.testing.assert_allclose(radial_vector_original, radial_vector_reversed)

    finitesize_frame_original = finitesize_coil.finite_size_frame(s)
    finitesize_frame_reversed = reversed_finitesize_coil.finite_size_frame(s_reverse)
    # the tangent vector will be reversed, but the radial vector the same, so
    # we need to reverse the tangent vector in the reversed frame to compare
    finitesize_frame_reversed_corrected = finitesize_frame_reversed.at[...,1,:].set(-finitesize_frame_reversed[...,1,:])

    onp.testing.assert_allclose(finitesize_frame_original, finitesize_frame_reversed_corrected)

    width_phi = 0.245
    width_radial = 0.123
    finitesize_original = finitesize_coil.finite_size(s, width_phi, width_radial)
    finitesize_reversed = reversed_finitesize_coil.finite_size(s_reverse, width_phi, width_radial)

    # finite size has    
    #  The finite size is in the following order:
    # v_0 : + radial, + phi
    # v_1 : - radial, + phi
    # v_2 : - radial, - phi
    # v_3 : + radial, - phi
    # since phi -> -phi in the reversed coil
    # we need to swap v_0 with v_3 and v_1 with v_2 to compare
    # i can just flip the arrays around the first axis.    
    finitesize_reversed_corrected = finitesize_reversed[:, ::-1, :]
    onp.testing.assert_allclose(finitesize_original, finitesize_reversed_corrected)

def _get_finitesize_coilset(coils_jax, frame_class : Type[jsb.coils.base_coil.FiniteSizeMethod]):
    #additional_args = additional_arguments_per_coil(frame_class, coils_jax, len(coils_jax))
    finitesize_coilset = [jsb.coils.base_coil.FiniteSizeCoil(coil_jax, frame_class.from_coil(coil_jax, *additional_arguments_per_coil(frame_class, i, len(coils_jax)))) for i, coil_jax in enumerate(coils_jax)]
    return finitesize_coilset

@pytest.mark.parametrize("frame_class", classes_discrete)
def test_finitesize_coilset_reverse_discrete(_get_all_discrete_coils, frame_class):
    coils_jaxsbgeom, coilset_sbgeom = _get_all_discrete_coils
    coils_finitesize = _get_finitesize_coilset(coils_jaxsbgeom, frame_class)
    [check_reverse_finite_size(coils_finitesize[i]) for i in range(len(coils_jaxsbgeom))]

@pytest.mark.parametrize("frame_class", classes)
def test_finitesize_coilset_reverse_fourier(_get_all_fourier_coils, frame_class):
    coils_jaxsbgeom, coilset_sbgeom = _get_all_fourier_coils
    coils_finitesize = _get_finitesize_coilset(coils_jaxsbgeom, frame_class)
    [check_reverse_finite_size(coils_finitesize[i]) for i in range(len(coils_jaxsbgeom))]