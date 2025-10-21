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



def _check_single_vectorized_internal(fun):
    s_0 = 0.1
    s_1 = jnp.linspace(0.0, 1.0, 10)
    s_2, _ = jnp.meshgrid(jnp.linspace(0.0, 1.0, 5), jnp.linspace(0.0, 1.0, 4), indexing='ij')
    s_3 = jnp.ones((3,4,5)) * jnp.linspace(0.0, 1.0, 5)

    fun(s_0)
    fun(s_1)
    fun(s_2)
    fun(s_3)


def _get_coil_files():
    return ["/home/tbogaarts/stellarator_paper/base_data/vmecs/HELIAS3_coils_all.h5", "/home/tbogaarts/stellarator_paper/base_data/vmecs/HELIAS5_coils_all.h5", "/home/tbogaarts/stellarator_paper/base_data/vmecs/squid_coilset.h5"]


@pytest.fixture(scope="session", params = _get_coil_files())
def _get_all_discrete_coils(request):    
    coilset_sbgeom = SBGeom.Coils.Discrete_Coil_Set_From_HDF5(request.param)    
    coilset_jaxsbgeom = jsb.coils.CoilSet([jsb.coils.DiscreteCoil.from_positions(coilset_sbgeom[i].Get_Vertices()) for i in range(coilset_sbgeom.Number_of_Coils())])
    
    return coilset_jaxsbgeom, coilset_sbgeom


#=================================================================================================================================================
#                                                   Tests for DiscreteCoil
#=================================================================================================================================================

def _sampling_s(n_s : int = 1000):
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