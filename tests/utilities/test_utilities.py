import jax
jax.config.update("jax_enable_x64", True)

import jax_sbgeom 
import scipy.interpolate
import jax.numpy as jnp
import pytest
import numpy as onp

def _interpolation_cases():
    # 1. Monotone increasing
    x1 = jnp.linspace(0, 1, 6)
    y1 = jnp.array([0, 0.5, 1.0, 1.5, 2.0, 2.5])

    # 2. Monotone decreasing
    x2 = jnp.linspace(0, 1, 6)
    y2 = jnp.array([2.5, 2.0, 1.5, 1.0, 0.5, 0.0])

    # 3. Flat regions
    x3 = jnp.array([0, 1, 2, 3, 4], dtype=float)
    y3 = jnp.array([1, 1, 2, 2, 3], dtype=float)

    # 4. Local extrema
    x4 = jnp.linspace(0, 4, 5)
    y4 = jnp.array([0, 1, 0, -1, 0])  # peak at 1, trough at -1

    # 5. Non-uniform spacing
    x5 = jnp.array([0, 0.2, 0.5, 1.0, 1.7])
    y5 = jnp.array([0, 0.1, 0.7, 1.5, 2.0])

    # 6. Mixed monotone + flat + extrema
    x6 = jnp.array([0, 1, 2, 3, 4, 5, 6])
    y6 = jnp.array([0, 1, 1, 0.5, 2, 2, 1])
    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6)]


@pytest.mark.parametrize("x, y", _interpolation_cases())
def test_pchip_derivatives(x, y):
    # Compute derivatives using our JAX implementation
    jax_derivs = jax_sbgeom.jax_utils.utils._pchip_derivatives(x, y)

    # Compute derivatives using SciPy's PchipInterpolator
    xnp, ynp = onp.array(x), onp.array(y)
    
    scipy_derivs = scipy.interpolate.PchipInterpolator._find_derivatives(xnp, ynp)

    # Compare the results
    onp.testing.assert_allclose(jax_derivs, scipy_derivs, rtol=1e-5, atol=1e-8)



@pytest.mark.parametrize("x, y", _interpolation_cases())
def test_pchip_interpolation(x, y):
    # Compute derivatives using our JAX implementation
    
    s_new               = jnp.linspace(x[0], x[-1], 50)
    jax_interpolation   = jax_sbgeom.jax_utils.utils.pchip_interpolation(x, y, s_new)        
    scipy_interpolation = scipy.interpolate.PchipInterpolator(x,y)(s_new)

    # Compare the results
    onp.testing.assert_allclose(jax_interpolation, scipy_interpolation)
    