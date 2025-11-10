import jax_sbgeom as jsb
import pytest
import jax
import jax.numpy as jnp 
import numpy as onp
jax.config.update("jax_enable_x64", True)

    
def test_spline_interpolation():
    n_period = 2.664
    t = jnp.linspace(0,  1 /n_period, 112) + jnp.sin(2 * jnp.pi * n_period * jnp.linspace(0,  1 /n_period, 112)) * 0.01 # non uniform
    y = jnp.sin(2 * jnp.pi * n_period * t)
    bspline = jsb.jax_utils.splines.periodic_interpolating_spline(t, y, 3)
    t_test = jnp.linspace(0, 1 / n_period, 1000)
    y_test = jnp.sin(2 * jnp.pi * n_period * t_test)
    y_spline = bspline(t_test)

    onp.testing.assert_allclose(y_test, y_spline, atol=1e-9)

def test_scipy_vs_jax_bspline():    
    from scipy.interpolate import make_interp_spline
    n_period = 2.664
    t = jnp.linspace(0,  1 /n_period, 112) + jnp.sin(2 * jnp.pi * n_period * jnp.linspace(0,  1 /n_period, 112)) * 0.01 # non uniform
    y = jnp.sin(2 * jnp.pi * n_period * t)
    bspline = jsb.jax_utils.splines.periodic_interpolating_spline(t, y, 3)
    t_test = jnp.linspace(0, 1 / n_period, 1000)
    y_test = jnp.sin(2 * jnp.pi * n_period * t_test)
    y_spline = bspline(t_test)

    tck = make_interp_spline(t,y, bc_type='periodic')    
    onp.testing.assert_allclose(tck.t, bspline.t, atol=1e-9)
    onp.testing.assert_allclose(tck.c, onp.array(bspline.c), atol=1e-9)