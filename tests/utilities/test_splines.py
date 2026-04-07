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

def test_periodic_knots_period():
    """make_periodic_knots with period=2π/nfp should equal (1/nfp) * make_periodic_knots with default period."""
    from jax_sbgeom.jax_utils.splines import make_periodic_knots
    nfp = 5
    n, k = 8, 3
    t_default = make_periodic_knots(n, k)                              # period 2π
    t_scaled  = make_periodic_knots(n, k, period=2 * jnp.pi / nfp)    # period 2π/nfp
    onp.testing.assert_allclose(t_scaled, t_default / nfp, atol=1e-12)


def test_periodic_bspline_default_period():
    """Adding period kwarg with default 2π must not change existing behaviour."""
    from jax_sbgeom.jax_utils.splines import periodic_bspline
    key = jax.random.PRNGKey(0)
    c = jax.random.normal(key, (10,))
    theta = jnp.linspace(0, 2 * jnp.pi, 50, endpoint=False)
    onp.testing.assert_allclose(
        periodic_bspline(theta, c, 3),
        periodic_bspline(theta, c, 3, period=2 * jnp.pi),
        atol=1e-12,
    )


def test_periodic_bspline_custom_period():
    """periodic_bspline with period P at phi == same spline with period 2π at phi*(2π/P)."""
    from jax_sbgeom.jax_utils.splines import periodic_bspline
    nfp = 5
    period = 2 * jnp.pi / nfp
    key = jax.random.PRNGKey(1)
    c = jax.random.normal(key, (10,))
    phi     = jnp.linspace(0, period, 50, endpoint=False)
    phi_2pi = phi * nfp   # same points rescaled to [0, 2π)
    onp.testing.assert_allclose(
        periodic_bspline(phi,     c, 3, period=period),
        periodic_bspline(phi_2pi, c, 3),
        atol=1e-12,
    )


def test_greville_abscissa_period():
    """Greville abscissae with period P should equal default abscissae * (P / 2π)."""
    from jax_sbgeom.jax_utils.splines import greville_abscissa_periodic_bspline
    nfp = 5
    period = 2 * jnp.pi / nfp
    n, k = 8, 3
    g_default = greville_abscissa_periodic_bspline(n, k)
    g_scaled  = greville_abscissa_periodic_bspline(n, k, period=period)
    onp.testing.assert_allclose(g_scaled, g_default / nfp, atol=1e-12)
    assert jnp.all(g_scaled >= 0) and jnp.all(g_scaled < period)


@pytest.fixture
def bspline_2d_setup():
    """Random coefficients and evaluation points shared across 2D tests."""
    nfp = 5
    n_theta, n_phi, k = 8, 6, 3
    key = jax.random.PRNGKey(42)
    c = jax.random.normal(key, (n_theta, n_phi))
    theta = jnp.linspace(0, 2 * jnp.pi,       37, endpoint=False)
    phi   = jnp.linspace(0, 2 * jnp.pi / nfp, 31, endpoint=False)
    return dict(nfp=nfp, n_theta=n_theta, n_phi=n_phi, k=k, c=c, theta=theta, phi=phi)


def test_2d_separability(bspline_2d_setup):
    """For a rank-1 coefficient matrix c[i,j] = a[i]*b[j], f_2d == f_theta * f_phi."""
    from jax_sbgeom.jax_utils.splines import periodic_bspline, periodic_bspline_2d
    s = bspline_2d_setup
    nfp, k = s['nfp'], s['k']
    key_a, key_b = jax.random.split(jax.random.PRNGKey(7))
    a = jax.random.normal(key_a, (s['n_theta'],))
    b = jax.random.normal(key_b, (s['n_phi'],))
    c_rank1 = a[:, None] * b[None, :]

    eval_2d = jax.vmap(jax.vmap(
        lambda th, ph: periodic_bspline_2d(th, ph, c_rank1, k, k, period_phi=2*jnp.pi/nfp),
        in_axes=(None, 0)), in_axes=(0, None))
    f_2d = eval_2d(s['theta'], s['phi'])  # [n_theta_eval, n_phi_eval]

    f_theta = periodic_bspline(s['theta'], a, k)                               # [n_theta_eval]
    f_phi   = periodic_bspline(s['phi'],   b, k, period=2 * jnp.pi / nfp)     # [n_phi_eval]
    expected = f_theta[:, None] * f_phi[None, :]

    onp.testing.assert_allclose(f_2d, expected, atol=1e-12)


def test_2d_theta_only(bspline_2d_setup):
    """c[i,j] = a[i] for all j: result is independent of phi (partition of unity in phi)."""
    from jax_sbgeom.jax_utils.splines import periodic_bspline, periodic_bspline_2d
    s = bspline_2d_setup
    nfp, k = s['nfp'], s['k']
    a = jax.random.normal(jax.random.PRNGKey(1), (s['n_theta'],))
    c_theta_only = jnp.broadcast_to(a[:, None], (s['n_theta'], s['n_phi']))

    eval_2d = jax.vmap(jax.vmap(
        lambda th, ph: periodic_bspline_2d(th, ph, c_theta_only, k, k, period_phi=2*jnp.pi/nfp),
        in_axes=(None, 0)), in_axes=(0, None))
    f_2d = eval_2d(s['theta'], s['phi'])  # [n_theta_eval, n_phi_eval]

    f_1d = periodic_bspline(s['theta'], a, k)  # [n_theta_eval]
    onp.testing.assert_allclose(f_2d, jnp.broadcast_to(f_1d[:, None], f_2d.shape), atol=1e-12)


def test_2d_phi_only(bspline_2d_setup):
    """c[i,j] = b[j] for all i: result is independent of theta (partition of unity in theta)."""
    from jax_sbgeom.jax_utils.splines import periodic_bspline, periodic_bspline_2d
    s = bspline_2d_setup
    nfp, k = s['nfp'], s['k']
    b = jax.random.normal(jax.random.PRNGKey(2), (s['n_phi'],))
    c_phi_only = jnp.broadcast_to(b[None, :], (s['n_theta'], s['n_phi']))

    eval_2d = jax.vmap(jax.vmap(
        lambda th, ph: periodic_bspline_2d(th, ph, c_phi_only, k, k, period_phi=2*jnp.pi/nfp),
        in_axes=(None, 0)), in_axes=(0, None))
    f_2d = eval_2d(s['theta'], s['phi'])  # [n_theta_eval, n_phi_eval]

    f_1d = periodic_bspline(s['phi'], b, k, period=2 * jnp.pi / nfp)  # [n_phi_eval]
    onp.testing.assert_allclose(f_2d, jnp.broadcast_to(f_1d[None, :], f_2d.shape), atol=1e-12)


def test_2d_periodic_theta(bspline_2d_setup):
    """f_2d(theta, phi) == f_2d(theta + 2π, phi)."""
    from jax_sbgeom.jax_utils.splines import periodic_bspline_2d
    s = bspline_2d_setup
    nfp, k, c = s['nfp'], s['k'], s['c']
    phi0 = s['phi'][0]
    f      = jax.vmap(lambda th: periodic_bspline_2d(th,              phi0, c, k, k, period_phi=2*jnp.pi/nfp))(s['theta'])
    f_wrap = jax.vmap(lambda th: periodic_bspline_2d(th + 2*jnp.pi,  phi0, c, k, k, period_phi=2*jnp.pi/nfp))(s['theta'])
    onp.testing.assert_allclose(f, f_wrap, atol=1e-12)


def test_2d_periodic_phi(bspline_2d_setup):
    """f_2d(theta, phi) == f_2d(theta, phi + 2π/nfp)."""
    from jax_sbgeom.jax_utils.splines import periodic_bspline_2d
    s = bspline_2d_setup
    nfp, k, c = s['nfp'], s['k'], s['c']
    theta0 = s['theta'][0]
    period_phi = 2 * jnp.pi / nfp
    f      = jax.vmap(lambda ph: periodic_bspline_2d(theta0, ph,              c, k, k, period_phi=2*jnp.pi/nfp))(s['phi'])
    f_wrap = jax.vmap(lambda ph: periodic_bspline_2d(theta0, ph + period_phi, c, k, k, period_phi=2*jnp.pi/nfp))(s['phi'])
    onp.testing.assert_allclose(f, f_wrap, atol=1e-12)


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