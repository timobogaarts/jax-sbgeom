### Experimental!!
import jax.numpy as jnp
import jax
from functools import partial
class Constant:
    MU_0 = 4e-7 * jnp.pi  # Permeability of free space

@jax.jit
def _biot_savart_internal(current : float, position : jnp.ndarray, delta_line_segment : jnp.ndarray, query_point : jnp.ndarray):
    '''
    Biot-Savart law for a single line element and single query point

    Parameters
    ----------
    current : float
        Current in the coil
    position : jnp.ndarray
        Position of the line element [3,]
    tangent : jnp.ndarray
        Tangent vector of the line element [3,]
    query_point : jnp.ndarray
        Point at which to evaluate the field [3,]

    Returns
    -------
    jnp.ndarray
        Magnetic field at the query point [3,]
    '''
    r_vec = query_point - position
    r_mag = jnp.linalg.norm(r_vec, axis=-1)
    dl_x_r = jnp.cross(delta_line_segment, r_vec, axis=-1)    
    B = (Constant.MU_0 * current) / (4.0 * jnp.pi) * dl_x_r / (r_mag**3)[..., None]    
    return B

_vmapped_biot_savart = jax.jit(jax.vmap(_biot_savart_internal, in_axes=(0, 0, 0, None)))


def biot_savart_single(currents : jnp.ndarray, positions : jnp.ndarray, delta_line_segments : jnp.ndarray, query_point : jnp.ndarray):
    '''
    Biot-Savart law for multiple line elements and a single query point

    Parameters
    ----------
    currents : jnp.ndarray
        Currents in the coil segments [N,]
    positions : jnp.ndarray
        Positions of the line elements [N, 3]
    delta_line_segments : jnp.ndarray
        Line element vectors [N, 3]
    query_point : jnp.ndarray
        Point at which to evaluate the field [3,]

    Returns
    -------
    jnp.ndarray
        Magnetic field at the query point [3,]
    '''    
    B_segments = _vmapped_biot_savart(currents, positions, delta_line_segments, query_point) # shape (N, 3)
    B_total = jnp.sum(B_segments, axis=0) # shape (3,)
    return B_total

_biot_savart_multiple = jax.jit(jax.vmap(biot_savart_single, in_axes=(None, None, None, 0)))

@jax.jit
def biot_savart(currents : jnp.ndarray, positions : jnp.ndarray, delta_line_segments : jnp.ndarray, query_points : jnp.ndarray):
    '''
    Biot-Savart law for multiple line elements and multiple query points

    Parameters
    ----------
    currents : jnp.ndarray
        Currents in the coil segments [N,]
    positions : jnp.ndarray
        Positions of the line elements [N, 3]
    delta_line_segments : jnp.ndarray
        Line element vectors [N, 3]
    query_points : jnp.ndarray
        Points at which to evaluate the field [M, 3]

    Returns
    -------
    jnp.ndarray
        Magnetic field at the query points [M, 3]
    '''    
    return _biot_savart_multiple(currents, positions, delta_line_segments, query_points)


@partial(jax.jit, static_argnums = (4,))
def biot_savart_batch(currents : jnp.ndarray, positions : jnp.ndarray, delta_line_segments : jnp.ndarray, query_points : jnp.ndarray, batch_size : int  = None):
    '''
    Biot-Savart law for multiple line elements and multiple query points

    Parameters
    ----------
    currents : jnp.ndarray
        Currents in the coil segments [N,]
    positions : jnp.ndarray
        Positions of the line elements [N, 3]
    delta_line_segments : jnp.ndarray
        Line element vectors [N, 3]
    query_points : jnp.ndarray
        Points at which to evaluate the field [M, 3]

    Returns
    -------
    jnp.ndarray
        Magnetic field at the query points [M, 3]
    '''    
    if batch_size is None:
        batch_size = query_points.shape[0]
    f_batch = lambda x : biot_savart_single(currents, positions, delta_line_segments, x)
    return jax.lax.map(f_batch, query_points, batch_size=batch_size)

def create_coilset_total_arrays(jax_coilset, currents, number_of_samples_per_coil):
    coil_samples     = jax_coilset.position(jnp.linspace(0,1, number_of_samples_per_coil, endpoint=False))    
    currents_stacked = jnp.stack([currents] * number_of_samples_per_coil, axis=-1)    
    coil_diff        = jnp.roll(coil_samples, -1, axis=1) - coil_samples
    return currents_stacked.reshape(-1) , coil_samples.reshape(-1,3), coil_diff.reshape(-1,3)