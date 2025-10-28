import jax 
import jax_sbgeom
from . import CoilSet
from functools import partial

import jax.numpy as jnp

@jax.jit
def _s_softplus(d_i : jnp.ndarray, minimum_distance : float = 1e-5):    
    '''
    Compute normalized arc length s in [0, 1] using softplus regularization to ensure positive segment lengths.
    Parameters
    ----------
    d_i : jnp.ndarray [n_coils, n_points-1]
        Unregularized segment lengths between consecutive points along each coil.
    Returns
    -------
    s_c : jnp.ndarray [n_coils, n_points]
        Normalized cumulative arc length along each coil, ranging from 0 to 1.
    '''
    soft_plus = jax.nn.softplus(d_i)
    d = soft_plus + minimum_distance

    s_c = jnp.cumsum(d, axis=1)
    dc = s_c[:, -1] - s_c[:, 0]

    return s_c / dc[:, None]

def _coil_surface_distance_loss(s_arr : jnp.ndarray, coilset : CoilSet):    
        '''
        Computes the distance between adjacent coils, sampled at s_arr.

        \\sum_{ij} (coil_i(s_j) - coil_{i+1}(s_j))^2

        Normalised by the distance between coil centres.

        Parameters:
        ----------
        s_arr: jnp.ndarray [n_coils, n_s]
            Sampled arc length positions along each coil
        coilset: CoilSet
            CoilSet containing the coils
        Returns:
        -------
        loss: float
            Distance loss, lower is better  

        '''
        positions   = coilset.position_different_s(s_arr)  # [n_coils, n_s, 3]
        obj         = jnp.sum((positions - jnp.roll(positions, 1, axis=0))**2)         
        centre_diff = jnp.sum((coilset.centre() - jnp.roll(coilset.centre(), shift=1, axis=0  ))**2) * s_arr.shape[1] # multiplied by the number of sample points along the coil
        return obj / centre_diff

@jax.jit
def _uniformity_loss(x : jnp.ndarray):
        '''
        Computes the uniformity loss of points in x:

        \\sum_{ij} (d_{ij} - 1/(N_i -1))^2

        Parameters:
        ----------
        x: jnp.ndarray [n_coils, n_points]
            Points along each coil
        Returns:
        -------
        loss: float
            Uniformity loss, lower is better        

        '''
        dx = jnp.diff(x, axis=1)
        ideal = 1.0 / (x.shape[1]-1)
        return jnp.sum(jnp.sum((dx - ideal)**2, axis=1))

@jax.jit
def _repulsion_loss(x : jnp.ndarray, p : int = 2, eps : float = 1e-6):
        '''
        Computes a repulsion loss of points in x:

        \\sum_{i<j} 1 / (d_{ij}^p + eps) 

        It is normalised by the repulsion loss of a uniform distribution minus one, so that a uniform distribution gives 0 repulsion loss (ideal).

        Parameters:
        ----------
        x: jnp.ndarray [n_coils, n_points]
            Points along each coil
        p: int
            Power of the repulsion
        eps: float
            Small number to avoid division by zero
        Returns:
        -------
        loss: float
            Repulsion loss, lower is better
        '''
        def coil_loss(points):        
            diff = points[:, None] - points[None, :]
            dist = jnp.abs(diff) + jnp.eye(len(points)) * 1e6
            rep = 1.0 / (dist**p + eps)
            return jnp.sum(jnp.triu(rep, k=1))

        # vmap over coils, then sum    
        losses = jax.vmap(coil_loss)(x)
        losses_base = coil_loss(jnp.linspace(0.0,1.0, x.shape[1]))
        
        return jnp.sum(losses) / (losses.shape[0] * losses_base) - 1.0 


@partial(jax.jit, static_argnums=(1,))
def _create_total_s(d_i : jnp.ndarray, n_coils : int):
    '''
    Create total s array from d_i vector.

    Simply reshapes the d_i vector and computes s using softplus regularization.

    Parameters:
    ----------
    d_i : jnp.ndarray [n_coils * (n_points - 1)]
        Unregularized segment lengths between consecutive points along each coil.
    n_coils : int
        Number of coils.
    Returns:
    -------
    s_c : jnp.ndarray [n_coils, n_points]
        Normalized cumulative arc length along each coil, ranging from 0 to 1.        
    '''
    return _s_softplus(d_i.reshape((n_coils, -1)))


@partial(jax.jit, static_argnums=(2,3,4))
def coil_surface_loss(d_i : jnp.ndarray, coilset : CoilSet, n_coils : int, uniformity_loss_weight : float, repulsive_loss_weight : float):
    '''
    Compute total coil surface loss.

    Parameters:
    ----------
    d_i : jnp.ndarray [n_coils * (n_points - 1)]
        Unregularized segment lengths between consecutive points along each coil.
    coilset : CoilSet
        CoilSet containing the coils.
    n_coils : int
        Number of coils.
    uniformity_loss_weight : float
        Weight of the uniformity loss.
    repulsive_loss_weight : float
        Weight of the repulsion loss.
    Returns:
    -------
    total_loss : float
        Total coil surface loss.


    '''
    total_s_array = _create_total_s(d_i, n_coils)
    return _coil_surface_distance_loss(total_s_array, coilset) + uniformity_loss_weight * _uniformity_loss(total_s_array) + repulsive_loss_weight * _repulsion_loss(total_s_array)
