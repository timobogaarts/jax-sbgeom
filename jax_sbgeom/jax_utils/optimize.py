import jax
import optax
from functools import partial
import jax.numpy as jnp
from typing import Callable, Tuple
from dataclasses import dataclass

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class OptimizationSettings:
    '''
    Settings for optimization routines.

    Attributes
    ----------
    max_iterations : int
        Maximum number of iterations for the optimizer
    tolerance : float
        Tolerance for convergence   
    '''
    max_iterations : int
    tolerance      : float

@partial(jax.jit, static_argnums=(2,3,4))
def run_lbfgs_step(params : optax.Params, opt_state : optax.OptState, loss_fn : Callable[[optax.Params], float], optimizer : optax.GradientTransformationExtraArgs, value_and_grad_function :  Callable[[optax.Params], Tuple[float, optax.Params]]):    
    '''
    Run a single step of L-BFGS optimization.

    Parameters
    ----------
    params : optax.Params
        Current parameters for optimization
    opt_state : optax.OptState
        Current optimizer state
    loss_fn : Callable[[optax.Params], float]
        Loss function to minimize
    optimizer : optax.GradientTransformationExtraArgs
        L-BFGS optimizer
    value_and_grad_function : Callable[[optax.Params], Tuple[float, optax.Params]]
        Function to compute value and gradient of the loss function
    Returns
    -------
    optax.Params
        Updated parameters after the optimization step
    optax.OptState
        Updated optimizer state after the optimization step
    '''
    value, grad = value_and_grad_function(params, state=opt_state)
    updates, opt_state = optimizer.update(
            grad, opt_state, params=params, value=value, grad=grad, value_fn=loss_fn
        )
    params = optax.apply_updates(params, updates)
    return params, opt_state


@partial(jax.jit, static_argnums=(1, 2))
def run_optimization_lbfgs(initial_values : optax.Params, loss_fn : Callable[[optax.Params], float], settings : OptimizationSettings):
    '''
    Run L-BFGS optimization on a given loss function.

    Parameters
    ----------
    initial_values : optax.Params
        Initial parameters for optimization
    loss_fn : Callable[[optax.Params], float]
        Loss function to minimize
    settings : OptimizationSettings
        Settings for the optimization
    Returns
    -------
    optax.Params
        Optimized parameters
    '''
    lbfgs                   = optax.lbfgs()
    value_and_grad_function = optax.value_and_grad_from_state(loss_fn)    
    opt_state               = lbfgs.init(initial_values)
    params                  = initial_values

    def continuing_criterion(carry):
        _, state = carry 
        iter_num = optax.tree.get(state, 'count')
        grad     = optax.tree.get(state, 'grad')
        err      = optax.tree.norm(grad)
        return (iter_num == 0) | ((iter_num < settings.max_iterations) & (err >= settings.tolerance))
            
    return jax.lax.while_loop(continuing_criterion, 
        lambda carry: run_lbfgs_step(carry[0], carry[1], loss_fn, lbfgs, value_and_grad_function), 
        (params, opt_state))