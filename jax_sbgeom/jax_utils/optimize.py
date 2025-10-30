import jax
import optax
from functools import partial
import jax.numpy as jnp
from typing import Callable, Tuple
from dataclasses import dataclass

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class OptimizationSettings:
    max_iterations : int
    tolerance      : float

@partial(jax.jit, static_argnums=(2,3,4))
def run_lbfgs_step(params : optax.Params, opt_state : optax.OptState, loss_fn : Callable[[optax.Params], float], optimizer : optax.GradientTransformationExtraArgs, value_and_grad_function :  Callable[[optax.Params], Tuple[float, optax.Params]]):    
    value, grad = value_and_grad_function(params, state=opt_state)
    updates, opt_state = optimizer.update(
            grad, opt_state, params=params, value=value, grad=grad, value_fn=loss_fn
        )
    params = optax.apply_updates(params, updates)
    return params, opt_state


@partial(jax.jit, static_argnums=(1, 2))
def run_optimization_lbfgs(initial_values : optax.Params, loss_fn : Callable[[optax.Params], float], settings : OptimizationSettings):
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