import jax 
import jax.numpy as jnp

import numpy as onp
def stack_jacfwd(fun, argnums):
    jacfwd_internal = jax.jacfwd(fun, argnums = argnums)
    def jac_stack_wrap(*args):                
        return jnp.stack(jacfwd_internal(*args), axis=-1)    
    return jac_stack_wrap


