import jax 
import jax.numpy as jnp
def stack_jacfwd(fun, argnums):
    def jac_stack_wrap(*args):
        jacrevs = jax.jacfwd(fun, argnums = argnums)(*args) 
        return jnp.stack(jacrevs, axis=-1)    
    return jac_stack_wrap