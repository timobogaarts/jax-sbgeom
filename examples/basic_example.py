"""Basic example demonstrating jax-sbgeom usage."""
import jax
import jax.numpy as jnp
from jax_sbgeom import __version__


def main():
    """Run a basic example with JAX."""
    print(f"jax-sbgeom version: {__version__}")
    print(f"JAX version: {jax.__version__}")
    
    # Example: Create a simple array and perform operations
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    print(f"\nOriginal array: {x}")
    
    # Perform a simple JAX operation
    y = jnp.sqrt(x)
    print(f"Square root: {y}")
    
    # Example with gradients
    def square_sum(x):
        return jnp.sum(x ** 2)
    
    grad_fn = jax.grad(square_sum)
    gradient = grad_fn(x)
    print(f"\nGradient of sum of squares: {gradient}")


if __name__ == "__main__":
    main()
