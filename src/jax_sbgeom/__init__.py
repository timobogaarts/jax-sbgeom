"""jax-sbgeom: A JAX-based package for geometric operations."""

from . import flux_surfaces, jax_utils, coils, interfaces

__version__ = "0.1.0"

# Import main modules here as the package grows
__all__ = ["__version__", "flux_surfaces", "jax_utils", "coils", "interfaces"]
