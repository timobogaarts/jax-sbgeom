# jax-sbgeom

A JAX-based Python package for differentiable stellarator geometry.

Documentation: https://ipp-srs.github.io/jax-sbgeom/

Key features:

- Differentiable stellarator fourier series evaluation as in VMEC
- Extension beyond the last-closed flux surface using normal vectors from the surfaces or custom extensions
- Fitting of surfaces to VMEC representation, possibly using equal-arclength parametrisation
- Meshing of layered geometries of flux surfaces using closed/open surfaces and tetrahedral volume meshes.
- Differentiable parametrised coils, using discrete or Fourier implementations.
- Finite sizes computed using variety of methods, including rotation-minimized.
- Meshing of finite-size coils
- Coil winding surface optimization
- DAGMC interface for Monte-Carlo codes
- BVH construction, ray-BVH traversal and shortest distances to mesh for flux surface fitting
- Spline interpolations

For a showcase of some of these features, see [T.J. Bogaarts and F. Warmer](tbd). Alternatively, the examples reproduce exactly the figures given in the paper.

# Installation

For GPU acceleration, JAX should be installed by the user with the appropriate CUDA libraries before pip install.
Default is just CPU.

PyVista is used in the examples with jupyter notebooks but is not a necessary dependency.

To use the examples:
```
pip install 'jupyterlab' 'pyvista[all]'
```

# Tests
Tests have been developed for a large fraction of the functions, but coverage might not be 100%. All data is provided in the tests folder. Raytracing tests require trimesh & embreex.