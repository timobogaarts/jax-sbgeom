.. jax-sbgeom documentation master file, created by
   sphinx-quickstart on Thu Jan  8 13:19:34 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to jax-sbgeom's documentation!
======================================

:code:`jax-sbgeom` is a fully `JAX <https://github.com/jax-ml/jax>`_-based stellarator geometry library.

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

Installation
----------------

Installation from this repository is possible using pip from this repository:

.. code::

   pip install .

JAX should be installed by the user with the appropriate CUDA backend for GPU support. See `here <https://docs.jax.dev/en/latest/installation.html>`_.

Some examples use the `PyVista <https://pyvista.org/>`_ library for visualization, which can be installed with:

.. code::

   pip install pyvista[all]


Why another library?
-----------------------
Similar libraries exist, see e.g. `Parastell <https://github.com/svalinn/parastell>`_. We opted to develop this from the following reasons:

- **Differentiability**: JAX's autodiff capabilities allow for gradient-based optimization and sensitivity analysis. Although the package itself uses it sparingly, for applications such as blanket optimization, differentiable geometry is crucial.
- **Volume meshing**: deterministic neutronics needs efficient meshes. Structured meshes in flux surface coordinates are an efficient way to produce uniform meshes in layered blanket geometry.
- **Timing**: Development of this package started when no other libraries were available.


Examples
-----------
The examples require some VMEC data. Due to copyright, they are not shipped with this library.

.. toctree::
   :maxdepth: 1
   :caption: Contents:      
   
   flux_surfaces.rst
   coils.rst
   examples.rst
   api_reference.rst   


