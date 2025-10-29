# jax-sbgeom

A JAX-based Python package for stellarator geometry


# Installation

The dependency on jax is purposefully left out. It can be installed with 

```
pip install jax
```
or with CUDA features 
```
pip install jax[cuda]
```

PyVista is used in the examples with jupyter notebooks.

Use
```
pip install 'jupyterlab' 'pyvista[all]'
```


# Tests
Tests use SBGeom itself to verify against. 