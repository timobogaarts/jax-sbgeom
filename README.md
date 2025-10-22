# jax-sbgeom

A JAX-based Python package for stellarator geometry

## Structure

The whole package is entirely functional, using dataclasses with static dispatching to reduce code duplication for e.g. different coils or flux surfaces.

All dataclass methods directly call a functional version of their method, possibly using their own data as arguments. Therefore, if one desires, one could entirely forego the 
static dispatching of dataclasses and just use the underlying functions. These are however written in some cases (i.e. meshing of surfaces) using static dispatch, so one has to 
stitch the functions together.