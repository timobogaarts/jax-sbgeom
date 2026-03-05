Coils 
======

We describe the basis ideas of the coils in :code:`jax-sbgeom` here. For more details, see the API reference and the examples.


Coils are a mapping:

.. math::
    s \to \mathbf{r}(s)

Given such a coil implemention (whether it be interpolation between discrete points, a fourier representation or a user-defined coil),
the :class:`jax_sbgeom.coils.Coil` class can be used to compute the tangent, normal, and curvature.


Finite size coils
~~~~~~~~~~~~~~~~~

A finite size is defined by a finite size frame at each point along the coil. In fact, if we only have the radial vector 
(a vector which describes a "radial" direction), we immediately have the frame, since we have the tangent and the last 
vector can be obtained by a cross product. :class:`jax_sbgeom.coils.FiniteSizeCoil` then wraps a coil and a finite-size method, such an implementation,
and can be used for meshing and frame computation.

Implementations include :class:`jax_sbgeom.coils.CentroidFrame`, :class:`jax_sbgeom.coils.RotationMinimizedFrame` and :class:`jax_sbgeom.coils.FrenetSerretFrame`.


Coilset 
~~~~~~~

A set of batched coils can be wrapped in a :class:`jax_sbgeom.coils.CoilSet` object, which can be used for meshing and position computation of multiple coils at once.
Similarly, a :class:`jax_sbgeom.coils.FiniteSizeCoilSet` can be used for a set of finite-size coils.

