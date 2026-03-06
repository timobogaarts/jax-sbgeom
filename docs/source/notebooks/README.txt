These notebooks are built automatically whenever they are changed. However, they require proprietary data which cannot be put on the 
repository. The strategy is then to build them with the cache enabled locally and push the updated cache.

Therefore, if one changes an example, we have to rebuild the cache using the proprietary data.

Q1:

Why don't we use CI to automatically do this every time?

A1:

CI doesn't have access to the proprietary data. Furthermore, don't want to rebuild on the CI resources every time; the cache is commited and used because of that.

Q2:

Why use this at all instead of just manually executing the notebooks?

A2

Plotly doesn't seem to render when this is done. Especially for 3D visualizations, we kind of want interactive plots, so we have to do it this way.
Furthermore, with more examples, this provides an easy way to run all examples instead of one-by-one.

