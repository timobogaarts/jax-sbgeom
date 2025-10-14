# jax-sbgeom

A JAX-based Python package for geometric operations.

## Installation

Install from source:

```bash
git clone https://github.com/timobogaarts/jax-sbgeom.git
cd jax-sbgeom
pip install -e .
```

Or install dependencies directly:

```bash
pip install jax jaxlib numpy
```

## Development Installation

To install with development dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

```python
import jax.numpy as jnp
from jax_sbgeom import __version__

print(f"jax-sbgeom version: {__version__}")

# Your JAX-based geometric operations here
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.sqrt(x)
```

## Examples

Check out the `examples/` directory for more detailed usage examples:

```bash
python examples/basic_example.py
```

## Package Structure

```
jax-sbgeom/
├── jax_sbgeom/          # Main package directory
│   └── __init__.py      # Package initialization
├── examples/            # Example scripts
│   ├── README.md        # Examples documentation
│   └── basic_example.py # Basic usage example
├── pyproject.toml       # Package configuration
├── setup.py             # Setup script
├── .gitignore           # Git ignore file
└── README.md            # This file
```

## Dependencies

- JAX >= 0.4.0
- jaxlib >= 0.4.0
- numpy >= 1.20.0

## License

MIT