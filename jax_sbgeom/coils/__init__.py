from .base_coil import Coil
from .fourier_coil import FourierCoil
from .discrete_coil import DiscreteCoil
from .coilset import CoilSet
from .coil_meshing import _mesh_finite_sized_lines_connectivity
__all__ = ["Coil", "CoilSet", "FourierCoil", "DiscreteCoil"]