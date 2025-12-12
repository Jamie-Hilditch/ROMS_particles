"""Particle Kernels."""

import types
from typing import Callable, Iterable, Mapping, NamedTuple, Self

import numpy.typing as npt

from ..fields import FieldData

type KernelFunction = Callable[[NamedTuple, dict[str, npt.NDArray], dict[str, FieldData]], None]

DEFAULT_PARTICLE_FIELDS: dict[str, npt.DTypeLike] = {
    "status": np.uint8,
    "zidx": np.float64,
    "yidx": np.float64,
    "xidx": np.float64,
}

class ParticleKernel:
    """A kernel to be execute on particles."""

    def __init__(
        self, 
        pyfunc: KernelFunction, 
        particle_fields: dict[str, npt.DTypeLike],
        scalars: dict[str, npt.DTypeLike],
        simulation_fields: Iterable[str],
        cfunc: KernelFunction | None = None
    ):
        self._pyfunc = pyfunc
        self._particle_fields = {
            field: np.dtype(dtype) for field, dtype in particle_fields.items()
        }
        self._scalars = {
            scalar: np.dtype(dtype) for scalar, dtype in scalars.items()
        }
        self._simulation_fields = frozenset(simulation_fields)
        self._cfunc = cfunc

    @property
    def particle_fields(self) -> Mapping[str, np.dtype]:
        """The required particle fields and their dtypes."""
        return types.MappingProxyType(self._particle_fields)

    @property
    def scalars(self) -> Mapping[str, np.dtype]:
        """The required scalar fields and their dtypes."""
        return types.MappingProxyType(self._scalars)

    @property
    def simulation_fields(self) -> frozenset[str]:
        """The required simulation fields."""
        return self._simulation_fields

    @classmethod 
    def chain(
        cls,
        *kernels: Self
    ) -> Self:
        """Create a ParticleKernel by merging multiple kernels."""
        pyfunc, cfunc = _chain_kernel_functions(kernels)
        particle_fields = _merge_particle_fields(kernels)
        scalars = _merge_scalars(kernels)
        simulation_fields = _merge_simulation_fields(kernels)

        return cls(
            pyfunc,
            particle_fields,
            scalars,
            simulation_fields,
            cfunc
        )

    def chain_with(self, *others: Self) -> Self:
        """Chain this kernel with other kernels."""
        return ParticleKernel.chain(self, *others)

#############################################
### Kernel chaining and merging functions ###
##############################################

# Define a C function pointer type
ctypedef void (*CFuncPtr)(object particles, object scalars, object fielddata)

cdef CFuncPtr _make_chained_cfunc(tuple cfuncs):
    """
    Return a C function pointer that calls all cfuncs in the list.
    Returns NULL if the list is empty.
    """
    if not cfuncs:
        return NULL

    # Define the chained C function at module scope
    cdef void chained_cfunc(object particles, object scalars, object fielddata):
        cdef int i
        for i in range(len(cfuncs)):
            cfuncs[i](particles, scalars, fielddata)

    return chained_cfunc


def _chain_kernel_functions(
    kernels: Iterable[ParticleKernel]
) -> tuple[KernelFunction, KernelFunction | None]:
    """Chain multiple kernel functions into a single function.
    
    Returns both a python and a Cython function.
    """

    cfuncs = tuple(k._cfunc for k in kernels)
    if None in cfuncs:
        cfunc = None
    else:
        cfunc = _make_chained_cfunc(list(cfuncs))

    if cfunc is None:
        pyfuncs = tuple(k._pyfunc for k in kernels)
        def pyfunc(particles, scalars, fielddata):
             for fn in pyfuncs:
                 fn(particles, scalars, fielddata)
    else:
        def pyfunc(particles, scalars, fielddata):
            cfunc(particles, scalars, fielddata)    
    return pyfunc, cfunc

def _merge_particle_fields(
    kernels: Iterable[ParticleKernel]
) -> dict[str, np.dtype]:
    """Merge particle fields from multiple kernels."""
    merged_fields: dict[str, np.dtype] = DEFAULT_PARTICLE_FIELDS.copy()
    for kernel in kernels:
        for field, dtype in kernel.particle_fields.items():
            if field in merged_fields:
                if merged_fields[field] != dtype:
                    raise TypeError(
                        f"Conflicting dtypes for particle field '{field}': "
                        f"{merged_fields[field]} vs {dtype}"
                    )
            else:
                merged_fields[field] = dtype
    return merged_fields

def _merge_scalars(
    kernels: Iterable[ParticleKernel]
) -> dict[str, np.dtype]:
    """Merge scalar fields from multiple kernels."""
    merged_scalars: dict[str, np.dtype] = {}
    for kernel in kernels:
        for scalar, dtype in kernel.scalars.items():
            if scalar in merged_scalars:
                if merged_scalars[scalar] != dtype:
                    raise TypeError(
                        f"Conflicting dtypes for scalar '{scalar}': "
                        f"{merged_scalars[scalar]} vs {dtype}"
                    )
            else:
                merged_scalars[scalar] = dtype

    return merged_scalars

def _merge_simulation_fields(
    kernels: Iterable[ParticleKernel]
) -> frozenset[str]:
    """Merge simulation fields from multiple kernels."""
    merged_fields: set[str] = set()
    for kernel in kernels:
        merged_fields.update(kernel.simulation_fields)
    return frozenset(merged_fields)