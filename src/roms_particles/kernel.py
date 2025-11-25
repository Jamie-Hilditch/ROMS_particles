"""Submodule for constructing and running kernels on particles."""

import numba 
import numpy.typing as npt

from typing import NamedTuple, Callable, Iterable

type Particle = npt.NDArray

class Tinfo(NamedTuple):
    time: float
    dt: float
    t_idx: float
    dt_idx: float

class FieldData(NamedTuple):
    data: npt.NDArray
    dmask: tuple[int, int, int]
    offsets: tuple[float,...]

type KernelFunction = Callable[[Particle, Tinfo, FieldData, ...], None]

class ParticleKernel:
    """A kernel to be executed on particles."""

    def __init__(self, 
        kernel_function: KernelFunction,
        particle_fields: Iterable[str],
        simulation_fields: Iterable[str],
    ) -> None:
        self._kernel_function = numba.jit(kernel_function, nogil=True, fastmath=True)
        self._particle_fields = set(particle_fields)
        self._simulation_fields = tuple(simulation_fields)

    @property
    def particle_fields(self) -> set[str]:
        """The particle fields required by this kernel."""
        return self._particle_fields

    @property
    def simulation_fields(self) -> tuple[str]:
        """The simulation fields required by this kernel."""
        return self._simulation_fields

    def chain_with(self, other: "ParticleKernel") -> "ParticleKernel":
        """Create a ParticleKernel by chaining this kernel with another."""

        combined_particle_fields = set(self.particle_fields).union(other.particle_fields)
        combined_simulation_fields = tuple(set(self.simulation_fields).union(other.simulation_fields))
        
        # find the indices of the fields in the combined tuple
        # these are tuples of ints so numba will treat them as compile time constants
        first_indices = tuple(combined_simulation_fields.index(f) for f in self.simulation_fields)
        second_indices = tuple(combined_simulation_fields.index(f) for f in other.simulation_fields)

        def chained_kernel(
            p: Particle, 
            tinfo: Tinfo, 
            *field_data: FieldData
        ) -> None:
            first_field_data = [field_data[i] for i in first_indices]
            second_field_data = [field_data[i] for i in second_indices]

            self._kernel_function(
                p, 
                tinfo, 
                *first_field_data
            )
            other._kernel_function(
                p, 
                tinfo, 
                *second_field_data
            )

        return ParticleKernel(
            chained_kernel,
            combined_particle_fields,
            combined_simulation_fields
        )

    @classmethod
    def from_sequence(
        cls, 
        kernels: Iterable["ParticleKernel"]
    ) -> "ParticleKernel":
        """Create a ParticleKernel by combining a sequence of ParticleKernels."""
        kernel_iter = iter(kernels)
        combined_kernel = next(kernel_iter)
        for kernel in kernel_iter:
            combined_kernel = combined_kernel.chain_with(kernel)
        return combined_kernel