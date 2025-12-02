"""Submodule for particle kernel launchers."""

import abc
from typing import Callable, Iterable, Self

import numpy as np
import numpy.typing as npt

from .fieldset import Fieldset
from .kernel import FieldData, ParticleKernel
from .kernel_tools import unsafe_inverse_linear_interpolation
from .spatial_arrays import BBox

# -------------------------------
# FieldData provider descriptor
# -------------------------------


class FieldDataProvider:
    """
    Descriptor for declaring a FieldData provider on a Launcher.
    Registers the provider under a reserved name.
    """

    def __init__(self, reserved_name: str):
        self.reserved_name = reserved_name

    def __set_name__(self, owner: type[Self], method_name: str) -> None:
        self.method_name = method_name
        # Initialize the class dict if it doesn't exist
        if not hasattr(owner, "_field_data_providers"):
            owner._field_data_providers = {}
        # Add this descriptor to the class registry
        owner._field_data_providers[self.reserved_name] = self

    def __get__(self, instance: Self, owner: type[Self]):
        if instance is None:
            return self
        # Return a callable that matches the FieldData signature
        return getattr(instance, self.method_name)


# -------------------------------
# Convenience decorator
# -------------------------------


def register_field_data_provider(reserved_name: str):
    """
    Decorator to register a method as a FieldData provider.
    Usage:
        @register_field_data_provider("__dt")
        def get_dt_field_data(self, time_index, bbox):
            ...
    """

    def decorator(func: Callable[[Self, float, BBox], FieldData]):
        provider = FieldDataProvider(reserved_name)
        provider.method_name = func.__name__  # store the method
        return provider

    return decorator


# -------------------------------
# Launcher base class
# -------------------------------


class Launcher(abc.ABC):
    """Class to launch particle kernels with required fields."""

    _field_data_providers: dict[str, FieldDataProvider] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Start with a copy of parent's providers to preserve inheritance
        cls._field_data_providers = {}
        for base in reversed(cls.__mro__):
            providers = getattr(base, "_field_data_providers", {})
            cls._field_data_providers.update(providers)

    def __init__(
        self,
        fieldset: Fieldset,
        index_padding: int = 0,
    ) -> None:
        self._fieldset = fieldset
        if index_padding < 0:
            raise ValueError("index_padding must be non-negative")
        self._index_padding = index_padding

    def field_data_keys(self) -> Iterable[str]:
        """Get the keys of the FieldData providers registered with this launcher."""
        return self._field_data_providers.keys()

    @property
    def fieldset(self) -> Fieldset:
        """The Fieldset used by this launcher."""
        return self._fieldset

    @property
    def index_padding(self) -> int:
        """The index padding used by this launcher."""
        return self._index_padding

    def construct_bbox(
        self,
        particles: npt.NDArray,
    ) -> BBox:
        """Construct a bounding box around the given particles with index padding."""

        z_indices = particles["z_idx"]
        y_indices = particles["y_idx"]
        x_indices = particles["x_idx"]

        z_min = z_indices.min() - self._index_padding
        z_max = z_indices.max() + self._index_padding

        y_min = y_indices.min() - self._index_padding
        y_max = y_indices.max() + self._index_padding

        x_min = x_indices.min() - self._index_padding
        x_max = x_indices.max() + self._index_padding

        return BBox(
            z_min=z_min,
            z_max=z_max,
            y_min=y_min,
            y_max=y_max,
            x_min=x_min,
            x_max=x_max,
        )

    def get_field_data(self, name: float, time_index: float, bbox: BBox) -> FieldData:
        """Get the field data at a given time index.

        Parameters
        ----------
        time_index : float
            Time index.
        bbox : BBox
            Bounding box to extract data from defined in terms of centered grid indices.

        Returns
        -------
        FieldData
            Namedtuple containing the field data array, the dimension mask, and offsets.
        """
        # First check if it's a launcher field
        if name in self._field_data_providers:
            return self._field_data_providers[name](time_index, bbox)
        # Otherwise get from fieldset
        if name in self.fieldset:
            return self.fieldset.get_field_data(name, time_index, bbox)

        raise ValueError(f"Field {name} not found in launcher or fieldset.")

    @abc.abstractmethod
    def kernels(self) -> Iterable[ParticleKernel]:
        """All the kernels attached to this launcher."""
        pass

    def launch_kernel(
        self, kernel: ParticleKernel, particles: npt.NDArray, time_index: float
    ) -> None:
        """Launch a kernel."""
        # Construct the bounding box around the particles.
        bbox = self.construct_bbox(particles)

        # gather the field data required by the kernel
        field_data = []
        for name in kernel.simulation_fields:
            field_data.append(self.get_field_data(name, time_index, bbox))

        # call the vectorized kernel function
        kernel._vector_kernel_function(particles, *field_data)


class TimestepLauncher(Launcher):
    """Launcher that completes a single particle advection time step."""

    def __init__(
        self,
        fieldset: Fieldset,
        time_array: npt.NDArray,
        dt: float,
        time: float = 0.0,
        index_padding: int = 0,
    ) -> None:
        super().__init__(fieldset, index_padding)

        # check time array is valid
        if time_array.size != fieldset.t_size:
            raise ValueError(
                "Time array size does not match fieldset time dimension size."
            )
        if not np.all(np.diff(time_array) > 0):
            raise ValueError("Time array must be strictly increasing.")
        self._time_array = time_array

        # store timestep, current time and current time index
        self._dt = dt
        self._time = time
        self._tidx = self.get_time_index(time)

    @property
    def dt(self) -> float:
        """The time step for this launcher."""
        return self._dt

    @property
    def time(self) -> float:
        """The current time for this launcher."""
        return self._time

    @property
    def tidx(self) -> float:
        """The current time index for this launcher."""
        return self._tidx

    @register_field_data_provider("__dt")
    def dt_field_data(self, tidx: float, bbox: BBox) -> FieldData:
        """get dt as field data."""
        return FieldData(array=np.asarray(self._dt), dmask=(0, 0, 0), offsets=())

    @register_field_data_provider("__time")
    def time_field_data(self, tidx: float, bbox: BBox) -> FieldData:
        """get time as field data."""
        return FieldData(array=np.asarray(self._time), dmask=(0, 0, 0), offsets=())

    @register_field_data_provider("__tidx")
    def tidx_field_data(self, tidx: float, bbox: BBox) -> FieldData:
        """get time index as field data."""
        return FieldData(array=np.asarray(self._tidx), dmask=(0, 0, 0), offsets=())

    def get_time_index(self, time: float) -> float:
        """Get the time index corresponding to the given time."""
        if time < self._time_array[0] or time > self._time_array[-1]:
            raise ValueError("Time is out of bounds of the time array.")

        return unsafe_inverse_linear_interpolation(self._time_array, time)

    def advance_time(self) -> None:
        """Advance the current time by dt and update the time index."""
        self._time += self._dt
        self._tidx = self.get_time_index(self._time)

    @abc.abstractmethod
    def timestep_particles(self, particles: npt.NDArray) -> None:
        """Timestep the particles by one time step."""
        pass


class RK2Launcher(TimestepLauncher):
    """Launcher for RK2 particle kernels.

    Implements two-stage second order explicit Runge-Kutta integration for particle advection.
    Explicit second-order RK2 schemes are defined by a single parameter alpha and have Butcher tableau:
        0   |
      alpha |       alpha
    -----------------------------------------
            |  1 - 1 / 2 alpha    1 / 2 alpha
    """

    def __init__(
        self,
        fieldset: Fieldset,
        time_array: npt.NDArray,
        rk_step_1_kernel: ParticleKernel,
        rk_step_2_kernel: ParticleKernel,
        rk_update_kernel: ParticleKernel,
        dt: float,
        time: float = 0.0,
        alpha: float = 2 / 3,
        index_padding: int = 0,
    ) -> None:
        super().__init__(fieldset, time_array, dt, time, index_padding)
        self._rk_step_1_kernel = rk_step_1_kernel
        self._rk_step_2_kernel = rk_step_2_kernel
        self._rk_update_kernel = rk_update_kernel
        self._alpha = alpha

    @property
    def dt(self) -> float:
        """The time step used by this launcher."""
        return self._dt

    @property
    def alpha(self) -> float:
        """The RK2 alpha parameter used by this launcher."""
        return self._alpha

    @register_field_data_provider("__RK2_alpha")
    def get_alpha_field_data(self, tidx: float, bbox: BBox) -> FieldData:
        """get alpha as field data."""
        return FieldData(array=np.asarray(self._alpha), dmask=(0, 0, 0), offsets=())

    def kernels(self) -> tuple[ParticleKernel, ParticleKernel, ParticleKernel]:
        return (
            self._rk_step_1_kernel,
            self._rk_step_2_kernel,
            self._rk_update_kernel,
        )

    def timestep_particles(self, particles: npt.NDArray) -> None:
        """Launch the RK2 kernels to timestep the particles."""
        # Stage 1
        self.launch_kernel(self._rk_step_1_kernel, particles, self._tidx)

        # Compute intermediate time and time index
        intermediate_time = self._time + self._alpha * self._dt
        intermediate_tidx = self.get_time_index(intermediate_time)

        # Stage 2
        self.launch_kernel(self._rk_step_2_kernel, particles, intermediate_tidx)

        # Update particle positions
        self.launch_kernel(self._rk_update_kernel, particles, self._tidx)

        # Advance time
        self.advance_time()
