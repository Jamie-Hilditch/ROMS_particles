"""Submodule for timestepping classes."""

import abc
import itertools
from typing import Iterator

import numpy as np
import numpy.typing as npt

from .fields import FieldData
from .kernels import ParticleKernel, ParticleStatus, is_active
from .launcher import Launcher, ScalarSource
from .particles import Particles

type T = np.float64 | np.datetime64
type D = np.float64 | np.timedelta64


class Timestepper(abc.ABC):
    """Class that handles particle advection timestepping."""

    # scalar data sources
    _dt_scalar = ScalarSource("_dt", lambda self, tidx: self._normalised_dt)
    _time_scalar = ScalarSource("_time", lambda self, tidx: self.time)
    _tidx_scalar = ScalarSource("_tidx", lambda self, tidx: self._tidx)

    def __init__(
        self,
        time_array: npt.NDArray[T],
        dt: D,
        *,
        time_unit: D | None = None,
    ) -> None:
        super().__init__()

        # check time_array is strictly increasing
        if np.any(time_array[1:] <= time_array[:-1]):  # type: ignore[operator]
            raise ValueError("time_array must be strictly increasing.")
        self._time_array = time_array

        # first set the time unit
        # this fixes the time types
        if time_unit is None:
            # use a default value of 1 if times are dimensionless else error
            if isinstance(dt, np.floating):
                time_unit = np.float64(1.0)
            else:
                raise ValueError("time_unit must be specified for dimensional time.")
        self._time_unit = time_unit

        # now set the timestep which has the same type as time_unit
        self.set_dt(dt)

        # initialise time, time_index and iteration
        self.set_time(self._time_array[0])
        self.set_iteration(0)

        # default value for index padding
        self._index_padding = 0

        # initialise empty lists for kernels
        self._initialisation_kernels: list[ParticleKernel] = []
        self._pre_step_kernels: list[ParticleKernel] = []
        self._post_step_kernels: list[ParticleKernel] = []

    def get_time_index(self, time: T) -> np.float64:
        """Get the time index corresponding to the given time.

        Args:
            time: The time to get the index for.

        Returns:
            float64: The time index corresponding to the given time.

        Raises:
            ValueError: If time is out of bounds of the time array.
            TypeError (from numpy): If time is not compatible with the time array.
        """
        time_array = self._time_array
        if time < time_array[0] or time > time_array[-1]:
            raise ValueError("Time is out of bounds of the time array.")

        idx = np.searchsorted(time_array, time, side="right") - 1
        t0 = time_array[idx]
        t1 = time_array[idx + 1]
        fraction = (time - t0) / (t1 - t0)
        return idx + fraction

    def set_dt(self, dt: D) -> None:
        """Set the time step for this timestepper."""
        # convert dt to timestep_type
        try:
            self._normalised_dt = np.float64(dt / self._time_unit)  # type: ignore[operator]
        except Exception as e:
            raise TypeError(f"dt must be of the same type as time_unit={self._time_unit!r}") from e

    def set_time(self, time: T) -> None:
        """Set the current time and update the time index."""
        # check time + dt is valid
        try:
            _ = time + self.dt  # type: ignore[operator]
        except Exception as e:
            raise TypeError(f"time must be compatible with dt={self.dt!r}") from e

        self._tidx = self.get_time_index(time)
        self._time = time

    def set_iteration(self, iteration: int) -> None:
        """Set the current iteration for this timestepper."""
        if iteration < 0:
            raise ValueError("Iteration must be non-negative.")
        self._iteration = iteration

    def add_initialisation_kernel(self, *kernels: ParticleKernel) -> None:
        """Add kernels to be launched during initialisation."""
        self._initialisation_kernels.extend(kernels)

    def add_pre_step_kernel(self, *kernels: ParticleKernel) -> None:
        """Add kernels to be launched before each timestep."""
        self._pre_step_kernels.extend(kernels)

    def add_post_step_kernel(self, *kernels: ParticleKernel) -> None:
        """Add kernels to be launched after each timestep."""
        self._post_step_kernels.extend(kernels)

    @property
    def time_unit(self) -> D:
        """The time unit for this timestepper."""
        return self._time_unit

    @property
    def dt(self) -> D:
        """The time step for this timestepper."""
        return self._normalised_dt * self._time_unit

    @property
    def time(self) -> T:
        """The current time for this timestepper."""
        return self._time

    @property
    def time_array(self) -> npt.NDArray[T]:
        """The time array for this timestepper."""
        return self._time_array

    @property
    def iteration(self) -> int:
        """The current iteration for this timestepper."""
        return self._iteration

    @property
    def tidx(self) -> np.float64:
        """The current time index for this timestepper."""
        return self._tidx

    @property
    def index_padding(self) -> int:
        """The index padding required by this timestepper."""
        return self._index_padding

    @property
    def forward_in_time(self) -> np.bool:
        """Whether the timestepper is advancing time forwards."""
        return self._normalised_dt > 0

    @property
    def initialisation_kernels(self) -> list[ParticleKernel]:
        """The list of initialisation kernels used by this timestepper."""
        return self._initialisation_kernels

    @property
    def pre_step_kernels(self) -> list[ParticleKernel]:
        """The list of pre-step kernels used by this timestepper."""
        return self._pre_step_kernels

    @property
    def post_step_kernels(self) -> list[ParticleKernel]:
        """The list of post-step kernels used by this timestepper."""
        return self._post_step_kernels

    @property
    def kernels(self) -> Iterator[ParticleKernel]:
        """Get the kernels used by this timestepper."""
        return itertools.chain(self._initialisation_kernels, self._pre_step_kernels, self._post_step_kernels)

    def advance_time(self) -> None:
        """Advance the current time by dt and update the time index."""
        self._time += self.dt  # type: ignore[operator]
        self._tidx = self.get_time_index(self._time)
        self._iteration += 1

    @abc.abstractmethod
    def step_particles(self, particles: Particles, launcher: Launcher) -> None:
        """Timestep the particles."""
        pass

    def step(self, particles: Particles, launcher: Launcher) -> None:
        """Timestep the particles by one time step."""
        # Launch pre-step kernels
        for kernel in self._pre_step_kernels:
            launcher.launch_kernel(kernel, particles, self._tidx)
        # Launch main step kernels
        self.step_particles(particles, launcher)
        # Advance time
        self.advance_time()
        # Launch post-step kernels
        for kernel in self._post_step_kernels:
            launcher.launch_kernel(kernel, particles, self._tidx)


class RK2Timestepper(Timestepper):
    """Timestepper implements RK2 particle kernels.

    Implements two-stage second order explicit Runge-Kutta integration for particle advection.
    Explicit second-order RK2 schemes are defined by a single parameter alpha and have Butcher tableau:
        0   |
      alpha |       alpha
    -----------------------------------------
            |  1 - 1 / 2 alpha    1 / 2 alpha
    """

    # scalar source
    _alpha_scalar = ScalarSource("_RK2_alpha", lambda self, tidx: self._alpha)

    def __init__(
        self,
        time_array: npt.NDArray[T],
        dt: D,
        rk_step_1_kernel: ParticleKernel,
        rk_step_2_kernel: ParticleKernel,
        rk_update_kernel: ParticleKernel,
        *,
        alpha: float = 2 / 3,
        time_unit: D | None = None,
        index_padding: int = 0,
    ) -> None:
        super().__init__(time_array, dt, time_unit=time_unit)
        self._rk_step_1_kernel = rk_step_1_kernel
        self._rk_step_2_kernel = rk_step_2_kernel
        self._rk_update_kernel = rk_update_kernel
        self._alpha = alpha
        self._index_padding = index_padding

    @property
    def alpha(self) -> float:
        """The RK2 alpha parameter used by this launcher."""
        return self._alpha

    @property
    def kernels(self) -> Iterator[ParticleKernel]:
        """Get the kernels used by this timestepper."""
        return itertools.chain(
            super().kernels,
            [
                self._rk_step_1_kernel,
                self._rk_step_2_kernel,
                self._rk_update_kernel,
            ],
        )

    def step_particles(self, particles: Particles, launcher: Launcher) -> None:
        """Launch the RK2 kernels to timestep the particles."""
        # Stage 1
        launcher.launch_kernel(self._rk_step_1_kernel, particles, self._tidx)
        # Compute intermediate time and time index
        intermediate_time = self.time + self._alpha * self.dt  # type: ignore[operator]
        intermediate_tidx = self.get_time_index(intermediate_time)
        # Stage 2
        launcher.launch_kernel(self._rk_step_2_kernel, particles, intermediate_tidx)
        # Update kernel
        launcher.launch_kernel(self._rk_update_kernel, particles, self._tidx)


class ABTimestepper(Timestepper):
    """Class for Adams-Bashforth timesteppers."""

    def __init__(
        self,
        time_array: npt.NDArray[T],
        dt: D,
        ab_kernel: ParticleKernel,
        *,
        time_unit: D | None = None,
        index_padding: int = 0,
    ) -> None:
        super().__init__(time_array, dt, time_unit=time_unit)
        self._ab_kernel = ab_kernel
        self._index_padding = index_padding

        # Add AB3 initialisation kernel
        self.add_initialisation_kernel(ab_initialisation_kernel)

    @property
    def kernels(self) -> Iterator[ParticleKernel]:
        """Get the kernels used by this timestepper."""
        return itertools.chain(super().kernels, [self._ab_kernel])

    def step_particles(self, particles: Particles, launcher: Launcher) -> None:
        """Launch the Adams-Bashforth kernel to timestep the particles."""
        # Launch Adams-Bashforth kernel
        launcher.launch_kernel(self._ab_kernel, particles, self._tidx)


# Kernel for AB3 initialisation
def _ab_initialisation_kernel_function(
    particles: Particles, scalars: dict[str, np.number], fields: dict[str, FieldData]
) -> None:
    """Kernel function to set status for AB3 initialisation."""
    status = particles.status
    idx = is_active(status)
    status[idx] = ParticleStatus.MULTISTEP_1


ab_initialisation_kernel = ParticleKernel(
    _ab_initialisation_kernel_function,
    particle_fields={"status": np.dtype(np.uint8)},
    scalars={},
    simulation_fields=[],
)
