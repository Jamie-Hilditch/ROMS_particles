"""Kernels for validating particle status."""

import numpy as np
import numpy.typing as npt

from ..particle_kernel import ParticleKernel


def _finite_indices(
    particle: npt.NDArray,
) -> None:
    """Kernel to check if particle positions are finite."""
    if (
        not np.isfinite(particle["zidx"])
        or not np.isfinite(particle["yidx"])
        or not np.isfinite(particle["xidx"])
    ):
        particle["status"] = 1 # Mark particle as inactive if any position is not finite
        
finite_indices_kernel = ParticleKernel(
    _finite_indices,
    particle_fields={
        "status": np.uint8,
        "zidx": np.float64,
        "yidx": np.float64,
        "xidx": np.float64,
    },
    scalars=(),
    simulation_fields=[],
    fastmath=False
)

def _inbounds(
    particle: npt.NDArray,
    zidx_min: float,
    zidx_max: float,
    yidx_min: float,
    yidx_max: float,
    xidx_min: float,
    xidx_max: float,
) -> None:
    """Kernel to check if particle indices are in bounds."""
    if (
        particle["zidx"] < zidx_min
        or particle["zidx"] > zidx_max
        or particle["yidx"] < yidx_min
        or particle["yidx"] > yidx_max
        or particle["xidx"] < xidx_min
        or particle["xidx"] > xidx_max
    ):
        particle["status"] = 2 # Mark particle as inactive if any index is out of bounds
        
inbounds_kernel = ParticleKernel(
    _inbounds,
    particle_fields={
        "status": np.uint8,
        "zidx": np.float64,
        "yidx": np.float64,
        "xidx": np.float64,
    },
    scalars=("zidx_min", "zidx_max", "yidx_min", "yidx_max", "xidx_min", "xidx_max"),
    simulation_fields=[],
)

validate_indices_kernel = ParticleKernel.from_sequence(
    [finite_indices_kernel, inbounds_kernel])