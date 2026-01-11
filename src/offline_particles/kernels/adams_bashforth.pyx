"""Kernels for implementing Adams-Bashforth schemes."""

from cython.parallel cimport prange

from .status cimport STATUS

import functools

import numpy as np

from ._kernels import ParticleKernel

cdef void _ab2_update(particles, scalars, fielddata, field, tendency_field_0, tendency_field_1):
    # unpack required particle fields
    cdef unsigned char[::1] status
    cdef double[::1] f, df0, df1
    status = particles.status
    f = particles[field]
    df0 = particles[tendency_field_0]
    df1 = particles[tendency_field_1]

    # unpack scalars
    cdef double dt = scalars["_dt"]

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(nparticles, schedule='static', nogil=True):
        if status[i] & STATUS.INACTIVE:
            continue

        # handle initialization steps
        if status[i] == STATUS.MULTISTEP_1:
            # if on first step use forward Euler, i.e. set prior step derivatives equal to current
            df1[i] = df0[i]

        # update field using AB2 scheme
        f[i] += dt * (df0[i] * 1.5 - df1[i] * 0.5)

        # shift derivatives for next time step
        df1[i] = df0[i]
        df0[i] = 0.0  # reset current tendency for next accumulation

cdef void _ab2_bump_status(particles, scalars, fielddata):
    # unpack required particle fields
    cdef unsigned char[::1] status
    status = particles.status

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(nparticles, schedule='static', nogil=True):
        if status[i] & STATUS.INACTIVE:
            continue

        # update status to indicate multistep has been initialized
        if status[i] == STATUS.MULTISTEP_1:
            status[i] = STATUS.NORMAL

cdef void _ab3_update(particles, scalars, fielddata, field, tendency_field_0, tendency_field_1, tendency_field_2):
    # unpack required particle fields
    cdef unsigned char[::1] status
    cdef double[::1] f, df0, df1, df2
    status = particles.status
    f = particles[field]
    df0 = particles[tendency_field_0]
    df1 = particles[tendency_field_1]
    df2 = particles[tendency_field_2]

    # unpack scalars
    cdef double dt = scalars["_dt"]

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(nparticles, schedule='static', nogil=True):
        if status[i] & STATUS.INACTIVE:
            continue

        # handle initialization steps
        if status[i] == STATUS.MULTISTEP_1:
            # if on first step use forward Euler, i.e. set prior step derivatives equal to current
            df2[i] = df1[i]
            df1[i] = df0[i]
        elif status[i] == STATUS.MULTISTEP_2:
            # if on second step set df2 to be consistent with AB2
            df2[i] = 2.0 * df1[i] - df0[i]

        # update field using AB3 scheme
        f[i] += dt * (df0[i] * 23.0 - df1[i] * 16.0 + df2[i] * 5.0) / 12.0

        # shift derivatives for next time step
        df2[i] = df1[i]
        df1[i] = df0[i]
        df0[i] = 0.0  # reset current tendency for next accumulation

cdef _ab3_bump_status(particles, scalars, fielddata):
    # unpack required particle fields
    cdef unsigned char[::1] status
    status = particles.status

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(nparticles, schedule='static', nogil=True):
        if status[i] & STATUS.INACTIVE:
            continue

        # update status to indicate multistep has been initialized
        if status[i] == STATUS.MULTISTEP_1:
            status[i] = STATUS.MULTISTEP_2
        elif status[i] == STATUS.MULTISTEP_2:
            status[i] = STATUS.NORMAL

# python wrappers
cpdef ab2_update(particles, scalars, fielddata, field, tendency_field_0, tendency_field_1):
    """
    Update particle field using 2nd-order Adams-Bashforth scheme.
    """
    _ab2_update(particles, scalars, fielddata, field, tendency_field_0, tendency_field_1)

cpdef ab2_bump_status(particles, scalars, fielddata):
    """
    Bump particle status after Adams-Bashforth 2nd-order update.
    """
    _ab2_bump_status(particles, scalars, fielddata)

cpdef ab3_update(particles, scalars, fielddata, field, tendency_field_0, tendency_field_1, tendency_field_2):
    """
    Update particle field using 3rd-order Adams-Bashforth scheme.
    """
    _ab3_update(particles, scalars, fielddata, field, tendency_field_0, tendency_field_1, tendency_field_2)

cpdef ab3_bump_status(particles, scalars, fielddata):
    """
    Bump particle status after Adams-Bashforth 3rd-order update.
    """
    _ab3_bump_status(particles, scalars, fielddata)

# kernels


def ab2_update_kernel(
    field: str, tendency_field_0: str | None = None, tendency_field_1: str | None = None, dtype: np.dtype = np.float64
) -> ParticleKernel:
    """
    Create an Adams-Bashforth 2nd-order update kernel for the specified field.

    Parameters
    ----------
    field : str
        The name of the particle field to be updated.
    tendency_field_0 : str | None, optional
        The field string the tendency from the current time (e.g., '_dz0' for vertical position).
        If None, defaults to '_d' + field + '0'.
    tendency_field_1 : str | None, optional
        The field string the tendency from the prior time (e.g., '_dz1' for vertical position).
        If None, defaults to '_d' + field + '1'.

    Returns
    -------
    ParticleKernel
        The Adams-Bashforth 2nd-order update kernel.
    """
    if tendency_field_0 is None:
        tendency_field_0 = "_d" + field + "0"
    if tendency_field_1 is None:
        tendency_field_1 = "_d" + field + "1"

    kernel_func = functools.partial(
        _ab2_update,
        field=field,
        tendency_field_0=tendency_field_0,
        tendency_field_1=tendency_field_1
    )

    return ParticleKernel(
        kernel_func,
        particle_fields={
            "status": np.uint8,
            field: dtype,
            tendency_field_0: dtype,
            tendency_field_1: dtype,
        },
        scalars={
            "_dt": np.float64,
        },
        simulation_fields=[],
    )


ab2_bump_status_kernel = ParticleKernel(
    _ab2_bump_status,
    particle_fields={
        "status": np.uint8,
    },
    scalars={},
    simulation_fields=[],
)


def ab3_update_kernel(
    field: str,
    tendency_field_0: str | None = None,
    tendency_field_1: str | None = None,
    tendency_field_2: str | None = None,
    dtype: np.dtype = np.float64
) -> ParticleKernel:
    """
    Create an Adams-Bashforth 3nd-order update kernel for the specified field.

    Parameters
    ----------
    field : str
        The name of the particle field to be updated.
    tendency_field_0 : str | None, optional
        The field string the tendency from the current time (e.g., '_dz0' for vertical position).
        If None, defaults to '_d' + field + '0'.
    tendency_field_1 : str | None, optional
        The field string the tendency from the prior time (e.g., '_dz1' for vertical position).
        If None, defaults to '_d' + field + '1'.
    tendency_field_2 : str | None, optional
        The field string the tendency from two time steps prior (e.g., '_dz2' for vertical position).
        If None, defaults to '_d' + field + '2'.

    Returns
    -------
    ParticleKernel
        The Adams-Bashforth 3nd-order update kernel.
    """
    if tendency_field_0 is None:
        tendency_field_0 = "_d" + field + "0"
    if tendency_field_1 is None:
        tendency_field_1 = "_d" + field + "1"
    if tendency_field_2 is None:
        tendency_field_2 = "_d" + field + "2"

    kernel_func = functools.partial(
        _ab3_update,
        field=field,
        tendency_field_0=tendency_field_0,
        tendency_field_1=tendency_field_1,
        tendency_field_2=tendency_field_2
    )

    return ParticleKernel(
        kernel_func,
        particle_fields={
            "status": np.uint8,
            field: dtype,
            tendency_field_0: dtype,
            tendency_field_1: dtype,
            tendency_field_2: dtype,
        },
        scalars={
            "_dt": np.float64,
        },
        simulation_fields=[],
    )


ab3_bump_status_kernel = ParticleKernel(
    _ab3_bump_status,
    particle_fields={
        "status": np.uint8,
    },
    scalars={},
    simulation_fields=[],
)
