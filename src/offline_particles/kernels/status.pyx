"""Particle status codes."""

from enum import IntEnum

cdef enum ParticleStatus:
    # bit flag for active/ inactive particles
    STATUS_INACTIVE = 1 << 7  # reserve final bit for inactive flag

    # normal state
    STATUS_NORMAL = 0

    # error states
    STATUS_NONFINITE = 1 | STATUS_INACTIVE
    STATUS_OUT_OF_DOMAIN = 2 | STATUS_INACTIVE

    # Reserved for multistep initialization
    STATUS_MULTISTEP_1 = 10
    STATUS_MULTISTEP_2 = 11


class STATUS(IntEnum):
    # inactive flag
    INACTIVE = STATUS_INACTIVE

    # normal state
    NORMAL = STATUS_NORMAL

    # error states
    NONFINITE = STATUS_NONFINITE
    OUT_OF_DOMAIN = STATUS_OUT_OF_DOMAIN

    # Reserved for multistep initialization
    MULTISTEP_1 = STATUS_MULTISTEP_1
    MULTISTEP_2 = STATUS_MULTISTEP_2
