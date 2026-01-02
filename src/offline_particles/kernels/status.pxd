cdef enum ParticleStatus:
    # bit flag for active/ inactive particles
    STATUS_INACTIVE

    # normal state
    STATUS_NORMAL

    # error states
    STATUS_NONFINITE
    STATUS_OUT_OF_DOMAIN

    # Reserved for multistep initialization
    STATUS_MULTISTEP_1
    STATUS_MULTISTEP_2
