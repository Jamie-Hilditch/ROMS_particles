from enum import IntEnum

import numpy as np
import numpy.typing as npt

class ParticleStatus(IntEnum):
    # inactive flag
    INACTIVE: int

    # normal state
    NORMAL: int

    # error states
    NONFINITE: int
    OUT_OF_DOMAIN: int

    # Reserved for multistep initialization
    MULTISTEP_1: int
    MULTISTEP_2: int

def is_inactive(status: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]: ...
def is_active(status: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]: ...
