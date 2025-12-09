"""Core functions for writing kernels in Cython."""

cimport cython
import numpy as np

cdef inline void unpack_fielddata_1d(FieldData fd, double[::1] *array, double *offset):
    array[0] = np.ascontiguousarray(fd.array, dtype=np.float64)
    offset[0] = fd.offsets[0]


cdef inline void unpack_fielddata_2d(FieldData fd, double[:, ::1] *array, double *offset_0, double *offset_1):
    array[0] = np.ascontiguousarray(fd.array, dtype=np.float64)    
    offset_0[0] = fd.offsets[0]
    offset_1[0] = fd.offsets[1]

cdef inline void unpack_fielddata_3d(FieldData fd, double[:, :, ::1] *array, double *offset_0, double *offset_1, double *offset_2):
    array[0] = np.ascontiguousarray(fd.array, dtype=np.float64)    
    offset_0[0] = fd.offsets[0]
    offset_1[0] = fd.offsets[1]
    offset_2[0] = fd.offsets[2]

