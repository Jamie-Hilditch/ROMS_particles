"""Functions for interpolation."""

cdef inline Py_ssize_t truncate_index(double idx, Py_ssize_t max_idx) noexcept nogil
cdef inline double linear_interpolation(const double[::1] array, double idx) noexcept nogil
cdef inline double bilinear_interpolation(const double[:, ::1] array, double idx0, double idx1) noexcept nogil
cdef inline double trilinear_interpolation(const double[:, :, ::1] array, double idx0, double idx1, double idx2) noexcept nogil