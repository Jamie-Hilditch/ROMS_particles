"""Submodule for handling vertical coordinate transformations in ROMS.

We follow the same variable naming conventions as ROMS:
- zidx: vertical index
- Nz: total number of vertical rho levels
- hc: critical depth
- sigma: sigma coordinate (uniformly spaced between -1 and 0)
- h: bathymetric depth
- C: stretching function value
- S: S-coordinate value (stretched sigma coordinate)
- zeta: free surface elevation
- z: physical vertical coordinate
"""

cdef inline double _sigma_coordinate(double zidx, int Nz) noexcept nogil
cdef inline double _S_coordinate(double hc, double sigma, double h, double C) noexcept nogil
cdef inline double _z_coordinate(double S, double h, double zeta) noexcept nogil
cdef inline double compute_z(double zidx, int Nz, double hc, double h, double C, double zeta) noexcept nogil
cdef inline double _S_from_z(double z, double h, double zeta) noexcept nogil
cdef inline double _sigma_from_Cidx(int Cidx, double C_offset, int Nz) noexcept nogil
cdef inline Py_ssize_t _compute_Cidx_from_S(double S, double hc, int NZ, double h, double zeta, const double[::1] C, double C_offset) noexcept nogil
cdef inline double _zidx_from_S(double S, double hc, int NZ, double h, double zeta, const double[::1] C, double C_offset) noexcept nogil
cdef inline double compute_zidx(double z, double h, double zeta, double hc, int NZ, const double[::1] C, double C_offset) noexcept nogil
