"""Advection scheme froms ROMS data with vertical velocity w.

Horizontal advection is done in index space. However, ROMS uses a sigma
coordinate system in the vertical, so vertical advection requires special treatment.
This module implements vertical advection in physical space using the vertical
velocity w and then transforms that into index space.
"""

cimport cython 
from cython.parallel cimport prange

from ..core cimport unpack_fielddata_1d, unpack_fielddata_2d, unpack_fielddata_3d
from ..interpolation cimport trilinear_interpolation, bilinear_interpolation, linear_interpolation
from .vertical_coordinate cimport S_from_z, z_from_S, compute_zidx_from_S


cdef void _rk2_step_1(Particles particles, Scalars scalars, FieldData fielddata):
    # unpack required particle fields
    cdef unsigned char status[::1] = particles.status
    cdef double[::1] zidx = particles.zidx
    cdef double[::1] yidx = particles.yidx
    cdef double[::1] xidx = particles.xidx
    cdef double[::1] z = particles.z
    cdef double[::1] dxidx1 = particles._dxidx1
    cdef double[::1] dyidx1 = particles._dyidx1
    cdef double[::1] dz1 = particles._dz1

    # unpack scalars 
    cdef double hc = scalars["hc"]
    cdef int NZ = scalars["NZ"]

    # unpack 3D field data
    cdef double[:, :, ::1] readonly u_array, v_array, w_array
    cdef double u_offz, u_offy, u_offx
    cdef double v_offz, v_offy, v_offx
    cdef double w_offz, w_offy, w_offx
    unpack_fielddata_3d(fielddata["u"], &u_array, &u_offz, &u_offy, &u_offx)
    unpack_fielddata_3d(fielddata["v"], &v_array, &v_offz, &v_offy, &v_offx)
    unpack_fielddata_3d(fielddata["w"], &w_array, &w_offz, &w_offy, &w_offx)
    
    # unpack 2D field data
    cdef double[:, ::1] readonly dx_array, dy_array, h_array, zeta_array
    cdef double dx_offy, dx_offx
    cdef double dy_offy, dy_offx
    cdef double h_offy, h_offx
    cdef double zeta_offy, zeta_offx
    unpack_fielddata_2d(fielddata["dx"], &dx_array, &dx_offy, &dx_offx)
    unpack_fielddata_2d(fielddata["dy"], &dy_array, &dy_offy, &dy_offx)
    unpack_fielddata_2d(fielddata["h"], &h_array, &h_offy, &h_offx)
    unpack_fielddata_2d(fielddata["zeta"], &zeta_array, &zeta_offy, &zeta_offx)


    # unpack 1D field data
    cdef double[::1] readonly C_array
    cdef double C_offz
    unpack_fielddata_1d(fielddata["C"], &C_array, &C_offz)

    # loop over particles
    cdef Py_ssize_t nparticles = status.shape[0]

    with nogil:
        for i in prange(nparticles, schedule='static'):
            # skip inactive particles
            if status[i] != 0:
                continue 

            # first compute z 
            cdef double h_value = bilinear_interpolation(
                h_array, 
                yidx[i] + h_offy, 
                xidx[i] + h_offx
            )
            cdef double zeta_value = bilinear_interpolation(
                zeta_array, 
                yidx[i] + zeta_offy, 
                xidx[i] + zeta_offx
            )
            cdef double sigma_value = sigma_coordinate(zidx[i], NZ)
            cdef double C_value = linear_interpolation(C_array, zidx[i] + C_offz)
            cdef double S_value = S_coordinate(hc, sigma_value, h_value, C_value)
            z[i] = z_from_S(S_value, h_value, zeta_value)

            # then interpolate horizontal velocities in index space
            cdef double u_value = trilinear_interpolation(
                u_array,    
                zidx[i] + u_offz,
                yidx[i] + u_offy,
                xidx[i] + u_offx
            )
            cdef double v_value = trilinear_interpolation(
                v_array,
                zidx[i] + v_offz,
                yidx[i] + v_offy,
                xidx[i] + v_offx
            )
            cdef double dx_value = bilinear_interpolation(
                dx_array,
                yidx[i] + dx_offy,
                xidx[i] + dx_offx
            )
            cdef double dy_value = bilinear_interpolation(
                dy_array,
                yidx[i] + dy_offy,
                xidx[i] + dy_offx
            )
            dxidx1[i] = u_value / dx_value
            dyidx1[i] = v_value / dy_value

            # finally do vertical advection in physical space
            dz1[i] = trilinear_interpolation(
                w_array,
                zidx[i] + w_offz,
                yidx[i] + w_offy,
                xidx[i] + w_offx
            )

           

cdef void _rk2_step_2(Particles particles, Scalars scalars, FieldData fielddata):
    # unpack required particle fields
    cdef unsigned char status[::1] = particles.status
    cdef double[::1] zidx = particles.zidx
    cdef double[::1] yidx = particles.yidx
    cdef double[::1] xidx = particles.xidx
    cdef double[::1] z = particles.z
    cdef double[::1] dxidx1 = particles._dxidx1
    cdef double[::1] dyidx1 = particles._dyidx1
    cdef double[::1] dz1 = particles._dz1
    cdef double[::1] dxidx2 = particles._dxidx2
    cdef double[::1] dyidx2 = particles._dyidx2
    cdef double[::1] dz2 = particles._dz2

    # unpack scalars 
    cdef double dt = scalars["_dt"]
    cdef double alpha = scalars["_RK2_alpha"]
    cdef double hc = scalars["hc"]
    cdef int NZ = scalars["NZ"]

     # unpack 3D field data
    cdef double[:, :, ::1] readonly u_array, v_array, w_array
    cdef double u_offz, u_offy, u_offx
    cdef double v_offz, v_offy, v_offx
    cdef double w_offz, w_offy, w_offx
    unpack_fielddata_3d(fielddata["u"], &u_array, &u_offz, &u_offy, &u_offx)
    unpack_fielddata_3d(fielddata["v"], &v_array, &v_offz, &v_offy, &v_offx)
    unpack_fielddata_3d(fielddata["w"], &w_array, &w_offz, &w_offy, &w_offx)
    
    # unpack 2D field data
    cdef double[:, ::1] readonly dx_array, dy_array, h_array, zeta_array
    cdef double dx_offy, dx_offx
    cdef double dy_offy, dy_offx
    cdef double h_offy, h_offx
    cdef double zeta_offy, zeta_offx
    unpack_fielddata_2d(fielddata["dx"], &dx_array, &dx_offy, &dx_offx)
    unpack_fielddata_2d(fielddata["dy"], &dy_array, &dy_offy, &dy_offx)
    unpack_fielddata_2d(fielddata["h"], &h_array, &h_offy, &h_offx)
    unpack_fielddata_2d(fielddata["zeta"], &zeta_array, &zeta_offy, &zeta_offx)


    # unpack 1D field data
    cdef double[::1] readonly C_array
    cdef double C_offz
    unpack_fielddata_1d(fielddata["C"], &C_array, &C_offz)

    # loop over particles
    cdef Py_ssize_t nparticles = status.shape[0]

    with nogil:
        for i in prange(nparticles, schedule='static'):
            # skip inactive particles
            if status[i] != 0:
                continue 

            # intermediate positions
            cdef double z_int = z[i] + alpha * dt * dz1[i]
            cdef double yidx_int = yidx[i] + alpha * dt * dyidx1[i]
            cdef double xidx_int = xidx[i] + alpha * dt * dxidx1[i]

            # compute intermediate zidx
            cdef double h_value = bilinear_interpolation(
                h_array, 
                yidx_int + h_offy, 
                xidx_int + h_offx
            )
            cdef double zeta_value = bilinear_interpolation(
                zeta_array, 
                yidx_int + zeta_offy, 
                xidx_int + zeta_offx
            )
            cdef double S_value = S_from_z(z_int, h_value, zeta_value)
            cdef double zidx_int = compute_zidx_from_S(S_value, hc, NZ, h_value, zeta_value, C_array, C_offz)

            # interpolate horizontal velocities in index space
            cdef double u_value = trilinear_interpolation(
                u_array,    
                zidx_int + u_offz,
                yidx_int + u_offy,
                xidx_int + u_offx
            )
            cdef double v_value = trilinear_interpolation(
                v_array,
                zidx_int + v_offz,
                yidx_int + v_offy,
                xidx_int + v_offx
            )
            cdef double dx_value = bilinear_interpolation(
                dx_array,
                yidx_int + dx_offy,
                xidx_int + dx_offx
            )
            cdef double dy_value = bilinear_interpolation(
                dy_array,
                yidx_int + dy_offy,
                xidx_int + dy_offx
            )
            dxidx2[i] = u_value / dx_value
            dyidx2[i] = v_value / dy_value

            # finally do vertical advection in physical space
            dz2[i] = trilinear_interpolation(
                w_array,
                zidx_int + w_offz,
                yidx_int + w_offy,
                xidx_int + w_offx
            )
            

cdef void _rk2_update(Particles particles, Scalars scalars, FieldData fieldata):
    # unpack required particle fields
    cdef unsigned char status[::1] = particles.status
    cdef double[::1] zidx = particles.zidx
    cdef double[::1] yidx = particles.yidx
    cdef double[::1] xidx = particles.xidx
    cdef double[::1] z = particles.z
    cdef double[::1] dxidx1 = particles._dxidx1
    cdef double[::1] dyidx1 = particles._dyidx1
    cdef double[::1] dz1 = particles._dz1
    cdef double[::1] dxidx2 = particles._dxidx2
    cdef double[::1] dyidx2 = particles._dyidx2
    cdef double[::1] dz2 = particles._dz2

    # unpack scalars 
    cdef double dt = scalars["_dt"]
    cdef double alpha = scalars["_RK2_alpha"]
    cdef double hc = scalars["hc"]
    cdef int NZ = scalars["NZ"]
    
    # unpack 2D field data
    cdef double[:, ::1] readonly h_array, zeta_array
    cdef double h_offy, h_offx
    cdef double zeta_offy, zeta_offx
    unpack_fielddata_2d(fieldata["h"], &h_array, &h_offy, &h_offx)
    unpack_fielddata_2d(fieldata["zeta"], &zeta_array, &zeta_offy, &zeta_offx)

    # unpack 1D field data
    cdef double[::1] readonly C_array
    cdef double C_offz
    unpack_fielddata_1d(fieldata["C"], &C_array, C_offz)

    # loop over particles
    cdef Py_ssize_t nparticles = status.shape[0]

    # rk constants
    cdef double b1 = 1.0 / (2.0 * alpha)
    cdef double b2 = 1.0 - b1

    with nogil:
        for i in prange(nparticles, schedule='static'):
            # skip inactive particles
            if status[i] != 0:
                continue 

            # update positions
            z[i] = z[i] + b1 * dt * dz1[i] + b2 * dt * dz2[i]
            yidx[i] = yidx[i] + b1 * dt * dyidx1[i] + b2 * dt * dyidx2[i]
            xidx[i] = xidx[i] + b1 * dt * dxidx1[i] + b2 * dt * dxidx2[i]

            # compute intermediate zidx
            cdef double h_value = bilinear_interpolation(
                h_array, 
                yidx[i] + h_offy, 
                xidx[i] + h_offx
            )
            cdef double zeta_value = bilinear_interpolation(
                zeta_array, 
                yidx[i] + zeta_offy, 
                xidx[i] + zeta_offx
            )
            cdef double S_value = S_from_z(z[i], h_value, zeta_value)
            zidx[i] = compute_zidx_from_S(S_value, hc, NZ, h_value, zeta_value, C_array, C_offz)
