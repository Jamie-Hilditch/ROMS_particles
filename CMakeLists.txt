cmake_minimum_required(VERSION 3.21)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C VERSION ${SKBUILD_PROJECT_VERSION})

# Find pacakges
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED) 

# Enable Cython
find_program(CYTHON_EXECUTABLE cython)

# Recursively find all .pyx files inside src/offline_particles/kernels
file(GLOB_RECURSE CYTHON_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/offline_particles/kernels/*.pyx"
)

set(GENERATED_SOURCES "")

# Generate C source from each .pyx file
foreach(pyx_file ${CYTHON_SOURCES})
    get_filename_component(fname ${pyx_file} NAME_WE)
    get_filename_component(dir ${pyx_file} DIRECTORY)

    # Compute module path (offline_particles/kernels/.../name)
    file(RELATIVE_PATH rel_path
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${pyx_file}"
    )

    string(REPLACE "/" "." module_name ${rel_path})
    string(REPLACE ".pyx" "" module_name ${module_name})

    set(c_file "${CMAKE_CURRENT_BINARY_DIR}/${fname}.c")

    add_custom_command(
        OUTPUT ${c_file}
        COMMAND ${CYTHON_EXECUTABLE} -3 -X boundscheck=False
                -o ${c_file} ${pyx_file}
        DEPENDS ${pyx_file}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    python_add_library(${module_name}
        MODULE
        WITH_SOABI
        ${c_file}
    )

    list(APPEND GENERATED_SOURCES ${c_file})
endforeach()