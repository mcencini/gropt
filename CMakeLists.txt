cmake_minimum_required(VERSION 3.18)
project(gropt LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Python COMPONENTS Development.Module REQUIRED)

set(PYTHON_EXT_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/python/gropt")

# Collect all C sources and mark them as C (not C++)
file(GLOB GROPT_C_SOURCES "${CMAKE_SOURCE_DIR}/src/*.c")
set_source_files_properties(${GROPT_C_SOURCES} PROPERTIES LANGUAGE C)

# Pybind11 module
pybind11_add_module(_gropt_binding
    "${CMAKE_SOURCE_DIR}/python/gropt/_gropt_binding.cpp"
    ${GROPT_C_SOURCES}
)

target_include_directories(_gropt_binding PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# Output directory and name
set_target_properties(_gropt_binding PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PYTHON_EXT_OUTPUT_DIR}"
    OUTPUT_NAME "_gropt_binding"
)
# For multi-config generators (MSVC, Xcode), also set per-config output dirs
foreach(cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${cfg}" cfg_uc)
    set_target_properties(_gropt_binding PROPERTIES
        "LIBRARY_OUTPUT_DIRECTORY_${cfg_uc}" "${PYTHON_EXT_OUTPUT_DIR}"
    )
endforeach()