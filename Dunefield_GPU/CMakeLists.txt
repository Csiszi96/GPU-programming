cmake_minimum_required(VERSION 3.18)

project(CUDA_dunefield LANGUAGES CXX CUDA)
message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

# find_package(pybind11 CONFIG REQUIRED)
add_subdirectory(external/pybind11)

file(GLOB SOURCES 
    src/*.cpp
    src/*.cu
)
add_executable(dunefield ${SOURCES})
# pybind11_add_module(dunefield ${SOURCES})

# add_executable(dunefield src/dunefield.cu src/gpu.cu src/field.cu src/gpu_func.cu)

target_include_directories(dunefield PRIVATE ${PROJECT_SOURCE_DIR}/include)