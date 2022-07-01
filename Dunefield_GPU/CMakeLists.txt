cmake_minimum_required(VERSION 3.18)

project(CUDA_dunefield LANGUAGES CXX CUDA)
message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")

add_subdirectory(external/pybind11)

file(GLOB SOURCES 
    src/*.cpp
    src/*.cu
)

pybind11_add_module(gpu_dunefield ${SOURCES})

set_target_properties(gpu_dunefield 
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

target_include_directories(gpu_dunefield PRIVATE ${PROJECT_SOURCE_DIR}/include)