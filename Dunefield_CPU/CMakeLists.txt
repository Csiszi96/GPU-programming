cmake_minimum_required(VERSION 3.10)

project(sand_dune)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
file(GLOB SRC "src/*.cpp")

add_subdirectory(external/pybind11)

pybind11_add_module(cpu_dunefield  ${SRC})

set_target_properties(cpu_dunefield 
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

target_include_directories(cpu_dunefield 
    PRIVATE ${PROJECT_SOURCE_DIR}/include)