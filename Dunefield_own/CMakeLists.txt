cmake_minimum_required(VERSION 3.10)

project(sand_dune)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
file(GLOB SRC "src/*.cpp")

# find_package(pybind11 REQUIRED)
add_subdirectory(external/pybind11)

pybind11_add_module(dunefield  ${SRC})
# add_executable(dunefield ${SRC})
# add_executable(dunefield "src/main.cpp")

target_include_directories(dunefield
    PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
)

# set(CMAKE_CXX_STANDARD 17)
# set_property(TARGET dunefield PROPERTY CXX_STANDARD 17)

# project(HelloVSCode
#     LANGUAGES CXX
# )

# add_executable(${PROJECT_NAME} main.cpp)