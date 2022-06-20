cmake_minimum_required(VERSION 3.10)

project(sand_dune)
file(GLOB SRC "src/*.cpp")

add_executable(simulation ${SRC})
# add_executable(simulation "src/main.cpp")

# find_package(pybind11 CONFIG REQUIRED)
add_subdirectory(external/pybind11)

target_include_directories(simulation
    PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
)

# set(CMAKE_CXX_STANDARD 17)
# set_property(TARGET simulation PROPERTY CXX_STANDARD 17)

# project(HelloVSCode
#     LANGUAGES CXX
# )

# add_executable(${PROJECT_NAME} main.cpp)