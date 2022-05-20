cmake_minimum_required(VERSION 3.10)

project(sand_dune)

file(GLOB SRC "src/*.cpp")

add_executable(simulation ${SRC})

target_include_directories(simulation
    PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
)