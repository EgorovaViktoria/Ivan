cmake_minimum_required(VERSION 3.16)
project(IvanAlgorithmsCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(ivan_algorithms_lib
    src/algorithms.cpp
)

target_include_directories(ivan_algorithms_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(ivan_algorithms
    src/main.cpp
)

target_link_libraries(ivan_algorithms
    PRIVATE
        ivan_algorithms_lib
)

