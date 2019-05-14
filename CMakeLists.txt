cmake_minimum_required(VERSION 3.5)
project(wasmpp)
set(CMAKE_VERBOSE_MAKEFILE ON)

option(BUILD_TESTS "Build GTest-based tests" OFF)
add_subdirectory(third_party/wabt)
add_compile_options(-std=c++11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
include_directories(${PROJECT_SOURCE_DIR})
# Extra includes to use wabt
include_directories(${PROJECT_SOURCE_DIR}/third_party/wabt)
include_directories(${PROJECT_SOURCE_DIR}/third_party/wabt/src)
include_directories(${CMAKE_BINARY_DIR}/third_party/wabt/)
file(GLOB_RECURSE SRC_FILES src/*/*.cpp)
add_executable(${PROJECT_NAME} ${SRC_FILES} src/main.cc)
target_link_libraries(${PROJECT_NAME} wabt)

