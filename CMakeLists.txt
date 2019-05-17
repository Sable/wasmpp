cmake_minimum_required(VERSION 3.5)
project(wasmpp)
set(CMAKE_VERBOSE_MAKEFILE ON)

option(BUILD_TESTS "Build GTest-based tests" OFF)
add_compile_options(-std=c++11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
include_directories(${PROJECT_SOURCE_DIR})

# Include WABT
add_subdirectory(third_party/wabt)
# Extra includes to use wabt
include_directories(${PROJECT_SOURCE_DIR}/third_party/wabt)
include_directories(${PROJECT_SOURCE_DIR}/third_party/wabt/src)
include_directories(${CMAKE_BINARY_DIR}/third_party/wabt/)

# Create wasmpp library
file(GLOB_RECURSE SRC_FILES src/wasmpp/*.cc src/wasmpp/*/*.cc)
add_library(${PROJECT_NAME} ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} wabt)

# Create learn-or application
set(NN_APP "nn")
file(GLOB_RECURSE NN_SRC_FILES src/apps/nn-builder/*.cc src/apps/nn-builder/*/*.cc)
add_executable(${NN_APP} ${NN_SRC_FILES})
target_link_libraries(${NN_APP} wabt wasmpp)

