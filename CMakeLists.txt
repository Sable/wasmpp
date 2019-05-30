cmake_minimum_required(VERSION 3.5)
set(CMAKE_VERBOSE_MAKEFILE ON)

option(BUILD_TESTS "Build GTest-based tests" OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
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

# All targets
set(WABT wabt)
set(WASMPP wasmpp)
set(NN_BUILDER nnb)
set(LOGIC logic)
set(MNIST mnist)

# Create wasmpp library
file(GLOB_RECURSE WASMPP_FILES src/wasmpp/*.cc)
add_library(${WASMPP} ${WASMPP_FILES})
target_link_libraries(${WASMPP} ${WABT})

# Create neural network library
file(GLOB_RECURSE NN_BUILDER_FILES src/nn-builder/*/*.cc)
add_library(${NN_BUILDER} ${NN_BUILDER_FILES})
target_link_libraries(${NN_BUILDER} ${WASMPP})

# Create logic example
file(GLOB_RECURSE LOGIC_FILES src/nn-builder/examples/logic.cc)
add_executable(${LOGIC} ${LOGIC_FILES})
target_link_libraries(${LOGIC} ${NN_BUILDER})

# Create mnist example
file(GLOB_RECURSE MNIST_FILES src/nn-builder/examples/mnist.cc)
add_executable(${MNIST} ${MNIST_FILES})
target_link_libraries(${MNIST} ${NN_BUILDER})
