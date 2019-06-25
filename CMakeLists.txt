cmake_minimum_required(VERSION 3.5)
set(CMAKE_VERBOSE_MAKEFILE ON)

option(BUILD_TESTS "Build GTest-based tests (for wabt)" OFF)
option(USE_WABT_EXPERIMENTAL "Use wabt-experimental instead of wabt" OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(WABT_DIR_NAME "wabt")
include_directories(${PROJECT_SOURCE_DIR})

if(USE_WABT_EXPERIMENTAL)
    set(WABT_DIR_NAME "${WABT_DIR_NAME}-experimental")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWABT_EXPERIMENTAL")
endif()

# Include WABT
message(">> Using ${WABT_DIR_NAME} for building the project")
add_subdirectory(third_party/${WABT_DIR_NAME})
# Extra includes to use wabt
include_directories(${PROJECT_SOURCE_DIR}/third_party/${WABT_DIR_NAME})
include_directories(${PROJECT_SOURCE_DIR}/third_party/${WABT_DIR_NAME}/src)
include_directories(${CMAKE_BINARY_DIR}/third_party/${WABT_DIR_NAME})

# All targets
set(WABT wabt)
set(WASMPP wasmpp)
set(NN_BUILDER nnb)
set(NN_BUILDER_JS nnb_js)
set(NN_TEST nn-test)
set(MNIST mnist)

# Create wasmpp library
file(GLOB_RECURSE WASMPP_FILES src/wasmpp/*.cc)
add_library(${WASMPP} ${WASMPP_FILES})
target_link_libraries(${WASMPP} ${WABT})

# Create neural network library
file(GLOB_RECURSE NN_BUILDER_FILES src/nn-builder/src/*/*.cc)
add_library(${NN_BUILDER} ${NN_BUILDER_FILES})
target_link_libraries(${NN_BUILDER} ${WASMPP})

# Create emscripten bindings
if(EMSCRIPTEN)
    add_executable(${NN_BUILDER_JS} src/nn-builder/emscripten/bindings.cc)
    target_link_libraries(${NN_BUILDER_JS} ${NN_BUILDER})
    set_target_properties(${NN_BUILDER_JS} PROPERTIES LINK_FLAGS "      \
    --bind                                                              \
    --pre-js ${PROJECT_SOURCE_DIR}/src/nn-builder/emscripten/utils.js   \
    -s WASM=1                                                           \
    -s ALLOW_MEMORY_GROWTH=1                                            \
    -s NO_EXIT_RUNTIME=0                                                \
    ")
endif()

# Create test cases
file(GLOB_RECURSE NN_TEST_FILES src/nn-builder/tests/*.cc)
add_executable(${NN_TEST} ${NN_TEST_FILES})
target_link_libraries(${NN_TEST} ${NN_BUILDER})

# Create mnist example
add_executable(${MNIST} src/nn-builder/examples/cpp/mnist.cc)
target_link_libraries(${MNIST} ${NN_BUILDER})
