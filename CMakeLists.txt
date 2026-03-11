cmake_minimum_required(VERSION 3.10)

# Use clang as compiler
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

project(NeuralForge)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/core/layers/Convolution.cpp
    src/core/layers/FullyConnected.cpp
    src/core/layers/MaxPool.cpp
    src/core/layers/Relu.cpp
    src/core/layers/Sigmoid.cpp
    src/core/loss/CrossEntropyLoss.cpp
    src/core/optimizers/SGD.cpp
    src/math/Matrix.cpp
    src/util/DataLoader.cpp
)

# Create library
add_library(NeuralForge ${SOURCES})

# Optional: Create test executable (if tests exist)
if(EXISTS "${CMAKE_SOURCE_DIR}/tests")
    file(GLOB TEST_SOURCES "tests/*.cpp")
    if(TEST_SOURCES)
        enable_testing()
        add_executable(NeuralForge_tests ${TEST_SOURCES})
        target_link_libraries(NeuralForge_tests NeuralForge)
    endif()
endif()
