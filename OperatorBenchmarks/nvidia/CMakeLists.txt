enable_language(CUDA)

find_package(CUDNN)

set(CUDNN_INCLUDE_DIR ${CUDNN_INCLUDE_DIR};${CMAKE_CURRENT_SOURCE_DIR}/helpers)

add_subdirectory(convolution)
add_subdirectory(gemm)