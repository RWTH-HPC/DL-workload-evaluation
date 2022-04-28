add_executable(cudnn_gemm cudnn_gemm.cu)

find_package(CUBLAS)

# include headers
target_include_directories(cudnn_gemm PRIVATE ${CUBLAS_INCLUDE_DIR};${CUDNN_INCLUDE_DIR})

# link prebuild library
target_link_libraries(cudnn_gemm cudnn curand ${CUBLAS_LIBRARY})

add_definitions("-std=c++11")