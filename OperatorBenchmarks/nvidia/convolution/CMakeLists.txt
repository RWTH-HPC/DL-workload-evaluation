add_executable(cudnn_conv cudnn_convolution.cu)

# include headers
target_include_directories(cudnn_conv PRIVATE ${CUDNN_INCLUDE_DIR})

# link prebuild library
target_link_libraries(cudnn_conv cudnn curand)

add_definitions("-std=c++11")