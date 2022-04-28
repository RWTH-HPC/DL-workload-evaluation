message(STATUS "Searching for CuBLAS")

# Search for headers
find_path(CUBLAS_INCLUDE_DIR cublas.h
        HINTS $ENV{CUDA_ROOT}/include "usr/local/include/")

# Search for Lib
find_path(CUBLAS_LIBRARY_DIR libcublas.so
        HINTS $ENV{CUDA_ROOT}/lib64/lib64 "usr/local/lib64/")

set(CUBLAS_LIBRARY ${CUBLAS_LIBRARY_DIR}/libcublas.so)
message(STATUS "CUBLAS found at ${CUBLAS_INCLUDE_DIR} and ${CUBLAS_LIBRARY_DIR}")

# Error if headers or libs are missing
if(${CUBLAS_INCLUDE_DIR} STREQUAL "CUBLAS_INCLUDE_DIR-NOTFOUND")
    message(WARNING "cublas.h not found in CUDA_ROOT/inlude")
endif()

if(${CUBLAS_LIBRARY_DIR} STREQUAL "CUBLAS_LIBRARY_DIR-NOTFOUND")
    message(WARNING "libcublas.so not found in CUDNN_ROOT/lib64/lib64")
endif()