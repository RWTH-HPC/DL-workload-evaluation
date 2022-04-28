message(STATUS "Searching for CuDNN")
# This uses CUDNN_ROOT as well as the standard install locations
# to find CUDNN

# Search for headers
find_path(CUDNN_INCLUDE_DIR cudnn.h
        HINTS $ENV{CUDNN_ROOT}/include "usr/local/include/"
        PATH_SUFFIXES include/ )

# Search for Lib
find_path(CUDNN_LIBRARY_DIR libcudnn.so
        HINTS $ENV{CUDNN_ROOT}/lib64 "usr/local/lib64/")

set(CUDNN_LIBRARY ${CUDNN_LIBRARY_DIR}/libcudnn.so)
message(STATUS "CUDNN found at ${CUDNN_INCLUDE_DIR}")

# Error if headers or libs are missing
if(${CUDNN_INCLUDE_DIR} STREQUAL "CUDNN_INCLUDE_DIR-NOTFOUND")
    message(WARNING "cudnn.h not found in CUDNN_ROOT/inlude")
endif()

if(${CUDNN_LIBRARY_DIR} STREQUAL "CUDNN_LIBRARY_DIR-NOTFOUND")
    message(WARNING "libcudnn.so not found in CUDNN_ROOT/lib64")
endif()