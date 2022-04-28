message(STATUS "Searching for MklDNN")
# This uses MKLDNN_INCLUDE_DIR and MKLDNN_LIBRARY_DIR as well as the standard install locations
# to find MKL-DNN

# Search for headers
find_path(MKLDNN_INCLUDE_DIR mkldnn.hpp
        HINTS $ENV{MKLDNN_INCLUDE_DIR} "usr/local/include/"
        PATH_SUFFIXES include/ )

# Search for Lib
find_path(MKLDNN_LIBRARY_DIR libmkldnn.so
        HINTS $ENV{MKLDNN_LIBRARY_DIR} "usr/local/lib64/")

set(MKLDNN_LIBRARY ${MKLDNN_LIBRARY_DIR}/libmkldnn.so)
message(STATUS "MKLDNN found at ${MKLDNN_INCLUDE_DIR}")

# Error if headers or libs are missing
if(${MKLDNN_INCLUDE_DIR} STREQUAL "MKLDNN_INCLUDE_DIR-NOTFOUND")
    message(WARNING "mkldnn.hpp not found in MKLDNN_INCLUDE_DIR")
endif()

if(${MKLDNN_LIBRARY_DIR} STREQUAL "MKLDNN_LIBRARY_DIR-NOTFOUND")
    message(WARNING "libmkldnn.so not found in MKLDNN_LIBRARY_DIR")
endif()