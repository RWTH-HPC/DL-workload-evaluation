add_executable(intel_conv std_conv_bench.cpp)

# include headers
target_include_directories(intel_conv PRIVATE ${MKLDNN_INCLUDE_DIR})

# link prebuild library
add_library(mkldnn SHARED IMPORTED)
target_link_libraries(intel_conv mkldnn)
set_target_properties(mkldnn PROPERTIES IMPORTED_LOCATION ${MKLDNN_LIBRARY})

add_definitions(-DUSE_MKLDNN)
