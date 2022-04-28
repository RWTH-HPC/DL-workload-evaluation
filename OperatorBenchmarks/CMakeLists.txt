cmake_minimum_required(VERSION 3.10)
set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")
project(OperatorBenchmarks)

set(CMAKE_CXX_STANDARD 11)

# Function to load Modules
function(MODULE_LOAD LOAD)
    set(CMD tclsh)
    set(MODULES /usr/local_rwth/modules/src/modulecmd.tcl sh load ${LOAD})
    execute_process(COMMAND ${CMD} ${MODULES} RESULT_VARIABLE res OUTPUT_VARIABLE out)
endfunction()

add_subdirectory(intel)
add_subdirectory(nvidia)