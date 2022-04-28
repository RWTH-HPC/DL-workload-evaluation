#load packages necessary for all intel based benchmarks
find_package(MklDNN)

# load OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -std=c++11 -O3")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp -std=c++11 -O3 -lrt")
endif()

#find vpu

add_subdirectory(convolution)
#add_subdirectory(simple_operators)
#add_subdirectory(gemm)
#add_subdirectory(spmm)
#add_subdirectory(rnn)