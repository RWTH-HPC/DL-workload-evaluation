#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <cstdint>
#include <sstream>

#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include "tensor.h"

#ifndef PAD_KERNELS
#define PAD_KERNELS 1
#endif


/*
Usage:

The default precision is set based on the architecture and mode.

By default, the program runs the benchmark in training mode.

bin/gemm_bench

To run inference mode, use the following command:

bin/gemm_bench inference


To change the precision for training/inference, use:

bin/gemm_bench train <precision>
bin/gemm_bench inference <precision>

Supported precision types:

For Maxwell GPUS:
float for training and inference

For Pascal GPUS:
float, half for training
float, half, int8 for inference

*/

struct gemm_problem {
    int m;
    int n;
    int k;
    int a_t;
    int b_t;
    int iters;              //Benchmark repetitions
    int id;
};

template <typename T1, typename T2>
float time_gemm(Tensor<T1> A, Tensor<T1> B, Tensor<T2> C, bool a_t, bool b_t, cublasHandle_t cublas_handle) {

    const int alpha = 1.f;
    const int beta  = 1.f;

    int m = C.dims()[0];
    int k = a_t ? A.dims()[0] : A.dims()[1];
    int n = C.dims()[1];

    int numRepeats = 400;
    cublasStatus_t stat;

    cudaDataType_t A_type = CUDA_R_32F;
    cudaDataType_t B_type = CUDA_R_32F;
    cudaDataType_t C_type = CUDA_R_32F;
    cudaDataType_t compute_type = CUDA_R_32F;
    cublasGemmAlgo_t algo;

    if (std::is_same<T1, uint16_t>::value) {
        A_type = CUDA_R_16F;
        B_type = CUDA_R_16F;
        C_type = CUDA_R_16F;
        compute_type = CUDA_R_16F;
    }

    if (std::is_same<T1, uint8_t>::value) {
        A_type = CUDA_R_8I;
        B_type = CUDA_R_8I;
        C_type = CUDA_R_32I;
        compute_type = CUDA_R_32I;
    }

    algo = CUBLAS_GEMM_DFALT_TENSOR_OP;

    stat = cublasGemmEx(cublas_handle,
                a_t ? CUBLAS_OP_T : CUBLAS_OP_N,
                b_t ? CUBLAS_OP_T : CUBLAS_OP_N,
                m,
                n,
                k,
                &alpha,
                A.begin(), A_type, A.dims()[0],
                B.begin(), B_type, B.dims()[0],
                &beta,
                C.begin(), C_type, C.dims()[0],
                compute_type,
                algo);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("sgemm failed");
    }

    cudaDeviceSynchronize();

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < numRepeats; ++i) {

        stat = cublasGemmEx(cublas_handle,
                    a_t ? CUBLAS_OP_T : CUBLAS_OP_N,
                    b_t ? CUBLAS_OP_T : CUBLAS_OP_N,
                    m,
                    n,
                    k,
                    &alpha,
                    A.begin(), A_type, A.dims()[0],
                    B.begin(), B_type, B.dims()[0],
                    &beta,
                    C.begin(), C_type, C.dims()[0],
                    compute_type,
                    algo);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("sgemm failed");
        }
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();

    return static_cast<float>(std::chrono::duration<double, std::milli>(end - start).count() / numRepeats);

}

int main(int argc, char **argv) {
    cudaFree(0);

    int inference = 0;

    std::string precision;
    precision = "float";

    cublasHandle_t cublas_handle;
    cublasStatus_t status = cublasCreate(&cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS init failed" << std::endl;
    }

    status = cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS math mode failed" << std::endl;
    }

    if(argc == 1){
        std::cout << "No inputfile specified." << std::endl;
        return 1;
    }
    std::string input_filepath = argv[1];
    std::cout << "Inputfilepath: " << input_filepath << std::endl;
    std::ifstream fin(input_filepath, std::ifstream::in);  //ifstream to read from
    if (!fin.is_open())  {  std::cerr << "failed to open file\n"; return 1; }

    std::string output_filepath = input_filepath.substr(0, input_filepath.size() - 4) + "_out.csv";
    std::cout << "Outputfilepath: " << output_filepath << std::endl;
    std::ofstream fout(output_filepath, std::ofstream::out);  //ifstream to write to
    if (!fout.is_open())  {  std::cerr << "failed to open file\n"; return 1; }

    fout << "id,ms" << std::endl;

    std::vector<gemm_problem> problems;
    std::string linestr;
    while (std::getline(fin, linestr) && !linestr.empty())
    {
        std::cout << linestr << std::endl;
        std::stringstream astream(linestr);
        gemm_problem prob;

        //Accepted order: ID, M, N, K, a_t, b_t
        std::vector<int*> prob_struct = {
                &prob.id,
                &prob.m, &prob.n, &prob.k,
                &prob.a_t, &prob.b_t};
        std::string tmp;
        for(int *value : prob_struct){

            std::getline(astream, tmp, ',');
            *value = std::stoi(tmp);
        }
        problems.push_back(prob);
    }

    curandGenerator_t curand_gen;

    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 123ULL);

    if (inference) {
        std::cout << std::setw(45) << "Running inference benchmark " << std::endl;
    } else {
        std::cout << std::setw(45) << "Running training benchmark " << std::endl;
    }

    std::cout << std::setw(30) << "Times" << std::endl;
    std::cout << std::setfill('-') << std::setw(88) << "-" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "    m       n      k      a_t     b_t      precision        time (ms) ";

    if (PAD_KERNELS && precision == "int8" && inference)
        std::cout << " pad_kerenels  ";


    std::cout << std::endl;

    int pad_kernels_count = 0;
    int p_idx = 0;

    for (gemm_problem p : problems) {
        float time_ms;
        bool skip_kernel = false;
        bool need_padding = false;

        if (p.m == 0 || p.n == 0 || p.k == 0){
            std::cout << "Invalid params." << std::endl;
            fout << p_idx << "," << "0.0" << std::endl;
            p_idx ++;
            continue;
        }


        int pad_m;
        pad_m = p.m;
        if (precision == "int8") {
            if (pad_m%4) {
                pad_kernels_count++;
                if (PAD_KERNELS) {
                    pad_dim(pad_m, 4);
                    need_padding = true;
                } else {
                    skip_kernel = true;
                }
            }
        }

        std::cout << std::setw(7) << p.m;
        std::cout << std::setw(7) << p.n;
        std::cout << std::setw(7) << p.k;
        std::cout << std::setw(7) << p.a_t ? "t" : "n";
        std::cout << std::setw(7) << p.b_t ? "t" : "n";

        std::stringstream ss;
        ss << "Unsupported precision requested. Precision: " << precision << " Inference: " << inference;

        if (precision == "half") {
            auto a = rand<uint16_t>({p.a_t ? p.k : p.m, p.a_t ? p.m : p.k}, curand_gen);
            auto b = rand<uint16_t>({p.b_t ? p.n : p.k, p.b_t ? p.k : p.n}, curand_gen);
            auto c = zeros<uint16_t>({p.m, p.n});
            std::cout << std::setw(13) << precision;
            time_ms = time_gemm<uint16_t, uint16_t>(a, b, c, p.a_t, p.b_t, cublas_handle);
        } else if (precision == "float") {
            auto a = rand<float>({p.a_t ? p.k : p.m, p.a_t ? p.m : p.k}, curand_gen);
            auto b = rand<float>({p.b_t ? p.n : p.k, p.b_t ? p.k : p.n}, curand_gen);
            auto c = zeros<float>({p.m, p.n});
            std::cout << std::setw(13) << precision;
            time_ms = time_gemm<float, float>(a, b, c, p.a_t, p.b_t, cublas_handle);
        } else {
            throw std::runtime_error(ss.str());
        }
        std::cout << std::setw(20) << std::setprecision(6);
        fout << p_idx << "," << time_ms << std::endl;
        p_idx ++;

        if (skip_kernel) {
            std::cout << "Not Supported";
        } else {
            std::cout << time_ms;
        }

        std::cout << std::endl;
    }
    fout.close();

    cublasDestroy(cublas_handle);
    curandDestroyGenerator(curand_gen);

    return 0;
}
