/*
 * Based on: https://github.com/baidu-research/DeepBench/blob/master/code/nvidia/conv_bench.cu
 * Original License Apache 2.0 (https://github.com/baidu-research/DeepBench/blob/master/LICENSE)
 */

#include <iomanip>
#include <memory>
#include <chrono>
#include <vector>
#include <tuple>

#include <cuda.h>
#include <cudnn.h>
#include <curand.h>

#include <cuda_profiler_api.h>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include "../helpers/tensor.h"
#include "../helpers/cudnn_helper.h"

#include <iostream>
#include <fstream>
#include <sstream>

#ifndef PAD_KERNELS
#define PAD_KERNELS 1
#endif

#ifndef USE_TENSOR_CORES
#define USE_TENSOR_CORES 0
#endif

struct conv_problem {
    int minibatch;          //Number of minibatches
    int w;                  //Width of input
    int h;                  //Height of input
    int ic;
    int oc;
    int groups;             // Number of groups
    int fw;                 //Filter/Kernel width
    int fh;                 //Filter/Kernel height
    int stride_w, stride_h; //Strides
    int pad_w, pad_h;       //Padding
    int iters;              //Benchmark repetitions
    int id;
};

// T1 is used as the data type for inputs, weights and outputs.
// T2 is used to describe the compute precision.
template <typename T1, typename T2>
class cudnnCNN {
    TensorDescriptor4d<T1> x_desc_;
    TensorDescriptor4d<T1> h_desc_;

    FilterDescriptor4d<T1> w_desc_;

    std::vector<int> output_dims_;
    int num_repeats_;

    size_t fwd_workspace_size_;
    size_t bwd_inputs_workspace_size_;
    size_t bwd_params_workspace_size_;

    Tensor<float> fwd_workspace_;
    Tensor<float> bwd_inputs_workspace_;
    Tensor<float> bwd_params_workspace_;

    cudnnConvolutionFwdAlgo_t fwd_algo_;
    cudnnConvolutionBwdDataAlgo_t bwd_inputs_algo_;
    cudnnConvolutionBwdFilterAlgo_t bwd_params_algo_;

    const float alpha_ = 1.f;
    const float beta_  = 0.f;

    ConvolutionDescriptor<T2> conv_desc_;
    CudnnHandle cudnn_handle_;

public:

    cudnnCNN(int w, int h, int c, int n, int k, int r, int s,
             int groups,
             int pad_w, int pad_h, int wstride, int hstride,
             bool fixed_alg, cudnnConvolutionFwdAlgo_t fwd_alg, cudnnConvolutionBwdDataAlgo_t bwdd_alg, cudnnConvolutionBwdFilterAlgo_t bwdf_alg)
            :
            cudnn_handle_(),
            conv_desc_(pad_h, pad_w, hstride, wstride)
    {
        int out_h, out_w, out_c, out_n;

        cudnnTensorFormat_t format;
        // For int8 inference, the supported format is NHWC
        if (std::is_same<T1, uint8_t>::value) {
            format = CUDNN_TENSOR_NHWC;
        } else {
            format = CUDNN_TENSOR_NCHW;
        }
        //format = CUDNN_TENSOR_NHWC;

        CHECK_CUDNN_ERROR(cudnnSetConvolutionGroupCount(conv_desc_.desc(), groups));

        x_desc_ = TensorDescriptor4d<T1>(format, n, c, h, w);
        w_desc_ = FilterDescriptor4d<T1>(format, k, c/groups, r, s);



#if (USE_TENSOR_CORES)
        cudnnSetConvolutionMathType(conv_desc_.desc(), CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
        printf("TensorOP Mathtype set.\n");
#endif
        // Get output dimensions
        CHECK_CUDNN_ERROR(cudnnGetConvolution2dForwardOutputDim(conv_desc_.desc(),
                                                                x_desc_.desc(),
                                                                w_desc_.desc(),
                                                                &out_n,
                                                                &out_c,
                                                                &out_h,
                                                                &out_w));

        h_desc_ = TensorDescriptor4d<T1>(format, out_n, out_c, out_h, out_w);

        output_dims_ = {out_w, out_h, out_c, out_n};
        //printf("Outdims: (%i, %i, %i, %i)", out_n, out_c, out_h, out_w);
        printf("Inputdims: (%i, %i, %i, %i, %i, %i, %i, %i)", n, c, h, w, k, r, s, groups);

        // Pick forward convolution algorithm
        cudnnConvolutionFwdAlgoPerf_t fwd_perf;
        int ret_count;

        if (std::is_same<T1, uint8_t>::value) {
            //Note: cuDNN only supports IMPLICIT_PRECOMP_GEMM for int8 data type.
            fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        } else if (!fixed_alg){
            CHECK_CUDNN_ERROR(cudnnFindConvolutionForwardAlgorithm(cudnn_handle_.handle(),
                                                                   x_desc_.desc(),
                                                                   w_desc_.desc(),
                                                                   conv_desc_.desc(),
                                                                   h_desc_.desc(),
                                                                   1,
                                                                   &ret_count,
                                                                   &fwd_perf));
            fwd_algo_ = fwd_perf.algo;
        } else {
            fwd_algo_ = fwd_alg;
        }
#if (USE_TENSOR_CORES)
        // Tensor Op math only supports IMPLICIT_PRECOMP_GEMM algorithm
        fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
#endif
        if (std::is_same<T1, uint8_t>::value) {
            //Note: cudnn workspace size function doesn't work for INT8_CONFIG
            fwd_workspace_size_= 1073741824;
        } else {
            // Set fwd workspace size
            CHECK_CUDNN_ERROR(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle_.handle(),
                                                                      x_desc_.desc(),
                                                                      w_desc_.desc(),
                                                                      conv_desc_.desc(),
                                                                      h_desc_.desc(),
                                                                      fwd_algo_,
                                                                      &fwd_workspace_size_));
        }

        fwd_workspace_ = zeros<float>(std::vector<int>{static_cast<int>(fwd_workspace_size_ / sizeof(float)), 1});

            cudnnConvolutionBwdFilterAlgoPerf_t filter_perf;

            if (std::is_same<T1, uint8_t>::value) {

                fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

            }
            if(!fixed_alg) {
                CHECK_CUDNN_ERROR(cudnnFindConvolutionBackwardFilterAlgorithm(cudnn_handle_.handle(),
                                                                              x_desc_.desc(),
                                                                              h_desc_.desc(),
                                                                              conv_desc_.desc(),
                                                                              w_desc_.desc(),
                                                                              1,
                                                                              &ret_count,
                                                                              &filter_perf));
                bwd_params_algo_ = filter_perf.algo;
            } else {
                bwd_params_algo_ = bwdf_alg;
            }
#if (USE_TENSOR_CORES)
            // Tensor Op math only supports this algorithm.
            bwd_params_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
#endif

        // Backward params workspace
        CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle_.handle(),
                                                                         x_desc_.desc(),
                                                                         h_desc_.desc(),
                                                                         conv_desc_.desc(),
                                                                         w_desc_.desc(),
                                                                         bwd_params_algo_,
                                                                         &bwd_params_workspace_size_));



        bwd_params_workspace_ = zeros<float>(std::vector<int>{static_cast<int>(bwd_params_workspace_size_ / sizeof(float)), 1});

        cudnnConvolutionBwdDataAlgoPerf_t data_perf;
        if (!fixed_alg) {
            CHECK_CUDNN_ERROR(cudnnFindConvolutionBackwardDataAlgorithm(cudnn_handle_.handle(),
                                                                        w_desc_.desc(),
                                                                        h_desc_.desc(),
                                                                        conv_desc_.desc(),
                                                                        x_desc_.desc(),
                                                                        1,
                                                                        &ret_count,
                                                                        &data_perf));
            bwd_inputs_algo_ = data_perf.algo;
        } else {
            bwd_inputs_algo_ = bwdd_alg;
        }
#if (USE_TENSOR_CORES)
        //Tensor Op math only supports this algorithm.
        bwd_inputs_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
#endif

        CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle_.handle(),
                                                                       w_desc_.desc(),
                                                                       h_desc_.desc(),
                                                                       conv_desc_.desc(),
                                                                       x_desc_.desc(),
                                                                       bwd_inputs_algo_,
                                                                       &bwd_inputs_workspace_size_));

        bwd_inputs_workspace_ = zeros<float>(std::vector<int>{static_cast<int>(bwd_inputs_workspace_size_ / sizeof(float)), 1});


    }

    std::vector<int> get_output_dims() { return output_dims_; }

    std::string get_fwd_algo_string() {
        if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)
            return "IMPLICIT_GEMM";
        else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)
            return "IMPLICIT_PRECOMP_GEMM";
        else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_GEMM)
            return "GEMM";
        else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_DIRECT)
            return "DIRECT";
        else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_FFT)
            return "FFT";
        else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING)
            return "FFT_TILING";
        else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)
            return "WINOGRAD";
        else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED)
            return "WINOGRAD_NONFUSED";
        else {
            std::stringstream ss;
            ss << "Illegal algorithm passed to get_fwd_algo_string. Algo: " << fwd_algo_ << std::endl;
            throw std::runtime_error(ss.str());
        }
    }


    void forward(Tensor<T1> x, Tensor<T1> filter, Tensor<T1> h) {

        // Convolution forward.
        CHECK_CUDNN_ERROR(cudnnConvolutionForward(cudnn_handle_.handle(),
                                                  &alpha_,
                                                  x_desc_.desc(),
                                                  x.begin(),
                                                  w_desc_.desc(),
                                                  filter.begin(),
                                                  conv_desc_.desc(),
                                                  fwd_algo_,
                                                  fwd_workspace_.begin(),
                                                  fwd_workspace_size_,
                                                  &beta_,
                                                  h_desc_.desc(),
                                                  h.begin()));

    }

    void backward_params(Tensor<T1> x, Tensor<T1> delta, Tensor<T1> dW) {

        CHECK_CUDNN_ERROR(cudnnConvolutionBackwardFilter(cudnn_handle_.handle(),
                                                         &alpha_,
                                                         x_desc_.desc(),
                                                         x.begin(),
                                                         h_desc_.desc(),
                                                         delta.begin(),
                                                         conv_desc_.desc(),
                                                         bwd_params_algo_,
                                                         bwd_params_workspace_.begin(),
                                                         bwd_params_workspace_size_,
                                                         &beta_,
                                                         w_desc_.desc(),
                                                         dW.begin()));


    }

    void backward_inputs(Tensor<T1> filter, Tensor<T1> delta, Tensor<T1> dX) {

        CHECK_CUDNN_ERROR(cudnnConvolutionBackwardData(cudnn_handle_.handle(),
                                                       &alpha_,
                                                       w_desc_.desc(),
                                                       filter.begin(),
                                                       h_desc_.desc(),
                                                       delta.begin(),
                                                       conv_desc_.desc(),
                                                       bwd_inputs_algo_,
                                                       bwd_inputs_workspace_.begin(),
                                                       bwd_inputs_workspace_size_,
                                                       &beta_,
                                                       x_desc_.desc(),
                                                       dX.begin()));

    }
};

template <typename T1, typename T2>
std::tuple<float, float, float, std::string> time_cnn(
        int k, int c, int r, int s,
        int n, int h, int w, int groups,
        int pad_h, int pad_w,
        int hstride, int wstride,
        int num_repeats,
        curandGenerator_t curand_gen,
        bool fixed_alg,
        std::string alg
) {

    cudnnConvolutionFwdAlgo_t fwd_alg;
    cudnnConvolutionBwdDataAlgo_t bwdd_alg;
    cudnnConvolutionBwdFilterAlgo_t bwdf_alg;

    if (fixed_alg) {
        if (alg == "IMPLICIT_PRECOMP_GEMM") {
            fwd_alg = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
            bwdd_alg = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
            bwdf_alg = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3;
        }
        if (alg == "IMPLICIT_GEMM") {
            fwd_alg = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
            bwdd_alg = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
            bwdf_alg = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3;
        }
        if (alg == "GEMM") {
            fwd_alg = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
            bwdd_alg = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
            bwdf_alg = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3;
        }
        if (alg == "FFT") {
            fwd_alg = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
            bwdd_alg = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
            bwdf_alg = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3;
        }
        if (alg == "FFT_TILING") {
            fwd_alg = CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;
            bwdd_alg = CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING;
            bwdf_alg = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING;
        }
        if (alg == "WINOGRAD") {
            fwd_alg = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
            bwdd_alg = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD;
            bwdf_alg = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED;
        }
        if (alg == "WINOGRAD_NONFUSED") {
            fwd_alg = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
            bwdd_alg = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED;
            bwdf_alg = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED;
        }
    }

    cudnnCNN<T1, T2> cnn(w, h, c, n, k, r, s, groups, pad_w, pad_h, wstride, hstride, fixed_alg, fwd_alg, bwdd_alg, bwdf_alg);

    // Allocate memory for filter
    auto filter = rand<T1>(std::vector<int>{s, r, c, k}, curand_gen);

    // Allocate memory for input
    auto input = rand<T1>(std::vector<int>{w, h, c, n}, curand_gen);

    // Allocate memory for output tensor
    auto output = zeros<T1>(cnn.get_output_dims());


    std::string fwd_algo_s = cnn.get_fwd_algo_string();

    //Warm up
    cnn.forward(input, filter, output);

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();
    cudaProfilerStart();
    cudaEventRecord(start);

    for (int i = 0; i < num_repeats; ++i) {
        cnn.forward(input, filter, output);
    }
    cudaProfilerStop();
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    float fwd_time = milliseconds / (float)num_repeats;

    float bwd_inputs_time = 0;
    float bwd_params_time = 0;

    // Allocate memory for backward pass wrt weights
    auto delta = rand<T1>(cnn.get_output_dims(), curand_gen);
    auto dW = zeros<T1>(std::vector<int>{s, r, c, k});

    // Warm up backward
    cnn.backward_params(input, delta, dW);

    cudaDeviceSynchronize();
    cudaEventRecord(start);

    for (int i = 0; i < num_repeats; ++i) {
        // Backward pass wrt weights
        cnn.backward_params(input, delta, dW);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    bwd_params_time = milliseconds / (float)num_repeats;
    //Allocate memory for backward pass wrt inputs
    auto dX = zeros<T1>(std::vector<int>{w, h, c, n});

    //Warm up backward inputs
    cnn.backward_inputs(filter, delta, dX);

    cudaDeviceSynchronize();
    cudaEventRecord(start);

    for (int i = 0; i < num_repeats; ++i) {
        // Backward pass wrt weights
        cnn.backward_inputs(filter, delta, dX);

    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    bwd_inputs_time = milliseconds / (float)num_repeats;

    return std::tuple<float, float, float, std::string>(fwd_time, bwd_inputs_time, bwd_params_time, fwd_algo_s);

}

int main(int argc, char **argv) {

    int num_repeats = 300;

    //Set for specific algorithms.
    bool fixed_alg = false;
    std::string alg;

    std::string precision;
    //TODO: Set Precision
    precision = "float";

    // Handles to various cuda libraries, structures
    curandGenerator_t curand_gen;

    cudaFree(0);

    // Initialize curand_gen and set appropriate seed.
    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 123ULL);

    if(argc == 1){
        std::cout << "No inputfile specified." << std::endl;
        return 1;
    }
    std::string input_filepath = argv[1];
    std::cout << "Inputfilepath: " << input_filepath << std::endl;
    std::ifstream fin(input_filepath, std::ifstream::in);  //ifstream to read from
    if (!fin.is_open())  {  std::cerr << "failed to open file\n"; return 1; }

    std::vector<conv_problem> problems;
    std::string linestr;
    int pad2, pad3;
    while (std::getline(fin, linestr) && !linestr.empty())
    {
        std::cout << linestr << std::endl;
        std::stringstream astream(linestr);
        conv_problem prob;

        //Accepted order: ID, N, C, H, W, K, fh, fw, groups, pad_h, pad_w, stride_h, stride_w
        std::vector<int*> prob_struct = {
                &prob.id,
                &prob.minibatch, &prob.ic, &prob.h, &prob.w,
                &prob.oc, &prob.groups, &prob.fh, &prob.fw,
                &prob.pad_h, &prob.pad_w, &pad2, &pad3,
                &prob.stride_h, &prob.stride_w};
        std::string tmp;
        for(int *value : prob_struct){

            std::getline(astream, tmp, ',');
            *value = std::stoi(tmp);
        }
        problems.push_back(prob);
    }

    std::string output_filepath = input_filepath.substr(0, input_filepath.size() - 4) + "_out.csv";
    std::cout << "Outputfilepath: " << output_filepath << std::endl;
    std::ofstream fout(output_filepath, std::ofstream::out);  //ifstream to write to
    if (!fout.is_open())  {  std::cerr << "failed to open file\n"; return 1; }

    fout << "id,ms" << std::endl;

    int pad_kernels_count = 0;

    int p_idx = 0;
    for (conv_problem p : problems) {

        bool skip_kernel = false;
        bool need_padding = false;

        int padded_c, padded_w, padded_h;
        int pad_value;
        int groups;

        padded_c = p.ic;
        padded_h = p.h;
        padded_w = p.w;
        groups = p.groups;

        if (precision == "int8") {
            pad_value = 4;
            if (p.ic % pad_value || p.w % pad_value || p.h % pad_value) {
                pad_kernels_count++;
                if (PAD_KERNELS) {
                    pad_dim(padded_c, pad_value);
                    pad_dim(padded_h, pad_value);
                    pad_dim(padded_w, pad_value);
                    need_padding = true;
                } else {
                    skip_kernel = true;
                }
            }
        }
#if (USE_TENSOR_CORES)
        // Tensor cores need channels to be a multiple of 8. So, added padding for some kernels.
        pad_value = 8;
        if (p.ic % pad_value) {
            pad_kernels_count++;
            if (PAD_KERNELS) {
                pad_dim(padded_c, pad_value);
                need_padding = true;
            } else {
                skip_kernel = true;
            }
        }
#endif

        float fwd_time, bwd_inputs_time, bwd_params_time;
        std::string fwd_algo_s;

        std::stringstream ss;
        ss << "Unsupported precision requested. Precision: " << precision;


//        int k, int c, int r, int s, int n, int h, int w, int groups, int pad_h, int pad_w, int hstride, int wstride, int num_repeats,

        try {


            if (precision == "float") {
                std::tie(fwd_time, bwd_inputs_time, bwd_params_time, fwd_algo_s) =
                        time_cnn<float, float>(p.oc, padded_c, p.fh, p.fw, p.minibatch, padded_h, padded_w, groups,
                                               p.pad_h, p.pad_w, p.stride_h, p.stride_w, num_repeats, curand_gen,
                                               fixed_alg, alg);
            } else if (precision == "half") {
                std::tie(fwd_time, bwd_inputs_time, bwd_params_time, fwd_algo_s) =
                        time_cnn<uint16_t, uint16_t>(p.oc, padded_c, p.fh, p.fw, p.minibatch, padded_h, padded_w,
                                                     groups, p.pad_h, p.pad_w, p.stride_h, p.stride_w, num_repeats,
                                                     curand_gen, fixed_alg, alg);
            } else if ((precision == "int8")) {
                std::tie(fwd_time, bwd_inputs_time, bwd_params_time, fwd_algo_s) =
                        time_cnn<uint8_t, int>(p.oc, padded_c, p.fh, p.fw, p.minibatch, padded_h, padded_w, groups,
                                               p.pad_h, p.pad_w, p.stride_h, p.stride_w, num_repeats, curand_gen,
                                               fixed_alg, alg);
            } else {
                throw std::runtime_error(ss.str());
            }

            if (skip_kernel) {
                std::cout << "Not Supported";
            } else {
                std::cout << (fwd_time) << ",";
            }

        } catch (const std::exception& e) {
            printf("Skipped: (%i, %i, %i, %i, %i, %i, %i, %i) err: (%s)", p.minibatch, padded_c, padded_h, padded_w, p.oc, p.fh, p.fw, groups, e.what());
            bwd_inputs_time = 0;
            bwd_params_time = 0;
            fwd_time = 0;
        }

        std::cout << std::setw(24) << (bwd_inputs_time) << ",";
        std::cout << std::setw(24) << (bwd_params_time) << ",";
        std::cout << std::setw(19) << (fwd_time + bwd_inputs_time + bwd_params_time) << ",";

        fout << p_idx << "," << (fwd_time + bwd_inputs_time + bwd_params_time) << std::endl;

        //printf("Times: %f, %f, %f\n", fwd_time, bwd_inputs_time, bwd_params_time);

        if (USE_TENSOR_CORES && PAD_KERNELS) {
            std::cout << std::setw(15) <<  need_padding;
        }


        std::cout << std::setw(25) << fwd_algo_s << ",";
        std::cout << std::endl;
        p_idx ++;
    }

    fout.close();

    if (precision == "int8") {
        std::cout << " Total kernels ";
        if (PAD_KERNELS)
            std::cout << "padded: " << pad_kernels_count << std::endl;
        else
            std::cout << "skipped: " << pad_kernels_count << std::endl;
    }

    // Destroy all the handles
    curandDestroyGenerator(curand_gen);
    return 0;

}
