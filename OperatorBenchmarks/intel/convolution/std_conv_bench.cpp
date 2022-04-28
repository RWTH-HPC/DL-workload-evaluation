/*******************************************************************************
* Copyright 2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* Based on https://github.com/baidu-research/DeepBench/blob/master/code/intel/convolution/mkl_conv/std_conv_bench.cpp
*******************************************************************************/

#include <stdio.h>
#include <float.h>
#include <time.h>
#include <assert.h>

#include <stdexcept>
#include <tuple>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#include "omp.h"

struct conv_problem {
    int minibatch;          //Number of minibatches
    int w;                  //Width of input
    int h;                  //Height of input
    int ic;
    int oc;
    int fw;                 //Filter/Kernel width
    int fh;                 //Filter/Kernel height
    int stride_w, stride_h; //Strides
    int pad_w, pad_h;       //Padding
    int iters;              //Benchmark repetitions
    int id;
};

#define FWD_CONVOLUTION   0
#define BWD_F_CONVOLUTION 1
#define BWD_D_CONVOLUTION 2

#define PREC_F32 0
#define PREC_U8S8U8 1

#define TRAINING 0
#define INFERENCE_SERVER 1
#define INFERENCE_DEVICE 2

#define ITERS 1000

// Calculates convolution output dimension using the definition from Caffe
static inline int calc_out_dim(
        int input_dim, int filter_dim, int padd, int stride)
{
    return (input_dim - filter_dim + 2 * padd) / stride + 1;
}

struct bench_result {
    double min_ms;
    double avg_ms;
};

// Returns milliseconds since the start of the epoch
static inline double ms_timer()
{
    struct timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv);
    return (1000000000ll * tv.tv_sec + tv.tv_nsec) / 1e6;
}

// Benchmarking loop
template <typename Func>
static inline bench_result timeit(int niters, Func func)
{
    const double max_ms_total = 3E3; // max milliseconds per problem
    func(); // Warmup
    bench_result result = {DBL_MAX, 0};
    int iters_done = 0;
    for (; iters_done < niters && result.avg_ms < max_ms_total; iters_done++) {
        double ms = ms_timer();
        func();
        ms = ms_timer() - ms;
        result.avg_ms += ms;
        result.min_ms = std::min(result.min_ms, ms);
    }
    result.avg_ms /= iters_done;
    return result;
}

template <typename T>
static inline void rand_fill(T *data, size_t size)
{
    static bool initialized = false;
    if (!initialized) {
        srand48(1);
        initialized = true;
    }
#pragma omp parallel for
    for (size_t i = 0; i < size / sizeof(T); i++)
        data[i] = static_cast<T>(drand48());
}

#define COMPUTE_BWD_BIAS 1

#include "mkldnn.hpp"

#define CHECK(f) do { \
    mkldnn_status_t s = f; \
    if (s != mkldnn_success) { \
        printf("[%s:%d] error: %s returns %d\n", __FILE__, __LINE__, #f, s); \
        exit(2); \
    } \
} while(0)

using namespace mkldnn;

static bench_result bench_conv(conv_problem prob, int mode, int precision)
{
    engine eng(engine::kind::cpu, 0);

    int groups = 1;

    // std::cout << "(" << prob.minibatch << "," << prob.ic << "," << prob.h << "," << prob.w << "," << prob.oc << ","
    //         << prob.fh << "," << prob.fw << "," << prob.pad_h << "," << prob.pad_w << "," << prob.stride_h << "," << prob.stride_w << ")" << std::endl;

    memory::data_type src_dt, dst_dt, filter_dt, bias_dt;
    switch (precision) {
    case PREC_U8S8U8:
        src_dt = memory::data_type::u8;
        dst_dt = memory::data_type::u8;
        filter_dt = memory::data_type::s8;
        bias_dt = memory::data_type::s32;
        break;
    default:
        src_dt = dst_dt = filter_dt = bias_dt = memory::data_type::f32;
    }

    //Source memory descriptor NCHW
    memory::desc src_d({prob.minibatch, prob.ic, prob.h, prob.w},
            src_dt, memory::format_tag::any);
    //Dest memory descriptor NCHW
    memory::desc dst_d({prob.minibatch, prob.oc,
            calc_out_dim(prob.h, prob.fh, prob.pad_h, prob.stride_h),
            calc_out_dim(prob.w, prob.fw, prob.pad_w, prob.stride_w)},
            dst_dt, memory::format_tag::any);

    /*printf("In Dim: (%i, %i, %i, %i), Out Dim: (%i, %i, %i, %i)",
           src_d.data.dims[0], src_d.data.dims[1], src_d.data.dims[2], src_d.data.dims[3],
           dst_d.data.dims[0], dst_d.data.dims[1], dst_d.data.dims[2], dst_d.data.dims[3]);*/

    memory::dims fsizes = {prob.oc / groups, prob.ic / groups, prob.fh, prob.fw};

    if (groups != 1) {
        fsizes.insert(fsizes.begin(), groups);
    }

    memory::desc filter_d(fsizes, filter_dt, memory::format_tag::any);
    memory::desc bias_d({prob.oc}, bias_dt, memory::format_tag::any);
    memory::dims strides = {prob.stride_h, prob.stride_w};
    memory::dims padding = {prob.pad_h, prob.pad_w};

    std::shared_ptr<primitive> conv;
    std::shared_ptr<memory> src;    // Source Data
    std::shared_ptr<memory> dst;    // Destination Data
    std::shared_ptr<memory> filter; // Weights Data
    std::shared_ptr<memory> bias;   // Bias Data

    try {
        //Create the Descriptor for the Convolution
        convolution_forward::desc fwd_conv_d = {prop_kind::forward_training, algorithm::convolution_auto,
            src_d, filter_d, bias_d, dst_d,
            strides, padding, padding, padding_kind::zero};

        //derive the primitive descriptor
        auto fwd_conv_pd = convolution_forward::primitive_desc(fwd_conv_d, eng);
        std::unordered_map< int, memory > conv_arguments;

        if (mode == FWD_CONVOLUTION) {
            src.reset(new memory(fwd_conv_pd.src_desc(), eng));
            dst.reset(new memory(fwd_conv_pd.dst_desc(), eng));
            filter.reset(new memory(fwd_conv_pd.weights_desc(), eng));
            bias.reset(new memory(fwd_conv_pd.bias_desc(), eng));
            conv.reset(new convolution_forward(fwd_conv_pd));
            conv_arguments = { { MKLDNN_ARG_SRC, (*src) },
                { MKLDNN_ARG_WEIGHTS, (*filter) },
                { MKLDNN_ARG_BIAS, (*bias) },
                { MKLDNN_ARG_DST, (*dst) } };
        } else if (mode == BWD_D_CONVOLUTION) {
            auto bwd_d_conv_pd = convolution_backward_data::primitive_desc(
                    {algorithm::convolution_auto, src_d, filter_d, dst_d,
                    strides, padding, padding, padding_kind::zero}, eng,
                    fwd_conv_pd);
            src.reset(new memory(bwd_d_conv_pd.diff_src_desc(), eng));
            dst.reset(new memory(bwd_d_conv_pd.diff_dst_desc(), eng));
            filter.reset(new memory(bwd_d_conv_pd.weights_desc(), eng));
//            bias.reset(new memory(bwd_d_conv_pd.diff_bias_desc(), eng));
            conv.reset(new convolution_backward_data(bwd_d_conv_pd));
            conv_arguments = { {MKLDNN_ARG_DIFF_SRC, (*src)}, {MKLDNN_ARG_DIFF_DST, (*dst)}, {MKLDNN_ARG_WEIGHTS, (*filter)} };
        } else if (mode == BWD_F_CONVOLUTION) {
            auto bwd_f_conv_pd = convolution_backward_weights::primitive_desc(
                    {algorithm::convolution_auto, src_d, filter_d,
                    bias_d,
                    dst_d,
                    strides, padding, padding, padding_kind::zero}, eng,
                    fwd_conv_pd);
            src.reset(new memory(bwd_f_conv_pd.src_desc(), eng));
            dst.reset(new memory(bwd_f_conv_pd.diff_dst_desc(), eng));
            filter.reset(new memory(bwd_f_conv_pd.diff_weights_desc(), eng));
            bias.reset(new memory(bwd_f_conv_pd.diff_bias_desc(), eng));
            conv.reset(new convolution_backward_weights(bwd_f_conv_pd));
            conv_arguments = { {MKLDNN_ARG_SRC, (*src)}, {MKLDNN_ARG_DIFF_DST, (*dst)}, {MKLDNN_ARG_DIFF_BIAS, (*bias)}, {MKLDNN_ARG_DIFF_WEIGHTS, (*filter)}};
        } else
            throw std::runtime_error("Invalid benchmarking mode");

        for (const auto &m : {src, dst, filter, bias}) {
            if (!m.get() || !m->get())
                continue;
            void *data = m->get_data_handle();
            auto pd = m->get_desc();
            size_t size = pd.get_size();
            switch (pd.data.data_type) {
            case memory::data_type::f32:
                rand_fill(static_cast<float *>(data), size);
                break;
            case memory::data_type::u8:
                rand_fill(static_cast<uint8_t *>(data), size);
                break;
            case memory::data_type::s8:
                rand_fill(static_cast<int8_t *>(data), size);
                break;
            case memory::data_type::s32:
                rand_fill(static_cast<int32_t *>(data), size);
                break;
            default:
                assert(!"Unsupported data type!\n");
            }
        }

        stream s(eng);

        return timeit(prob.iters,
                [&](){
                    (*conv).execute(s, conv_arguments);
                });
    }
    catch (mkldnn::error err) {
        printf("Error: %s\n", err.message);
        return (bench_result){0.0,0.0};
    }
}

int main(int argc, char *argv[])
{
    if(argc == 1){
        std::cout << "No inputfile specified." << std::endl;
        return 1;
    }
    std::string input_filepath = argv[1];
    std::cout << "Inputfilepath: " << input_filepath << std::endl;
    std::ifstream fin(input_filepath, std::ifstream::in);  //ifstream to read from
    if (!fin.is_open())  {  std::cerr << "failed to open file\n"; return 1; }

    std::vector<int> modes
            = {FWD_CONVOLUTION, BWD_F_CONVOLUTION, BWD_D_CONVOLUTION};

    std::vector<conv_problem> problems;
    std::string linestr;
    while (std::getline(fin, linestr) && !linestr.empty())
    {
        std::cout << linestr << std::endl;
        std::stringstream astream(linestr);
        conv_problem prob;

        //Accepted order: ID, N, C, H, W, K, fh, fw, pad_h, pad_w, stride_h, stride_w
        std::vector<int*> prob_struct = {
                &prob.id,
                &prob.minibatch, &prob.ic, &prob.h, &prob.w,
                &prob.oc, &prob.fh, &prob.fw,
                &prob.pad_h, &prob.pad_w,
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
    for (conv_problem p : problems) {
        p.iters = ITERS;
        bench_result acc_result = {0, 0};
        for(auto m : modes) {
            bench_result r = bench_conv(p, m, PREC_F32);
            acc_result.min_ms += r.min_ms;
            acc_result.avg_ms += r.avg_ms;
        }
        fout << p.id << "," << acc_result.avg_ms << std::endl;
        std::cout << "Result: " << acc_result.avg_ms << "ms" << std::endl;
    }
    fout.close();
}
