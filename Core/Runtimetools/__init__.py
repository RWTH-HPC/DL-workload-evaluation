from Runtimetools.runtime import Runtime, Benchmark, OperatorBenchmark, TrainingBenchmark

# Register Benchmarks here:
benchmark_list = [
    #TrainingBenchmark(name="Ex_lm_Transformer2",
    #                      exe="/home/sw693484/pytorch-examples/word_language_model/main.py",
    #                      parameters=" --cuda --epochs 1 --model Transformer --batch_size 128 --nhid=512 --data /rwthfs/rz/cluster/home/sw693484/pytorch-examples/word_language_model/data/wikitext-2"),

    OperatorBenchmark(name="GPU_GEMM",
                      synonyms=["MatMul"],
                      exe="OperatorBenchmarks/build/nvidia/gemm/cudnn_gemm",
                      param_descr=["M","N","K","a_t","b_t"],
                      permutation=["ab,bc -> abc00"]),

    TrainingBenchmark(name="Resnet50",
                            exe="/home/sw693484/pytorch-examples/imagenet/main_gpu.py",
                            parameters="--epochs 1 -b 128 -a resnet50 /hpcwork/sw693484/Imagenet"),

    TrainingBenchmark(name="Resnext50",
                            exe="/home/sw693484/pytorch-examples/imagenet/main_gpu.py",
                            parameters="--epochs 1 -b 128 -a resnext50_32x4d /hpcwork/sw693484/Imagenet"),

    # OperatorBenchmark(name="CPU_Convolution2D",
    #           synonyms=["Conv"],
    #           exe="OperatorBenchmarks/build/intel/convolution/intel_conv",
    #           param_descr=["N", "C", "H", "W", "K", "kernel_shape_0", "kernel_shape_1", "pads_0", "pads_1", "strides_0", "strides_1"]
    #           ),

    # OperatorBenchmark(name="GPU_Convolution2D",
    #           synonyms=["Conv"],
    #           exe="OperatorBenchmarks/build/nvidia/convolution/cudnn_conv",
    #           param_descr=["N", "C", "H", "W", "K", "kernel_shape_0", "kernel_shape_1", "pads_0", "pads_1", "strides_0", "strides_1"]
    #           ),

    OperatorBenchmark(name="GPU_Convolution2D",
                      synonyms=["Conv"],
                      exe="OperatorBenchmarks/build/nvidia/convolution/cudnn_conv",
                      param_descr=["N","C","H","W","K", "group", "kernel_shape_0", "kernel_shape_1", "pads_0", "pads_1", "pads_2", "pads_3", "strides_0", "strides_1"])

    #TrainingBenchmark(name="Resnet50_cifar10",
    #                  exe="TrainingBenchmarks/Resnet50_cifar10.py",
    #                  parameters=" --batch-size 128 --epochs 1",
    #                  ),

    #TrainingBenchmark(name="Resnet50_v2_Imagenet",
    #                      exe="TrainingBenchmarks/Resnet50_imagenet.py",
    #                      parameters=" -b 128 --gpu 0 --epochs 1 -a resnet50 /hpcwork/sw693484/Imagenet"
    #                      ),

    #TrainingBenchmark(name="Transformer_gpt2",
    #                      exe="/home/sw693484/transformers/examples/language-modeling/run_language_modeling.py",
    #                      parameters=" --output_dir=/home/sw693484/transformers/examples/language-modeling/output " +
    #                                 "--model_type=gpt2 --model_name_or_path=gpt2 --do_train " +
    #                                 "--train_data_file=/home/sw693484/transformers/data/wikitext-2-raw/wiki.train.raw " +
    #                                 "--per_gpu_train_batch_size=2 --overwrite_output_dir")

    #TrainingBenchmark(name="Ex_lm_RNN_TANH",
    #                  exe="/home/sw693484/pytorch-examples/word_language_model/main.py",
    #                  parameters=" --cuda --epochs 1 --model RNN_TANH --batch_size 128 --nhid=512"),

    #TrainingBenchmark(name="Ex_lm_LSTM",
    #                      exe="/home/sw693484/pytorch-examples/word_language_model/main.py",
    #                      parameters=" --cuda --epochs 1 --model LSTM --batch_size 128 --nhid=512 --data /rwthfs/rz/cluster/home/sw693484/pytorch-examples/word_language_model/data/wikitext-2"),

    #TrainingBenchmark(name="Ex_lm_LSTM3",
    #                      exe="/home/sw693484/pytorch-examples/word_language_model/main.py",
    #                      parameters=" --cuda --epochs 1 --model LSTM --batch_size 128 --nlayers=3 --nhid=512 --data /rwthfs/rz/cluster/home/sw693484/pytorch-examples/word_language_model/data/wikitext-2")


]

pytorch_op_name_variants = {
    "Reshape" : ["view"],
    "Flatten" : ["squeeze"],
    "LSTM" : ["rnn"],
    "MatMul" : ["mm", "bmm"]
}

pytorch_cpu_minor_ops = [
    "aten::empty",
    "aten::as_strided",
    "aten::as_strided_"
]

pytorch_op_parameter_order = {
    #"mkldnn_convolution" : ["N", "C", "H", "W", "K", "pads_0", "kernel_shape_0", "kernel_shape_1"],
    #"mkldnn_convolution_backward" : ["N", "C", "H", "W", "N", "K", "out_0", "out_1", "K", "C", "kernel_shape_0", "kernel_shape_1"]
}

# Some operations use the same operation in the backend for forward, backward etc.
# MatrixMultiplication for example is unfolded into mm forward, mm backward weights, mm backwards bias
# Only the order of the input dimensions changes but the operation is the same
# The result: One Onnx Operator turns into multiple variants
pytorch_op_unfolding = {
    "MatMul" : ["abc,de -> ea*b,a*bc",
                "abc,de -> a*bc,ce",
                "abc,de -> a*be,ec",
                "abc,acd -> abc,acd",
                "abc,acd -> abd,adc",
                "abc,acd -> adb,abc"
                ],
    # "Conv" : ["nchwkergfjpoiusa -> nchw,nch/sw/a,ncfj",
    #           "nchwkergfjpoiusa -> nchw,nch/sw/a,ncfj"]


#[['? - n',
#  '? - c',
#  '? - h',
#  '? - w',
#  'K - k',
#  'dilations_0 - e',
#  'dilations_1 - r',
#  'group - g',
#  'kernel_shape_0 - f',
#  'kernel_shape_1 - j',
#  'pads_0 - p',
#  'pads_1 - o',
#  'pads_2 - i',
#  'pads_3 - u',
#  'strides_0 - s',
#  'strides_1 - a'
#    ]
}
