from BenchmarkDatabase.elements import Operator

default_operators = [
    Operator("Conv", ["N", "C", "H", "W", "K",
                      "dilations_0", "dilations_1",
                      "group",
                      "kernel_shape_0", "kernel_shape_1",
                      "pads_0", "pads_1", "pads_2", "pads_3",
                      "strides_0", "strides_1"
                      ]),
    Operator("Relu", ["N", "C", "H", "W"]),
    Operator("BatchNormalization", ["N", "C", "H", "W",
                                    "epsilon", "momentum", "spatial"
                                    ]),
    Operator("GlobalAveragePool", ["N", "C", "H", "W"]),
    Operator("MaxPool", ["N", "C", "H", "W",
                         "kernel_shape_0", "kernel_shape_1",
                         "pads_0", "pads_1", "pads_2", "pads_3",
                         "strides_0", "strides_1"
                         ]),
    Operator("Gemm", ["A", "B", "alpha", "beta", "transA", "transB"])
]


