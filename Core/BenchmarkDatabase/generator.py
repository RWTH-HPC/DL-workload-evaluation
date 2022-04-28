from BenchmarkDatabase.elements import Operator, Operation, Measurement, Target, Time

def op_benchmark_generator(operator : Operator, initial_vals):
    # TODO: Generate sensible data automaticly
    #op_benchmark_generator(Operator("MatMul", [["M", "N"],["N", "K"]]), [33278, 4480, 512]))
    return [
        Operation(operator, [[16639, 2240], [2240, 256]]),
        Operation(operator, [[33278, 2240], [2240, 256]]),
        Operation(operator, [[49917, 2240], [2240, 256]]),

        Operation(operator, [[16639, 4480], [4480, 256]]),
        Operation(operator, [[33278, 4480], [4480, 256]]),
        Operation(operator, [[49917, 4480], [4480, 256]]),

        Operation(operator, [[16639, 6720], [6720, 256]]),
        Operation(operator, [[33278, 6720], [6720, 256]]),
        Operation(operator, [[49917, 6720], [6720, 256]]),

        Operation(operator, [[16639, 2240], [2240, 512]]),
        Operation(operator, [[33278, 2240], [2240, 512]]),
        Operation(operator, [[49917, 2240], [2240, 512]]),

        Operation(operator, [[16639, 4480], [4480, 512]]),
        Operation(operator, [[33278, 4480], [4480, 512]]),
        Operation(operator, [[49917, 4480], [4480, 512]]),

        Operation(operator, [[16639, 6720], [6720, 512]]),
        Operation(operator, [[33278, 6720], [6720, 512]]),
        Operation(operator, [[49917, 6720], [6720, 512]]),

        Operation(operator, [[16639, 2240], [2240, 768]]),
        Operation(operator, [[33278, 2240], [2240, 768]]),
        Operation(operator, [[49917, 2240], [2240, 768]]),

        Operation(operator, [[16639, 4480], [4480, 768]]),
        Operation(operator, [[33278, 4480], [4480, 768]]),
        Operation(operator, [[49917, 4480], [4480, 768]]),

        Operation(operator, [[16639, 6720], [6720, 768]]),
        Operation(operator, [[33278, 6720], [6720, 768]]),
        Operation(operator, [[49917, 6720], [6720, 768]])
    ]