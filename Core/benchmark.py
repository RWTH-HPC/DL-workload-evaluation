from BenchmarkDatabase.database import Database, Target, runtime_filepath
from BenchmarkDatabase.elements import Time, Operator, Operation
from OnnxUtils import extraction
import Runtimetools
from Runtimetools.runtime import Runtime, TrainingBenchmark
from BenchmarkDatabase.generator import op_benchmark_generator
import argparse
import os
import logging

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        help="Path to an onnx file that should be processed.",
                        default="../models/")
    args = parser.parse_args()

    with Database() as db:



        #--batch_size 128 --nhid=512 --nhead=8 --emsize
        #bs, hiddensize, heads, emsize
        # configurations = [[64, 512, 8, 200],[128, 512, 8, 200],[256, 512, 8, 200],[512, 512, 8, 200],
        #                   [128, 128, 8, 200],[128, 256, 8, 200],[128, 512, 8, 200],[128, 1024, 8, 200],
        #                   [128, 512, 4, 200],[128, 512, 8, 200],[128, 512, 20, 200],[128, 512, 40, 200],
        #                   [128, 512, 10, 40],[128, 512, 10, 100],[128, 512, 10, 200],[128, 512, 10, 400]]

        # for config in configurations:
        #     Runtimetools.benchmark_list.append(TrainingBenchmark(name="Ex_lm_Transformer{}_{}_{}_{}".format(
        #         config[0], config[1], config[2], config[3]),
        #               exe="/home/sw693484/pytorch-examples/word_language_model/main.py",
        #               parameters=(" --cuda --epochs 1 --model Transformer --batch_size {} --nhid={} --nhead={} --emsize={}" +
        #                          " --data /rwthfs/rz/cluster/home/sw693484/pytorch-examples/word_language_model/data/wikitext-2").format(
        #                              config[0], config[1], config[2], config[3])),)

        #rt = Runtime("CPU_x48_Platinum_8160")
        rt = Runtime("GPU_volta_v100")
        rt.generate_onnx(rt.training_benchmarks())

        # Dict of operations per extracted Network
        network_ops = {}
        # Extract operations + operators from the network
        #onnx_paths = [os.path.join(args.path, x) for x in os.listdir(args.path) if x.endswith(".onnx")]
        onnx_paths = [os.path.join(runtime_filepath, x) for x in os.listdir(runtime_filepath) if x.endswith(".onnx")]
        for file in onnx_paths:
            operations = extraction.extract_operations(os.path.join(args.path, file), 128)
            print("Added {} operations extracted from {}.".format(len(operations), file))
            # for op in operations:
            #     print(op)
                # if(op.operator == Operator("MatMul", ("M", "N", "K"))):
                #     print(op)
            db.add_operations(operations, Target.OP)
            network_ops[os.path.splitext(os.path.basename(file))[0]] = Database.op_list_to_dict(operations)
            #for op in operations:
            #    print("Op measurement: " + str(db.get_measurement(op)))
        # return

        #db.add_operations(op_benchmark_generator(Operator("MatMul", [["M", "N"],["N", "K"]]), [33278, 4480, 512]), Target.OP)

        # Run operator benchmark
        operations = db.get_operations_by_type(Target.OP)

        print("Runtimetools running.")
        measurements = rt.run_all_operator_benchmarks(operations)
        db.add_measurements(measurements, Target.OP)

        print("Training Benchmarks.")

        for benchmark in rt.training_benchmarks():
            measurements = rt.run_training(network_ops[benchmark.name], benchmark)
            db.add_measurements(measurements, Target.TRAINING)

        ''' TODO:
        Measure base model performance. Estimate base model performance
        measure and predict for all other model variants. Fill in prediction gaps with base model.
        '''
        #try:
        for net, netops in network_ops.items():
            print("Results of {}:".format(net))
            db.log_op_results(netops)
        #except Exception:
        #    logging.exception("logging crashed")
	
#        for net, netops in network_ops.items():
#            print("Results of {}:".format(net))
#            db.log_op_results(netops)

        #db.log_op_and_training_differences()

if __name__ == '__main__':
    main()


