import os
import csv
import ast
import torch
import Runtimetools
import BenchmarkDatabase
from BenchmarkDatabase.elements import Operation, Operator, Measurement, Time
from typing import List
import subprocess
import re
from abc import ABC, abstractmethod
import numpy as np
from itertools import combinations, chain
from modulefinder import ModuleFinder
from OnnxUtils.run_summary import summarise_training_run

class Runtime(object):
    """
    The Runtime collects information about the system, the available benchmarks and is
    also the base class to run Benchmarks from.
    Attributes:
        active_system (str): Indicator for the current system. TODO: Maybe extend?
        available_benchmarks (Benchmark): List of all benchmarks that were found
    """
    def __init__(self, system):
        self.active_system = system
        self.available_benchmarks = self.detect_available_benchmarks()


    def detect_available_benchmarks(self):
        """
        Try to find all executables and build a list of benchmarks with associated executables.
        This uses the Runtimetools.benchmark_list as basis. To add new benchmarks this list needs to be extended.
        :return: List of available benchmarks
        """
        print("Runtime: Detecting available benchmarks.")
        av_benchmarks = []
        for benchmark in Runtimetools.benchmark_list:
            if benchmark.available():
                av_benchmarks.append(benchmark)
                print("Found {}".format(benchmark))
        return av_benchmarks

    def run_all_operator_benchmarks(self, operations) -> List[Measurement]:
        """
        Benchmark all operations that can be associated with an available Benchmark
        :param operations: The operations that should be measured
        """
        measurements = []
        for benchmark in self.available_benchmarks:
            if isinstance(benchmark, OperatorBenchmark):
                print("Running benchmark {} with synonyms: {}.".format(benchmark.name, benchmark.synonyms))
                for synonym in benchmark.synonyms:
                    if synonym in operations:
                        print("{} operations with this synonym.".format(len(operations[synonym])))
                        for dop in operations[synonym]:
                            print(dop)
                        measurements += benchmark.run(operations[synonym], self.active_system)
        return measurements

    def run_training(self, operations, benchmark) -> List[Measurement]:
        measurements = []
        print("Running training benchmark {}.".format(benchmark.name))
        measurements += benchmark.run(operations, self.active_system)
        return measurements

    def training_benchmarks(self):
        return [x for x in self.available_benchmarks if isinstance(x, TrainingBenchmark)]

    def operator_benchmarks(self):
        return [x for x in self.available_benchmarks if isinstance(x, OperatorBenchmark)]

    def generate_onnx(self, trainings):
        for training in trainings:
            if isinstance(training, TrainingBenchmark):
                training.generate_onnx()


class Benchmark(ABC):
    @abstractmethod
    def run(self, operations:List[Operation], system: str) -> List[Measurement]:
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def available(self) -> bool:
        pass

class OperatorBenchmark(Benchmark):
    """
    Each benchmark has a reference to an executable and a descriptor for the accepted parameters.
    :param name: Descriptive name for the benchmark
    :param synonyms: Synonyms contain the name and additional strings that are used to match
    onnx-extracted operators and the benchmark
    :param param_descriptors: A list of parameter descriptions that indicate what values are
    accepted by the benchmark executable. If the parameters do not match the order of parameters governs the execution.
    """
    def __init__(self, name, synonyms, exe, param_descr, permutation=None):
        self.name = name
        self.synonyms = synonyms + [name]
        if (os.path.isabs(exe)):
            self.executable_path = exe
        else:
            self.executable_path = BenchmarkDatabase.root_path + exe
        self.param_descriptors = param_descr
        self.permutation = permutation

    def __str__(self):
        return "Benchmark {}: {} [{}]".format(self.name, self.executable_path, self.param_descriptors)

    def available(self):
        return os.path.isfile(self.executable_path)


    def run(self, operations:List[Operation], system: str) -> List[Measurement]:
        """
        Run the benchmarks on the input list of operations
        :param operations: The operations that should be executed. The operators of the operations
        should match with a synonym or the benchmark name
        """

        # Write an inputfile into the runtimefiles directory
        measurements = []
        operator_assignment = {}
        filepath = BenchmarkDatabase.runtime_filepath + self.name + "_test_ops.csv"
        csv = open(filepath, "w")
        for idx, op in enumerate(operations):
            if op.operator.name in self.synonyms:
                csv.write("{}, ".format(idx))
                operator = op.operator

                if self.permutation is None:
                    if operator.name not in operator_assignment:
                        # Try to rewrite the operation operator to align with the benchmark params
                        operator_assignment[operator.name] = [None] * len(self.param_descriptors)
                        for a_idx, param_desc in enumerate(operator.parameter_descriptors):
                            for b_idx, b_param in enumerate(self.param_descriptors):
                                if b_param == param_desc:
                                    operator_assignment[operator.name][b_idx] = a_idx
                                    break
                        for ass in operator_assignment[operator.name]:
                            if ass is None:
                                print("Could not align by parameter descriptors. Trying simple order. " +
                                      "To prevent this make sure the benchmark param_descriptors match the " +
                                      "database operator descriptors.")
                                # Assignment of parameters failed. Not all descriptors have a match
                                # Assign by order
                                if len(self.param_descriptors) > len(operator.parameter_descriptors):
                                    print("Can't run operation: {}\nNot enough parameters!".format(op))
                                else:
                                    operator_assignment[operator.name] = range(0,len(self.param_descriptors))
                                    break

                    assignment = operator_assignment[operator.name]
                    print("{} -> {}".format(assignment, op.parameters))
                    for j, ass in enumerate(assignment):
                        csv.write(str(op.parameters[ass]))
                        if j < len(assignment) - 1:
                            csv.write(",")
                        else:
                            csv.write("\n")
                else:
                    self.write_ops_with_permutation(csv, op)
        csv.close()

        # Run executable with the input file
        subprocess.run([self.executable_path, filepath])
        output_filepath = filepath.partition(".csv")[0] + "_out.csv"
        csv = open(output_filepath)
        form = re.compile('\d+,\s*\d+\.{0,1}\d*(\n|$)')
        lines = csv.readlines()
        header = lines[0]
        units = 'ms'

        if len(header.split(',')) > 1 and form.match(header) is None:
            units = header.split(',')[1].rstrip("\n")
            lines.pop(0)

        for line in lines:
            if form.match(line) is not None:
                id, ms = line.split(',')
                id = int(id)
                ms = float(ms)
                if id < len(operations):
                    measurements.append(Measurement(operations[id], system, Time(ms, units)))
                else:
                    print("Measurement in {} references wrong indices.".format(output_filepath))

        return measurements


    def write_ops_with_permutation(self, csv, op):
        for permutation in self.permutation:
            [ls, rs] = permutation.split(" -> ")
            params = TrainingBenchmark.flatten(op.parameters)
            ls = ls.replace(',','')
            if len(ls) != len(params):
                print(ls)
                print("Wrong number of parameters for alignment. {} <-> {}".format(permutation, params))
                #TODO: Add a*bc->ab type permutators
                csv.write('0,0,0,0,0\n')
                break
            assign_dict = {}
            for idx,param in enumerate(params):
                if ls[idx] not in assign_dict.keys():
                    assign_dict[ls[idx]] = param
                elif assign_dict[ls[idx]] != param:
                    print("Assumed to be the same. Maybe wrong permutator? {} <- {}".format(op.parameters, permutation))
            rs = rs.replace(',','')
            for i,l in enumerate(rs):
                if l.isdigit():
                    csv.write(str(l))
                else:
                    csv.write(str(assign_dict[l]))
                if i < len(rs) - 1:
                    csv.write(',')
                else:
                    csv.write('\n')



class TrainingBenchmark(Benchmark):
    def __init__(self, name, exe, parameters=""):
        self.name = name
        self.params = parameters
        if (os.path.isabs(exe)):
            self.executable_path = exe
        else:
            self.executable_path = BenchmarkDatabase.root_path + exe

    def __str__(self):
        return "Benchmark {}: {} Params:{}".format(self.name, self.executable_path, self.params)

    def available(self):
        return os.path.isfile(self.executable_path)

    def run(self, operations, system: str):
        print("Launch params: {}".format(self.params))

        """
        #Check if nvprof needs to be linked.
        finder = ModuleFinder()
        finder.run_script(self.executable_path)
        if "Runtimetools.gpu_diagnostics" in finder.modules:
            # Running with emit_nvtx and the nvprof does not work at the moment. Use generic_diagnostics as a workaround
            #nv_out_path = BenchmarkDatabase.runtime_filepath + "nvtrace.prof"
            #print("GPU Diagnostics output: " + BenchmarkDatabase.runtime_filepath + "nvtrace.prof" )
            #subprocess.call(["nvprof --profile-from-start off -o " + nv_out_path + " -f -- python " + self.executable_path + " " + self.params],
            #                shell=True)
            #nprof = torch.autograd.profiler.load_nvprof(nv_out_path)
        else:
        """
        print("CPU/Generic Diagnostics")
        subprocess.call([" python " + self.executable_path + " " + self.params],
                        shell=True)

        outputfilepath = BenchmarkDatabase.runtime_filepath + os.path.splitext(os.path.basename(self.executable_path))[0] + "_out.csv"
        raw_operation_dict = {}

        with open(outputfilepath, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            #format: time, name, shape
            for row in csvreader:
                [time, name, shape] = row
                if name not in raw_operation_dict:
                    raw_operation_dict[name] = [(time, shape)]
                else:
                    raw_operation_dict[name].append((time, shape))

            summarise_training_run(raw_operation_dict)

            correlations = []
            filter = ["backward", '_']
            #Try to find correlated operations like conv + conv_backward
            for raw_op in raw_operation_dict:
                max_lms = 0
                max_el = ""
                filtered_raw_op = raw_op
                for f in filter:
                    filtered_raw_op = filtered_raw_op.lower().replace(f, '')
                for other_op in raw_operation_dict:
                    if other_op != raw_op:
                        lms = self.longest_matching_sequence(other_op, raw_op, filter)
                        if len(lms) > max_lms:
                            max_lms = len(lms)
                            max_el = other_op
                if max_lms >= min(len(filtered_raw_op), len(max_el)) * 0.7:
                    correlations.append(set({raw_op, max_el}))
                else:
                    correlations.append(set({raw_op}))

            #reduce pairs of correlations to sets
            fully_joined = False
            while not fully_joined:
                fully_joined = True
                for (i,a),(j,b) in combinations(enumerate(correlations), 2):
                    if not a.isdisjoint(b):
                        correlations[i] = a|b
                        correlations.pop(j)
                        fully_joined = False
                        break

            # Correlations now holds sets of operators that should map to the same onnx operation

            print("Operationtypes executed when training the Network:")
            for corr in correlations:
                print(corr)
            print("")

            #TODO: Deal with combined functions. Example: mm Addmm Gemm Add

            #for type, ops in operations.items():
            #    for op in ops:
            #        print(op.operator.name)

            print("Assignment extracted Onnx operation -> Group of measured pytorch backend operations")
            op_aligned_correlations = {}
            for op_type, ops in operations.items():
                max_lms = 0
                max_corrset = ""
                max_match = ""
                op_variants = [op_type]
                if op_type in Runtimetools.pytorch_op_name_variants:
                    op_variants += Runtimetools.pytorch_op_name_variants[op_type]
                # Determine the matching set of operations in the network by name
                op_max = op_type
                for corrset in correlations:
                    tmax_lms = 0
                    tmax = ""
                    for op_type_variant in op_variants:
                        for corr in corrset:
                            lms = self.longest_matching_sequence(op_type_variant, corr, ["backward", '_'])
                            if len(lms) > tmax_lms:
                                tmax_lms = len(lms)
                                tmax = corr
                                op_max = op_type_variant
                    lms = self.longest_matching_sequence(op_max, tmax, ["backward", '_'])
                    if len(lms) > max_lms:
                        max_lms = len(lms)
                        max_corrset = corrset
                        max_match = tmax
                    for op_type_variant in op_variants:
                        if op_type_variant == corr:
                            max_corrset = corrset
                            op_max = ""
                            break
                    if op_max == "":
                        break
                if max_lms >= min(len(max_match), len(op_max)) * 0.5:
                    print("{} -> {}".format(op_type, max_corrset))
                    op_aligned_correlations[op_type] = max_corrset
                else:
                    print("No matching operation found in the training profile for {}.".format(op_type))

            print("Operations not assigned to any extracted onnx operator:")
            for corr in correlations:
                is_in = False
                for _,val in enumerate(op_aligned_correlations.items()):
                    if corr in val:
                        is_in = True
                        break
                if not is_in:
                    print(corr)

            # Check if the input dimensions align
            # op_lst = ops.copy()
            op_align_dict = []
            output_measurements = []
            for _, (op_type, corr) in enumerate(op_aligned_correlations.items()):
                unassigned_t = None
                ops = operations[op_type] # Onnx Operations
                '''if op_type == "MatMul":
                    print("Aligning {} with {}".format(op_type, corr))
                    print("A:")
                    for op in ops:
                        print("{} {} {}".format(op.operator.name, op.parameters, op.operator.parameter_descriptors))
                    print("B:")
                    for net_op in corr:
                        for (time, shape) in raw_operation_dict[net_op]:
                            print("{} {} {} {}".format(corr, net_op, shape, time))
                        #for op in ops:'''

                #if op_type in Runtimetools.pytorch_op_unfolding:
                #    (out_measures, unassigned_t) = self.align_strategy_guided(op_type, ops, corr, raw_operation_dict)
                #else:
                (out_measures, unassigned_t) = self.align_strategy_default(op_type, ops, corr, raw_operation_dict)

                for (op, time) in out_measures:
                    output_measurements.append(Measurement(op, system, time))


            if unassigned_t is not None:
                print("Total uncorrelated operations: " + str(unassigned_t))

                """for idx, op in enumerate(ops):
                    op_align_dict.append([])
                    for corr in max_corrset:
                        max_ir = 0
                        best_param = ()
                        for (time, shape) in raw_operation_dict[corr]:
                            op_params_a = set(self.flatten(ast.literal_eval(shape)))
                            op_params_b = set(self.flatten(op.parameters))
                            min_elem = min(len(op_params_a), len(op_params_b))
                            inter = len(op_params_a & op_params_b)

                            # print("A: {} B: {}".format(op_params_a, op_params_b))
                            if len(op_params_a) > 0 and len(op_params_b) > 0 and inter >= min_elem * 0.7:
                                int_ratio = inter / min_elem
                                if int_ratio > max_ir:
                                    max_ir = int_ratio
                                    best_param = (time, shape)
                                    #print("Align List:{} {}[{}] -> {}[{}] & {}".format(op.operator.name, op_params_a, len(op_params_a), op_params_b, len(op_params_b), op_params_a & op_params_b))
                                # op_align_dict[idx].append((time, shape))
                        if len(best_param) > 0:
                            op_align_dict[idx].append(best_param)
                if len(op_align_dict[idx]) > 0:
                    print("Align List:{} {} -> {}".format(op.operator.name, op.parameters, op_align_dict[idx]))"""


                    #continue
                    #print("___________________")
                    #print("A:")
                    #for op in ops:
                    #    print("{} {}".format(op.operator.name, self.flatten(op.parameters)))
                    #print("___________________")
                    #print("B:")
                    #for corr in max_corrset:
                    #    for (time, shape) in raw_operation_dict[corr]:
                    #        print("{} {}".format(corr, self.flatten(ast.literal_eval(shape))))
            return output_measurements


    def align_strategy_default(self, op_type, ops, corr, raw_operation_dict):
        #print("Align strategy default summary: ")
        #print("Aligning operations of type " + op_type)
        unassigned_t = None
        raw_ops = {}  # Training low level Operations
        for net_op in corr:
            raw_ops[net_op] = raw_operation_dict[net_op].copy()

        '''
        print("Raw ops:")
        for rop_name, rop_value in raw_ops.items():
            print(rop_name)
            for el in rop_value:
                print(el) # Align this
        print("Ops:")
        for op in ops:
            print(op) # With this
        print("End Strategy summary.")'''

        output_measurements = []
        # try to assign an onnx operation to the training operations
        tsum = None#Time.fromString("0ms")
        #for op in ops:
        unrolled_raw = []
        #for net_op in corr:
            #print(raw_ops[net_op])
            #unrolled_raw += raw_ops[net_op]
        for rop_name, rop_value in raw_ops.items():
            unrolled_raw += rop_value

        '''print("Combined:")
        for el in unrolled_raw:
            print(el)

        print("net_op in corr: " + str(len(corr)))'''

        align_matrix = np.zeros((len(ops), len(unrolled_raw)))
        op_index = 0
        for op in ops:
            permutator = range(0, len(self.flatten(op.parameters)))
            op_align_quality = [self.count_overlap(self.flatten(op.parameters),
                              self.flatten(ast.literal_eval(x[1])),
                              permutator) for x in unrolled_raw]
            align_matrix[op_index, :] = op_align_quality
            op_index = op_index + 1
            '''
            net_op_offset = 0
            #old pre rewritten
            for net_op in corr:
                if net_op in Runtimetools.pytorch_op_parameter_order:
                    permutator = self.get_rewrite_permutation(op.operator.parameter_descriptors,
                                                              Runtimetools.pytorch_op_parameter_order[net_op])
                else:
                    permutator = range(0, len(self.flatten(op.parameters)))
                if len(raw_ops[net_op]) > 0:
                    if op.operator.name == "MatMul":
                        print("A: {}, B: {}".format(self.flatten(op.parameters),
                                       raw_ops[net_op]))
                    net_op_align_quality = [self.count_overlap(self.flatten(op.parameters),
                                       self.flatten(ast.literal_eval(x[1])),
                                       permutator) for x in raw_ops[net_op]]
                    align_matrix[op_index, net_op_offset:(net_op_offset + len(net_op_align_quality))] = net_op_align_quality
                    net_op_offset += len(raw_ops[net_op])
                    #best_fit_op = raw_ops[net_op].index(max(raw_ops[net_op], key=lambda val:
                    #self.count_overlap(self.flatten(op.parameters),
                    #                   self.flatten(ast.literal_eval(val[1])),
                    #                   permutator)))
                    #mtime = raw_ops[net_op][best_fit_op][0]
                    #print("{}{} -> {}{} {}".format(op.operator.name, op.parameters, net_op,
                    #                               raw_ops[net_op][best_fit_op][1], mtime))
                    #tsum = Time.fromString(mtime) + tsum
                    #output_measurements.append((op, Time.fromString(mtime)))
                    #del raw_ops[net_op][best_fit_op]
            '''
        #avail_op_indices = [x for x in range(0, len(ops))]
        #for net_op in corr:
        #    for i, rop in enumerate(raw_ops[net_op]):
        listed_op = list(ops.copy())
        listed_raw = list(unrolled_raw.copy())
        for i in range(0, min(len(ops), len(unrolled_raw))):
            (u, v) = np.unravel_index(align_matrix.argmax(), align_matrix.shape)
            align_matrix = np.delete(align_matrix, u, 0)
            align_matrix = np.delete(align_matrix, v, 1)
            mtime = listed_raw[v][0]
            op = listed_op[u]
            print("{}{} -> {} {}".format(op.operator.name, op.parameters,
                                           listed_raw[v][1], mtime))
            output_measurements.append((op, Time.fromString(mtime)))
            listed_raw.pop(v)
            listed_op.pop(u)

        for raw in unrolled_raw:
            if tsum is None:
                tsum = Time.fromString(raw[0])
            else:
                tsum += Time.fromString(raw[0])

        print("All {} operations took {}".format(op_type, tsum))

        if len(listed_raw) > 0:
            for residue in listed_raw:
                #print("{} {}".format(residue[1], residue[0]))
                unassigned_t = Time.fromString(residue[0]) + unassigned_t
            print("{} remaining in the network operations. ({},{})".format(len(listed_raw), op_type, unassigned_t))

        #for net_op in corr:
        #    if len(raw_ops[net_op]) > 0:
        #        print("{} remaining in the network operations. ({})".format(len(raw_ops[net_op]), net_op))
        #        for op in raw_ops[net_op]:
        #            unassigned_t = Time.fromString(op[0]) + unassigned_t
        #            print("{}".format(op))

        return (output_measurements, unassigned_t)


    def generate_onnx(self):
        onnx_dir = BenchmarkDatabase.runtime_filepath + self.name + ".onnx"
        if not os.path.isfile(onnx_dir):
            arguments = ["python"] + [self.executable_path] + self.params.strip().split(' ')
            process = subprocess.Popen(arguments, stdin=subprocess.PIPE)
            process.stdin.write(onnx_dir.encode('utf-8'))
            process.communicate()[0]


    @staticmethod
    def get_rewrite_permutation(a, b):
        permutation = [-1] * len(a)
        q_counter = 0
        for ia, lita in enumerate(a,0):
            if lita == "?":
                a[ia] = "?" + str(q_counter)
                q_counter += 1

        for ia, lita in enumerate(a, 0):
            for ib, litb in enumerate(b, 0):
                if lita == b[ib]:
                    permutation[ia] = ib
                    break
        return permutation

    @staticmethod
    def count_overlap(a, b, p):
        if len(a) == 0 or len(b) == 0:
            return 0
        cnt = 0
        #print("A:{}, B:{}, p:{}, count:{}".format(a, b, p, cnt))
        for idx in range(len(a)):
            if idx < len(b) and a[idx] == b[p[idx]] and p[idx] >= 0:
                cnt += 1
        #set(a).union(set(b))
        #print("A:{}, B:{}, p:{}, count:{}".format(a, b, p, cnt))
        return cnt

    @staticmethod
    def longest_matching_sequence(a, b, filter):
        if len(a) > len(b):
           a, b = b, a
        for f in filter:
            a = a.lower().replace(f, '')
            b = b.lower().replace(f, '')
        seq_starts = range(len(b))
        match = ""
        for seq_length in range(1, len(a) + 1):
            new_starts = []
            for s in seq_starts:
                if s+seq_length <= len(a) and a[s:(s+seq_length)] in b:
                    match = a[s:(s+seq_length)]
                    new_starts.append(s)
            seq_starts = new_starts
        return match

    @staticmethod
    def flatten(test_list):
        if isinstance(test_list, list):
            if len(test_list) == 0:
                return []
            first, rest = test_list[0], test_list[1:]
            return TrainingBenchmark.flatten(first) + TrainingBenchmark.flatten(rest)
        else:
            return [test_list]