import torch
from contextlib import ExitStack
import csv
import os
import __main__ as main
from BenchmarkDatabase import runtime_filepath
import sys
from abc import ABC, abstractmethod
import time
import numpy as np

from Runtimetools import pytorch_cpu_minor_ops


def generate_onnx(path, net, input):
    if hasattr(net, "module"):
        m = net.module
    else:
        m = net
    print(input)
    if type(input) is dict:
        input = list(input.items())
    torch.onnx.export(m,
                      input,
                      path,
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=False)
    sys.exit("Onnx exported to: " + path)


def print_children(event, indent):
    for child in event.cpu_children:
        print(indent + "Child: Name:{} ID:{} Shape:{} Kernels:{} Keys:{}".format(child.name, child.id,
                                                                                 child.input_shapes,
                                                                                 child.kernels, child.key))
    for child in event.cpu_children:
        print_children(child, indent + "   ")


class op_performance_analysis(ABC):
    def __init__(self, batch_id, input, net, cuda):
        self.batch_idx = batch_id
        self.do_profile = False
        self.input = input
        self.net = net
        self.cuda = cuda

    def __enter__(self):
        with ExitStack() as stack:
            if not sys.stdin.isatty():
                for line in sys.stdin:
                    self.onnx_out = line
                    generate_onnx(self.onnx_out, self.net, self.input)
            self.do_profile = self.batch_idx == 3
            stack = self.set_profiler(stack)#torch.autograd.profiler.profile(enabled=self.do_profile, record_shapes=True)
            #stack.enter_context(self.prof)
            self._stack = stack.pop_all()
        return self

    @abstractmethod
    def set_profiler(self, stack):
        print("Base class method.")
        return

class op_performance_analysis_generic(op_performance_analysis):

    """
        Generic pytorch profiling that saves a csv with the relevant events.
        use_cuda utilises cuda_events and has not been tested or validated. TODO!
    """

    def set_profiler(self, stack):
        self.prof = torch.autograd.profiler.profile(enabled=self.do_profile, use_cuda=self.cuda, record_shapes=True)
        if self.do_profile:
            self.start_time = time.time()
        stack.enter_context(self.prof)
        return stack

    def __exit__(self, exception_type, exception_value, traceback):
        self._stack.__exit__(exception_type, exception_value, traceback)
        if self.do_profile:
            print("Batch execution time: {}s".format(time.time() - self.start_time))
            self.prof.function_events.populate_cpu_children()
            #self.prof.export_chrome_trace("gpu_trace_11_06.trace")
            outputfilepath = os.path.splitext(os.path.basename(main.__file__))[0]
            with open(runtime_filepath + outputfilepath + "_out.csv", 'w', newline='') as csvfile:
                for event in self.prof.function_events:
                    #print(dir(event))
                    #print("Name:{} ID:{} Shape:{} Kernels:{} Keys:{} Children:{}".format(event.name, event.id, event.input_shapes,
                    #                                                         event.kernels, event.key, event.cpu_children))
                    #is_in = False
                    #for other in self.prof.function_events:
                    #    for pchilds in other.cpu_children:
                    #        if pchilds.id == event.id:
                    #            is_in = True
                    #if not is_in:
                    #    print("Name:{} ID:{} Shape:{} Kernels:{} Keys:{}".format(event.name, event.id, event.input_shapes,
                    #                                                           event.kernels, event.key))
                    #    print_children(event, "   ")
                    #continue
                    #print(dir(event))
                    print(event)
                    if self.prof.use_cuda:
                        if len(event.cpu_children) == 0 and len(event.input_shapes) > 0:
                            csvwriter = csv.writer(csvfile, delimiter=',')
                            csvwriter.writerow([event.cuda_time_str, event.name, event.input_shapes])
                    else:
                        csvwriter = csv.writer(csvfile, delimiter=',')
                        if len(event.cpu_children) != 0:
                            #subEventsAreMinor = True
                            #for childEventId in event.cpu_children:
                            #    print(childEventId)
                            #    idFound = False
                            #    for other in self.prof.function_events:
                            #        if other.id == childEventId and len(other.cpu_children) != 0:
                            #            subEventsAreMinor = False
                            #            idFound = True
                            #            break
                                #        if other.name not in pytorch_cpu_minor_ops:
                                #            subEventsAreMinor = False
                                #        idFound = True
                                #        break
                            #    if not idFound:
                            #        subEventsAreMinor = False
                            #if subEventsAreMinor:
                            subEventsAreMinor = True
                            for childEvent in event.cpu_children:
                                if childEvent.name not in pytorch_cpu_minor_ops:
                                    subEventsAreMinor = False

                            if subEventsAreMinor:
                                print(event)
                                csvwriter.writerow([event.cpu_time_str, event.name, event.input_shapes])
                        elif event.name not in pytorch_cpu_minor_ops:
                            csvwriter.writerow([event.cpu_time_str, event.name, event.input_shapes])








