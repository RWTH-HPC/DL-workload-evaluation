import torch
import os
import csv
from Runtimetools.benchmark_diagnostics import op_performance_analysis
from BenchmarkDatabase import runtime_filepath
import __main__ as main

class op_performance_analysis_cpu(op_performance_analysis):
    def set_profiler(self, stack):
        self.prof = torch.autograd.profiler.profile(enabled=self.do_profile, record_shapes=True)
        stack.enter_context(self.prof)
        return stack

    def __exit__(self, exception_type, exception_value, traceback):
        self._stack.__exit__(exception_type, exception_value, traceback)
        if self.do_profile:
            self.prof.function_events.populate_cpu_children()
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
                    if len(event.cpu_children) == 0 and len(event.input_shapes) > 0:
                        csvwriter = csv.writer(csvfile, delimiter=',')
                        csvwriter.writerow([event.cpu_time_str, event.name, event.input_shapes])