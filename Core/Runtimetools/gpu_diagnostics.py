import torch
import os
import csv
from Runtimetools.benchmark_diagnostics import op_performance_analysis
from BenchmarkDatabase import runtime_filepath
import __main__ as main

class op_performance_analysis_gpu(op_performance_analysis):
    """
        When Tested with cuda 10.2 and 10 + cudnn 7.65/7.4 this did not work.
        Running into Assert issues when loading the nvprofiling data or no nvprofile got written.
        Use op_performance_analysis_generic with use_cuda for now
    """
    def set_profiler(self, stack):
        #self.prof = torch.autograd.profiler.profile(enabled=self.do_profile, use_cuda=True, record_shapes=True)
        #stack.enter_context(self.prof)
        self.prof = torch.cuda.profiler.profile()
        stack.enter_context(self.prof)
        stack.enter_context(torch.autograd.profiler.emit_nvtx(enabled=self.do_profile, record_shapes=True))
        return stack

    def __exit__(self, exception_type, exception_value, traceback):
        self._stack.__exit__(exception_type, exception_value, traceback)
