import reframe as rfm
import reframe.utility.sanity as sn
from nekrs import NekRSCase, NekRSTest
import os


@rfm.simple_test
class NekRSKershawTest(NekRSTest):
    def __init__(self):
        super().__init__(nekrs_case=NekRSCase("kershaw"))

    # Match "flops/rank" at start of line to avoid matching output during setup.
    @performance_function("flops/s/rank", perf_key="BPS5")
    def bps5_performance(self):
        return sn.extractsingle(r"^flops/rank: (\S+)", self.stdout, 1, float, 0)

    @performance_function("flops/s/rank", perf_key="BP5")
    def bp5_performance(self):
        return sn.extractsingle(r"^flops/rank: (\S+)", self.stdout, 1, float, 1)

    @performance_function("flops/s/rank", perf_key="BP6")
    def bp6_performance(self):
        return sn.extractsingle(r"^flops/rank: (\S+)", self.stdout, 1, float, 2)

    @performance_function("flops/s/rank", perf_key="BP6PCG")
    def bp6pcg_performance(self):
        return sn.extractsingle(r"^flops/rank: (\S+)", self.stdout, 1, float, 3)
