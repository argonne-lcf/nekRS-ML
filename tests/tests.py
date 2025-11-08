import reframe as rfm
import reframe.utility.sanity as sn
from nekrs import NekRSTest, NekRSMLOfflineTest, NekRSMLOnlineTest
import os


@rfm.simple_test
class NekRSKershawTest(NekRSTest):
    num_nodes = parameter([1])
    ranks_per_node = parameter([2])

    def __init__(self):
        super().__init__(
            "kershaw", "kershaw", nn=self.num_nodes, rpn=self.ranks_per_node
        )

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


@rfm.simple_test
class NekRSTGVOffline(NekRSMLOfflineTest):
    num_nodes = parameter([1])
    ranks_per_node = parameter([2])

    def __init__(self):
        super().__init__(
            case="tgv",
            directory="tgv_gnn_offline",
            nn=self.num_nodes,
            rpn=self.ranks_per_node,
            time_dependency="time_independent",
            target_loss=1.6206e-04,
        )
        self.tags |= {"tgv_offline"}


@rfm.simple_test
class NekRSTGVOfflineCoarseMesh(NekRSMLOfflineTest):
    num_nodes = parameter([1])
    ranks_per_node = parameter([2])

    def __init__(self):
        super().__init__(
            case="tgv",
            directory="tgv_gnn_offline_coarse_mesh",
            nn=self.num_nodes,
            rpn=self.ranks_per_node,
            time_dependency="time_independent",
            target_loss=1.6206e-04,
        )
        self.tags |= {"tgv_offline_coarse_mesh"}


@rfm.simple_test
class NekRSTGVOfflineTraj(NekRSMLOfflineTest):
    num_nodes = parameter([1])
    ranks_per_node = parameter([4])

    def __init__(self):
        super().__init__(
            case="tgv",
            directory="tgv_gnn_offline_traj",
            nn=self.num_nodes,
            rpn=self.ranks_per_node,
            time_dependency="time_dependent",
            target_loss=6.9076e-01,
        )
        self.tags |= {"tgv_offline_traj"}


@rfm.simple_test
class NekRSTGVOnline(NekRSMLOnlineTest):
    num_nodes = parameter([1])
    ranks_per_node = parameter([4])

    def __init__(self):
        super().__init__(
            case="tgv",
            directory="tgv_gnn_online",
            nn=self.num_nodes,
            rpn=self.ranks_per_node,
            time_dependency="time_independent",
            client="smartredis",
            target_loss=1.6206e-04,
        )
        self.tags |= {"tgv_online"}


@rfm.simple_test
class NekRSTGVOnlineTraj(NekRSMLOnlineTest):
    num_nodes = parameter([1])
    ranks_per_node = parameter([8])

    def __init__(self):
        super().__init__(
            case="tgv",
            directory="tgv_gnn_online_traj",
            nn=self.num_nodes,
            rpn=self.ranks_per_node,
            time_dependency="time_dependent",
            client="smartredis",
            target_loss=6.9076e-01,
        )
        self.tags |= {"tgv_online_traj"}
