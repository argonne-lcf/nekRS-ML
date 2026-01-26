import reframe as rfm
import reframe.utility.sanity as sn
from nekrs import NekRSTest, NekRSMLOfflineTest, NekRSMLOnlineTest
import os


@rfm.simple_test
class NekRSTGVOffline(NekRSMLOfflineTest):
    num_nodes = parameter([1])
    ranks_per_node = parameter([2])
    model = parameter(["dist-gnn"])

    def __init__(self):
        super().__init__(
            case="tgv",
            directory="tgv_gnn_offline",
            nn=self.num_nodes,
            rpn=self.ranks_per_node,
            model=self.model,
            time_dependency="time_independent",
            target_loss=1.6206e-04,
        )
        self.tags |= {f"tgv_offline_{self.model}"}


@rfm.simple_test
class NekRSTGVOfflineCoarseMesh(NekRSMLOfflineTest):
    num_nodes = parameter([1])
    ranks_per_node = parameter([2])
    model = parameter(["dist-gnn"])

    def __init__(self):
        super().__init__(
            case="tgv",
            directory="tgv_gnn_offline_coarse_mesh",
            nn=self.num_nodes,
            rpn=self.ranks_per_node,
            model=self.model,
            time_dependency="time_independent",
            target_loss=1.6206e-04,
        )
        self.tags |= {f"tgv_offline_coarse_mesh_{self.model}"}


@rfm.simple_test
class NekRSTGVOfflineTraj(NekRSMLOfflineTest):
    num_nodes = parameter([1])
    ranks_per_node = parameter([4])
    model = parameter(["dist-gnn"])

    def __init__(self):
        super().__init__(
            case="tgv",
            directory="tgv_gnn_offline_traj",
            nn=self.num_nodes,
            rpn=self.ranks_per_node,
            model=self.model,
            time_dependency="time_dependent",
            target_loss=6.9076e-01,
        )
        self.tags |= {f"tgv_offline_traj_{self.model}"}


@rfm.simple_test
class NekRSTGVOnline(NekRSMLOnlineTest):
    num_nodes = parameter([1])
    ranks_per_node = parameter([4])
    model = parameter(["dist-gnn"])

    def __init__(self):
        super().__init__(
            case="tgv",
            directory="tgv_gnn_online",
            nn=self.num_nodes,
            rpn=self.ranks_per_node,
            model=self.model,
            time_dependency="time_independent",
            client="smartredis",
            target_loss=1.6206e-04,
        )
        self.tags |= {f"tgv_online_{self.model}"}


@rfm.simple_test
class NekRSTGVOnlineTraj(NekRSMLOnlineTest):
    num_nodes = parameter([1])
    ranks_per_node = parameter([8])
    model = parameter(["dist-gnn"])

    def __init__(self):
        super().__init__(
            case="tgv",
            directory="tgv_gnn_online_traj",
            nn=self.num_nodes,
            rpn=self.ranks_per_node,
            model=self.model,
            time_dependency="time_dependent",
            client="smartredis",
            target_loss=6.9076e-01,
        )
        self.tags |= {f"tgv_online_traj_{self.model}"}
