import reframe as rfm
import reframe.utility.sanity as sn
from nekrs import NekRSTest, NekRSMLOfflineTest, NekRSMLOnlineTest
import os


@rfm.simple_test
class TGVOffline(NekRSMLOfflineTest):
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
        self.tags |= {"tgv_offline"}


@rfm.simple_test
class TGVOfflineCoarseMesh(NekRSMLOfflineTest):
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
        self.tags |= {"tgv_offline_coarse_mesh"}


@rfm.simple_test
class TGVOfflineTraj(NekRSMLOfflineTest):
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
        self.tags |= {"tgv_offline_traj"}


@rfm.simple_test
class TurbChannelOffline(NekRSMLOfflineTest):
    num_nodes = parameter([1])
    ranks_per_node = parameter([12])
    model = parameter(["sr-gnn"])

    def __init__(self):
        super().__init__(
            case="turbChannel",
            directory="turbChannel_srgnn",
            nn=self.num_nodes,
            rpn=self.ranks_per_node,
            model=self.model,
            epochs=5,
            n_element_neighbors=12,
            n_messagePassing_layers=6,
            time_dependency="time_independent",
        )
        self.tags |= {"turbchannel_offline"}


@rfm.simple_test
class TGVOnline(NekRSMLOnlineTest):
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
        self.tags |= {"tgv_online"}


@rfm.simple_test
class TGVOnlineTraj(NekRSMLOnlineTest):
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
        self.tags |= {"tgv_online_traj"}


@rfm.simple_test
class TGVOnlineTrajAdios(NekRSMLOnlineTest):
    num_nodes = parameter([1])
    ranks_per_node = parameter([8])
    model = parameter(["dist-gnn"])

    def __init__(self):
        super().__init__(
            case="tgv",
            directory="tgv_gnn_online_traj_adios",
            nn=self.num_nodes,
            rpn=self.ranks_per_node,
            model=self.model,
            time_dependency="time_dependent",
            client="adios",
            target_loss=6.9076e-01,
        )
        self.tags |= {"tgv_online_traj_adios"}
