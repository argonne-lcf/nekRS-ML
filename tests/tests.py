import reframe as rfm
import reframe.utility.sanity as sn
from nekrs import (
    NekRSMLOfflineFldTest,
    NekRSMLOfflineRepartTest,
    NekRSMLOfflineTest,
    NekRSMLOnlineTest,
    EnsembleTest,
)
import os


TGV_TRANSFORM_OPTS = [
    "transform_x=true",
    "transform_y=true",
    "transform_z=true",
]


@rfm.simple_test
class TGVOffline(NekRSMLOfflineTest):
    num_nodes = parameter([1])
    ranks_per_node = parameter([1, 2, 4])

    def __init__(self):
        super().__init__(
            case="tgv",
            directory="tgv_gnn_offline",
            nn=self.num_nodes,
            rpn=self.ranks_per_node,
            time_dependency="time_independent",
            target_loss=2.7161e-04,
            extra_opts=TGV_TRANSFORM_OPTS,
        )
        self.tags |= {"tgv_offline"}


@rfm.simple_test
class TGVOfflineCoarseMesh(NekRSMLOfflineTest):
    num_nodes = parameter([1])
    ranks_per_node = parameter([2])

    def __init__(self):
        super().__init__(
            case="tgv",
            directory="tgv_gnn_offline_coarse_mesh",
            nn=self.num_nodes,
            rpn=self.ranks_per_node,
            time_dependency="time_independent",
            target_loss=2.7161e-04,
            extra_opts=TGV_TRANSFORM_OPTS,
        )
        self.tags |= {"tgv_offline_coarse_mesh"}


@rfm.simple_test
class TGVOfflineRepart(NekRSMLOfflineRepartTest):
    num_nodes = parameter([1])
    # nekRS always runs on nekrs_ranks=2 and writes the usual gnn_outputs
    # directory; the repartition CLI redistributes it to 2 and 4 ranks
    # before training to check that the loss is independent of the rank
    # count (rpn=4 exercises an actual 2 -> 4 repartitioning).
    ranks_per_node = parameter([2, 4])
    # rcb is pure Python; parrsb builds the C shim against the install's
    # libparRSB.a/libgs.a in a prerun step (build_parrsb_shim.sh) and
    # checks the topology-aware partitioner end to end. Same target_loss:
    # the loss is independent of the partitioning method.
    repart_method = parameter(["rcb", "parrsb"])

    def __init__(self):
        super().__init__(
            case="tgv",
            directory="tgv_gnn_offline",
            nn=self.num_nodes,
            rpn=self.ranks_per_node,
            nekrs_ranks=2,
            repartition_method=self.repart_method,
            time_dependency="time_independent",
            target_loss=2.7161e-04,
            extra_opts=TGV_TRANSFORM_OPTS,
        )
        self.tags |= {"tgv_offline_repart"}


@rfm.simple_test
class TGVOfflineFld(NekRSMLOfflineFldTest):
    num_nodes = parameter([1])
    # nekRS always runs on nekrs_ranks=2 and writes only a .f checkpoint;
    # the graph and training data are reconstructed from it at 2 and 4
    # ranks to check that the loss is independent of the rank count
    # (rpn=4 exercises an actual 2 -> 4 repartitioning).
    ranks_per_node = parameter([2, 4])

    def __init__(self):
        super().__init__(
            case="tgv",
            directory="tgv_gnn_offline_fld",
            nn=self.num_nodes,
            rpn=self.ranks_per_node,
            nekrs_ranks=2,
            periodic="xyz",
            time_dependency="time_independent",
            target_loss=2.7161e-04,
            extra_opts=TGV_TRANSFORM_OPTS,
        )
        self.tags |= {"tgv_offline_fld"}


@rfm.simple_test
class TGVOfflineTraj(NekRSMLOfflineTest):
    num_nodes = parameter([1])
    # Run with 1, 2, and 4 ranks to check consistency of Dist-GNN model
    ranks_per_node = parameter([1, 2, 4])

    def __init__(self):
        super().__init__(
            case="tgv",
            directory="tgv_gnn_offline_traj",
            nn=self.num_nodes,
            rpn=self.ranks_per_node,
            time_dependency="time_dependent",
            target_loss=6.6139e-01,
            extra_opts=TGV_TRANSFORM_OPTS,
        )
        self.tags |= {"tgv_offline_traj"}


@rfm.simple_test
class TGVOfflineTrajGT(NekRSMLOfflineTest):
    num_nodes = parameter([1])
    # Run with 1, 2, and 4 ranks to check consistency of the Graph Transformer model
    ranks_per_node = parameter([1, 2, 4])

    def __init__(self):
        super().__init__(
            case="tgv",
            directory="tgv_gt_offline_traj",
            nn=self.num_nodes,
            rpn=self.ranks_per_node,
            time_dependency="time_dependent",
            target_loss=3.1556e-01,
            extra_opts=["model_name=graph_transformer", *TGV_TRANSFORM_OPTS],
        )
        self.tags |= {"tgv_offline_traj_gt"}


@rfm.simple_test
class TurbChannelSRGNN(NekRSMLOfflineTest):
    num_nodes = parameter([1])
    ranks_per_node = parameter([12])

    def __init__(self):
        super().__init__(
            case="turbChannel",
            directory="turbChannel_srgnn",
            nn=self.num_nodes,
            rpn=self.ranks_per_node,
            model="sr-gnn",
            epochs=5,
            n_element_neighbors=12,
            n_messagePassing_layers=6,
            time_dependency="time_independent",
        )
        self.tags |= {"turbchannel_srgnn"}


@rfm.simple_test
class TGVOnline(NekRSMLOnlineTest):
    num_nodes = parameter([1])
    ranks_per_node = parameter([2, 4])

    def __init__(self):
        super().__init__(
            case="tgv",
            directory="tgv_gnn_online",
            nn=self.num_nodes,
            rpn=self.ranks_per_node,
            time_dependency="time_independent",
            client="smartredis",
            target_loss=2.7161e-04,
            extra_opts=TGV_TRANSFORM_OPTS,
        )
        self.tags |= {"tgv_online"}


@rfm.simple_test
class TGVOnlineTraj(NekRSMLOnlineTest):
    num_nodes = parameter([1])
    ranks_per_node = parameter([4])

    def __init__(self):
        super().__init__(
            case="tgv",
            directory="tgv_gnn_online_traj",
            nn=self.num_nodes,
            rpn=self.ranks_per_node,
            time_dependency="time_dependent",
            client="smartredis",
            target_loss=6.6139e-01,
            extra_opts=TGV_TRANSFORM_OPTS,
        )
        self.tags |= {"tgv_online_traj"}


@rfm.simple_test
class TGVOnlineTrajAdios(NekRSMLOnlineTest):
    num_nodes = parameter([1])
    ranks_per_node = parameter([4])

    def __init__(self):
        super().__init__(
            case="tgv",
            directory="tgv_gnn_online_traj_adios",
            nn=self.num_nodes,
            rpn=self.ranks_per_node,
            time_dependency="time_dependent",
            client="adios",
            target_loss=6.6139e-01,
            extra_opts=TGV_TRANSFORM_OPTS,
        )
        self.tags |= {"tgv_online_traj_adios"}


@rfm.simple_test
class PeriodicHillEnsemble(EnsembleTest):
    num_members = parameter([4])
    nodes_per_member = parameter([1])
    ranks_per_node = parameter([12])

    def __init__(self):
        super().__init__(
            case="periodicHill",
            directory="periodicHill_ensemble",
            members=self.num_members,
            nodes_per_member=self.nodes_per_member,
            rpn=self.ranks_per_node,
            gen_args=["--hillScale", f"0.8,1.2,{self.num_members}"],
        )
        self.tags |= {"periodichill_ensemble"}
