import os
import re
import sys
from typing import Optional, Union, Tuple
import logging
from omegaconf import DictConfig
import numpy as np
from time import sleep, perf_counter

# Import SmartRedis
try:
    from smartredis import Client, Dataset
except ModuleNotFoundError:
    pass

# Import ADIOS2
try:
    from adios2 import Stream, Adios, bindings
except ModuleNotFoundError:
    pass

log = logging.getLogger(__name__)


class OnlineClient:
    """Class for the online training client"""

    def __init__(self, cfg: DictConfig, comm) -> None:
        self.client = None
        self.backend = cfg.client.backend
        self.comm = comm
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.local_rank = int(os.getenv("PALS_LOCAL_RANKID"))
        self.local_size = int(os.getenv("PALS_LOCAL_SIZE"))

        # Initialize timers
        self.timers = self.setup_timers()

        # Initialize the client backend
        clients = ["smartredis", "adios"]
        if self.backend not in clients:
            sys.exit(
                f"Client {self.backend} not implemented. "
                f"Available options are: {clients}"
            )
        self.init_client(cfg)

    def setup_timers(self) -> dict:
        """Setup timer dictionary to collect time spent on client ops"""
        timers = {}
        timers["init"] = []
        timers["data"] = []
        timers["meta_data"] = []
        return timers

    def init_client(self, cfg: DictConfig) -> None:
        """Initialize the client based on the specified backend"""
        tic = perf_counter()
        if self.backend == "smartredis":
            self.db_nodes = cfg.client.db_nodes
            SSDB = os.getenv("SSDB")
            if self.db_nodes == 1:
                self.client = Client(address=SSDB, cluster=False)
            else:
                self.client = Client(address=SSDB, cluster=True)
        elif self.backend == "adios":
            self.engine = cfg.client.adios_engine
            self.transport = cfg.client.adios_transport
            self.adios = Adios(self.comm)
            self._bp_io_counter = 0
            self.client = self.adios.declare_io("streamIO")
            self.client.set_engine(self.engine)
            parameters = {
                "DataTransport": self.transport,  # options: MPI, WAN, UCX, RDMA
                "OpenTimeoutSecs": "600",  # number of seconds writer waits on Open() for reader
                "AlwaysProvideLatestTimestep": "False",  # True means reader will see only the newest available step
            }
            self.client.set_parameters(parameters)
            self.solutionStream = None
        # set by get_graph_data_from_stream
        self.graph_source = None
        self.repart = None
        self.N_list = None
        self.num_edges_list = None
        self.fieldOffset_list = None
        self.timers["init"].append(perf_counter() - tic)

    def _create_bp_io(self) -> "IO":
        """Create a uniquely-named IO configured for BP5 file access.

        The path-based Stream(path, mode, comm) constructor defaults to
        engine type 'File', which is no longer registered in newer ADIOS2
        builds.  This helper creates an IO with engine explicitly set to
        'BP5' so it can be passed to the IO-based Stream constructor.
        """
        self._bp_io_counter += 1
        io = self.adios.declare_io(f"bp_io_{self._bp_io_counter}")
        io.set_engine("BP5")
        return io

    def file_exists(self, file_name: str) -> bool:
        """Check if a file (or key) exists"""
        tic = perf_counter()
        if self.backend == "smartredis":
            return self.client.key_exists(file_name)
        self.timers["meta_data"].append(perf_counter() - tic)

    def get_array(self, file_name) -> np.ndarray:
        """Get an array frpm staging area / simulation"""
        tic = perf_counter()
        if self.backend == "smartredis":
            if isinstance(file_name, str):
                while True:
                    if self.file_exists(file_name):
                        array = self.client.get_tensor(file_name)
                        break
                    else:
                        sleep(0.5)
                        t_elapsed = perf_counter() - tic
                        if t_elapsed > 300:
                            sys.exit(f"Could not find {file_name} in DB")
            else:
                array = file_name.get_tensor("data")
        if self.backend == "adios":
            var_name = file_name.split(".")[0]
            with Stream(self._create_bp_io(), file_name, "r") as stream:
                stream.begin_step()
                arr = stream.inquire_variable(var_name)
                shape = arr.shape()
                count = int(shape[0] / self.size)
                start = count * self.rank
                if self.rank == self.size - 1:
                    count += shape[0] % self.size
                array = stream.read(var_name, [start], [count])
                stream.end_step()
        self.timers["data"].append(perf_counter() - tic)
        return array

    def put_array(
        self,
        file_name: str,
        array: np.ndarray,
        global_ids: Optional[np.ndarray] = None,
    ) -> None:
        """Put/send an array to staging area / simulation.

        Under adios the array is written as one global BP array over the
        whole reader communicator, not one file per rank: the per-rank
        _rank_R_size_S suffix is stripped from file_name and the blocks are
        concatenated in rank order. Layout is component-major with no
        alignStride padding, matching the nekRS-facing convention of
        in_u/out_u (but unpadded, since the reader has no fieldOffset).

        global_ids, when given, is written alongside as <var>_global_ids so
        a consumer can scatter the result back onto the simulation mesh
        without knowing how many ranks produced it -- necessary as soon as
        the ML rank count is decoupled from the nekRS rank count.
        """
        if self.backend == "smartredis":
            self.client.put_tensor(file_name, array)
        elif self.backend == "adios":
            tic = perf_counter()
            var = re.sub(r"_rank_\d+_size_\d+$", "", file_name)
            arr = np.ascontiguousarray(array)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            nloc, ncols = arr.shape
            counts = self.comm.allgather(nloc)
            start, total = sum(counts[: self.rank]), sum(counts)
            with Stream(f"{var}.bp", "w", self.comm) as stream:
                stream.begin_step()
                stream.write(
                    var,
                    np.ascontiguousarray(arr.T).reshape(-1),
                    [total * ncols],
                    [start * ncols],
                    [nloc * ncols],
                )
                if global_ids is not None:
                    gid = np.ascontiguousarray(
                        np.asarray(global_ids).reshape(-1), dtype=np.int64
                    )
                    if gid.size != nloc:
                        raise ValueError(
                            f"global_ids has {gid.size} entries but the "
                            f"array has {nloc} rows"
                        )
                    stream.write(
                        f"{var}_global_ids", gid, [total], [start], [nloc]
                    )
                stream.end_step()
            self.timers["data"].append(perf_counter() - tic)

    def get_file_list(self, list_name: str) -> list:
        """Get the list of files to read"""
        tic = perf_counter()
        if self.backend == "smartredis":
            # Ensure the list of DataSets is available
            while True:
                list_length = self.client.get_list_length(list_name)
                if list_length == 0:
                    sleep(1)
                    continue
                else:
                    break

            # Grab list of datasets
            file_list = self.client.get_datasets_from_list(list_name)
        self.timers["meta_data"].append(perf_counter() - tic)
        return file_list

    def get_file_list_length(self, list_name: str) -> int:
        """Get the length of the file list"""
        tic = perf_counter()
        if self.backend == "smartredis":
            list_length = self.client.get_list_length(list_name)
        self.timers["meta_data"].append(perf_counter() - tic)
        return list_length

    def _import_repartition(self):
        """The repartition package lives one level up so models can share it."""
        pkg_parent = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        )
        if pkg_parent not in sys.path:
            sys.path.insert(0, pkg_parent)
        from repartition import AdiosSource, Repartitioner

        return AdiosSource, Repartitioner

    def _open_bp_read(self, path: str):
        """Open a BP file for reading, tolerating a serial adios2 build."""
        self._import_repartition()
        from repartition import open_bp_read

        return open_bp_read(path, self.comm)

    def _wait_for_graph(self, path: str = "graph.bp", timeout: float = 600.0):
        """Block until nekRS has finished writing graph.bp.

        gnn.cpp writes the file in a single step and closes it, so waiting
        for the directory to appear is not enough -- the metadata may not be
        flushed yet. Opening it and requiring the variables we need is the
        cheapest reliable completion test.
        """
        tic = perf_counter()
        while True:
            if os.path.exists(path):
                try:
                    with self._open_bp_read(path) as stream:
                        stream.begin_step()
                        have = set(stream.available_variables())
                        stream.end_step()
                    if {"N", "num_edges", "pos_node"} <= have:
                        return
                except Exception:
                    pass
            if perf_counter() - tic > timeout:
                raise TimeoutError(
                    f"{path} did not become readable within {timeout:.0f}s"
                )
            sleep(1)

    def get_graph_data_from_stream(self, method: str = "rcb") -> dict:
        """Get the entire set of graph datasets from a stream.

        graph.bp is written by the nekRS ranks, so it has W writer blocks
        while this communicator has self.size readers. When the two agree
        each rank reads its own block directly (the original path). When
        they differ the graph is repartitioned onto this communicator by
        whole elements, which also fixes the routing used later for the
        in_u/out_u solution stream.
        """
        tic = perf_counter()
        graph_data = {}
        if self.backend == "adios":
            self._wait_for_graph("graph.bp")
            AdiosSource, Repartitioner = self._import_repartition()
            src = AdiosSource("graph.bp", comm=self.comm)
            self.graph_source = src

            # writer-side per-block metadata, kept for the solution stream
            self.N_list = [int(n) for n in src.n_per_src]
            self.num_edges_list = [int(e) for e in src.num_edges_per_src]
            self.fieldOffset_list = [int(f) for f in src.fo_per_src]
            graph_data["Np"] = src.Np

            if src.src_size == self.size:
                self.repart = None
                graph_data.update(self._read_own_graph_block(src))
            else:
                if self.rank == 0:
                    log.info(
                        "Repartitioning online graph from %d nekRS ranks to "
                        "%d ML ranks (method=%s)",
                        src.src_size,
                        self.size,
                        method,
                    )
                self.repart = Repartitioner(src, self.comm, method=method)
                arrs = self.repart.graph_arrays()
                graph_data["pos"] = arrs["pos"]
                graph_data["global_ids"] = arrs["global_ids"].reshape(-1)
                graph_data["local_unique_mask"] = arrs["local_unique_mask"]
                graph_data["halo_unique_mask"] = arrs["halo_unique_mask"]
                graph_data["edge_index"] = arrs["edge_index"].astype(np.int64).T
        self.timers["data"].append(perf_counter() - tic)
        return graph_data

    def _read_own_graph_block(self, src) -> dict:
        """Read writer block self.rank straight out of graph.bp (W == M).

        Within a block pos_node and edge_index are component-major, hence
        the order="F" reshapes; the masks and global_ids are plain per-node
        scalars.
        """
        r = self.rank
        n = self.N_list[r]
        e = self.num_edges_list[r]
        n_off = int(src.node_offsets[r])
        e_off = int(src.edge_offsets[r])
        out = {}
        with self._open_bp_read("graph.bp") as stream:
            stream.begin_step()
            out["pos"] = stream.read("pos_node", [n_off * 3], [n * 3]).reshape(
                (-1, 3), order="F"
            )
            out["edge_index"] = (
                stream
                .read("edge_index", [e_off * 2], [e * 2])
                .reshape((-1, 2), order="F")
                .T.astype(np.int64)
            )
            out["global_ids"] = stream.read("global_ids", [n_off], [n])
            out["local_unique_mask"] = stream.read(
                "local_unique_mask", [n_off], [n]
            )
            out["halo_unique_mask"] = stream.read(
                "halo_unique_mask", [n_off], [n]
            )
            stream.end_step()
        return out

    def _read_own_field_block(self, name: str, ncols: int) -> np.ndarray:
        """Read writer block self.rank of a solution variable (W == M).

        in_u/out_u are component-major with a per-writer stride of
        fieldOffset = alignStride(N), not N, so the N rows of each component
        must be read separately; a single contiguous N*ncols read silently
        picks up padding and shears the components whenever N % 32 != 0.
        """
        r = self.rank
        n = self.N_list[r]
        fo = self.fieldOffset_list[r]
        base = sum(self.fieldOffset_list[:r]) * ncols
        out = np.empty((n, ncols), dtype=np.float64)
        for c in range(ncols):
            out[:, c] = self.solutionStream.read(
                name, [base + c * fo], [n]
            ).reshape(-1)
        return out

    def get_train_data_from_stream(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the solution from a stream"""
        self.comm.Barrier()
        tic = perf_counter()
        if self.backend == "adios":
            if self.solutionStream is None:
                if self.rank == 0:
                    log.info("Opening ADIOS2 solutionStream ...")
                self.solutionStream = Stream(
                    self.client, "solutionStream", "r", self.comm
                )
            if self.N_list is None:
                raise RuntimeError(
                    "get_graph_data_from_stream() must run before "
                    "get_train_data_from_stream(): the solution stream is "
                    "laid out in the writer's blocks, whose sizes only "
                    "graph.bp announces"
                )

            # Status options are: bindings.StepStatus.OtherError, bindings.StepStatus.NotReady, bindings.StepStatus.EndOfStream, bindings.StepStatus.OK
            # status = self.solutionStream.step_status()

            self.solutionStream.begin_step()
            
            # stream.read() gets data now, Mode.Sync is default
            # see
            #   - https://github.com/ornladios/ADIOS2/blob/67f771b7a2f88ce59b6808cc4356159d86255f1d/python/adios2/stream.py#L331
            #   - https://github.com/ornladios/ADIOS2/blob/67f771b7a2f88ce59b6808cc4356159d86255f1d/python/adios2/engine.py#L123)
            ticc = perf_counter()
            if self.repart is not None:
                self.graph_source.attach_field_stream(self.solutionStream)
                inputs = self.repart.read_field("in_u", ncols=3)
                outputs = self.repart.read_field("out_u", ncols=3)
            else:
                inputs = self._read_own_field_block("in_u", 3)
                outputs = self._read_own_field_block("out_u", 3)
            transfer_time = perf_counter() - ticc

            self.solutionStream.end_step()
        self.timers["data"].append(perf_counter() - tic)
        return inputs, outputs, transfer_time

    def stop_nekRS(self) -> None:
        """Communicate to nekRS to stop running and exit cleanly"""
        MLrun = 0
        tic = perf_counter()
        if self.backend == "smartredis":
            if self.db_nodes == 1:
                if self.rank % self.local_size == 0:
                    self.put_array("check-run", np.int32(np.array([MLrun])))
            else:
                if self.rank == 0:
                    self.put_array("check-run", np.int32(np.array([MLrun])))
        elif self.backend == "adios":
            # Communicate to nekRS to stop
            with Stream(self._create_bp_io(), "check-run.bp", "w") as stream:
                if self.rank == 0:
                    stream.write("check-run", np.int32([MLrun]))

            # Close solution stream
            if self.solutionStream is not None:
                self.solutionStream.close()
        self.timers["meta_data"].append(perf_counter() - tic)
