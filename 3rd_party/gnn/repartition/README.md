# repartition — rank-count-independent nekRS graph partitioning

A model-agnostic Python package that decouples ML training/inference from
the rank count of the nekRS run that produced the mesh graph and data.
Any model that consumes nekRS's element-based GLL graph (dist-gnn today,
others tomorrow) can use it to run on an arbitrary number of MPI ranks.

**Core dependencies: numpy + mpi4py only.** (The dist-gnn halo-metadata
integration in the CLI additionally uses torch / torch_geometric and the
`dist-gnn` sources; it is optional and skippable.)

## What it does

nekRS's gnn plugin writes, per simulation rank, an element-based GLL graph:
nodes are the GLL points of whole elements (element-major, blocks of
`Np=(p+1)^3`), coincident copies share a `global_id`, and every edge lies
inside a single element. This package:

1. **Reads** that graph from any source, at any source rank count:
   - `BinSource` — the binary `gnn_outputs_poly_*` directories,
   - `FldSource` — plain nekRS `.f` field files: the graph is reconstructed
     from the mesh record's coordinates (distributed coincidence matching
     with optional periodic folding — verified to reproduce nekRS's
     `mesh->globalIds` classes exactly), and any field record (U, P, T,
     S##) becomes node data;
2. **Partitions** whole elements onto the current communicator
   (`block` contiguous split or `rcb` recursive coordinate bisection);
3. **Redistributes** elements with a single `Alltoallv`, keeping a reusable
   `Routing` so any node-level field laid out in source order (snapshots,
   trajectories, checkpoints) can be moved identically;
4. **Regenerates** the five per-rank arrays that define the distributed
   graph for the new partition: `pos_node`, `global_ids`, `edge_index`,
   `local_unique_mask`, `halo_unique_mask` — with semantics identical to
   what the nekRS plugin would have written had the simulation run on this
   communicator. Model-specific halo metadata is derivable from these
   (e.g. dist-gnn's `create_halo_info_par` works unchanged at the new
   size).

Design notes, correctness argument and validation record:
`doc/graph_repartitioning_plan.md` at the repo root.

## Python API (for any model)

```python
import sys; sys.path.insert(0, "<repo>/3rd_party/gnn")
from repartition import BinSource, Repartitioner
from repartition.fld import FldSource

# from gnn_outputs binaries (source rank count autodetected):
rp = Repartitioner(BinSource("gnn_outputs_poly_7"), comm, method="rcb")

# or purely from a .f file (periodic axes fold coincidence classes):
src = FldSource("case0.f00000", periodic=(True, True, True))
rp = Repartitioner(src, comm, method="rcb")

arrays = rp.graph_arrays()   # dict: pos (N,3) f8, global_ids (N,1) i8,
                             # edge_index (E,2) i4, local/halo masks (N,) i4
u = rp.read_field(spec, ncols=3)   # any node field, routed consistently:
                                   #  BinSource: spec = lambda s: path of
                                   #    source rank s's file
                                   #  FldSource: spec = (path, "U")
```

`arrays` and `read_field` outputs are node-consistent with each other:
row i of a routed field is the value at graph node i.

## CLI (materialize files on disk)

Run from `3rd_party/gnn` (or with it on `PYTHONPATH`):

```sh
# repartition an existing gnn_outputs dir (plus fld_* snapshots and/or a
# trajectory) to M ranks:
mpirun -n M python -m repartition.cli \
    --src-dir gnn_outputs_poly_7 --out-dir gnn_outputs_poly_7_M \
    --method rcb --fld \
    --traj-dir traj_poly_7/tinit_0.000000_dtfactor_10 --traj-out traj_M

# reconstruct graph + training data purely from .f checkpoint files:
mpirun -n M python -m repartition.cli \
    --fld-mesh case0.f00000 --out-dir gnn_from_fld --periodic xyz
# time-dependent (u_step_* trajectory from a sequence of .f files):
mpirun -n M python -m repartition.cli \
    --fld-mesh case0.f00000 --out-dir gnn_from_fld --periodic xyz \
    --fld-traj case0.f0000{0..5} --fld-traj-out traj_from_fld
```

The CLI always writes the five arrays + `Np` file named
`*_rank_r_size_M`. By default it also writes dist-gnn's three halo `.npy`
files (`halo_info`, `node_degree`, `edge_weights`) so dist-gnn training
runs unchanged; pass `--no-halo` for other models.

dist-gnn can alternatively skip the CLI entirely: its trainer autodetects
a rank-count mismatch in `gnn_outputs_path` and repartitions in memory
(config keys `gnn_outputs_size`, `repartition_method`).

## Tests

```sh
cd 3rd_party/gnn
python repartition/tests/gen_synthetic.py --out /tmp/synth \
    --nex 4 --ney 3 --nez 2 --poly 3 --src-size 4
mpirun -n M python repartition/tests/test_consistency.py \
    --src /tmp/synth --method rcb    # any M
```

The MPI test checks mask invariants, global reduced-edge-set invariance,
node-degree accounting, and one full halo-consistent aggregation round
against a serial reference. End-to-end: the `tgv_gnn_offline`,
`tgv_gnn_offline_traj` and `tgv_gnn_offline_fld` examples reproduce their
ReFrame `target_loss` values when trained at any rank count from
repartitioned or `.f`-only data.
