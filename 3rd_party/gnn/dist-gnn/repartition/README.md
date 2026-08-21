# repartition — rank-count-independent graph partitioning for dist-gnn

Decouples dist-gnn training/inference from the rank count of the nekRS run
that produced the graph and data. Whole spectral elements are redistributed
across the current MPI communicator and the five per-rank arrays the
trainer consumes (`pos_node`, `global_ids`, `edge_index`,
`local_unique_mask`, `halo_unique_mask`) are regenerated for the new
partition; the existing halo machinery (`create_halo_info_par`) then works
unchanged at the new size. Design + validation record:
`doc/graph_repartitioning_plan.md` at the repo root.

## Usage

Repartition an existing `gnn_outputs` directory (plus `fld_*` snapshots
and/or a trajectory) to M ranks:

```sh
mpirun -n M python -m repartition.cli \
    --src-dir gnn_outputs_poly_7 --out-dir gnn_outputs_poly_7_M \
    --method rcb --fld \
    --traj-dir traj_poly_7/tinit_0.000000_dtfactor_10 --traj-out traj_M
```

Reconstruct graph + training data purely from nekRS `.f` checkpoint files
(no gnn_outputs / traj needed; write checkpoints with
`nrs->writeCheckpoint(time, tstep, true, true)`):

```sh
# time-independent (fld_u / fld_p snapshots from the U/P records):
mpirun -n M python -m repartition.cli \
    --fld-mesh tgv0.f00000 --out-dir gnn_from_fld --periodic xyz

# time-dependent (u_step_* trajectory from a sequence of .f files):
mpirun -n M python -m repartition.cli \
    --fld-mesh tgv0.f00000 --out-dir gnn_from_fld --periodic xyz \
    --fld-traj tgv0.f0000{0..5} --fld-traj-out traj_from_fld
```

`--periodic xyz` folds periodic images into one coincidence class when
assigning global ids from coordinates (required for periodic cases like
TGV to reproduce nekRS's topology-aware numbering exactly).

Then train/infer as usual with `gnn_outputs_path` (and `traj_data_path`)
pointing at the new directories — no trainer changes needed.

In-memory API (used by the CLI, available for direct wiring):

```python
from repartition import BinSource, Repartitioner
rp = Repartitioner(BinSource("gnn_outputs_poly_7"), comm, method="rcb")
arrays = rp.graph_arrays()          # the five trainer arrays
u = rp.read_field(path_fn, ncols=3) # any node field, routed consistently
```

## Tests

```sh
python repartition/tests/gen_synthetic.py --out /tmp/synth \
    --nex 4 --ney 3 --nez 2 --poly 3 --src-size 4
mpirun -n M python repartition/tests/test_consistency.py \
    --src /tmp/synth --method rcb    # any M
```

The MPI test checks mask invariants, global reduced-edge-set invariance,
node-degree accounting, and one full halo-consistent aggregation round
against a serial reference. End-to-end: the `tgv_gnn_offline` /
`tgv_gnn_offline_traj` examples reproduce their ReFrame `target_loss`
values when trained at any rank count from repartitioned or `.f`-only
data (see the plan doc's progress log for the exact commands).
