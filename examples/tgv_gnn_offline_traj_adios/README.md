# TGV offline training with ADIOS2 BP5 data exchange

Taylor-Green vortex case that hands both the **graph** and the **training
data** to dist-gnn through ADIOS2 BP5 files, then trains offline.

This is the file-based counterpart to `tgv_gnn_online_traj_adios`: same ADIOS2
layout, but no SST transport and no live reader. It replaces the two
non-scalable offline paths — the one-file-per-rank binaries of
`tgv_gnn_offline_traj` and the `fld -> gnn_outputs` conversion of
`tgv_gnn_offline_fld`.

## What nekRS writes

| file | written by | contents |
|---|---|---|
| `graph.bp` | `graph->gnnWriteADIOS(client)` | the five graph arrays plus the per-writer `N`/`num_edges`/`field_offset` manifest |
| `trainingData.bp` | `tgen->trajGenWriteBP(...)` | one ADIOS step per snapshot: `u` (padded component-major) and the scalars `tstep` and `time` |

`gnnWriteADIOS` must run first — it is the sole setter of the client field
offsets and the sole writer of the block manifest that the reader needs to
interpret `trainingData.bp`.

Unlike the SST path, only the current snapshot is written per step; the Dist-GNN trainer
pairs consecutive steps, similarly to the POSIX
`u_step_<tstep>.bin` files.

`trajGenWriteBP` takes the same arguments and the same `field_name` vocabulary
(`"velocity"`, `"pressure"`, `"all"`) as `trajGenWrite`, `trajGenWriteDB` and
`trajGenWriteSST`, so switching a case between transports is a one-word edit in
the UDF. A case needing fields outside that vocabulary calls
`graph->writeToFileBP(nrs, client, fields, time, tstep)` directly with its own
`std::vector<bpField_t>`.

## Pipeline

The trainer selects the reader from the shape of the paths it is given: a
`gnn_outputs_path` ending in `.bp` is read through `AdiosSource` and
repartitioned onto the ML rank count in memory, and a `traj_data_path` ending in
`.bp` has its `(x, y)` pairs built by walking the ADIOS steps. Nothing is
written to disk in between.

```bash
# 1. nekRS at SIM_RANKS=2 -> graph.bp + trainingData.bp
# 2. train at ML_RANKS=4 
mpiexec -n $ML_RANKS python .../dist-gnn/main.py \
    gnn_outputs_path=$PWD/graph.bp traj_data_path=$PWD/trainingData.bp \
    time_dependency=time_dependent target_loss=6.6139e-01
```

A BP5 graph is always repartitioned, at every ML rank count: `graph.bp` is
written in the nekRS *writers'* blocks, and even at `ML_RANKS == SIM_RANKS` the
partitioner does not reproduce the writer's element assignment, so there is no
native layout to fall back on. `gnn_outputs_size` is neither needed nor
consulted here — the writer count comes out of the file itself.

### Alternative: materialize a POSIX tree first

`repartition.cli` converts the same two files into the `gnn_outputs` /
`traj_poly` tree the `.bin` trainer path globs. It is the slower route — it
writes the intermediate arrays to disk and reads them back — but it is useful
for inspecting those arrays, and for a trainer build without ADIOS2:

```bash
mpiexec -n $ML_RANKS python -m repartition.cli \
    --graph-bp ./graph.bp --train-bp ./trainingData.bp \
    --train-bp-mode traj \
    --out-dir ./gnn_outputs_poly_7 --traj-out ./traj_poly_7 --method parrsb
mpiexec -n $ML_RANKS python .../dist-gnn/main.py \
    gnn_outputs_path=$PWD/gnn_outputs_poly_7 traj_data_path=$PWD/traj_poly_7 \
    time_dependency=time_dependent target_loss=6.6139e-01
```

Both routes repartition with the same code and reach the same loss.

## Running

```bash
./gen_run_script <system> $NEKRS_HOME 
qsub run.sh
```

`SIM_RANKS_PER_NODE` and `ML_RANKS_PER_NODE` are independent; set them
differently to move data across a rank-count change. Both settings go through
the partitioner (see above), so the difference is in how much data moves, not in
which code runs. The training loss is rank-invariant either way, so
`target_loss=6.6139e-01` is the acceptance gate in both.
