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
| `trainingData.bp` | `tgen->trajGenWriteBP(...)` | one ADIOS step per snapshot: `u` (padded component-major) and the scalar `tstep` |

`gnnWriteADIOS` must run first — it is the sole setter of the client field
offsets and the sole writer of the block manifest that the reader needs to
interpret `trainingData.bp`.

Unlike the SST path, only the current snapshot is written per step; the reader
pairs consecutive steps, exactly as dist-gnn does for the POSIX
`u_step_<tstep>.bin` files.

## Pipeline

```bash
# 1. nekRS at SIM_RANKS -> graph.bp + trainingData.bp
# 2. repartition to ML_RANKS (may differ from SIM_RANKS)
mpiexec -n $ML_RANKS python -m repartition.cli \
    --graph-bp ./graph.bp --train-bp ./trainingData.bp \
    --train-bp-mode traj \
    --out-dir ./gnn_from_bp --traj-out ./traj_from_bp --method parrsb
# 3. train
mpiexec -n $ML_RANKS python .../dist-gnn/main.py \
    gnn_outputs_path=$PWD/gnn_from_bp traj_data_path=$PWD/traj_from_bp \
    time_dependency=time_dependent target_loss=6.6139e-01
```

Step 2 materializes the exact filenames the trainer already globs, so dist-gnn
itself is unchanged and `gnn_outputs_size` stays 0.

## Running

```bash
./gen_run_script <system> $NEKRS_HOME --venv_path <venv>
SIM_RANKS_PER_NODE=2 ML_RANKS_PER_NODE=4 ./nrsrun_aurora <nodes>
qsub run.sh
```

Setting `SIM_RANKS_PER_NODE != ML_RANKS_PER_NODE` exercises the repartitioner;
setting them equal exercises the pass-through case. The training loss is
rank-invariant, so `target_loss=6.6139e-01` is the acceptance gate in both.
