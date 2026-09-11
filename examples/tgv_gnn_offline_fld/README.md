# Offline GNN training from nekRS .f checkpoint files only

This example demonstrates the nekRS → dist-gnn training pipeline **through
standard .f field files**, with no coupling between the simulation and the
ML side: nekRS writes a single `.f` checkpoint (no `gnn` plugin, no
`gnn_outputs` or trajectory directories), and the graph and training data
are reconstructed from that file by the
[`repartition`](../../3rd_party/gnn/repartition/README.md) package — **at
an arbitrary number of training ranks**, independent of how many ranks
nekRS ran on.

It is the `.f`-file counterpart of [tgv_gnn_offline](../tgv_gnn_offline):
the same Taylor-Green-Vortex case, the same time-independent modeling task
(predict pressure from the three velocity components at every mesh node),
and the same training target loss (2.7161e-04 after 100 iterations),
regardless of the rank count used for nekRS or for training. Because this
works on any archived `.f` files, the same pipeline can train models on
past simulation data without rerunning nekRS.

How it works:

- `UDF_ExecuteStep()` calls `nrs->writeCheckpoint(time, tstep, true, true)`
  which writes `tgv0.f00000` containing the mesh coordinates (X), velocity
  (U) and pressure (P) records in double precision. That is the only
  output nekRS produces here.
- The `repartition` CLI reads the mesh record, reassigns global node ids by
  coordinate coincidence matching (`--periodic xyz` folds the periodic
  images into one class, reproducing exactly the topology-aware numbering
  nekRS builds from the periodic BCs), partitions whole spectral elements
  onto the training communicator, and writes the graph arrays, the
  dist-gnn halo files, and the `fld_u`/`fld_p` training snapshots from the
  U/P records.
- dist-gnn training then runs unchanged on the reconstructed directory.

## Building nekRS

See [tgv_gnn_offline](../tgv_gnn_offline/README.md#building-nekrs) — the
requirements and build scripts are identical.

## Running the example

**From a compute node** execute:
```sh
./gen_run_script <system_name> </path/to/nekRS> [--venv_path </path/to/venv>]
```
and then run the generated script:
```sh
./run.sh
```

The `run.sh` script is composed of three steps:

- **nekRS** (on `SIM_RANKS_PER_NODE` ranks, default 2) evaluates the TGV
  initial condition and writes the single checkpoint `tgv0.f00000`.
- **Graph + data reconstruction** (on `ML_RANKS_PER_NODE` ranks, default 4
  — deliberately different from the nekRS rank count) via the repartition
  CLI, producing `./gnn_from_fld`.
- **GNN training** on the same `ML_RANKS_PER_NODE` ranks; the final loss
  is validated against the target 2.7161e-04.

## Running the pipeline manually

The three steps, spelled out (any `M` works for the ML steps):

```sh
# 1. nekRS writes tgv0.f00000 (N ranks, any N)
mpiexec -n N $NEKRS_HOME/bin/nekrs --setup tgv.par --backend <backend>

# 2. reconstruct graph + training data at M ranks (any M)
export PYTHONPATH=$NEKRS_HOME/3rd_party/gnn:$PYTHONPATH
mpiexec -n M python -m repartition.cli \
    --fld-mesh tgv0.f00000 --out-dir ./gnn_from_fld \
    --method rcb --periodic xyz

# 3. train at the same M ranks
mpiexec -n M python $NEKRS_HOME/3rd_party/gnn/dist-gnn/main.py \
    halo_swap_mode=all_to_all_opt layer_norm=True \
    gnn_outputs_path=$PWD/gnn_from_fld target_loss=2.7161e-04 \
    transform_x=true transform_y=true transform_z=true
```

Notes:

- `--periodic xyz` is required for this fully periodic case; for
  non-periodic cases omit it (or list only the periodic axes).
- Time-dependent training from a sequence of `.f` files works the same way
  with `--fld-traj <files...> --fld-traj-out <dir>`; see the
  [repartition README](../../3rd_party/gnn/repartition/README.md).
- Design notes and the validation record for the reconstruction machinery
  are in [doc/graph_repartitioning_plan.md](../../doc/graph_repartitioning_plan.md).
