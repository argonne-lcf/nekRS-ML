# Plan: decouple dist-gnn rank count from nekRS via Python-side graph repartitioning

Date: 2026-08-21. Status: living document — tracks design, verified facts, task breakdown, and progress.

## Problem

Training and inference of the dist-gnn model are hard-bound to run on exactly the same
number of ranks as the nekRS simulation that produced the graph (`gnn_outputs_poly_*`,
`graph.bp`) and the training data. In the shooting workflow this forces inference onto
half the available GPUs. More generally it prevents training on archived data at an
arbitrary rank count.

**Scope boundary (set 2026-09-09).** This work covers the forward direction only:

    nekRS (N/2 nodes) --> online training (N/2 nodes) --> inference (N nodes)

Everything nekRS produces -- `graph.bp`, the `in_u`/`out_u` SST stream,
`checkpoint.bp` -- is repartitioned on read, so the ML side runs at any rank count.
The return direction, inference --> nekRS, is **explicitly deferred**: the rolled-out
solution in `checkpt_u.bp` is written but nothing reads it back, and closing that loop
is not required for this change to be done. See "Deferred: the inference --> nekRS
return path" below for what it would take.

## Verified facts the design rests on (from source reading)

1. **Graph nodes are the GLL points of whole elements**, stored element-major in blocks
   of `Np=(p+1)^3` per element (`src/plugins/gnn.cpp:117`, `gnn_connectivity.cpp:48-50`).
2. **Every edge lives inside a single element.** The GLL stencil (`get_graph_nodes`,
   `gnn_connectivity.cpp:75-824`) and the optional multiscale p1 edges
   (`add_p1_neighbors`) never cross element boundaries. The "coincident-copy
   augmentation" edges (`gnn.cpp:495-594, 665-769`) are *derivable* from global ids
   (they union the neighbor lists of same-gid copies on a rank). Therefore **moving
   whole elements moves the entire graph**, and intra-element edges of any one element
   block are a template identical for all elements.
3. **`global_ids` (nek `glo_num`) are partition-independent**: coincident GLL points
   share a gid across elements and ranks. Only element-*surface* lattice points can be
   coincident; element-interior points have globally unique gids.
4. **Both masks are pure functions of (gid, partition)** (`gnn.cpp:358-773`):
   - `local_unique_mask[i]=1` iff node i is the first on-rank copy (min local id) of a
     gid whose copies are all on this rank (incl. never-shared nodes).
   - `halo_unique_mask[i]=1` iff node i is the first on-rank copy of a gid also present
     on ≥1 other rank. Every sharing rank marks one copy (per-rank representative, not
     a global owner).
5. **All downstream halo machinery is already pure Python** and consumes only the five
   per-rank arrays `pos, global_ids, edge_index, local_unique_mask, halo_unique_mask`:
   `create_halo_info_par.py` builds `halo_info` / `node_degree` / `edge_weights` from
   gids via Allgatherv; the trainer's online path even recomputes them on the fly
   (`trainer.py:989-1034`). Loss/stats weight nodes by `1/node_degree`; message passing
   halo-swaps edge aggregates and `index_add_`s them into owned copies — all driven by
   `halo_info`, which is derived from gids at the *current* world size.
6. **ADIOS `graph.bp` is a global concatenation** of per-writer-rank blocks, plus
   per-writer-rank `N` and `num_edges` arrays (`gnn.cpp:250-332`). A reader at any world
   size can slice it element-aligned; writer size is discoverable from `shape("N")`.
7. **parRSB is wrappable**: `parrsb_part_mesh(part, vtx, xyz, tag, nel, nv, opts, comm)`
   takes per-element corner-vertex global ids (8 for hex) + coords, returns dest rank
   per element; C API, `-fPIC` objects already built; comm passable from mpi4py. The
   corner vertices of each element are the GLL lattice corners, whose gids/coords we
   already have.
8. **.f files are element-granular and partition-independent** (132B ASCII header +
   endian probe + int32 global-element-id map + per-element field blocks with
   components interleaved per element). Reading them on M ranks needs no gslib — an
   `Alltoallv` keyed by destination rank replaces `sarray_transfer`
   (`3rd_party/nek5000/core/ic.f:1933-2210, 2581-2699`).

**Conclusion: no C/gslib wrapping is required for correctness.** Repartitioning =
(a) read elements from any source, (b) choose a destination rank per element,
(c) `Alltoallv` element blocks, (d) regenerate the five arrays, (e) let the existing
halo pipeline run. parRSB wrapping is an optional quality upgrade for step (b).

## Every rank-coupling that must be broken (verified list)

- File/key names embed `rank_{RANK}_size_{SIZE}` throughout `trainer.py`
  (graph: 751-774; halo: 946-957; traj: 1321, 1507-1517).
- ADIOS reads `N`/`num_edges` at offset `[self.rank]` (`client.py:169,173`) and slices
  1:1 writer-block→reader-rank; `get_array` splits `checkpoint.bp` `shape[0]/size`
  component-blind (`client.py:104-114`).
- Model checkpoint filename embeds SIZE: `name="...SIZE_%d_SEED_%d"` (`trainer.py:369,389`
  via `get_save_header`) — weights themselves are size-agnostic.
- `driver.py:156-159` launches inference with `mlprocs/ml_cpu_bind`;
  `nrsrun_aurora` emits no separate inference keys.
- halo buffers indexed by absolute rank id (`trainer.py:503-509`) — fine, they're built
  at current size once the five arrays are right.

## Design

New package `3rd_party/gnn/dist-gnn/repartition/`:

```
repartition/
  __init__.py      # repartition_graph(), repartition_fields(), ElementMap
  sources.py       # element-block readers:
                   #   BinSource      gnn_outputs_poly_* at any source size
                   #   AdiosSource    graph.bp / checkpoint.bp / solutionStream (phase 2)
                   #   FldSource      nekRS .f files (phase 3)
  partition.py     # dest-rank per element: 'block' | 'rcb' | 'parrsb'
  redistribute.py  # mpi4py Alltoallv of element records; deterministic receiver order
  rebuild.py       # edge template extraction + rep-mapped edges + mask regeneration
  cli.py           # mpirun -n M python -m repartition.cli --src gnn_outputs_poly_7
                   #   --src-size N [--out ...]   -> writes *_rank_r_size_M.* files
```

Key algorithms:
- **Element identity**: global ordinal = position in the concatenation of source ranks'
  element lists (source rank major). Deterministic; carried through redistribution so
  receiver sorts by ordinal. (True nek global element ids only needed for .f matching —
  phase 3 adds a tiny `gnn.cpp` write of `mesh global element ids` for that.)
- **Edge template**: intra-block edges of element 0 of source rank 0 (both endpoints in
  `[0,Np)`); by fact (2) this is the per-element stencil incl. multiscale edges, and
  augmentation edges never appear intra-block. Broadcast; regenerate all edges as
  `template + e*Np`. Then map every endpoint through `rep[i]` = min-local-id on-rank
  copy of `gid[i]`, dedup, drop self-loops. After `get_reduced_graph` this yields
  connectivity identical to nekRS's augmentation (proof: reduction keeps exactly the
  representatives; nekRS's unioned neighbor lists collapse to the same rep-rep edges).
- **Cross-rank sharing detection** (replaces `ogsHostGatherScatter` min/max +
  `ogsGsUnique`): rendezvous hash. Each rank sends its unique *surface-candidate* gids
  to home rank `gid % M`; home ranks count distinct source ranks per gid; flag
  `shared` returned to all senders. Masks then set per fact (4).
- **Field redistribution**: same ElementMap applied to any `(N_src_total, k)` array
  (traj snapshots, checkpoint); sources handle fieldOffset padding (truncate to
  `Ne*Np` per writer block) and component-major `checkpoint.bp` layout.

Integration:
- **Offline**: `cfg.gnn_outputs_size` (0 = autodetect from filenames). If != SIZE,
  `load_graph_data` routes through the repartition API and trajectory loading uses
  `repartition_fields`. Alternative zero-code-change path: run `repartition/cli.py`
  once to materialize size-M files, existing pipeline untouched. Both supported.
- **Online ADIOS**: `get_graph_data_from_stream` reads writer size from `shape("N")`,
  reads element-aligned slices, repartitions in memory; `in_u/out_u/checkpoint` reads
  use the ElementMap. This unlocks inference on 24 ranks in the shooting workflow AND
  training at any rank count online.
- **Checkpoint naming**: drop SIZE from `get_save_header` (keep a compat fallback that
  tries the old name on load).
- **Driver**: `nrsrun_aurora` emits `inferprocs/inferprocs_pn/infer_cpu_bind/infer_nodes`;
  `driver.py:launchInference` uses them (default 2x mlprocs over sim+train nodes).

## Validation (user requirement: existing ReFrame loss tests are the acceptance gate)

The ReFrame tests (`tests/tests.py`) already assert a rank-invariant `target_loss`
(e.g. `TGVOffline` at rpn 1/2/4 all check 2.7161e-04; `TGVOfflineTraj` 6.6139e-01 at
1/2/4) — dist-gnn's consistency machinery makes the loss independent of partitioning.
Everything below must reproduce those same losses.

1. **Synthetic unit test (no nekRS, CPU)**: generate an Nx×Ny×Nz hex box mesh with GLL
   lattices per element, exact integer gids from the conforming global lattice; write
   source files at size S; `mpirun -n M` repartition for M in {1,2,3,8}; assert
   (a) the gid-canonicalized global reduced edge set is invariant,
   (b) Σ_ranks Σ_nodes 1/node_degree = global unique node count,
   (c) one round of edge-aggregate + halo swap + index_add (the exact model consistency
       operation, using `create_halo_info_par` outputs at size M) matches the serial
       global-graph reference per gid to fp64 tolerance,
   (d) redistributed fields equal f(pos) locally.
2. **Existing examples / ReFrame must keep passing unchanged** (pure passthrough when
   source size == SIZE): run `tgv_gnn_offline` (and `_traj`) pipelines locally at 1/2/4
   ranks, losses at their targets.
3. **Repartitioned-graph loss test**: nekRS `tgv_gnn_offline` at S=4 → train at
   M ∈ {1,2,3,6,8} from the repartitioned graph/data → same target_loss (rel 1e-3).
4. **.f-only reconstruction (user-requested gold test, promoted from phase 3)**:
   - `tgv_gnn_offline` with nekRS writing ONLY .f checkpoint files (no gnn_outputs, no
     traj): Python reconstructs the graph (GLL coords from the mesh record; gids by
     exact/tolerance coordinate matching via rendezvous spatial hash; template stencil
     generated from the header's nx) and the training data (velocity/pressure records)
     from .f files, trains at arbitrary M, and hits the same target_loss.
   - Then `tgv_gnn_offline_traj` the same way (sequence of .f files as the trajectory).
5. **Aurora**: shooting workflow with inference on 24 ranks (user-run).

## Task breakdown & model assignment

Phase 1 (Fable, this session): core `repartition/` package (BinSource, block+rcb
partitioners, redistribute, rebuild), synthetic test harness, local mpirun validation,
CLI, trainer offline wiring. Hardest correctness-critical code done by the strongest model.

Phase 2 (Opus 4.7, next sessions — each item is self-contained). Phase 1
delivered and validated: core package, CLI materialization (bin dirs, fld
snapshots, trajectories, .f-only reconstruction incl. periodic folding), and
all offline loss-equality tests. Remaining:

1. **Online ADIOS path — the 24-GPU shooting-workflow inference** (the
   original motivating feature). **READER SIDE DONE 2026-08-26 (commit
   29914874)** — see the Progress log entry and the HPC handoff section at
   the end of this doc; validated only against a synthetic BP fixture, NOT
   yet against real nekRS output. The driver-knob sub-item below is still
   open. IMPLEMENTATION SPEC, verified against the writers by a reader
   agent 2026-08-21 (file:line refs checked at HEAD), retained as the
   layout reference:

   **Writer layouts (the ground truth the reader must honor):**
   - `graph.bp` (`gnn.cpp:249-330`, written on `_write_io` which has NO
     SetEngine — graph.bp and checkpoint.bp are ALWAYS BP files, never
     SST, regardless of `[ML] adiosEngine`; only `solutionStream` is SST):
     1-D global concatenations of per-writer blocks. Variables: `N` and
     `num_edges` (shape {W}, one int32 per writer — `shape("N")[0]` is
     the ONLY writer-size announcement), `pos_node` (float64,
     COMPONENT-major per block: [x(0..N-1), y(...), z(...)] — unlike the
     row-major .bin files), `global_ids` (int64), the two masks (int32),
     `edge_index` (int32, component-major per block [all nei, all own],
     block-LOCAL node ids), `Np` (int32, scalar; note defect: declared
     start {1} not {0} — CORRECTION, verified 2026-08-26: reading it
     without a selection returns **0**, not Np. BP5 defaults the selection
     to start=0/count=Shape, the block at start 1 fails the intersection
     test (`BP5Deserializer.cpp:1476`), no read is issued, and the Python
     binding returns its pre-zeroed buffer (`stream.py:379-380`). Only an
     out-of-bounds `[1],[1]` selection returns the value. `client.py:166`
     `int(stream.read("Np"))` therefore yields 0 today, silently.
     `AdiosSource._resolve_np` tries both selections and takes the
     candidate that is positive and divides every N_w, so it also keeps
     working if the writer is ever fixed. graph.bp has NO padding:
     N_w = Ne_w * Np exactly.
   - `in_u`/`out_u` (`trajGen.cpp:218-279`, SST `solutionStream`): per
     writer block `3 * fieldOffset_w` float64, component-major
     [u(0..fo-1), v, w], where fieldOffset_w = alignStride(N_w) =
     ceil(N_w/32)*32 (256B/8B alignment, `nekrsSys.hpp.in:167-175`) — up
     to 31 trailing pad doubles PER COMPONENT. Global block offsets use
     fieldOffset strides, NOT N. No step/time variable — the ADIOS step
     counter is the only cadence signal.
   - `checkpoint.bp` (`adiosStreamer.cpp:155-172`): one variable
     `checkpoint`, per-block `3 * fieldOffset_w` component-major —
     the COARSE GNN-mesh fieldOffset, globally offset by the true
     per-writer scan. Layout-identical to `in_u`/`out_u`, so one reader
     serves both. (This changed in `b4ce824b`; see Corrected above.)

   **AdiosSource (new, in repartition/):** mirror BinSource's surface —
   `Np`, `src_size` (= W = shape("N")[0]), `read_elements(comm)`,
   `read_node_field(...)`. `ne_per_src = N_list // Np` (exact),
   el_offsets = cumsum. `my_ordinal_range` / `_overlaps` are
   source-agnostic — lift them into a shared helper/base instead of
   copying. KEY DIFFERENCE vs BinSource: blocks are component-major, so
   an element-range slice of pos_node (or a field) is THREE disjoint
   sub-reads per overlapped block (x at blockstart+el0*Np, y at +N_s,
   z at +2*N_s) stacked into (nrows, 3) — the single-offset `_read_slice`
   shortcut does NOT carry over. Template: from writer block 0's
   edge_index, reshape order="F" first, then the same
   `(ei[:,0]<Np)&(ei[:,1]<Np)` filter. For in_u/out_u reads use
   fieldOffset_w strides (fixes the confirmed client.py padding bug: it
   slices by N_list — wrong start for every rank>0 and u/v/w component
   mixing whenever N_w % 32 != 0). `Repartitioner.read_field` takes a
   path-per-src-rank callable; generalize so AdiosSource can accept a
   variable name / open Stream instead.

   **Wiring:** `trainer.py:740` gates `_maybe_repartition_graph` behind
   `if not self.cfg.online:` — move the gate and add the AdiosSource
   branch next to the BinSource import (trainer.py:849). The halo-info
   on-the-fly path already works unchanged: under adios,
   `client.file_exists` returns None (client.py:81-86 has no adios
   branch) so `setup_halo` (trainer.py:1002-1143) always computes
   halo_info/node_degree/edge_weights from gids at the current M.
   `get_graph_data_from_stream` (client.py:150-216) currently reads
   `N[[rank]]` — the hard M==W coupling; replace its body with
   AdiosSource+Repartitioner. `get_train_data_from_stream`
   (client.py:218-254): keep the persistent SST stream, replace the
   slicing with AdiosSource block reads + `routing.route_node_array`.
   `get_array`/checkpoint (client.py:104-115 + trainer.py:1591): the
   naive shape/size split ALSO reshapes C-order against a
   component-major writer (existing bug) — replace with block-aware
   reads. The fine-vs-coarse mismatch noted here does not arise: post
   `b4ce824b` the checkpoint is interpolated to the coarse GNN mesh
   before it is written (see Corrected), so it routes with the graph.
   ALSO: `client.put_array` is a no-op under adios and
   `inference.py:313` pushes the rollout result through it — the adios
   shooting loop currently DROPS the inference result; an adios return
   path (e.g. a BP write mirroring check-run.bp, adiosStreamer.cpp:81-119
   reads it) is part of this task.
   - ~~`trainer.py:369/389` save_header embeds SIZE~~ DONE 2026-08-21:
     model name is now `POLY_%d_SEED_%d` (no SIZE); the restart load
     resolves legacy `POLY_p_SIZE_S_SEED_s` checkpoints (any S) via a
     glob fallback (`_resolve_legacy_ckpt`). Verified: py_compile + a
     filename-shape micro-test of the fallback (get_save_header appends
     the non-name input_dict values once; name is not repeated). NOT yet
     exercised end-to-end at M != S — do that with the inference smoke.
   - `driver.py:150-183 launchInference` + `nrsrun_aurora`: add
     `inferprocs`, `inferprocs_pn`, `infer_cpu_bind`, `infer_nodes` config
     keys (default: sim_nodes+train_nodes, 2x mlprocs) and use them.
   **Driver knobs:** `driver.py:150-183 launchInference` reuses
   mlprocs/mlprocs_pn/ml_cpu_bind/inference_nodes verbatim — add
   `inferprocs`, `inferprocs_pn`, `infer_cpu_bind`, `infer_nodes`
   (default: sim_nodes+train_nodes, 2x mlprocs); touch `assignNodes`
   (driver.py:64-88) and the nrsrun config generators. Latent script
   bugs to fix in passing: shooting nrsrun_aurora emits
   `ml_nodes: ${SIM_NODES}` (should be TRAIN_NODES) and defines
   INFERENCE_CPU_BIND_LIST without ever using it.

   **Local testing (verified feasible):** the venv now has the serial
   pip adios2 wheel 2.12.1 (installed 2026-08-21; per-rank BP-file reads
   work without MPI-adios2). No graph.bp exists on this machine and the
   local nekRS build has -DENABLE_ADIOS=OFF (BuildMeOnLocal:72), so the
   cheap fixture is a small Python BP writer that replicates the exact
   graph.bp / in_u / checkpoint layouts above from the in-repo ref dir
   `examples/tgv_gnn_offline_traj/ref/gnn_outputs_poly_7` (component-
   major, alignStride padding, W=4) — then assert AdiosSource-at-M
   arrays equal BinSource-at-M arrays for M in {1,2,3,8}. That
   equivalence test is the core acceptance gate; SST end-to-end needs
   an MPI-enabled adios2 (Aurora) or a local rebuild with ADIOS ON.

   Acceptance: AdiosSource==BinSource fixture test locally; then
   shooting workflow on 2 nodes with inference on all GPUs; SST smoke
   (nekRS 2 ranks, train 2, infer 4) on a machine with MPI adios2.
2. **Trainer in-memory wiring (offline)**: optional convenience so users
   skip the CLI: cfg keys `gnn_outputs_size` (0=autodetect via
   `BinSource.detect_size`) + `repartition_method`; in `load_graph_data`
   offline branch, route through Repartitioner when source size != SIZE;
   compute halo info on the fly (reuse online else-branch); route
   `load_field_data` / `load_trajectory` / `load_initial_condition` reads
   through `Repartitioner.read_field`. Acceptance: same losses as the CLI
   path on tgv_gnn_offline(_traj) without materializing files.
3. **ReFrame tests**: add parameterized variants that (a) repartition the
   nekRS output to a different rank count via the CLI and train
   (target_loss unchanged), (b) run the .f-only pipeline. Mirror the local
   commands recorded in the Progress log.
4. **parRSB wrapper** — DONE on macOS (see Progress log 2026-08-21,
   parRSB entry) via a C shim (`repartition/parrsb_shim.c`) + ctypes
   (`repartition/parrsb.py`), `method="parrsb"` in partition.py, CLI
   choice, quality metric `tests/partition_quality.py`. Remaining
   hand-off items for this task:
   a. **CMake integration**: DONE (see Progress log 2026-08-26). The shim
      is now a normal `parrsb_shim` SHARED target built by the nekRS build
      and installed with the package; `build_parrsb_shim.sh` is kept only
      as the standalone/ReFrame path.
   b. **HPC validation (Aurora/Polaris/Crux)**: `-fPIC` is already passed
      to the nek5000/parRSB builds by cmake/nek5000.cmake, so linking the
      shared shim should work; `build_parrsb_shim.sh` honors `MPICC` for
      Cray `cc`. Run the synthetic consistency matrix and
      partition_quality at a few hundred ranks; confirm no rank-0 memory
      spike (parRSB is fully distributed, unlike our RCB).
   c. **ReFrame**: DONE (local instantiation) — `TGVOfflineRepart` now
      parameterized over `repart_method ∈ {rcb, parrsb}`; for parrsb the
      test builds the shim into the stage dir via the installed
      `build_parrsb_shim.sh` (the CMake `install(DIRECTORY ...)` rule
      ships the .c/.sh with the package) and exports `PARRSB_SHIM_LIB`,
      so it does NOT depend on item a. Not yet run on ALCF CI; if the
      Cray wrapper isn't `mpicc`, export `MPICC=cc` in the test env.
   d. Optional: distributed edge-cut metric (partition_quality.py gathers
      (gid, rank) pairs to rank 0 — fine at test scale only).
5. **New example** `tgv_gnn_offline_fld` (or README section): the .f-only
   workflow the user requested, wired with the CLI commands from the
   Progress log; udf writes .f checkpoints only (writeCheckpoint), no
   gnn_outputs/traj needed.

## Corrected: two `checkpoint.bp` issues this plan raised are now fixed upstream

Both were real when this plan was written (`92f70b24`, 2026-08-26) and were fixed by
`b4ce824b` (Merge ALCF-4 benchmark changes, PR #81), which landed after it. They are
recorded here rather than deleted, because the plan's reader-side reasoning was built
on the pre-`b4ce824b` layout and the conclusions change.

1. **The checkpoint is on the COARSE GNN mesh, not the fine mesh.** The plan said
   `checkpoint.bp` is written from the fine mesh with `nrs->fieldOffset` and no
   interpolation, so an inference IC read would be silently truncated whenever
   `gnnPolynomialOrder != polynomialOrder` (2 vs 7 in the shooting example). That was
   true of the old no-argument `checkpoint()`, which did `o_U.copyTo(U, dim *
   nrs->fieldOffset)` itself. It now takes the field as an argument, and
   `turbChannel.udf:120-124` allocates `dim * graph->fieldOffset` and calls
   `graph->interpolateField(nrs, nrs->o_U, U, dim)` first — which interpolates
   fine->coarse when `gnnMeshPOrder < nekMeshPOrder` (`gnn.cpp:986-997`).

   Consequence: the checkpoint rows correspond **one-to-one with the graph.bp nodes**.
   That is what makes the size-agnostic read possible at all — the same element routing
   that places the graph on a rank places the checkpoint rows on it, so inference can
   run at a rank count nekRS never knew about. No truncation, no writer-side fix needed.

2. **The global shape is a true per-writer scan, not a uniform-fieldOffset assumption.**
   The plan said the global shape `_size * 3 * fieldOffset` assumes every writer has the
   same `fieldOffset`, which only holds for uniform element counts, and that no per-rank
   block-size record exists in the file. Both were true of the old declaration. It is now
   `{_global_field_offset * dim}, {_offset_field_offset * dim}, {_field_offset * dim}`
   (`adiosStreamer.cpp:162-166`), where `_global_field_offset` is an `MPI_Allreduce(SUM)`
   and `_offset_field_offset` a genuine `MPI_Allgather` + prefix scan
   (`gnn.cpp:249,266-274`). The layout is block-correct for heterogeneous element counts.

   Related, and not mentioned anywhere in this plan: `graph.bp` now publishes a per-writer
   `field_offset` variable, `{W},{rank},{1}` (`gnn.cpp:300`). `AdiosSource` reads it
   instead of recomputing `align_stride(N_w)`, which keeps the reader correct across a
   build with a different `dfloat` size or `ALIGN_SIZE_BYTES`; the recomputation remains
   as a fallback for older files and for the pre-existing test fixtures.

The standing caveat on both: verified by reading `b4ce824b` and the current sources, not
by running against real nekRS BP output. That validation is still open.

## Latent issues found while reading (pre-existing, not caused by this work)

- Stream offsets: writer uses `fieldOffset=alignStride(N)` (`trajGen.cpp:256`), reader
  slices by `N` (`client.py:233-246`) — only consistent when padding is zero.
- `trainer.py:1518` reshapes checkpoint C-order; stream reads use `order="F"`.
- `client.py:81-86` `file_exists` returns None for adios; `put_array` no-op for adios
  (`inference.py:313-315` result is silently dropped).
- `graph.bp` `Np` variable declared shape {1} start {1} (`gnn.cpp:303`) — off-by-one.
  Consequence is not cosmetic: `client.py:166` reads 0, and `trainer.py:370-373`
  then computes `poly = int(np.cbrt(0) - 1.0)` = -1 (the bare `except` never
  fires), so online model checkpoints are named with the wrong polynomial
  order. Worked around read-side; the one-character writer fix ({1}->{0}) is
  still worth making, and the workaround survives it.
- shooting `nrsrun_aurora:79` emits `ml_nodes: ${SIM_NODES}` (should be TRAIN_NODES);
  `INFERENCE_CPU_BIND_LIST` (line 17) is defined but never consumed.
- `inference.py:49-50` crashes off-PALS (`PALS_LOCAL_RANKID` no default).
- Offline a-priori `inference()` uses `data["test"]` / `stats["mean"]` keys that
  `setup_data` never creates.

## Deferred: the inference --> nekRS return path

Decision 2026-09-09 (user): **out of scope for now.** Focus is
nekRS + online training --> inference. Reading the rollout result back into nekRS to
actually shoot the solution forward comes later. Recorded here so the next person does
not mistake it for an oversight, and so the part that IS already done is not redone.

### What already works

`client.put_array` under adios (`client.py:147-200`) writes one global BP array over the
ML communicator -- the `_rank_R_size_S` suffix is stripped, blocks concatenate in ML-rank
order, component-major, unpadded. Given `global_ids` it writes a companion
`<var>_global_ids` with the same block decomposition.

`inference.py:315-319` passes `x[:n_nodes_local]` and `graph.global_ids[:n_nodes_local]`.
The slice is the point: `n_nodes_local` is the OWNED node count, halo rows excluded
(`trainer.py:1177`, `data_reduced.pos.shape[0]`). So the union over ML ranks covers each
mesh node exactly once -- no halo duplicates, no gaps.

The consequence worth keeping: `checkpt_u.bp` is already a complete, self-describing,
ML-rank-count-independent representation of the rolled-forward solution. **The data model
is settled**; a future consumer is not blocked on a format change, only on being written.
This predates the repartitioning work and was verified, not assumed.

### What is missing (three gaps, increasing size)

1. **No nekRS-side reader.** `adios_client_t` needs a `restart(dfloat*, int)` mirroring
   `checkpoint()`. Not a straight inverse: `checkpoint.bp` is in WRITER-block layout
   (strided by `fieldOffset`, offset by the per-writer scan) while `checkpt_u.bp` is in
   ML-rank layout, unpadded. A nekRS rank cannot select its rows by offset arithmetic --
   it must match on `global_ids`.

2. **That match is a distributed scatter.** nekRS rank r knows its own nodes' global ids
   (`gnn_t` computed them) and needs the rows carrying them, which sit in arbitrary blocks
   from arbitrary ML ranks. It is a gather-by-key -- the same rendezvous pattern
   `repartition/rebuild.py` already implements for the mask computation. The pieces exist
   in Python; nothing equivalent exists in C++.

3. **Coarse->fine prolongation does not exist.** `gnn_t::interpolateField`
   (`gnn.cpp:986-997`) handles `gnnMeshPOrder == nekMeshPOrder` (copy) and
   `gnnMeshPOrder < nekMeshPOrder` (fine->coarse). There is no `else` -- the coarse->fine
   direction silently leaves the buffer as passed. The shooting workflow is gnn p=2 vs
   nekRS p=7, so a restart needs a prolongation operator that is not written. Largest of
   the three and the easiest to miss, because the function exists and compiles.

### Related: the workflow does not currently loop

`driver.py:runner()` does `fineTune()` then `rollout()` and stops. There is no second
nekRS launch picking the solution back up, which the example README describes as the
point ("picked back up by nekRS for more model fine-tuning"). So gap (1) is not merely an
unwired function -- it is the step that would make the loop a loop. Whether the current
one-shot form is intentional for the benchmark configuration is unconfirmed.

## Progress log (updated 2026-09-09)

- [x] Subsystem deep-read (6 parallel readers) and design (this doc).
- [x] Core package `3rd_party/gnn/dist-gnn/repartition/` implemented:
      `sources.py` (BinSource), `fld.py` (FldSource: .f reader + distributed
      coordinate-coincidence gid assignment with periodic folding),
      `partition.py` (block, RCB), `redistribute.py` (Alltoallv element
      routing, reusable `Routing` for field data), `rebuild.py` (masks via
      rendezvous-hash sharing detection, template-tiled rep-mapped edges),
      `templates.py`, `cli.py` (materializes size-M gnn_outputs + halo .npy
      + fld/traj data, from either a bin dir or a .f file).
- [x] Synthetic MPI test suite (`repartition/tests/`): gen_synthetic.py
      (independent serial mask/edge implementation as cross-check) +
      test_consistency.py (6 checks incl. one full halo-consistent
      aggregation round vs serial reference). PASSES for src size 4 or 5 →
      M ∈ {1,2,3,4,6,7,8}, methods block and rcb, poly 2 and 3.
- [x] Real-data validation (validation item 3): nekRS `tgv_gnn_offline` run
      locally at 4 ranks (SERIAL backend, install from BuildMeOnLocal at
      ~/.local/nekrs-repart); baseline training reproduces the ReFrame
      target loss 2.7161e-04; CLI-repartitioned (RCB) training at
      M ∈ {2,3,4,6} ALL hit 2.7161/2e-04 (SUCCESS line printed).
      Bug found and fixed on the way: gid==0 (element-interior) nodes must
      get unique negative ids before edge-weight cantor pairing
      (gcon.update_global_ids in cli.write_halo_files) — real nekRS files
      have gid==0 entries; the synthetic mesh did not.
- [x] .f-ONLY GOLD TEST (validation item 4a): patched scratch tgv.udf with
      `nrs->writeCheckpoint(time, tstep, true, true)`; graph reconstructed
      purely from tgv0.f00000 coordinates (periodic folding on xyz), data
      from U/P records; training at M ∈ {1,2,4} ALL hit 2.7161e-04 —
      i.e. coordinate-based coincidence classes are exactly equivalent to
      mesh->globalIds including periodic identification.
- [x] tgv_gnn_offline_traj, binary trajectory routing (validation item 3
      time-dependent): baseline at 4 ranks hits target 6.6139e-01; CLI
      repartition (graph + traj) to M ∈ {2,6} → both hit 6.6139e-01.
- [x] .f-ONLY TRAJECTORY GOLD TEST (validation item 4b): scratch traj udf
      patched to `nrs->writeCheckpoint` every 10 steps; CLI `--fld-mesh
      tgv0.f00000 --fld-traj tgv0.f00000..5 --periodic xyz` reconstructs
      graph + u_step trajectory from .f files only; training at
      M ∈ {1,2,4} ALL hit 6.6139e-01.
- [x] Trainer in-memory wiring (Phase 2 item 2, pulled forward): cfg keys
      `gnn_outputs_size` / `repartition_method`; `load_graph_data` routes
      through Repartitioner when the current-size files are missing (or a
      different `gnn_outputs_size` is set); halo info computed on the fly
      (client-cache guarded for offline); `load_field_data` /
      `load_trajectory` / `load_initial_condition` route snapshots via
      `_load_snapshot`. Pure passthrough when files match the world size.
- [x] Coincidence-class cross-check: coordinate-based gids from the .f mesh
      (with periodic folding) induce EXACTLY the same partition of nodes as
      nekRS `mesh->globalIds` (bijection verified node-by-node on matched
      element orderings for the tgv case).
- [x] Trainer wiring regression: passthrough at matching size still hits
      2.7162e-04 with the repartitioner dormant; in-memory repartition
      (M=2 reading the size-4 dir directly, no CLI step) hits 2.7162e-04
      (time_independent) and 6.6139e-01 (time_dependent trajectory).
- [x] Package promoted to `3rd_party/gnn/repartition/` (model-agnostic,
      numpy+mpi4py core; the dist-gnn halo-file writing in the CLI is an
      optional integration, `--no-halo` to skip). Installed into NEKRS_HOME
      via CMakeLists install rule. dist-gnn imports it from the parent dir.
- [x] New example `examples/tgv_gnn_offline_fld`: nekRS writes ONLY a .f
      checkpoint (`nrs->writeCheckpoint`, no gnn plugin); the repartition
      CLI reconstructs graph + fld_u/fld_p data; training at an arbitrary
      rank count hits 2.7161e-04. Validated locally end-to-end (nekRS at 4
      ranks, training at 2 and 3). Run scripts default to
      SIM_RANKS_PER_NODE=2, ML_RANKS_PER_NODE=4 to showcase the decoupling.
      **VERIFIED ON AURORA 2026-09-09** (user-run, nekRS at 2 ranks ->
      repartition + train at 4): the first HPC validation of the
      repartitioner. Two portability fixes were needed to get there, both
      unrelated to the repartitioning math itself:
      (a) torch must be imported before `from mpi4py import MPI`
      initializes MPI, or the Aurora frameworks torch fails to load —
      the import now sits at the top of `repartition/__init__.py` (the
      module that actually runs first under `python -m repartition.cli`,
      ahead of the mpi4py-importing `.api` chain) and ahead of the
      mpi4py import in each standalone entry point (cli.py, tests/*.py);
      (b) the merge with main (`acd4b5a5`) changed the four
      `create_halo_info_par` halo functions to take `COMM, RANK, SIZE`
      explicitly instead of reading module globals; `trainer.py` was
      updated in that merge but `repartition/cli.py` and
      `tests/test_consistency.py` were not. All 8 call sites fixed and
      checked by AST arity/order comparison; a repo-wide scan of
      3rd_party/gnn finds no remaining mismatches.
- [x] ReFrame coverage (Phase 2 item 3): `TGVOfflineRepart` (variant a —
      nekRS at fixed nekrs_ranks=2 writes gnn_outputs, CLI repartitions
      graph + fld data with `--src-dir ... --fld`) and `TGVOfflineFld`
      (variant b — .f-only reconstruction with `--fld-mesh ... --periodic
      xyz` on the tgv_gnn_offline_fld example), both parameterized over
      rpn ∈ {2,4} against the standard target loss 2.7161e-04 (trainer
      check is math.isclose rel_tol=1e-3, so the 2.7161/2e-04 spread is
      fine). Shared machinery in `NekRSMLOfflineRepartTest` (tests/nekrs.py):
      decoupled-rank nekRS launch (mpiexec_n), PYTHONPATH export for the
      installed repartition package, CLI step, trainer opts. Validated
      locally via `reframe --system generic -l` (INSTANTIATION ONLY —
      these have still never been RUN on Aurora; see the open ReFrame
      task at the end of this log). Also added nrsrun_crux to the
      tgv_gnn_offline_fld example (mirrors tgv_gnn_offline's Crux script).
- [x] parRSB wrapper (Phase 2 item 4), branch worktree-parrsb-wrapper:
      `parrsb_shim.c` (one exported function wrapping `parrsb_part_mesh`
      with `MPI_Comm_f2c(fcomm)` — same pattern as nek5000's
      partitioner.c) + `build_parrsb_shim.sh` (mpicc, `-DMPI`, links
      libparRSB.a + nek5000-side libgs.a from $NEKRS_HOME; on macOS
      `-dynamiclib`, PIC by default; Linux `-shared`, libs are -fPIC) +
      `parrsb.py` (ctypes: vtx (Ne,8) int64 corner gids, xyz (Ne,8,3)
      f8, comm via py2f; parRSB averages xyz to centroids itself) +
      `partition.py` `method="parrsb"`: corner lattice indices
      [0, nq-1, nq^2-1, nq(nq-1)] × {k=0, k=nq-1} in nekRS hex vertex
      order (verified against ref data by a reader agent); gid==0
      corners (never-shared) get unique NEGATIVE labels from the element
      ordinal — aliasing them would glue unrelated elements. VALIDATED
      locally: test_consistency ALL PASS for M ∈ {1,2,3,4,6,8} (synth
      4x3x2 poly 3) + rcb/block regression; determinism across repeated
      calls confirmed; CLI end-to-end on real tgv ref gnn_outputs_poly_7
      (4 -> 2, includes the gid==0 halo-file path). Quality
      (tests/partition_quality.py, halo classes / halo copies / neighbor
      ranks): synth M=8 parrsb 314/684/mean 4.25 vs rcb 328/746/6.50 vs
      block 334/768/6.00 — acceptance (<= RCB) met with a strict win; on
      the perfectly symmetric periodic tgv 8x8x8 box all three methods
      tie exactly (equivalent cuts by symmetry). NOTE for real science
      meshes: parRSB is the only method without a rank-0 gather (our RCB
      gathers all centroids) — default choice at scale. LOSS GOLD TEST:
      trainer in-memory path (`gnn_outputs_size=4 repartition_method=parrsb`,
      M=2, tgv_gnn_offline_traj ref data, traj_data_path pointing at the
      tinit_0.000000_dtfactor_10 subdir) hits 6.6139e-01 exactly at step
      100 — "SUCCESS! GNN training validated!", log shows
      "Repartitioning graph from size 4 to 2 (method=parrsb)".
- [x] ReFrame parrsb variant (parRSB item c): `TGVOfflineRepart`
      parameterized over `repart_method ∈ {rcb, parrsb}` (4 variants with
      rpn ∈ {2,4}); `NekRSMLOfflineRepartTest` gained a
      `repartition_method` kwarg and, for parrsb, prerun cmds that build
      the shim into the stage dir (`parrsb_shim_cmds()`) and export
      PARRSB_SHIM_LIB. Validated via `reframe -C sites.py --system
      generic -c tests.py -l` (22 checks instantiate). CI run pending.
- [x] Checkpoint names made partition-independent (commit a6179d7b):
      model name drops SIZE (`POLY_p_SEED_s`); restart load falls back
      to legacy `POLY_p_SIZE_*_SEED_s` via glob. First slice of the
      online task. End-to-end M != S load still to be exercised.
- [x] Online ADIOS path fully specced (Phase 2 item 1 above rewritten
      from a verified deep-read of gnn.cpp/trajGen.cpp/adiosStreamer.cpp/
      client.py/driver.py): exact BP layouts (component-major blocks,
      alignStride(N)=ceil(N/32)*32 padding, W=shape("N")[0]), AdiosSource
      design (three sub-reads per block; shared _overlaps helper),
      trainer/client wiring points, driver knobs, and a LOCAL test
      recipe — graph.bp is always a plain BP file (writer IO never sets
      an engine), serial pip adios2 2.12.1 is now in the venv, so a
      Python-written fixture from the ref dir + AdiosSource==BinSource
      equivalence is the acceptance gate. Ready for Opus to implement.
- [x] parRSB shim CMake integration (parRSB item a): `parrsb_shim` SHARED
      target added at the end of `add_nek5000()` in `cmake/nek5000.cmake`
      — it must live there because `PARRSB_*`/`NEK5000_GS_*` are
      function-local and expand to empty paths at the top level. Links
      `${PARRSB_LIB_DIR}/libparRSB.a` + `${NEK5000_GS_LIB_DIR}/libgs.a`
      by path (the imported targets carry no usage requirements),
      `MPI::MPI_C`, `m`; `-DMPI`; PREFIX/SUFFIX forced to `lib`/`.so`
      because parrsb.py hardcodes that filename on every platform. Also
      fixed a latent bug on the neighbouring imported target: its
      IMPORTED_LOCATION was `${PARRSB_DIR}/lib/libparRSB.a`, but
      PARRSB_DIR is the *source* dir — the archive is installed one level
      up in PARRSB_LIB_DIR (harmless so far only because nothing links
      that target). New cache option `ENABLE_PARRSB_SHIM` (default ON) as
      an escape hatch if a platform link misbehaves. Install: the
      `install(DIRECTORY 3rd_party/gnn/repartition ...)` rule gained
      `PATTERN "libparrsb_shim.so" EXCLUDE` so a stale hand-built copy in
      the source tree can never shadow the real artifact, then the target
      file is installed to both `3rd_party/gnn/repartition/` (where
      parrsb.py looks first) and `lib/` (so $NEKRS_HOME/lib resolves it
      when the package is imported from a source checkout). blasLapack is
      deliberately NOT linked — `nm -u` shows no dgemm/mxm/blas
      undefineds, matching the known-good manual recipe; add
      `${BLASLAPACK_DIR}/libblasLapack.a` only if a Linux link proves
      otherwise. VALIDATED: configure+build clean, library exports the
      same 172 symbols as the manually built one, test_consistency
      parrsb ALL PASS for M ∈ {2,3,6} against the build-tree copy and
      M=3 against the installed copy, and a clean-prefix install with a
      stale 293056 B copy planted in the source tree still ships the
      311632 B CMake artifact to both destinations.
- [ ] **IMPORTANT — run the full ReFrame suite on Aurora.** No ReFrame
      test has ever been executed on ALCF CI; all 22 checks are
      instantiation-verified only (`reframe -C sites.py --system generic
      -l`), and the 6 repartitioning variants (TGVOfflineRepart x4,
      TGVOfflineFld x2) are the acceptance gate for this whole work item.
      BLOCKED as of 2026-09-09, for an environment reason unrelated to the
      code: the work is being done against the upcoming Aurora image/SDK,
      which is available on compute nodes but NOT on login nodes. ReFrame
      builds nekRS on the login node and runs on compute nodes, so the
      build would link against the old environment and the run would fail.
      Unblocks when the login- and compute-node environments are matched;
      the user will run it then. Expect two first-run snags: the parrsb
      shim build needs `MPICC=cc` if the Cray wrapper is not `mpicc`, and
      these tests predate the `create_halo_info_par` signature drift fixed
      2026-09-09 (CLI is fixed; a green run is what confirms it).
      Commands: `./tests/run.sh -t tgv_offline_repart` and
      `./tests/run.sh -t tgv_offline_fld` (add `-b` on the first run).
- [ ] Remaining: see Phase 2 tasks (online ADIOS path — implementation,
      spec is done; parRSB items b/d: HPC validation, optional
      distributed quality metric).

Local reproduction notes: python env at ~/.venvs/nekrs-gnn-repart
(mpi4py, torch, torch_geometric, hydra-core, einops, ruff); run nekRS with
`source envMac.sh` FIRST (else the udf JIT uses clang against a gcc-built
libnekrs → dlopen symbol errors); training locally needs
`master_addr=localhost` and `halo_swap_mode=all_to_all` (gloo cannot do the
unequal-size all_to_all_opt).

### 2026-09-09 — Phase 2 items 1-2: checkpoint read + driver knobs (written, UNVALIDATED)

Scope fixed with the user this session: forward direction only, nekRS + online training
--> inference. The inference --> nekRS return path is deferred (see "Deferred" above).

Written, **none of it executed** — no adios2 is importable on the Aurora login node
(`frameworks` ships none; the nekRS-built copy needs GLIBC 2.32, the compute-node image).
Static checks only: `ast.parse` on every touched Python file, `bash -n` on both run
scripts, ruff 186 -> 186 on the touched files (large pre-existing baseline, no regression).

- `client.get_checkpoint_from_file()` — reads `checkpoint.bp` through the same element
  routing that placed the graph, so it is rank-count-agnostic. Possible only because of
  correction (1) above: the checkpoint is on the coarse mesh, 1:1 with graph.bp nodes.
  Supporting changes: `_wait_for_graph` generalized to `_wait_for_bp(path, need)`;
  `_read_own_field_block` takes an optional `stream=`; `file_exists` returns an explicit
  `False` under adios instead of falling off the end returning `None`.
- `trainer.load_initial_condition` uses it; `load_graph_data` now honors
  `cfg.repartition_method` instead of hardcoding rcb.
- `AdiosSource` reads the writer-published `field_offset` (`gnn.cpp:300`) instead of
  recomputing `align_stride(N_w)`, falling back for older files and the existing fixtures.
- Driver knobs: `assignNodes()` slices inference nodes; `launchInference()` uses
  `inferprocs` / `inferprocs_pn` / `infer_cpu_bind`, read straight off the config.
  No backward-compatible fallback — decision 2026-09-09 (user): `config.yaml` is always
  generated by `nrsrun_<system>`, both of which emit all four keys, so a `runArg()`
  shim was dead weight. `infer_nodes` outside the job's node count is a hard exit. Also
  `skip = 0` unconditionally — inference runs alone, so no device skip even under a
  colocated deployment; the old code inherited the colocated skip and would have stranded
  half the GPUs.
- `nrsrun_aurora` + `nrsrun_polaris`: new `INFER_NODES` (0 => all job nodes) and
  `INFER_RANKS_PER_NODE`; fixed `ml_nodes: ${SIM_NODES}` -> `${TRAIN_NODES}`; consumed the
  previously-dead `INFERENCE_CPU_BIND_LIST`. Third bug found in passing: the colocated
  branch set `SIM_NODES=$nodes` before `nrsqsub_utils` defines `nodes`, so both were the
  empty string — changed to `$2`.
- `tests/bin_to_bp.py` emits `field_offset` and a `checkpoint.bp` fixture. **No test
  asserts against them yet** — that gate is still to be written.

Open, in priority order: (a) online ADIOS end-to-end on Aurora at W != M, the whole point
and never yet run; (b) a `checkpoint.bp` assertion in `test_online_client.py`; (c) the
full ReFrame suite, blocked on the login/compute image mismatch; (d) parRSB validation at
a few hundred ranks.

### 2026-08-26 — Phase 2 item 1: online ADIOS path (reader side complete)

Ground truth re-verified against the writers before any code was written;
three layout facts drove the implementation, all confirmed at source:

- `writeToFileBinary` (`gnn.cpp:49-56`, `index = j*nRows + i`) **transposes on
  write**, so the `.bin` files are interleaved rows while the BP variables are
  the raw component-major device buffers. `BinSource` and `AdiosSource` must
  therefore reshape differently for the same data — `pos_node` and
  `edge_index` need a per-writer-block `order="F"` reshape, and reshaping the
  whole global array at once is wrong for any W > 1.
- `in_u`/`out_u` stride by the padded `graph->fieldOffset = alignStride(N)`,
  and their global offsets are a **true per-writer scan** (`trajGen.cpp:225-247`),
  so heterogeneous element counts are handled correctly by the writer.
  `client.py:233-234` sliced by unpadded `N` — correct only because the TGV
  case has Np=512 and 512 % 32 == 0. Fixed.
- `Np` reads back as 0 (see above).

Done:
- `repartition/sources.py`: `ElementSource` base + `AdiosSource` (robust `Np`,
  `src_size` from `shape("N")[0]`, per-writer node/edge/fieldOffset scans,
  component-major sub-reads, template from writer block 0), `align_stride`,
  and `open_bp_read` (shared serial-adios2 fallback, exported).
- `Repartitioner.read_field` needed no change: it already forwards the spec to
  the source, so a variable name works wherever a path callable did.
- `dist-gnn/client.py`: `get_graph_data_from_stream` now reads graph.bp through
  `AdiosSource`, keeps the direct per-block read when W == M, and repartitions
  when W != M; `get_train_data_from_stream` routes through the repartitioner or
  the fieldOffset-correct block read; `put_array` has a real ADIOS branch
  (one global array over the reader comm, component-major, plus
  `<var>_global_ids`); `_wait_for_graph` waits for readable variables rather
  than for the directory to appear.
- `dist-gnn/inference.py`: the rollout result is no longer silently dropped
  under adios; the locally-unique rows are returned tagged with global ids.

Tests (all local, no nekRS needed):
- `tests/bin_to_bp.py` — serial bin -> BP fixture converter derived line by
  line from the C++ writers, reproducing the `Np` defect deliberately.
- `tests/test_adios_equiv.py` — the acceptance gate: AdiosSource == BinSource
  after repartitioning. Passes at M in {1,2,3,8} on an adversarial fixture
  (`/tmp/synth_pad`: W=4, Np=27, N=[189,216,189,216], pads=[3,8,3,8] — both
  non-uniform N and non-zero padding, neither of which the real TGV data has).
- `tests/test_online_client.py` — the W == M client path against the `.bin`
  files directly (not against another reader), including an assertion that the
  pre-fix contiguous `N*3` read is demonstrably wrong on this fixture.
- Five mutation negative controls run against the gate (interleaved pos read,
  C-order edge reshape, unpadded field stride, no-selection `Np`, unblocked
  global reshape); all five are detected, so the gate is not vacuously green.

Not yet done / explicitly out of scope of this change:
- **No validation against real nekRS BP output** — the local `gnn_outputs_*`
  and trajectory directories under `/tmp` are empty, so every check above runs
  on the synthetic fixture. This needs an HPC run and pairs naturally with
  parRSB item (b).
- `get_array` (checkpoint.bp) is untouched. The reason given here — a fine-mesh
  write with a uniform-fieldOffset global shape, needing a writer-side fix — no
  longer holds after `b4ce824b` (see Corrected above), so the read is now doable
  purely reader-side. `get_array` itself is still component-blind and
  padding-blind; the checkpoint no longer goes through it.
- Driver knobs (`inferprocs`, `infer_cpu_bind`, ...) and the
  `nrsrun_aurora:79` `ml_nodes: ${SIM_NODES}` bug. **Done 2026-09-09** (written,
  not yet run — see the 2026-09-09 log entry).
- New, unrelated, found while verifying: `smartRedis.cpp:149-150` copies
  `o_P` into `U` instead of `P`, clobbering the u-component with pressure and
  leaving `checkpt_p` all zeros. Affects the smartredis path only.
