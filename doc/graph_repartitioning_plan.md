# Plan: decouple dist-gnn rank count from nekRS via Python-side graph repartitioning

Date: 2026-08-21. Status: living document — tracks design, verified facts, task breakdown, and progress.

## Problem

Training and inference of the dist-gnn model are hard-bound to run on exactly the same
number of ranks as the nekRS simulation that produced the graph (`gnn_outputs_poly_*`,
`graph.bp`) and the training data. In the shooting workflow this forces inference onto
half the available GPUs. More generally it prevents training on archived data at an
arbitrary rank count.

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
   original motivating feature). Changes, with the couplings at:
   - `client.py:150-216 get_graph_data_from_stream`: writer size W =
     `inquire_variable("N").shape()[0]` (writer-rank counts array). Read the
     full `N`/`num_edges` arrays, compute writer-block offsets; assign each
     reader a contiguous global ELEMENT range (N is per-writer-block,
     element-aligned since N = Ne*Np; Np from the stream); read pos/gids
     element-aligned. Edge index: do NOT read per-writer edge blocks —
     extract the intra-element template from writer block 0's edges (both
     endpoints < Np after subtracting nothing; block 0 starts at local id 0)
     and rebuild, or simpler: feed pos/gids elements into
     `repartition.Repartitioner` with an `AdiosSource` that mirrors
     `BinSource` (implement `read_elements` reading the stream, and reuse
     partition/redistribute/rebuild unchanged). Then compute halo info on
     the fly exactly as the online path already does (trainer.py:989-1034).
   - `client.py:218-254 get_train_data_from_stream` (in_u/out_u): writer
     blocks are `graph->fieldOffset*3` long (component-major u,v,w per
     block, fieldOffset = alignStride(N) — trajGen.cpp:239-269); slice per
     writer block, truncate padding to N, reshape, then route through
     `Repartitioner.routing.route_node_array`. NOTE the existing code
     slices by N_list, which silently assumes zero padding — fix while
     touching this.
   - `client.py:104-114 get_array` for `checkpoint.bp`: same fieldOffset
     component-major layout per writer block (adiosStreamer.cpp:155-175);
     replace the naive shape[0]/size split with writer-block-aware,
     element-aligned reads + routing.
   - `trainer.py:369/389` save_header embeds SIZE — replace with a
     partition-independent name (e.g. drop SIZE, keep a fallback that also
     tries the old name on load). Without this, inference at M != training
     size cannot find the .tar.
   - `driver.py:150-183 launchInference` + `nrsrun_aurora`: add
     `inferprocs`, `inferprocs_pn`, `infer_cpu_bind`, `infer_nodes` config
     keys (default: sim_nodes+train_nodes, 2x mlprocs) and use them.
   Acceptance: shooting workflow on 2 nodes with inference on all GPUs;
   local smoke test with SST on a laptop (nekRS 2 ranks, train 2, infer 4).
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
   a. **CMake integration**: build `libparrsb_shim.so` during the nekRS
      build and install it with the repartition package (today users run
      `repartition/build_parrsb_shim.sh` manually against NEKRS_HOME).
      Natural place: after the parRSB external project in
      `cmake/nek5000.cmake` (link `${PARRSB_DIR}/lib/libparRSB.a` +
      nek5000-side `libgs.a`, `-DMPI`, includes from both installs), then
      install next to the repartition package (its CMake install rule
      already exists). Note: use the **nek5000-side gslib** (BLAS=2
      build), not nekRS's own gs_content copy (USE_NAIVE_BLAS) — parRSB
      was compiled against the former.
   b. **HPC validation (Aurora/Polaris/Crux)**: `-fPIC` is already passed
      to the nek5000/parRSB builds by cmake/nek5000.cmake, so linking the
      shared shim should work; `build_parrsb_shim.sh` honors `MPICC` for
      Cray `cc`. Run the synthetic consistency matrix and
      partition_quality at a few hundred ranks; confirm no rank-0 memory
      spike (parRSB is fully distributed, unlike our RCB).
   c. **ReFrame**: add a `method=parrsb` variant to `TGVOfflineRepart`
      (tests/nekrs.py hardcodes `--method rcb` in `repartition_cmd()`);
      needs the shim built in the CI environment (depends on a).
   d. Optional: distributed edge-cut metric (partition_quality.py gathers
      (gid, rank) pairs to rank 0 — fine at test scale only).
5. **New example** `tgv_gnn_offline_fld` (or README section): the .f-only
   workflow the user requested, wired with the CLI commands from the
   Progress log; udf writes .f checkpoints only (writeCheckpoint), no
   gnn_outputs/traj needed.

## Latent issues found while reading (pre-existing, not caused by this work)

- `checkpoint.bp` is written from the FINE mesh (`nrs->fieldOffset`, no interpolation,
  `adiosStreamer.cpp:155-175`) while graph/training data live on the coarse GNN mesh
  (`gnnPolynomialOrder=2` vs `polynomialOrder=7` in the shooting example) — inference IC
  reads it truncated to the coarse node count (`trainer.py:1080-1082`). Looks wrong
  unless orders match; verify with the workflow owner.
- Stream offsets: writer uses `fieldOffset=alignStride(N)` (`trajGen.cpp:256`), reader
  slices by `N` (`client.py:233-246`) — only consistent when padding is zero.
- `trainer.py:1518` reshapes checkpoint C-order; stream reads use `order="F"`.
- `client.py:81-86` `file_exists` returns None for adios; `put_array` no-op for adios
  (`inference.py:313-315` result is silently dropped).
- `graph.bp` `Np` variable declared shape {1} start {1} (`gnn.cpp:303`) — off-by-one.
- `inference.py:49-50` crashes off-PALS (`PALS_LOCAL_RANKID` no default).
- Offline a-priori `inference()` uses `data["test"]` / `stats["mean"]` keys that
  `setup_data` never creates.

## Progress log (updated 2026-08-21)

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
      locally via `reframe --system generic -l` (instantiation only; the
      run stage needs PBS + Lmod). Also added nrsrun_crux to the
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
      gathers all centroids) — default choice at scale.
- [ ] Remaining: see Phase 2 tasks (online ADIOS path is the big one;
      parRSB items a-d: CMake shim build+install, HPC validation,
      ReFrame parrsb variant, optional distributed quality metric).

Local reproduction notes: python env at ~/.venvs/nekrs-gnn-repart
(mpi4py, torch, torch_geometric, hydra-core, einops, ruff); run nekRS with
`source envMac.sh` FIRST (else the udf JIT uses clang against a gcc-built
libnekrs → dlopen symbol errors); training locally needs
`master_addr=localhost` and `halo_swap_mode=all_to_all` (gloo cannot do the
unequal-size all_to_all_opt).
