"""
Stage EnsembleLauncher directories for turbChannel_srgnn_workflow (step 3).

After a pilot nekRS run has written field checkpoints as ``<case>0.f#####`` (no
polynomial order in the filename), this script globs those files, takes a
comma-separated list of **polynomial orders** (``--p-orders``, e.g. ``7,2``),
and builds the **Cartesian product**: one ensemble member per ``(.f`` file,
``p``-order) pair.

1. Finds all ``<case>0.f*`` checkpoints under this example directory.
2. Creates one run directory per pair under ``./run_dir/<member>/``; member
   names tag both the frame and the order (e.g. ``turbChannel0_f00000_p7``).
3. Writes a per-member ``.par`` with ``[GENERAL]`` overrides: ``startFrom``,
   ``numSteps``, and ``polynomialOrder`` (from ``--p-orders``, not from the
   filename).
4. Symlinks each checkpoint into the member dir as ``restart.fld`` (target is
   the real ``<case>0.f#####`` next to the template), plus shared ``.re2``,
   ``.cache``, etc.

Then writes the three EnsembleLauncher JSON configs. Launch with ``el start``
as in ``periodicHill_ensemble``.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List

HERE = Path(__file__).resolve().parent
sys.path.append(os.path.join(os.environ["NEKRS_HOME"], "3rd_party", "ensembleLauncher"))

from nekrs_ensemble_utils import (
    setup_ensemble_dirs,
    write_ensemble_configs,
)

# Symlink name in each member cwd; ``startFrom`` in the generated .par must match.
RESTART_LINK_NAME = "restart.fld"


def _field_frame_index(path: Path) -> int:
    m = re.search(r"\.f(\d+)$", path.name, re.IGNORECASE)
    return int(m.group(1)) if m else -1


def parse_p_orders(values: List[int]) -> List[int]:
    """Comma-separated polynomial orders (each >= 1)."""
    if not values:
        raise ValueError("--p-orders produced an empty list")
    if any(v < 1 for v in values):
        raise ValueError(f"--p-orders values must be >= 1, got {values!r}")
    return values


def discover_pilot_checkpoints(base: Path, case_name: str) -> List[Path]:
    """All ``<case>0.f#####`` field files directly under ``base``."""
    pat = re.compile(rf"^{re.escape(case_name)}0\.f\d+$", re.IGNORECASE)
    out: List[Path] = []
    for p in base.glob(f"{case_name}0.f*"):
        if not p.is_file():
            continue
        if not pat.match(p.name):
            continue
        out.append(p)

    def sort_key(path: Path) -> tuple:
        return (_field_frame_index(path), path.name.lower())

    out.sort(key=sort_key)
    return out


def member_dir_name(snap: Path, case_name: str, poly_order: int) -> str:
    """Directory name: checkpoint stem + frame + polynomial order."""
    idx = _field_frame_index(snap)
    if idx >= 0:
        return f"{case_name}0_f{idx:05d}_p{poly_order}"
    safe = re.sub(r"[^\w.\-]+", "_", snap.name)
    return f"{safe}_p{poly_order}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build ensemble run dirs for turbChannel_srgnn_workflow: one member "
            "per (pilot <case>0.f##### checkpoint, --p-orders entry); restart "
            f"linked as {RESTART_LINK_NAME}."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "case",
        help="Case name (basename of .par/.udf/.usr/.re2), e.g. 'turbChannel'.",
    )
    p.add_argument(
        "--p-orders",
        required=True,
        help="Comma-separated polynomial orders for each checkpoint (e.g. 7,2).",
    )
    p.add_argument(
        "--num-steps",
        type=int,
        required=True,
        help="Number of steps to run for each ensemble member.",
    )
    p.add_argument(
        "--outdir",
        default=str(HERE / "run_dir"),
        help="Output directory for per-member run dirs and EL JSON configs.",
    )
    p.add_argument(
        "--ppn",
        type=int,
        default=12,
        help="MPI ranks per node per member (Aurora: 12).",
    )
    p.add_argument(
        "--nodes-per-member",
        type=int,
        default=1,
        help="Nodes assigned to each ensemble member.",
    )
    p.add_argument(
        "--ngpus-per-process",
        type=int,
        default=1,
        help="GPUs per MPI rank.",
    )
    p.add_argument(
        "--system",
        default="aurora",
        help="System name written into system_config.json.",
    )
    p.add_argument(
        "--backend",
        default="dpcpp",
        help="OCCA backend passed to nekrs --backend.",
    )
    p.add_argument(
        "--cpu-bind",
        default="",
        help="CPU bind string for EnsembleLauncher (comma-separated IDs for EL).",
    )
    p.add_argument(
        "--ensemble-name",
        default="turbChannel_checkpoint_ensemble",
        help="Name of the ensemble inside config.json.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    case_name = args.case

    cache_dir = HERE / ".cache"
    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"{cache_dir} not found; run nekrs --build-only first so all "
            "members can share the same .cache"
        )

    raw_orders = [int(x.strip()) for x in args.p_orders.split(",") if x.strip()]
    poly_orders = parse_p_orders(raw_orders)

    pattern = f"{case_name}0.f*"
    snapshots = discover_pilot_checkpoints(HERE, case_name)
    if not snapshots:
        raise FileNotFoundError(
            f"No pilot checkpoints matched under {HERE} for {pattern!r}. "
            "Run the upstream nekRS step first (expect "
            f"'{case_name}0.f#####' files in this directory)."
        )

    members = []
    for snap in snapshots:
        rel = snap.name
        for poly in poly_orders:
            name = member_dir_name(snap, case_name, poly)
            members.append(
                {
                    "name": name,
                    "par_overrides": {
                        "GENERAL": {
                            "startFrom": RESTART_LINK_NAME,
                            "numSteps": float(args.num_steps),
                            "polynomialOrder": int(poly),
                        }
                    },
                    "symlinks": {RESTART_LINK_NAME: rel},
                }
            )

    n_snap = len(snapshots)
    n_poly = len(poly_orders)
    print(
        f"[gen_ensemble_inputs] {len(members)} members "
        f"({n_snap} checkpoints × {n_poly} p-orders {poly_orders}): "
        f"{[m['name'] for m in members]}"
    )

    re2 = HERE / f"{case_name}.re2"
    symlink_files: List[str] = [".cache"]
    if re2.is_file():
        symlink_files.insert(0, f"{case_name}.re2")

    copy_files = [f"{case_name}.udf", f"{case_name}.usr"]
    box = HERE / f"{case_name}.box"
    if box.is_file():
        copy_files.append(f"{case_name}.box")

    member_dirs = setup_ensemble_dirs(
        case_name=case_name,
        members=members,
        base_dir=str(HERE),
        output_dir=args.outdir,
        copy_files=copy_files,
        symlink_files=symlink_files,
    )

    for m, d in zip(members, member_dirs):
        for dst_name in m.get("symlinks") or {}:
            link = d / Path(dst_name).name
            if not link.exists():
                raise RuntimeError(
                    f"Expected restart symlink missing: {link} (member {m['name']!r})."
                )

    paths = write_ensemble_configs(
        out_dir=args.outdir,
        member_dirs=member_dirs,
        case_name=case_name,
        nekrs_home=os.environ["NEKRS_HOME"],
        system_name=args.system,
        nodes_per_member=args.nodes_per_member,
        ppn=args.ppn,
        ngpus_per_process=args.ngpus_per_process,
        backend=args.backend,
        ensemble_name=args.ensemble_name,
        cpu_bind=args.cpu_bind or None,
    )

    print(
        f"[gen_ensemble_inputs] {len(member_dirs)} run directories under {args.outdir}"
    )
    for kind, path in paths.items():
        print(f"[gen_ensemble_inputs] wrote {kind:<8} -> {path}")
    print(
        "[gen_ensemble_inputs] launch with: "
        f"el start {paths['config']} "
        f"--system-config-file {paths['system']} "
        f"--launcher-config-file {paths['launcher']}"
    )


if __name__ == "__main__":
    main()
