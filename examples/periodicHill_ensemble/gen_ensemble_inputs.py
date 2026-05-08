"""
Generate ensemble run directories for the periodic hill case.

This script sweeps the ``[CASEDATA] hillScale`` parameter of
``periodicHill.par``, producing one run directory per member under
``./run_dir/``. Because the hill geometry is created at runtime in
``usrdat2()`` from a single base ``.re2`` file, every member shares the
same mesh and the same ``.cache``: those are *symlinked* into each member
directory rather than copied.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NEKRS_ML_ROOT = HERE.parents[1]
sys.path.append(str(NEKRS_ML_ROOT / "3rd_party" / "ensembleLauncher"))

from nekrs_ensemble_utils import (
    setup_ensemble_dirs,
    write_ensemble_configs,
)


CASE_NAME = "periodicHill"


def _parse_range(spec: str) -> np.ndarray:
    """Parse ``min,max,N`` into ``np.linspace(min, max, N)``.

    Also accepts a single comma-separated explicit list, e.g. ``0.8,1.0,1.2``.
    """
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) == 3:
        try:
            mn, mx, n = float(parts[0]), float(parts[1]), int(parts[2])
            if n >= 2 and abs(mx - mn) > 0:
                return np.linspace(mn, mx, n)
        except ValueError:
            pass
    return np.array([float(p) for p in parts], dtype=float)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate periodicHill ensemble run directories by sweeping "
            "[CASEDATA] hillScale."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--hillScale",
        default="0.8,1.5,4",
        help=(
            "hillScale sweep specified either as 'min,max,N' (linspace) or "
            "as an explicit comma-separated list of values."
        ),
    )
    p.add_argument(
        "--outdir",
        default=str(HERE / "run_dir"),
        help="Output directory for per-member run dirs and config.json.",
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
        help="Number of nodes assigned to each ensemble member.",
    )
    p.add_argument(
        "--ngpus-per-process",
        type=int,
        default=1,
        help="GPUs per MPI rank (Aurora tile-per-rank: 1).",
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
        default="list:1:8:16:24:32:40:53:60:68:76:84:92",
        help="mpiexec --cpu-bind argument forwarded via launcher_options.",
    )
    p.add_argument(
        "--ensemble-name",
        default="periodicHill_hillScale_sweep",
        help="Name of the ensemble inside the EnsembleLauncher config.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if "NEKRS_HOME" not in os.environ:
        raise EnvironmentError("NEKRS_HOME must be set before running this script")

    cache_dir = HERE / ".cache"
    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"{cache_dir} not found; run nekrs --build-only first so all "
            "members can share the same .cache"
        )

    hill_scales = _parse_range(args.hillScale)
    if hill_scales.size == 0:
        raise ValueError("hillScale sweep is empty")
    if (hill_scales <= 0).any():
        raise ValueError(
            "hillScale must be strictly positive (it appears in the denominator "
            "of hmax = 28.0 / hillScale in usrdat2)"
        )

    members = [
        {
            "name": f"hillScale_{v:.3f}",
            "par_overrides": {"CASEDATA": {"hillScale": float(v)}},
        }
        for v in hill_scales
    ]

    print(
        f"[gen_ensemble_inputs] sweeping hillScale over "
        f"{hill_scales.tolist()} ({len(members)} members)"
    )

    member_dirs = setup_ensemble_dirs(
        case_name=CASE_NAME,
        members=members,
        base_dir=str(HERE),
        output_dir=args.outdir,
        copy_files=[f"{CASE_NAME}.udf", f"{CASE_NAME}.usr"],
        symlink_files=[f"{CASE_NAME}.re2", ".cache"],
    )

    paths = write_ensemble_configs(
        out_dir=args.outdir,
        member_dirs=member_dirs,
        case_name=CASE_NAME,
        nekrs_home=os.environ["NEKRS_HOME"],
        nodes_per_member=args.nodes_per_member,
        ppn=args.ppn,
        ngpus_per_process=args.ngpus_per_process,
        backend=args.backend,
        cpu_bind=args.cpu_bind,
        ensemble_name=args.ensemble_name,
        sys_name=args.system,
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
