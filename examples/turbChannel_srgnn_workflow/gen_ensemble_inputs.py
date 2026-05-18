"""
Stage EnsembleLauncher directories for turbChannel_srgnn_workflow.

After the initial nekRS run has written high and low order checkpoints
(e.g. ``p20`` → polynomial order 2, ``p70`` → order 7), this script creates
a run directory and EnsembleLauncher JSON configs to run additional simulations
starting from each high and low order checkpoints.

1. Finds all requested ``<case>_p*.f*`` checkpoints under this example directory.
2. Creates one run directory per checkpoint under ``./run_dir/<member>/``.
3. Writes a per-member ``<case>.par`` file.
4. Symlinks each checkpoint into its member dir and copies ``.udf`` / ``.usr``.
5. Writes the three EnsembleLauncher JSON configs.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from typing import List, Sequence

HERE = Path(__file__).resolve().parent

sys.path.append(os.path.join(os.environ["NEKRS_HOME"], "3rd_party", "ensembleLauncher"))
from nekrs_ensemble_utils import (
    setup_ensemble_dirs,
    write_ensemble_configs,
)


def _field_frame_index(path: Path) -> int:
    m = re.search(r"\.f(\d+)$", path.name, re.IGNORECASE)
    return int(m.group(1)) if m else -1


def _pp_tag(path: Path, case_name: str) -> int:
    """Return the ``PP`` in ``<case>_p<PP>.f#####``."""
    m = re.search(rf"{re.escape(case_name)}_p(\d+)\.", path.name, re.IGNORECASE)
    if not m:
        raise ValueError(
            f"Checkpoint name {path.name!r} does not match "
            f"{case_name}_p<PP>.f##### (cannot read p-tag)."
        )
    return int(m.group(1))


def polynomial_order_from_checkpoint(path: Path, case_name: str) -> int:
    """Map nek field suffix ``p<PP>`` to ``[GENERAL] polynomialOrder`` (``PP / 10``)."""
    pp = _pp_tag(path, case_name)
    if pp % 10 != 0:
        raise ValueError(
            f"Expected p-tag in {path.name!r} to be a multiple of 10 (got p{pp}); "
            "nekRS multiscale naming here is p(10*polynomialOrder)."
        )
    return pp // 10


def p_file_suffixes_from_p_orders_arg(values: List[int]) -> List[int]:
    """Map ``--p-orders`` to filename digits ``PP`` in ``_pPP``.

    * If **every** value is in ``1..9``, treat as polynomial orders → ``PP = 10*N``.
    * If **every** value is ``>= 10`` and ``% 10 == 0``, treat as literal p-tags.
    """
    if not values:
        raise ValueError("--p-orders produced an empty list")
    if any(v < 1 for v in values):
        raise ValueError(f"--p-orders values must be >= 1, got {values!r}")

    all_poly_small = all(1 <= v <= 9 for v in values)
    all_literal_tags = all(v >= 10 and v % 10 == 0 for v in values)

    if all_poly_small:
        return [10 * v for v in values]
    if all_literal_tags:
        return values

    raise ValueError(
        "--p-orders must be either (a) all polynomial orders in 1..9, e.g. 7,2 "
        "→ …_p70.f*, …_p20.f*, or (b) all literal nek p-tags (each ≥10 and "
        f"divisible by 10), e.g. 70,20. Got: {values!r}"
    )


def discover_snapshots_for_case(
    base: Path, case_name: str, patterns: Sequence[str]
) -> List[Path]:
    seen: set[Path] = set()
    out: List[Path] = []
    for pattern in patterns:
        for p in base.glob(pattern):
            if not p.is_file():
                continue
            try:
                _pp_tag(p, case_name)
            except ValueError:
                continue
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            out.append(p)

    def sort_key(path: Path) -> tuple:
        try:
            pp = _pp_tag(path, case_name)
        except ValueError:
            pp = 0
        return (pp, _field_frame_index(path), path.name)

    out.sort(key=sort_key)
    return out


def member_name_for_snapshot(path: Path, case_name: str) -> str:
    """Filesystem-safe directory name; includes p-tag and frame index."""
    idx = _field_frame_index(path)
    pp = _pp_tag(path, case_name)
    if idx >= 0:
        return f"from_p{pp}_f{idx:05d}"
    safe = re.sub(r"[^\w.\-]+", "_", path.name)
    return f"cp_p{pp}_{safe}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "case",
        help="nekRS case basename (e.g. turbChannel for turbChannel.par / .re2).",
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
        help="Number of steps to run for each nekRS simulation.",
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

    # Check that the cache directory exists
    cache_dir = HERE / ".cache"
    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"{cache_dir} not found; run nekrs --build-only first so all "
            "members can share the same .cache"
        )

    # Parse the polynomial orders and generate the file patterns
    raw_orders = [int(x.strip()) for x in args.p_orders.split(",") if x.strip()]
    pp_tags = p_file_suffixes_from_p_orders_arg(raw_orders)
    patterns = [f"{case_name}_p{pp}.f*" for pp in pp_tags]
    pattern_desc = ", ".join(patterns)

    # Discover the snapshots for each polynomial order
    snapshots = discover_snapshots_for_case(HERE, case_name, patterns)
    if not snapshots:
        raise FileNotFoundError(
            f"No checkpoints matched under {HERE} for {pattern_desc!r}."
        )

    # Define the ensemble members
    members = []
    for snap in snapshots:
        poly = polynomial_order_from_checkpoint(snap, case_name)
        name = member_name_for_snapshot(snap, case_name)
        base_name = snap.name
        rel = snap.name
        members.append(
            {
                "name": name,
                "par_overrides": {
                    "GENERAL": {
                        "startFrom": base_name,
                        "numSteps": float(args.num_steps),
                        "polynomialOrder": int(poly),
                    }
                },
                "symlinks": {base_name: rel},
            }
        )

    print(
        f"[gen_ensemble_inputs] {len(members)} members from {pattern_desc!r}: "
        f"{[m['name'] for m in members]}"
    )

    # Link the re2 and cache files
    re2 = HERE / f"{case_name}.re2"
    symlink_files: List[str] = [".cache"]
    if re2.is_file():
        symlink_files.insert(0, f"{case_name}.re2")

    # Check that the par file exists
    par_path = HERE / f"{case_name}_simple.par"
    if not par_path.is_file():
        raise FileNotFoundError(f"missing {par_path} file")

    # Add the udf, usr, box files to the copy list
    copy_files: List[str] = []
    udf_path = HERE / f"{case_name}_simple.udf"
    if not udf_path.is_file():
        raise FileNotFoundError(f"missing {udf_path} file")
    copy_files.append(udf_path.name)

    usr = HERE / f"{case_name}.usr"
    if usr.is_file():
        copy_files.append(usr.name)

    box = HERE / f"{case_name}.box"
    if box.is_file():
        copy_files.append(box.name)

    # Create the run directories
    member_dirs = setup_ensemble_dirs(
        case_name=case_name,
        members=members,
        base_dir=str(HERE),
        output_dir=args.outdir,
        copy_files=copy_files,
        symlink_files=symlink_files,
        par_template=str(par_path.resolve()),
    )

    nek_udf_name = f"{case_name}.udf"
    for d in member_dirs:
        src = d / udf_path.name
        dst = d / nek_udf_name
        if not src.is_file():
            raise FileNotFoundError(f"expected copied {udf_path.name} in {d}")
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        shutil.move(str(src), str(dst))

    # Check that the checkpoint symlinks are present
    for m, d in zip(members, member_dirs):
        for dst_name in m.get("symlinks") or {}:
            link = d / Path(dst_name).name
            if not link.exists():
                raise RuntimeError(
                    f"Expected checkpoint symlink missing: {link} "
                    f"(member {m['name']!r})."
                )

    # Write the EnsembleLauncher JSON configs
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
