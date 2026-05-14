"""
Stage EnsembleLauncher directories for turbChannel_srgnn_workflow (step 3).

After a single-node nekRS run has written field checkpoints, ``--p-orders``
lists either **polynomial orders** in ``1..9`` (e.g. ``7,1`` → globs
``<case>_p70.f*`` and ``<case>_p10.f*``) or **literal p-tags** (e.g. ``70,10``)
when every value is ``≥ 10`` and divisible by ``10``.

1. Finds all requested checkpoints under this example directory.
2. Creates one run directory per checkpoint under ``./run_dir/<member>/``.
3. Writes a per-member ``.par`` from the template with ``[GENERAL]`` overrides:
   ``startFrom``, ``numSteps``, and ``polynomialOrder`` from the checkpoint
   (``p<PP>`` → ``polynomialOrder = PP / 10``, e.g. ``p70`` → ``7``).
4. Symlinks each checkpoint into its member dir, plus shared ``.re2``, ``.cache``, etc.

Then writes the three EnsembleLauncher JSON configs. Launch with ``el start``
as in ``periodicHill_ensemble``.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Sequence

HERE = Path(__file__).resolve().parent
sys.path.append(
    os.path.join(os.environ["NEKRS_HOME"], "3rd_party", "ensembleLauncher")
)

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
    """Map ``--p-orders`` integers to nek filename digits ``PP`` in ``_pPP``.

    * If **every** value is in ``1..9``, treat them as **polynomial orders** and
      use ``PP = 10 * N`` (``7,1`` → ``70``, ``10`` for ``p70``, ``p10`` files).
    * If **every** value is ``>= 10`` and divisible by ``10``, treat them as
      literal **p-tags** (``70,10`` → ``70``, ``10``).

    Mixed lists like ``1,70`` are rejected. For polynomial order ``>= 10``,
    pass literal p-tags (e.g. ``100`` for ``p100`` files).
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
        "--p-orders must be either (a) all polynomial orders in 1..9, e.g. 7,1 "
        "→ …_p70.f*, …_p10.f*, or (b) all literal nek p-tags (each ≥10 and "
        f"divisible by 10), e.g. 70,10. Got: {values!r}"
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

    def sort_key(p: Path) -> tuple:
        try:
            pp = _pp_tag(p, case_name)
        except ValueError:
            pp = 0
        return (pp, _field_frame_index(p), p.name)

    out.sort(key=sort_key)
    return out


def member_name_for_snapshot(path: Path, case_name: str) -> str:
    """Filesystem-safe directory name; includes p-tag so p70 and p10 differ."""
    idx = _field_frame_index(path)
    pp = _pp_tag(path, case_name)
    if idx >= 0:
        return f"from_p{pp}_f{idx:05d}"
    safe = re.sub(r"[^\w.\-]+", "_", path.name)
    return f"cp_p{pp}_{safe}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build per-checkpoint ensemble run dirs for turbChannel_srgnn_workflow "
            "(one member per .f; startFrom, numSteps, polynomialOrder from checkpoint)."
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
        help=(
            "Comma-separated integers: either polynomial orders 1..9 (e.g. 7,1 "
            "→ globs …_p70.f*, …_p10.f*) or literal p-tags (e.g. 70,10 when each "
            "value is ≥10 and divisible by 10)."
        ),
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

    raw_orders = [
        int(x.strip())
        for x in args.p_orders.split(",")
        if x.strip()
    ]
    pp_tags = p_file_suffixes_from_p_orders_arg(raw_orders)
    patterns = [f"{case_name}_p{pp}.f*" for pp in pp_tags]
    pattern_desc = ", ".join(patterns)

    snapshots = discover_snapshots_for_case(HERE, case_name, patterns)
    if not snapshots:
        raise FileNotFoundError(
            f"No checkpoints matched under {HERE} for {pattern_desc!r}. "
            "Run the upstream nekRS step first, or adjust --p-orders."
        )

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

    print(f"[gen_ensemble_inputs] {len(member_dirs)} run directories under {args.outdir}")
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
