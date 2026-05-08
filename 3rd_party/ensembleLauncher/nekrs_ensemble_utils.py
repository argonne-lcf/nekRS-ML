"""
General-purpose helpers for setting up nekRS ensembles to be launched with
EnsembleLauncher (https://github.com/argonne-lcf/ensemble_launcher).

The public entry points are:

* ``setup_ensemble_dirs``    -- create one run directory per ensemble member,
                                with copies of small case files (``.udf``,
                                ``.usr``, ``.oudf``, ...), symlinks to large /
                                shared ones (``.re2``, ``.cache``, restart
                                files, ...) and a per-member ``.par`` produced
                                from a base template by overriding section/key
                                entries (typically ``[CASEDATA]``).

* ``write_ensemble_configs`` -- write the three JSON files the EnsembleLauncher
                                CLI consumes (``el start <ensemble> --system-
                                config-file <sys> --launcher-config-file
                                <launcher>``):

                                  - ``config.json``         (ensembles block)
                                  - ``system_config.json``  (SystemConfig)
                                  - ``launcher_config.json``(LauncherConfig)

These helpers are intentionally agnostic to which parameter is being swept --
they just override entries in the ``.par`` file. Case-specific generators
(for example ``examples/periodicHill_ensemble/gen_ensemble_inputs.py``) are
expected to build the list of members and call into here.
"""

import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence


_SECTION_RE = re.compile(r"^\s*\[(?P<name>[^\]]+)\]\s*$")


def setup_ensemble_dirs(
    case_name: str,
    members: Sequence[Dict],
    base_dir: str = ".",
    output_dir: str = "run_dir",
    copy_files: Sequence[str] = (),
    symlink_files: Sequence[str] = (),
    par_template: Optional[str] = None,
    overwrite: bool = True,
) -> List[Path]:
    """Create one nekRS run directory per ensemble member.

    Parameters
    ----------
    case_name : str
        Base name used by nekRS (the ``.par``/``.udf``/``.usr``/``.re2`` files
        are expected to be ``<case_name>.<ext>``).
    members : sequence of dict
        One dict per ensemble member. Each dict must contain ``"name"`` (the
        per-member subdirectory name under ``output_dir``) and may contain
        ``"par_overrides"``: a mapping ``{section: {key: value, ...}, ...}``
        that is applied to the base ``.par`` (see ``apply_par_overrides``
        for matching/formatting rules).
    base_dir : str
        Directory containing the source case files (default: current working
        directory).
    output_dir : str
        Directory under which to write per-member subdirectories. Re-created
        from scratch when ``overwrite=True``.
    copy_files : sequence of str
        Files (paths relative to ``base_dir``) to *copy* into each member
        directory. Typical: ``periodicHill.udf``, ``periodicHill.usr``.
    symlink_files : sequence of str
        Files or directories (paths relative to ``base_dir``) to *symlink*
        into each member directory. Typical: ``periodicHill.re2``, ``.cache``,
        restart ``.fld`` files. Symlinks point at absolute resolved paths so
        the run directories remain valid regardless of cwd at launch time.
    par_template : str, optional
        Path to the base ``.par`` to use as a template. Defaults to
        ``<base_dir>/<case_name>.par``.
    overwrite : bool
        If ``True``, ``output_dir`` is removed first.

    Returns
    -------
    list of pathlib.Path
        Absolute paths to the created per-member directories, in the order
        of ``members``.
    """
    base = Path(base_dir).resolve()
    out = Path(output_dir).resolve()

    if out.exists() and overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    par_src = (
        Path(par_template).resolve()
        if par_template
        else (base / f"{case_name}.par").resolve()
    )
    if not par_src.is_file():
        raise FileNotFoundError(f"par template not found: {par_src}")
    with open(par_src, "r") as f:
        par_lines = f.readlines()

    dirs: List[Path] = []
    for member in members:
        if "name" not in member:
            raise ValueError("each member dict must include a 'name' key")
        d = out / member["name"]
        d.mkdir(parents=True, exist_ok=True)

        # Per-member .par
        overrides = member.get("par_overrides", {}) or {}
        with open(d / f"{case_name}.par", "w") as f:
            f.writelines(apply_par_overrides(par_lines, overrides))

        # Copy small files
        for rel in copy_files:
            src = base / rel
            if not src.is_file():
                raise FileNotFoundError(f"copy_files entry not found: {src}")
            shutil.copy(src, d / Path(rel).name)

        # Symlink large / shared paths
        for rel in symlink_files:
            src = (base / rel).resolve()
            if not src.exists():
                raise FileNotFoundError(f"symlink_files entry not found: {src}")
            dst = d / Path(rel).name
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            elif dst.is_dir():
                shutil.rmtree(dst)
            os.symlink(src, dst)

        dirs.append(d)

    return dirs


def apply_par_overrides(
    template_lines: Sequence[str],
    overrides: Dict[str, Dict[str, object]],
) -> List[str]:
    """Return a copy of ``template_lines`` with ``overrides`` applied.

    Section names and keys are matched case-insensitively. Existing keys are
    replaced in place (preserving leading whitespace and the original key
    spelling); keys that are not present in their section are appended at
    the end of the file inside a fresh section block. Numeric values are
    formatted with ``%.10g``; booleans become ``true``/``false``; everything
    else is ``str()``-coerced.
    """
    pending: Dict[str, Dict[str, str]] = {
        sect.lower(): {k: _fmt_par_value(v) for k, v in kv.items()}
        for sect, kv in overrides.items()
    }
    out: List[str] = []
    section: Optional[str] = None
    for line in template_lines:
        m = _SECTION_RE.match(line)
        if m:
            section = m.group("name").strip().lower()
            out.append(line)
            continue

        stripped = line.strip()
        if (
            section
            and stripped
            and not stripped.startswith("#")
            and "=" in stripped
        ):
            key, _, _rest = stripped.partition("=")
            key_lc = key.strip().lower()
            if section in pending and key_lc in pending[section]:
                value = pending[section].pop(key_lc)
                lead = line[: len(line) - len(line.lstrip())]
                out.append(f"{lead}{key.strip()} = {value}\n")
                continue
        out.append(line)

    leftover = {sect: kv for sect, kv in pending.items() if kv}
    if leftover:
        if out and not out[-1].endswith("\n"):
            out.append("\n")
        for sect, kv in leftover.items():
            out.append(f"\n[{sect.upper()}]\n")
            for k, v in kv.items():
                out.append(f"{k} = {v}\n")

    return out


def _fmt_par_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.10g}"
    return str(v)


def write_ensemble_configs(
    out_dir: str,
    member_dirs: Sequence[Path],
    case_name: str,
    nekrs_home: str,
    *,
    nodes_per_member: int = 1,
    ppn: int = 12,
    ngpus_per_process: int = 1,
    backend: str = "dpcpp",
    cpu_bind: Optional[str] = None,
    ensemble_name: str = "nekrs_ensemble",
    extra_nrs_args: str = "--device-id 0",
    sys_name: str = "aurora",
    ncpus_per_node: int = 104,
    ngpus_per_node: int = 12,
    cpus: Optional[Sequence[int]] = None,
    gpus: Optional[Sequence] = None,
    launcher_config: Optional[Dict[str, object]] = None,
) -> Dict[str, str]:
    """Write the three JSON files the EnsembleLauncher CLI consumes.

    Files written into ``out_dir``:

    * ``config.json``          -- the ``ensembles`` block. All members share
                                  ``cmd_template`` (``$NEKRS_HOME/bin/nekrs
                                  --setup <case> --backend <backend>``); only
                                  ``run_dir``/``launch_dir`` differ. The
                                  ``relation`` is ``"one-to-one"`` so member
                                  *i* runs in ``member_dirs[i]``.
    * ``system_config.json``   -- ``SystemConfig`` (per-node CPU / GPU
                                  counts and explicit ID lists). On Aurora,
                                  ``ncpus_per_node=104``, ``ngpus_per_node=12``
                                  (one tile per GPU process).
    * ``launcher_config.json`` -- ``LauncherConfig`` (executor / comm /
                                  reporting). Sensible defaults for an HPC
                                  job; override via ``launcher_config``.

    The trio is then launched with::

        el start <out_dir>/config.json \\
            --system-config-file   <out_dir>/system_config.json \\
            --launcher-config-file <out_dir>/launcher_config.json

    Per-task fields not set here -- e.g. ``cpu_affinity``, ``gpu_affinity`` --
    can be added by the caller by editing ``config.json`` after the fact.

    Returns a dict of {kind: written_path}.
    """
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    cmd = (
        f"{nekrs_home}/bin/nekrs --setup {case_name} "
        f"--backend {backend} {extra_nrs_args}"
    ).strip()
    dirs = [str(p) for p in member_dirs]

    # 1) ensemble config -- the ensembles block
    ensemble_cfg: Dict[str, object] = {
        "ensembles": {
            ensemble_name: {
                "nnodes": nodes_per_member,
                "ppn": ppn,
                "ngpus_per_process": ngpus_per_process,
                "launcher": "mpi",
                "relation": "one-to-one",
                "run_dir": dirs,
                "launch_dir": dirs,
                "cmd_template": cmd,
            }
        },
    }
    if cpu_bind:
        ensemble_cfg["ensembles"][ensemble_name]["launcher_options"] = {
            "cpu-bind": cpu_bind,
        }

    # 2) system config -- SystemConfig
    sys_cfg: Dict[str, object] = {
        "name": sys_name,
        "ncpus": ncpus_per_node,
        "ngpus": ngpus_per_node,
        "cpus": list(cpus) if cpus is not None else list(range(ncpus_per_node)),
        "gpus": list(gpus) if gpus is not None else list(range(ngpus_per_node)),
    }

    # 3) launcher config -- LauncherConfig
    launcher_cfg: Dict[str, object] = {
        "child_executor_name": "mpi",
        "task_executor_name": "mpi",
        "comm_name": "zmq",
        "report_interval": 10.0,
        "return_stdout": True,
        "worker_logs": True,
        "master_logs": True,
    }
    if launcher_config:
        launcher_cfg.update(launcher_config)

    paths = {
        "config": str(out / "config.json"),
        "system": str(out / "system_config.json"),
        "launcher": str(out / "launcher_config.json"),
    }
    with open(paths["config"], "w") as f:
        json.dump(ensemble_cfg, f, indent=4)
    with open(paths["system"], "w") as f:
        json.dump(sys_cfg, f, indent=4)
    with open(paths["launcher"], "w") as f:
        json.dump(launcher_cfg, f, indent=4)

    return paths
