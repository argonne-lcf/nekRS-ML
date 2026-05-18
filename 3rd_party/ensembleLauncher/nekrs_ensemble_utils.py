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

from ensemble_launcher.config import LauncherConfig, PolicyConfig, SystemConfig
from ensemble_launcher.helper_functions import get_nodes

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
        Optional ``"symlinks"``: ``{dest_basename: src_relpath_from_base, ...}``
        creates extra symlinks in that member directory (e.g. symlink each
        checkpoint ``.f`` file next to the per-member ``.par`` for ``startFrom``).
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

        # Optional per-member symlinks: dest filename in member dir -> path
        # relative to ``base_dir`` (same as ``copy_files`` / ``symlink_files``).
        extra_links = member.get("symlinks")
        if extra_links:
            if not isinstance(extra_links, dict):
                raise TypeError(
                    "member['symlinks'] must be a dict "
                    "{dest_basename: src_relpath_from_base}"
                )
            for dst_name, src_rel in extra_links.items():
                src = (base / str(src_rel)).resolve()
                if not src.is_file():
                    raise FileNotFoundError(
                        f"member['symlinks'] source not found: {src} "
                        f"(member={member['name']!r})"
                    )
                dst = d / Path(dst_name).name
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
    pending: Dict[str, Dict[str, tuple]] = {
        sect.lower(): {
            k.strip().lower(): (k.strip(), _fmt_par_value(v))
            for k, v in kv.items()
        }
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
                _, value = pending[section].pop(key_lc)
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
            for _, (orig_key, value) in kv.items():
                out.append(f"{orig_key} = {value}\n")

    return out


def _fmt_par_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.10g}"
    return str(v)


def write_ensemble_configs(
    out_dir: str,
    member_dirs: Sequence[Path],
    case_name: str,
    nekrs_home: str,
    system_name: str,
    *,
    nodes_per_member: int = 1,
    ppn: int = 12,
    ngpus_per_process: int = 1,
    backend: str = "dpcpp",
    cpu_bind: Optional[str] = None,
    ensemble_name: str = "nekrs_ensemble",
    extra_nrs_args: Optional[str] = None,
) -> Dict[str, str]:
    """Write the three JSON config files that the EnsembleLauncher CLI consumes.

    Files written into ``out_dir``:

    * ``config.json``          -- defines the ensemble
    * ``system_config.json``   -- defines the system resources
    * ``launcher_config.json`` -- defines the launcher configuration

    Returns a dict of {kind: written_path}.
    """
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    cmd = (
        f"{nekrs_home}/bin/nekrs --setup {case_name} "
        f"--backend {backend} --device-id 0" + (f" {extra_nrs_args}" if extra_nrs_args else "")
    ).strip()
    dirs = [str(p) for p in member_dirs]

    # Ensemble config
    ensemble_cfg: Dict[str, object] = {
        "ensembles": {
            ensemble_name: {
                "nnodes": nodes_per_member,
                "ppn": ppn,
                "ngpus_per_process": ngpus_per_process,
                "executor_name": "async_mpi",
                "relation": "one-to-one",
                "run_dir": dirs,
                "cmd_template": cmd,
                "cpu_affinity": cpu_bind,
                "stdout_file": "nekrs.out",
                "stderr_file": "nekrs.err",
            }
        },
    }

    # System config
    if system_name == "aurora":
        cpus = list(range(104))
        cpus.pop(52)
        cpus.pop(0)
        gpus = list(range(12))
    elif system_name == "polaris":
        cpus = list(range(32))
        gpus = list(range(4))
    else:
        raise ValueError(f"Unsupported system: {system_name}")

    sys_config = SystemConfig(
        name=system_name,
        cpus=cpus,
        ncpus=len(cpus),
        gpus=gpus,
        ngpus=len(gpus),
    )

    # Launcher config
    launcher_config = LauncherConfig(
        child_executor_name="async_mpi",
        task_executor_name="async_mpi",
        return_stdout=True,
        children_scheduler_policy="fixed_leafs_children_policy",
        policy_config=PolicyConfig(nlevels=2, leaf_nodes=len(get_nodes()) // nodes_per_member),
    )

    # Write configs
    paths = {
        "config": str(out / "config.json"),
        "system": str(out / "system_config.json"),
        "launcher": str(out / "launcher_config.json"),
    }
    with open(paths["config"], "w") as f:
        json.dump(ensemble_cfg, f, indent=4)
    with open(paths["system"], "w") as f:
        json.dump(sys_config.model_dump(), f, indent=4)
    with open(paths["launcher"], "w") as f:
        json.dump(launcher_config.model_dump(), f, indent=4)

    return paths
