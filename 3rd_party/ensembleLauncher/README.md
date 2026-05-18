# nekRS-ML helpers for [EnsembleLauncher](https://github.com/argonne-lcf/ensemble_launcher)

This directory ships a small, reusable Python module (`nekrs_ensemble_utils.py`)
that case-specific generator scripts use to set up nekRS ensembles deployed with
[EnsembleLauncher](https://github.com/argonne-lcf/ensemble_launcher).

What this directory provides:

- `setup_ensemble_dirs(...)` — given a base nekRS case (a `<case>.par`,
  `.udf`, `.usr`, `.re2`, plus an optional `.cache`) and a list of member
  dicts, this creates one run directory per member with:
  - a per-member `<case>.par` produced from the base by overriding entries in
    the `[CASEDATA]` section (or any other `.par` section);
  - **copies** of the small case files declared by the caller (typically
    `.udf`, `.usr`, `.oudf`); and
  - **symlinks** to the large/shared paths declared by the caller (typically
    `.re2` and `.cache`, both of which are identical across members when the
    sweep is over runtime parameters rather than over the mesh).
  - optional per-member **extra symlinks** via each member dict's ``symlinks``
    key: ``{dest_basename: src_relpath_from_base}`` (e.g. symlink one checkpoint
    ``.f`` file into each member directory for ``startFrom`` restarts).

  The split between copies and symlinks matters: `.cache` in particular is
  large and members must not race to rebuild it.

- `write_ensemble_configs(...)` — writes the three JSON files that the
  EnsembleLauncher CLI (`el start`) consumes for a uniform nekRS sweep:
  - `config.json`          — the `ensembles` block (same `cmd_template` for
    every member, only `run_dir`/`launch_dir` differ, paired one-to-one);
  - `system_config.json`   — `SystemConfig` (per-node CPU / GPU counts and
    explicit ID lists);
  - `launcher_config.json` — `LauncherConfig` (executor / comm / reporting
    defaults; tunable via the `launcher_config=` kwarg).

Both helpers are deliberately agnostic to which parameter is being swept;
case-specific generators are expected to build the list of member dicts and
call into here. See `examples/periodicHill_ensemble/gen_ensemble_inputs.py`
for a worked example that sweeps the periodic hill height through a
`[CASEDATA] hillScale` value.

## Quick example

```python
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, "../../3rd_party/ensembleLauncher"))
from nekrs_ensemble_utils import setup_ensemble_dirs, write_ensemble_configs

members = [
    {
        "name": f"hillScale_{v:.3f}",
        "par_overrides": {"CASEDATA": {"hillScale": v}},
    }
    for v in [0.8, 0.9, 1.0, 1.1, 1.2]
]

member_dirs = setup_ensemble_dirs(
    case_name="periodicHill",
    members=members,
    base_dir=here,
    output_dir=os.path.join(here, "run_dir"),
    copy_files=["periodicHill.udf", "periodicHill.usr"],
    symlink_files=["periodicHill.re2", ".cache"],
)

write_ensemble_configs(
    out_dir=os.path.join(here, "run_dir"),
    member_dirs=member_dirs,
    case_name="periodicHill",
    nekrs_home=os.environ["NEKRS_HOME"],
    nodes_per_member=1,
    ppn=12,
    cpu_bind="list:1:8:16:24:32:40:53:60:68:76:84:92",
)
```

The three resulting JSON files are then handed to the EnsembleLauncher CLI:

```sh
el start run_dir/config.json \
    --system-config-file   run_dir/system_config.json \
    --launcher-config-file run_dir/launcher_config.json
```

The CLI blocks until all members finish and writes `results.json` next to
the configs. See the upstream README for cluster-mode and other CLI options.

## Notes

- `apply_par_overrides` matches section names and keys case-insensitively.
  Existing keys are replaced in place (preserving the original key spelling
  and indentation); missing keys are appended in a fresh section block at
  the end of the file.
- Floats are formatted with `%.10g`, booleans become `true`/`false`,
  everything else is `str()`-coerced.
- Symlink targets are resolved to absolute paths so members remain valid
  if they are launched from a different cwd.
- The schema written by `write_ensemble_configs` follows the EnsembleLauncher
  CLI conventions: ensembles JSON uses `nnodes`, `ppn`, `ngpus_per_process`,
  `relation: one-to-one`, per-member `run_dir`/`launch_dir` lists, and
  `cmd_template`; system/launcher JSONs map directly onto `SystemConfig` and
  `LauncherConfig`. Adjust the field names in `nekrs_ensemble_utils.py` if
  you are pinning to a version of EnsembleLauncher that uses different names.
