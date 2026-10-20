"""Run the precomputed examples and update their run record.

The examples listed in ``examples/precomputed_runs.toml`` are too heavy to be
executed at each documentation build. This script runs them headless, writes
their animation in ``--media-dir`` (to be copied to the ``media/examples``
folder of the 3MAH/fedoo-docs repository) and records the software versions,
date, machine and wall time of the run in the ``run`` field of each entry.

The animation is taken, in this order of preference:

* from the ``movie`` table of the entry, if any: the result file
  ``movie.result`` (relative to the example folder) is read with
  :func:`fedoo.read_data` and :meth:`fedoo.MultiFrameDataSet.write_movie` is
  called with the other keys of the table (``field``, ``component``, plot
  options);
* otherwise, from a file with the name of ``media`` written by the example in
  its own folder or in its ``results`` subfolder.

An entry without ``media`` (an example with no animation to show) only gets
its run record.

Usage::

    python tools/run_precomputed.py                    # all examples
    python tools/run_precomputed.py contact/tube_compression.py
    python tools/run_precomputed.py --list

A failing example is reported and skipped: its run record is not updated, and
the script ends with a non-zero exit code listing the failures.
"""

import argparse
import datetime
import os
import platform
import re
import shutil
import subprocess
import sys
import time

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")
REGISTRY = os.path.join(EXAMPLES, "precomputed_runs.toml")


def _versions():
    import numpy
    import simcoon

    import fedoo

    del fedoo  # its installed metadata may be stale: use the source tree
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as file:
        fedoo_version = tomllib.load(file)["project"]["version"]
    versions = {
        "fedoo": fedoo_version,
        "simcoon": simcoon.__version__,
        "numpy": numpy.__version__,
    }
    try:
        import ipctk

        versions["ipctk"] = getattr(ipctk, "__version__", "installed")
    except ImportError:
        pass
    try:
        versions["commit"] = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return versions


def _toml_value(value):
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return repr(value)


def _write_run(path, run):
    """Replace the ``run = {...}`` line of the entry of ``path``."""
    with open(REGISTRY) as file:
        text = file.read()
    inline = "{" + ", ".join(f"{k} = {_toml_value(v)}" for k, v in run.items()) + "}"
    pattern = re.compile(
        r'(path = "' + re.escape(path) + r'"\n(?:(?!\[\[example\]\]).*\n)*?)run = .*\n'
    )
    text, n = pattern.subn(lambda m: m.group(1) + f"run = {inline}\n", text, count=1)
    if n != 1:
        raise RuntimeError(f"no 'run' field found for {path} in {REGISTRY}")
    with open(REGISTRY, "w") as file:
        file.write(text)


def _make_media(entry, folder, media_dir):
    if "media" not in entry:
        return None
    target = os.path.join(media_dir, entry["media"])
    os.makedirs(os.path.dirname(target), exist_ok=True)
    movie = dict(entry.get("movie", {}))
    if movie:
        import fedoo as fd

        result = movie.pop("result")
        data = fd.read_data(os.path.join(folder, result))
        data.write_movie(target, **movie)
        return target
    for subfolder in ("", "results"):
        produced = os.path.join(folder, subfolder, os.path.basename(entry["media"]))
        if os.path.isfile(produced):
            shutil.copy(produced, target)
            return target
    return None


def run(entry, media_dir):
    script = os.path.join(EXAMPLES, entry["path"])
    folder = os.path.dirname(script)
    # Run the fedoo of this source tree, not an installed one
    pythonpath = os.pathsep.join(filter(None, [ROOT, os.environ.get("PYTHONPATH")]))
    env = dict(
        os.environ, MPLBACKEND="Agg", PYVISTA_OFF_SCREEN="true", PYTHONPATH=pythonpath
    )
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    print(f"--- {entry['path']}", flush=True)
    start = time.perf_counter()
    subprocess.run([sys.executable, script], cwd=folder, env=env, check=True)
    wall_time = time.perf_counter() - start
    media = _make_media(entry, folder, media_dir)
    if media is None and "media" in entry:
        raise RuntimeError(
            f"{entry['path']}: no animation produced. Add a 'movie' table to its "
            f"entry in {REGISTRY}, or make the example write "
            f"{os.path.basename(entry['media'])!r}; the run record is not updated."
        )
    record = _versions()
    record.update(
        date=datetime.date.today().isoformat(),
        machine=f"{platform.system()} {platform.machine()}",
        wall_time_s=round(wall_time),
    )
    _write_run(entry["path"], record)
    print(f"    {wall_time:.0f} s, media: {media}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="*", help="examples to run (default: all)")
    parser.add_argument("--list", action="store_true", help="list the examples")
    parser.add_argument(
        "--media-dir",
        default=os.path.join(ROOT, "docs", "build", "precomputed_media"),
        help="folder where the animations are written",
    )
    args = parser.parse_args()

    with open(REGISTRY, "rb") as file:
        entries = tomllib.load(file)["example"]
    if args.list:
        for entry in entries:
            done = entry["run"].get("date", "never run")
            print(f"{entry['path']:55s} {done}")
        return
    selected = [e for e in entries if not args.paths or e["path"] in args.paths]
    unknown = set(args.paths) - {e["path"] for e in selected}
    if unknown:
        parser.error(f"not in {REGISTRY}: {', '.join(sorted(unknown))}")
    failures = []
    for entry in selected:
        try:
            run(entry, args.media_dir)
        except (subprocess.CalledProcessError, RuntimeError) as err:
            # keep going: the other examples still get their run record
            print(f"    FAILED: {err}", flush=True)
            failures.append(entry["path"])
    if failures:
        sys.exit("failed (run record not updated):\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()
