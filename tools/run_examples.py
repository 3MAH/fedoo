"""Run the gallery examples headless and report their status and wall time.

The examples executed at each documentation build are run, i.e. every
``examples/<category>/*.py`` of a gallery folder (a folder with a
``README.rst``), except the precomputed ones listed in
``examples/precomputed_runs.toml`` (run those with
``tools/run_precomputed.py``). Each script is run in its own folder, as
sphinx-gallery does, with ``MPLBACKEND=Agg`` and pyvista off screen. The
report (``examples_report.md`` by default) gives the status, wall time and,
for failures, the last lines of the output; the full output of each example
is written in ``--log-dir``.

Usage::

    python tools/run_examples.py                       # all executed examples
    python tools/run_examples.py basic contact/hertz_axisymmetric.py
    python tools/run_examples.py --timeout 600 --report report.md
    python tools/run_examples.py --precomputed         # the heavy ones too
"""

import argparse
import datetime
import os
import platform
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


def gallery_examples(include_precomputed=False):
    """Return the example paths (relative to ``examples/``) of the gallery."""
    with open(REGISTRY, "rb") as file:
        precomputed = {e["path"] for e in tomllib.load(file)["example"]}
    paths = []
    for folder, _, files in sorted(os.walk(EXAMPLES)):
        if not os.path.isfile(os.path.join(folder, "README.rst")):
            continue
        rel = os.path.relpath(folder, EXAMPLES)
        if rel == ".":
            continue
        for name in sorted(files):
            path = f"{rel}/{name}".replace(os.sep, "/")
            if name.endswith(".py") and (
                include_precomputed or path not in precomputed
            ):
                paths.append(path)
    return paths


def run(path, timeout, log_dir):
    script = os.path.join(EXAMPLES, path)
    pythonpath = os.pathsep.join(filter(None, [ROOT, os.environ.get("PYTHONPATH")]))
    env = dict(
        os.environ, MPLBACKEND="Agg", PYVISTA_OFF_SCREEN="true", PYTHONPATH=pythonpath
    )
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    log = os.path.join(log_dir, path.replace("/", "__")[: -len(".py")] + ".log")
    start = time.perf_counter()
    with open(log, "w") as out:
        try:
            proc = subprocess.run(
                [sys.executable, os.path.basename(script)],
                cwd=os.path.dirname(script),
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
            status = "ok" if proc.returncode == 0 else f"failed ({proc.returncode})"
        except subprocess.TimeoutExpired:
            status = f"timeout ({timeout} s)"
    wall_time = time.perf_counter() - start
    with open(log) as file:
        tail = file.read().splitlines()[-15:]
    return status, wall_time, log, tail


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "paths", nargs="*", help="examples or category folders (default: all)"
    )
    parser.add_argument(
        "--timeout", type=float, default=900, help="seconds per example"
    )
    parser.add_argument(
        "--precomputed", action="store_true", help="also run the precomputed examples"
    )
    parser.add_argument("--report", default=os.path.join(ROOT, "examples_report.md"))
    parser.add_argument(
        "--log-dir", default=os.path.join(ROOT, "docs", "build", "example_logs")
    )
    args = parser.parse_args()

    paths = gallery_examples(args.precomputed)
    if args.paths:
        # an example named explicitly is run even if it is a precomputed one
        wanted = [p.rstrip("/") for p in args.paths]
        paths = [
            p
            for p in gallery_examples(include_precomputed=True)
            if p in wanted or (p.split("/")[0] in wanted and p in paths)
        ]
        if not paths:
            parser.error(f"no gallery example matches {', '.join(args.paths)}")
    os.makedirs(args.log_dir, exist_ok=True)

    rows, failures = [], []
    for i, path in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {path}", flush=True)
        status, wall_time, log, tail = run(path, args.timeout, args.log_dir)
        print(f"    {status}, {wall_time:.0f} s", flush=True)
        rows.append((path, status, wall_time))
        if status != "ok":
            failures.append((path, log, tail))

    import numpy
    import simcoon

    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as file:
        fedoo_version = tomllib.load(file)["project"]["version"]
    with open(args.report, "w") as report:
        report.write(
            f"# Example runs, {datetime.date.today().isoformat()}\n\n"
            f"fedoo {fedoo_version}, simcoon {simcoon.__version__}, numpy "
            f"{numpy.__version__}, Python {platform.python_version()}, "
            f"{platform.system()} {platform.machine()}\n\n"
            f"{sum(s == 'ok' for _, s, _ in rows)}/{len(rows)} ok, total "
            f"{sum(t for _, _, t in rows) / 60:.1f} min\n\n"
            "| example | status | wall time (s) |\n|---|---|---|\n"
        )
        for path, status, wall_time in rows:
            report.write(f"| {path} | {status} | {wall_time:.0f} |\n")
        for path, log, tail in failures:
            report.write(f"\n## {path}\n\nLog: `{log}`\n\n```\n")
            report.write("\n".join(tail) + "\n```\n")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
