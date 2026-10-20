"""Create the conda environment used to run all the examples.

1. ``conda env create`` from ``tools/environment_examples.yml``;
2. ``conda install`` of the BLAS/LAPACK and sparse solver of the platform:

   * macOS: Apple Accelerate and python-mumps;
   * Linux and Windows x86_64: Intel MKL and pypardiso (which uses the same
     MKL as numpy and scipy);
   * arm64 Linux: OpenBLAS and python-mumps;

3. ``pip install -e .[all]``: fedoo from this source tree with all its
   optional dependencies. pip finds the solver already installed by conda.

Usage, from any conda environment (e.g. ``base``)::

    python tools/setup_examples_env.py               # environment fedoo-examples
    python tools/setup_examples_env.py --name other  # another name
    python tools/setup_examples_env.py --dry-run     # print the commands only
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(ROOT, "tools", "environment_examples.yml")


def platform_packages():
    """Return the conda packages (BLAS variant and sparse solver) to install."""
    if platform.system() == "Darwin":
        return ["libblas=*=*accelerate", "python-mumps"]
    if platform.machine().lower() in ("x86_64", "amd64"):
        return ["libblas=*=*mkl", "pypardiso"]
    return ["libblas=*=*openblas", "python-mumps"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", default="fedoo-examples")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda is None:
        sys.exit("conda not found: run this script from a conda environment")
    commands = [
        [conda, "env", "create", "-y", "-n", args.name, "-f", ENV_FILE],
        [conda, "install", "-y", "-n", args.name, "-c", "conda-forge"]
        + platform_packages(),
        [conda, "run", "--no-capture-output", "-n", args.name]
        + ["python", "-m", "pip", "install", "-e", f"{ROOT}[all]"],
    ]
    for command in commands:
        print("$ conda", " ".join(command[1:]))
        if not args.dry_run:
            subprocess.run(command, check=True)
    print(f"\nconda activate {args.name}")


if __name__ == "__main__":
    main()
