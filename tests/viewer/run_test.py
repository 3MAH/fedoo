"""Run the optional viewer suite independently of the current directory."""

from pathlib import Path
import sys
import pytest


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve().parent), *sys.argv[1:]]))
