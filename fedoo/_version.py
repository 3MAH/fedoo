from importlib.metadata import version, PackageNotFoundError


def _version_from_pyproject():
    try:
        import tomllib  # Python 3.11+
    except ImportError:  # Python <3.11
        import tomli as tomllib

    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"

    if not pyproject.exists():
        return "0.0.0.dev0"

    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})

    if "version" in project:
        return project["version"]

    return "0.0.0.dev0"


try:
    # if installed package (pip or conda)
    __version__ = version("fedoo")
except PackageNotFoundError:
    __version__ = _version_from_pyproject()
