"""Publishing a subset must preserve earlier media and run provenance."""

import importlib.util
import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import pytest

_spec = importlib.util.spec_from_file_location(
    "precomputed_records",
    Path(__file__).resolve().parents[1] / "tools" / "precomputed_records.py",
)
records = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(records)


def test_publish_merges_completed_runs_and_keeps_other_examples(tmp_path):
    generated, hosted = tmp_path / "generated", tmp_path / "hosted"
    records.save_record(hosted, {"path": "old.py"}, {"commit": "old"})
    (hosted / "old.gif").write_bytes(b"old")
    records.save_record(
        generated, {"path": "new.py", "media": "contact/new.gif"}, {"commit": "new"}
    )
    (generated / "contact").mkdir()
    (generated / "contact/new.gif").write_bytes(b"new")
    records.publish(generated, hosted)
    published = json.loads((hosted / "records.json").read_text())
    assert published["old.py"]["run"]["commit"] == "old"
    assert published["new.py"]["run"]["commit"] == "new"
    assert (hosted / "old.gif").read_bytes() == b"old"
    assert (hosted / "contact/new.gif").read_bytes() == b"new"


def test_publish_missing_media_keeps_previous_records(tmp_path):
    generated, hosted = tmp_path / "generated", tmp_path / "hosted"
    records.save_record(hosted, {"path": "old.py"}, {"commit": "old"})
    before = (hosted / "records.json").read_text()
    records.save_record(generated, {"path": "new.py", "media": "missing.gif"}, {})
    with pytest.raises(FileNotFoundError):
        records.publish(generated, hosted)
    assert (hosted / "records.json").read_text() == before


def test_registry_paths_match_case_sensitive_gallery_names():
    root = Path(__file__).resolve().parents[1] / "examples"
    paths = {p.relative_to(root).as_posix() for p in root.rglob("*.py")}
    with (root / "precomputed_runs.toml").open("rb") as file:
        registry = tomllib.load(file)
    assert all(entry["path"] in paths for entry in registry["example"])
