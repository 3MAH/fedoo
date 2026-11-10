"""Merge generated media and run records into a fedoo-docs checkout."""

import argparse
import json
import shutil
from pathlib import Path


def load_records(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else {}


def save_record(directory, entry, run):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = load_records(directory / "records.json")
    records[entry["path"]] = {
        "run": run,
        **({"media": entry["media"]} if "media" in entry else {}),
    }
    (directory / "records.json").write_text(json.dumps(records, indent=2) + "\n")


def publish(source, destination):
    source, destination = Path(source), Path(destination)
    records = load_records(destination / "records.json")
    updates = load_records(source / "records.json")
    if not updates:
        raise ValueError("No completed example runs to publish")
    for entry in updates.values():
        if "media" in entry:
            relative = Path(entry["media"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid media path: {relative}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, target)
    destination.mkdir(parents=True, exist_ok=True)
    records.update(updates)
    (destination / "records.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    publish(args.source, args.destination)
