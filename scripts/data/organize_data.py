#!/usr/bin/env python3
"""
Utility to create a clean data directory layout and link raw sources into
`data/processed/merged` without duplicating files.

Usage:
  python scripts/data/organize_data.py
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

TARGET_SUBDIRS = [
    "processed/merged",
    "processed/scalers",
    "interim",
    "features",
    "predictions",
    "external",
    "docs",
]


def ensure_dirs():
    for sub in TARGET_SUBDIRS:
        d = DATA_DIR / sub
        d.mkdir(parents=True, exist_ok=True)


def link_raw_sources():
    if not RAW_DIR.exists():
        print(f"Raw directory not found: {RAW_DIR}")
        return

    dest = DATA_DIR / "processed" / "merged"
    files = list(RAW_DIR.glob("*"))
    if not files:
        print(f"No files found in {RAW_DIR}")
        return

    for f in files:
        if f.is_file():
            dst = dest / f.name
            # Remove existing file/symlink to keep a single canonical copy in data/raw
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            rel_target = os.path.relpath(f, start=dest)
            dst.symlink_to(rel_target)
            print(f"Linked {dst} -> {rel_target}")


def create_readme():
    readme = DATA_DIR / "README.md"
    if readme.exists():
        print(f"README already exists at {readme}")
        return

    content = """
# Data directory layout

This folder contains data used by the project. The `organize_data.py`
script creates the following structure and **symlinks** original raw files
into `data/processed/merged` (no duplication):

- `data/raw/` : original source files (do not modify)
- `data/processed/merged/` : consolidated CSVs ready for processing
- `data/processed/scalers/` : saved scaler objects and normalization artifacts
- `data/interim/` : intermediate datasets during preprocessing
- `data/features/` : feature-engineered datasets
- `data/predictions/` : model output files (predictions)
- `data/external/` : optional external data sources (weather APIs etc.)
- `data/docs/` : data provenance and metadata files

Run `python scripts/data/organize_data.py` to (re)create folders and link
raw sources into `data/processed/merged`.
"""

    readme.write_text(content.strip() + "\n")
    print(f"Created {readme}")


def main():
    ensure_dirs()
    link_raw_sources()
    create_readme()


if __name__ == "__main__":
    main()
