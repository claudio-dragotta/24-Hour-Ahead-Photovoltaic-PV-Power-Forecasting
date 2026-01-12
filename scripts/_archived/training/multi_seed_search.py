"""ARCHIVE: multi_seed_search.py

Original: scripts/training/multi_seed_search.py
"""

#!/usr/bin/env python3
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pv_forecasting.models.multi_branch_tft import MultiBranchTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Multi-seed search for Multi-Branch Transformer")
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--keep-top", type=int, default=10)
    parser.add_argument("--outdir", type=str, default="outputs/multi_branch/seed_search")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print("Archived multi-seed search script (kept for reference)")


if __name__ == "__main__":
    main()
