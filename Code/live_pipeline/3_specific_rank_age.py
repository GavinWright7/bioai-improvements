#!/usr/bin/env python3
"""
Launcher for targeted ATP rankings enrichment (same as rankings_age.py).

Run:
  python Code/live_pipeline/3_specific_rank_age.py
"""
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rankings_age import main

    main()
