#!/usr/bin/env python3
"""
ESPN match-date enrichment — entry point (implementation in 2_dates.py).

Set INCLUDE_TODAY here, then run:
  python Code/live_pipeline/dates.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

INCLUDE_TODAY = True


def main() -> None:
    impl_path = Path(__file__).resolve().parent / "2_dates.py"
    spec = importlib.util.spec_from_file_location("_dates_impl", impl_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {impl_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.INCLUDE_TODAY = INCLUDE_TODAY
    mod.main()


if __name__ == "__main__":
    main()
