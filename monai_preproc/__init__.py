"""Additive MONAI-based pre-processing for the Paxos2025 pipeline.

Mirrors cropping/ + informative_frames/ (same split CSVs, same anti-leakage
val-fold protocol, same CLEAN_<video>.mp4 output naming so data/match_clips.py
and data/dataset.py keep working) but builds models/transforms on MONAI, with
a configurable experiment grid for cost/performance trade-offs.

Nothing in the existing cropping/, informative_frames/, data/ or scripts/
packages is modified by this package.
"""

from .configs import PreprocConfig, DEFAULT_GRID, ARCH_GRID, GRIDS, load_grid

__all__ = ["PreprocConfig", "DEFAULT_GRID", "ARCH_GRID", "GRIDS", "load_grid"]
