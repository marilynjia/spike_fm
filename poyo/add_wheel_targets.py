"""Add the POYO decoding target and train/valid/test splits to converted IBL sessions.

To each h5, add:
    data.behavior      IrregularTimeSeries  50 Hz wheel velocity (rad/s), wheel_vel (N, 1)
    data.train_domain  Interval             70 % of the trials
    data.valid_domain  Interval             10 % of the trials
    data.test_domain   Interval             20 % of the trials

Whole trials are assigned to splits at random, so every split covers the whole
session (engaged and disengaged periods alike). A time split would put the
last, mostly disengaged trials in validation only.

Usage:
    python add_wheel_targets.py ../data/ibl_bwm/*.h5
"""

import sys
from pathlib import Path

import h5py
import numpy as np

from torch_brain.data import Data, Interval, IrregularTimeSeries

SPLIT_FRACTIONS = {"train": 0.7, "valid": 0.1, "test": 0.2}
SEED = 0


def split_trials(trials, seed=SEED):
    """Randomly assign whole trials to splits; return {split: Interval}."""
    n = len(trials.start)
    order = np.random.default_rng(seed).permutation(n)
    n_train = round(SPLIT_FRACTIONS["train"] * n)
    n_valid = round(SPLIT_FRACTIONS["valid"] * n)
    picks = {
        "train": order[:n_train],
        "valid": order[n_train : n_train + n_valid],
        "test": order[n_train + n_valid :],
    }
    start, end = trials.start[:], trials.end[:]
    return {
        name: Interval(start=start[np.sort(idx)], end=end[np.sort(idx)])
        for name, idx in picks.items()
    }


def add_targets(path):
    with h5py.File(path, "r") as file:
        data = Data.from_hdf5(file)
        data.materialize()

    if not hasattr(data, "wheel"):
        raise RuntimeError(f"{path.name} has no wheel data; reconvert without --no-wheel")

    t, pos = data.wheel.timestamps, data.wheel.position
    grid = np.arange(t[0], t[-1], 0.02)  # 50 Hz
    vel = np.gradient(np.interp(grid, t, pos), 0.02)  # rad/s
    data.behavior = IrregularTimeSeries(
        timestamps=grid, wheel_vel=vel[:, None], domain="auto"
    )

    splits = split_trials(data.trials)
    for name, interval in splits.items():
        setattr(data, f"{name}_domain", interval)

    tmp = path.with_suffix(".tmp.h5")
    with h5py.File(tmp, "w") as file:
        data.to_hdf5(file)
    tmp.replace(path)

    print(f"{path.name}:")
    for name, interval in splits.items():
        inside = np.zeros(len(grid), dtype=bool)
        for a, b in zip(interval.start, interval.end):
            inside |= (grid >= a) & (grid < b)
        print(f"  {name:5s} {len(interval.start):4d} trials, "
              f"{(interval.end - interval.start).sum():6.0f} s, "
              f"wheel_vel std = {vel[inside].std():.3f}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        add_targets(Path(arg))
