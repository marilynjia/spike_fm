"""Ridge regression baseline: binned spike counts -> wheel velocity.

Features are each unit's spike counts (square-root transformed) in six windows
around every 50 Hz target time, from 200 ms before to 200 ms after. The ridge
penalty is picked on the validation split. Uses the same trial splits as POYO.

Usage (from spike_fm/):
    python models/ridge.py data/ibl_bwm/*.h5

Writes results/ridge/{valid,test}.npz and metrics.json in the same format as
poyo/eval.py, so the plotting scripts can compare the two.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from torch_brain.data import Data

WINDOWS = [(-0.2, -0.1), (-0.1, -0.05), (-0.05, 0.0), (0.0, 0.05), (0.05, 0.1), (0.1, 0.2)]
ALPHAS = [1e2, 1e3, 1e4, 1e5]
SPLITS = ["train", "valid", "test"]


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def split_index(timestamps, interval):
    """Indices of the timestamps that fall inside the interval."""
    idx = [
        np.arange(np.searchsorted(timestamps, a), np.searchsorted(timestamps, b))
        for a, b in zip(interval.start[:], interval.end[:])
    ]
    return np.concatenate(idx)


def features(data, t):
    """Spike counts of every unit in each window around the times t."""
    st, ui = data.spikes.timestamps[:], data.spikes.unit_index[:]
    n_units = len(data.units.id)
    X = np.zeros((len(t), n_units * len(WINDOWS)), np.float32)
    for u in range(n_units):
        su = st[ui == u]
        for k, (a, b) in enumerate(WINDOWS):
            counts = np.searchsorted(su, t + b) - np.searchsorted(su, t + a)
            X[:, u * len(WINDOWS) + k] = np.sqrt(counts)
    return X


def fit_session(path):
    with h5py.File(path, "r") as file:
        data = Data.from_hdf5(file)
        data.materialize()

    bt, v = data.behavior.timestamps, data.behavior.wheel_vel[:, 0]
    idx = {s: split_index(bt, getattr(data, f"{s}_domain")) for s in SPLITS}
    X = {s: features(data, bt[idx[s]]) for s in SPLITS}
    y = {s: v[idx[s]] for s in SPLITS}

    mu, sd = X["train"].mean(0), X["train"].std(0) + 1e-6
    X = {s: (x - mu) / sd for s, x in X.items()}
    ym = y["train"].mean()
    G = X["train"].T @ X["train"]
    c = X["train"].T @ (y["train"] - ym)

    best = None
    for alpha in ALPHAS:
        w = np.linalg.solve(G + alpha * np.eye(len(G), dtype=G.dtype), c)
        score = r2(y["valid"], X["valid"] @ w + ym)
        if best is None or score > best[0]:
            best = (score, alpha, w)
    _, alpha, w = best

    session_id = data.session.id
    print(f"{session_id}: alpha = {alpha:g}")
    preds = {s: X[s] @ w + ym for s in SPLITS}
    trials = dict(
        trial_start=data.trials.start,
        trial_end=data.trials.end,
        first_movement_time=data.trials.first_movement_time,
        choice=data.trials.choice,
    )
    return session_id, bt, idx, y, preds, trials, alpha


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/ridge"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    metrics = {}
    per_split = {s: {k: [] for k in ["session_id", "timestamps", "true", "pred", "trial"]}
                 for s in ["valid", "test"]}
    tables = {}
    for path in args.files:
        rid, bt, idx, y, preds, trials, alpha = fit_session(path)
        metrics[f"alpha/{rid}"] = alpha
        tables[rid] = trials
        for s in SPLITS:
            metrics[f"{s}/{rid}"] = r2(y[s], preds[s])
        for s in per_split:
            ts = bt[idx[s]]
            start, end = trials["trial_start"], trials["trial_end"]
            k = np.searchsorted(start, ts, side="right") - 1
            in_trial = (k >= 0) & (ts < end[np.clip(k, 0, None)])
            f = per_split[s]
            f["session_id"].append(np.full(len(ts), rid))
            f["timestamps"].append(ts)
            f["true"].append(y[s])
            f["pred"].append(preds[s])
            f["trial"].append(np.where(in_trial, k, -1))

    for s in SPLITS:
        metrics[f"{s}/avg"] = float(np.mean(
            [v for k, v in metrics.items() if k.startswith(f"{s}/") and not k.endswith("avg")]
        ))
    for s, f in per_split.items():
        arrays = {k: np.concatenate(v) for k, v in f.items()}
        for rid, table in tables.items():
            arrays.update({f"{rid}/{k}": v for k, v in table.items()})
        np.savez_compressed(args.out / f"{s}.npz", **arrays)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"valid R² = {metrics['valid/avg']:.3f}, test R² = {metrics['test/avg']:.3f}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
