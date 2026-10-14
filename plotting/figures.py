"""Wheel-velocity decoding figure: POYO vs the ridge baseline.

    A  recorded vs decoded velocity on a few example test trials
    B  R² per model on the validation and test splits
    C  velocity aligned to first wheel movement, averaged over test trials,
       split by the mouse's choice

Reads only saved results (poyo/eval.py and models/ridge.py).

Usage (from spike_fm/):
    python plotting/figures.py --poyo results/poyo_2026-10-14_08-51-51_epoch_020
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import style

ALIGN_WINDOW = (-0.2, 1.0)  # s around first movement
BIN = 0.02  # s, matches the 50 Hz targets


def load(result_dir, split):
    npz = np.load(Path(result_dir) / f"{split}.npz")
    metrics = json.loads((Path(result_dir) / "metrics.json").read_text())
    return npz, metrics


def by_trial(npz, session_id):
    """{trial index: (timestamps, true, pred)} for one session."""
    m = (npz["session_id"] == session_id) & (npz["trial"] >= 0)
    t, y, p, k = npz["timestamps"][m], npz["true"][m], npz["pred"][m], npz["trial"][m]
    return {int(i): (t[k == i], y[k == i], p[k == i]) for i in np.unique(k)}


def panel_examples(ax, poyo, ridge, n_trials=5, gap=0.6):
    """Concatenate a few test trials left to right, separated by a small gap."""
    trials = sorted(set(poyo) & set(ridge))
    # pick consecutive test trials with the most wheel movement, so the example is informative
    scores = [sum(np.abs(poyo[k][1]).mean() for k in trials[i:i + n_trials])
              for i in range(len(trials) - n_trials + 1)]
    chosen = trials[int(np.argmax(scores)):][:n_trials]

    offset = 0.0
    for j, k in enumerate(chosen):
        t, y, p = poyo[k]
        tr, _, pr = ridge[k]
        x = t - t[0] + offset
        ax.plot(x, y, color=style.COLORS["true"], lw=1.2, **({"label": style.LABELS["true"]} if not j else {}))
        ax.plot(tr - t[0] + offset, pr, color=style.COLORS["ridge"], lw=0.9,
                **({"label": style.LABELS["ridge"]} if not j else {}))
        ax.plot(x, p, color=style.COLORS["poyo"], lw=0.9,
                **({"label": style.LABELS["poyo"]} if not j else {}))
        if j:
            ax.axvline(offset - gap / 2, color=style.GRID, lw=0.8, zorder=0)
        offset = x[-1] + gap
    ax.axhline(0, color=style.GRID, lw=0.6, zorder=0)
    ax.set_xlim(0, offset - gap)
    ax.set_xlabel(f"Time (s) — {len(chosen)} test trials, separated by gray lines")
    ax.set_ylabel("Wheel velocity (rad/s)")
    ax.legend(loc="upper right", ncol=3, handlelength=1.2)


def panel_r2(ax, metrics):
    splits = ["valid", "test"]
    models = ["poyo", "ridge"]
    width = 0.36
    for i, model in enumerate(models):
        values = [metrics[model][f"{s}/avg"] for s in splits]
        x = np.arange(len(splits)) + (i - 0.5) * (width + 0.04)
        ax.bar(x, values, width, color=style.COLORS[model], label=style.LABELS[model])
        for xi, v in zip(x, values):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6,
                    color=style.INK)
    ax.set_xticks(range(len(splits)), ["Validation", "Test"])
    ax.set_ylabel("R²")
    ax.set_ylim(0, max(0.7, ax.get_ylim()[1] + 0.1))
    ax.yaxis.grid(True, color=style.GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", ncol=2, handlelength=1.0)


def aligned_mean(trials, table, choice, which):
    """Mean velocity around first movement over trials with the given choice."""
    edges = np.arange(ALIGN_WINDOW[0], ALIGN_WINDOW[1] + BIN / 2, BIN)
    rows = []
    for k, (t, y, p) in trials.items():
        fmt = table["first_movement_time"][k]
        if table["choice"][k] != choice or not np.isfinite(fmt):
            continue
        rel = t - fmt
        values = y if which == "true" else p
        idx = np.digitize(rel, edges) - 1
        ok = (idx >= 0) & (idx < len(edges) - 1)
        row = np.full(len(edges) - 1, np.nan)
        for b in np.unique(idx[ok]):
            row[b] = values[ok][idx[ok] == b].mean()
        rows.append(row)
    centers = edges[:-1] + BIN / 2
    return centers, np.nanmean(rows, axis=0), len(rows)


def panel_aligned(ax, poyo, ridge, table, choice, title):
    for name, trials, which in [("true", poyo, "true"), ("ridge", ridge, "pred"),
                                ("poyo", poyo, "pred")]:
        x, m, n = aligned_mean(trials, table, choice, which)
        ax.plot(x, m, color=style.COLORS[name], lw=1.2 if name == "true" else 0.9)
    ax.axvline(0, color=style.GRID, lw=0.8, zorder=0)
    ax.axhline(0, color=style.GRID, lw=0.6, zorder=0)
    ax.set_title(f"{title} (n = {n})", fontweight="normal", fontsize=7)
    ax.set_xlabel("Time from first movement (s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--poyo", type=Path, required=True)
    parser.add_argument("--ridge", type=Path, default=style.RESULTS_DIR / "ridge")
    parser.add_argument("--session", default=None, help="session id (default: first)")
    parser.add_argument("--name", default="fig_decoding")
    args = parser.parse_args()
    style.apply()

    poyo_npz, poyo_metrics = load(args.poyo, "test")
    ridge_npz, ridge_metrics = load(args.ridge, "test")
    session = args.session or str(poyo_npz["session_id"][0])
    poyo, ridge = by_trial(poyo_npz, session), by_trial(ridge_npz, session)
    table = {k: poyo_npz[f"{session}/{k}"] for k in ["first_movement_time", "choice"]}

    fig = plt.figure(figsize=(style.DOUBLE_COL, 4.2), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], width_ratios=[1, 1, 1])
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c1 = fig.add_subplot(grid[1, 1])
    ax_c2 = fig.add_subplot(grid[1, 2], sharey=ax_c1)

    panel_examples(ax_a, poyo, ridge)
    ax_a.set_title("A  Decoded wheel velocity, held-out test trials")
    panel_r2(ax_b, {"poyo": poyo_metrics, "ridge": ridge_metrics})
    ax_b.set_title("B  Decoding accuracy")
    # IBL choice: -1 / +1 is the wheel direction the mouse turned
    panel_aligned(ax_c1, poyo, ridge, table, -1, "C  Choice = −1")
    panel_aligned(ax_c2, poyo, ridge, table, 1, "Choice = +1")
    ax_c1.set_ylabel("Mean wheel velocity (rad/s)")
    plt.setp(ax_c2.get_yticklabels(), visible=False)

    fig.suptitle(f"Session {session[:8]}", x=0.99, ha="right", fontsize=6,
                 color=style.INK_MUTED, fontweight="normal")
    style.save(fig, args.name)


if __name__ == "__main__":
    main()
