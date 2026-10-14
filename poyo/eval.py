"""Evaluate a POYO checkpoint and save its predictions for plotting.

Usage (from poyo/):
    python eval.py outputs/2026-10-14/08-51-51/checkpoints/epoch_020.pt

Writes to ../results/<name>/ (name defaults to poyo_<run time>_<checkpoint>):
    <split>.npz    per sample: session_id, timestamps, true, pred, trial (-1 = none),
                   plus each session's trial table (trial_start, trial_end,
                   first_movement_time, choice) for event-aligned plots
    metrics.json   R² per session and split

The dataset settings (e.g. normalize_std) come from the run's .hydra/config.yaml,
so predictions are scored in the same units as during training and saved in rad/s.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchmetrics
from datasets.wrapper import PoyoDatasetWrapper
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm
from utils import move_to_device

from torch_brain.batching import collate
from torch_brain.models import POYO
from torch_brain.samplers import SequentialFixedWindowSampler
from torch_brain.utils.stitcher import stitch


def predict_split(model, ds, split, batch_size, device):
    """Run the model over one split; return {session_id: (timestamps, pred, true)}."""
    sampler = SequentialFixedWindowSampler(
        sampling_intervals=ds.get_sampling_intervals(split),
        window_length=model.sequence_length,
        step=model.sequence_length / 2.0,
    )
    loader = torch.utils.data.DataLoader(
        ds, sampler=sampler, batch_size=batch_size, collate_fn=collate
    )
    cache = {rid: ([], [], []) for rid in ds.recording_ids}
    with torch.no_grad():
        for X, Y in tqdm(loader, desc=split, leave=False):
            X, Y = move_to_device((X, Y), device)
            pred = model(**X, output_timestamps=Y["timestamps"])
            for i in range(len(pred)):
                m = Y["eval_mask"][i]
                t, p, y = cache[Y["session_id"][i]]
                t.append(Y["timestamps"][i][m] + Y["absolute_start"][i])
                p.append(pred[i][m])
                y.append(Y["values"][i][m])

    out = {}
    for rid, (t, p, y) in cache.items():
        if not t:
            continue
        t = torch.cat(t)
        # overlapping windows predict the same timestamp more than once; average them
        ts, p = stitch(t, torch.cat(p))
        _, y = stitch(t, torch.cat(y))
        out[rid] = (ts.cpu().numpy(), p.cpu().numpy(), y.cpu().numpy())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--splits", nargs="+", default=["valid", "test"])
    parser.add_argument("--name", default=None, help="results subfolder name")
    parser.add_argument("--results-dir", type=Path, default=Path("../results"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    run_dir = args.checkpoint.resolve().parent.parent
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    name = args.name or f"poyo_{run_dir.parent.name}_{run_dir.name}_{args.checkpoint.stem}"
    out_dir = args.results_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = POYO.load_pretrained(args.checkpoint).to(device).eval()
    base_ds = instantiate(cfg.dataset, root=cfg.data_root)
    base_ds.transform = instantiate(cfg.eval_transform)
    ds = PoyoDatasetWrapper(base_ds, model.tokenize)

    metrics = {"checkpoint": str(args.checkpoint)}
    for split in args.splits:
        results = predict_split(model, ds, split, args.batch_size, device)
        fields = {k: [] for k in ["session_id", "timestamps", "true", "pred", "trial"]}
        trial_tables = {}
        for rid, (ts, pred, true) in results.items():
            data = base_ds.get_recording(rid)
            cfg_r = data.readout_config
            pred = pred * cfg_r.normalize_std + cfg_r.normalize_mean  # back to rad/s
            true = true * cfg_r.normalize_std + cfg_r.normalize_mean
            metrics[f"{split}/{rid}"] = torchmetrics.functional.r2_score(
                torch.from_numpy(pred), torch.from_numpy(true)
            ).item()

            start, end = data.trials.start[:], data.trials.end[:]
            k = np.searchsorted(start, ts, side="right") - 1
            in_trial = (k >= 0) & (ts < end[np.clip(k, 0, None)])
            fields["session_id"].append(np.full(len(ts), rid))
            fields["timestamps"].append(ts)
            fields["true"].append(true[:, 0])
            fields["pred"].append(pred[:, 0])
            fields["trial"].append(np.where(in_trial, k, -1))
            trial_tables[rid] = dict(
                trial_start=start,
                trial_end=end,
                first_movement_time=data.trials.first_movement_time[:],
                choice=data.trials.choice[:],
            )

        session_r2 = [v for k, v in metrics.items() if k.startswith(f"{split}/")]
        metrics[f"{split}/avg"] = float(np.mean(session_r2))
        arrays = {k: np.concatenate(v) for k, v in fields.items()}
        for rid, table in trial_tables.items():
            arrays.update({f"{rid}/{k}": v for k, v in table.items()})
        np.savez_compressed(out_dir / f"{split}.npz", **arrays)
        print(f"{split}: R² = {metrics[f'{split}/avg']:.3f}")

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
