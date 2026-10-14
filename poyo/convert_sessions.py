"""Convert one IBL Brain-Wide Map session to a torch_brain `Data` file (HDF5).

Usage:
    python convert_sessions.py --eid <session id>          # all probes of the session
    python convert_sessions.py --pid <probe insertion id>  # one probe only

Output:
    <out-dir>/<eid>.h5, with this structure:

    data.spikes   IrregularTimeSeries  timestamps (s), unit_index (row of data.units)
    data.units    ArrayDict            id, probe, cluster_id, region, atlas_id, label, x, y, z, depth
    data.trials   Interval             start, end, event times, choice, contrasts, block prior
    data.wheel    IrregularTimeSeries  timestamps (s), position (rad)   [optional]
    data.session / data.subject / data.device / data.brainset   metadata

Then run add_wheel_targets.py on the h5 to add the decoding target and splits.

Requirements: pip install ONE-api ibllib torch_brain
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader

from torch_brain.data import (
    ArrayDict,
    BrainsetDescription,
    Data,
    DeviceDescription,
    Interval,
    IrregularTimeSeries,
    SessionDescription,
    SubjectDescription,
)

IBL_PUBLIC_URL = "https://openalyx.internationalbrainlab.org"


def load_probe(one, eid, pname, good_only):
    """Load the sorted spikes and the unit table of one probe.

    Returns (spike_times, spike_unit_index, units), where spike_unit_index
    points to a row of `units` (0 .. number of kept units - 1).
    """
    ssl = SpikeSortingLoader(eid=eid, pname=pname, one=one)
    spikes, clusters, channels = ssl.load_spike_sorting()
    clusters = ssl.merge_clusters(spikes, clusters, channels)

    cluster_id = np.asarray(clusters["cluster_id"])
    label = np.asarray(clusters["label"], dtype=np.float64)
    n = len(cluster_id)

    # label == 1 means that the unit passes all three IBL quality criteria
    keep_unit = (label == 1) if good_only else np.ones(n, dtype=bool)
    kept_ids = cluster_id[keep_unit]

    lookup = np.full(int(cluster_id.max()) + 1, -1, dtype=np.int64)
    lookup[kept_ids] = np.arange(len(kept_ids))

    unit_index = lookup[np.asarray(spikes["clusters"])]
    keep_spike = (unit_index >= 0) & np.isfinite(spikes["times"])

    def column(*names, dtype=np.float64, fill=np.nan):
        """Read a per-unit column; use `fill` if this probe does not have it."""
        for name in names:
            if name in clusters:
                return np.asarray(clusters[name])[keep_unit].astype(dtype)
        return np.full(len(kept_ids), fill).astype(dtype)

    units = dict(
        id=np.array([f"{pname}_{c}" for c in kept_ids], dtype=str),
        probe=np.array([pname] * len(kept_ids), dtype=str),
        cluster_id=kept_ids.astype(np.int64),
        region=column("acronym", "acronyms", dtype=str, fill="unknown"),
        atlas_id=column("atlas_id", dtype=np.int64, fill=-1),
        label=label[keep_unit],
        x=column("x"),
        y=column("y"),
        z=column("z"),
        depth=column("depths"),
    )
    return (
        np.asarray(spikes["times"], dtype=np.float64)[keep_spike],
        unit_index[keep_spike],
        units,
    )


def load_trials(one, eid):
    """Load the trials table as a torch_brain Interval (one row per trial)."""
    trials = one.load_object(eid, "trials", collection="alf")

    if "intervals" in trials:
        start, end = np.asarray(trials["intervals"], dtype=np.float64).T
    else:  # older table layout
        start = np.asarray(trials["intervals_0"], dtype=np.float64)
        end = np.asarray(trials["intervals_1"], dtype=np.float64)

    ok = np.isfinite(start) & np.isfinite(end) & (end > start)

    def col(name, dtype=np.float64):
        return np.asarray(trials[name], dtype=dtype)[ok]

    contrast_left = col("contrastLeft")
    contrast_right = col("contrastRight")

    return Interval(
        start=start[ok],
        end=end[ok],
        stim_on_time=col("stimOn_times"), # all event times can be NaN
        go_cue_time=col("goCue_times"),
        first_movement_time=col("firstMovement_times"),
        response_time=col("response_times"),
        feedback_time=col("feedback_times"),
        choice=col("choice", np.int64),  # -1 / +1 = wheel direction, 0 = no response
        feedback_type=col("feedbackType", np.int64),  # +1 = reward, -1 = error
        contrast_left=contrast_left,  # NaN if the stimulus was on the right
        contrast_right=contrast_right,  # NaN if the stimulus was on the left
        signed_contrast=np.nan_to_num(contrast_right) - np.nan_to_num(contrast_left),
        probability_left=col("probabilityLeft"),  # block prior: 0.2, 0.5 or 0.8
        # Time-valued fields must be listed here, so that torch_brain shifts
        # them together with start/end when it slices a window.
        timekeys=[
            "start",
            "end",
            "stim_on_time",
            "go_cue_time",
            "first_movement_time",
            "response_time",
            "feedback_time",
        ],
    )


def load_wheel(one, eid):
    """Load the wheel position trace as an IrregularTimeSeries (in radians)."""
    wheel = one.load_object(eid, "wheel", collection="alf")
    t = np.asarray(wheel["timestamps"], dtype=np.float64)
    pos = np.asarray(wheel["position"], dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(pos)
    return IrregularTimeSeries(timestamps=t[ok], position=pos[ok], domain="auto")


def convert_session(one, eid, pnames, good_only=True, with_wheel=True):
    """Build the torch_brain Data object for one session."""
    eid = str(eid)

    # 1. Spikes and units, for each probe. Unit indices of the later probes
    #    are shifted so that they point into the concatenated unit table.
    times, index, unit_tables, offset = [], [], [], 0
    for pname in pnames:
        t, i, u = load_probe(one, eid, pname, good_only)
        print(f"  {pname}: {len(u['id'])} units, {len(t):,} spikes")
        times.append(t)
        index.append(i + offset)
        unit_tables.append(u)
        offset += len(u["id"])

    times, index = np.concatenate(times), np.concatenate(index)
    if offset == 0 or len(times) == 0:
        raise RuntimeError("No units left after filtering. Try --all-units.")

    order = np.argsort(times, kind="stable")  # torch_brain needs sorted timestamps
    spikes = IrregularTimeSeries(
        timestamps=times[order], unit_index=index[order], domain="auto"
    )
    units = ArrayDict(
        **{k: np.concatenate([u[k] for u in unit_tables]) for k in unit_tables[0]}
    )

    # 2. Behaviour.
    fields = dict(spikes=spikes, units=units, trials=load_trials(one, eid))
    if with_wheel:
        try:
            fields["wheel"] = load_wheel(one, eid)
        except Exception as err:  # the wheel is optional
            print(f"  wheel not loaded ({type(err).__name__}: {err})")

    # 3. Metadata.
    details = one.get_details(eid)
    start_time = details.get("start_time")
    recording_date = datetime.fromisoformat(str(start_time)) if start_time else None

    return Data(
        brainset=BrainsetDescription(
            id="ibl_brain_wide_map",
            origin_version="0.0.0",
            derived_version="1.0.0",
            source=IBL_PUBLIC_URL,
            description="IBL Brain-Wide Map: Neuropixels recordings in mice "
            "during a visual decision-making task.",
        ),
        subject=SubjectDescription(
            id=str(details.get("subject", "unknown")), species="MUS_MUSCULUS"
        ),
        session=SessionDescription(id=eid, recording_date=recording_date),
        device=DeviceDescription(id="+".join(pnames), recording_tech="NEUROPIXELS_SPIKES"),
        domain="auto",
        **fields,
    )


def save(data, path):
    """Write the Data object to HDF5."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(data, "save"):
        data.save(str(path))
    else:
        import h5py

        with h5py.File(path, "w") as file:
            data.to_hdf5(file)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--eid", help="session id: converts all probes of the session")
    which.add_argument("--pid", help="probe insertion id: converts this probe only")
    parser.add_argument("--out-dir", default="data/ibl_bwm", help="output directory")
    parser.add_argument("--cache-dir", default=None, help="ONE download directory")
    parser.add_argument("--all-units", action="store_true",
                        help="keep all sorted units, not only those with label == 1")
    parser.add_argument("--no-wheel", action="store_true", help="do not store the wheel")
    args = parser.parse_args()

    one_args = dict(base_url=IBL_PUBLIC_URL, password="international", silent=True)
    if args.cache_dir:
        one_args["cache_dir"] = args.cache_dir
    one = ONE(**one_args)

    if args.pid:
        eid, pname = one.pid2eid(args.pid)
        pnames = [pname]
    else:
        eid = args.eid
        _, pnames = one.eid2pid(eid)
        pnames = sorted(pnames)
    eid = str(eid)
    print(f"Session {eid}, probes: {', '.join(pnames)}")

    data = convert_session(
        one, eid, pnames, good_only=not args.all_units, with_wheel=not args.no_wheel
    )

    path = Path(args.out_dir) / f"{eid}.h5"
    save(data, path)

    # Read the file again (lazily) to confirm that it is valid.
    check = Data.load(str(path))
    regions, counts = np.unique(np.asarray(check.units.region).astype(str), return_counts=True)
    top = ", ".join(f"{r} ({c})" for r, c in sorted(zip(regions, counts), key=lambda p: -p[1])[:8])
    print(f"Saved {path}")
    print(f"  units:  {len(check.units.id)}")
    print(f"  spikes: {len(check.spikes.timestamps):,}")
    print(f"  trials: {len(check.trials.start)}")
    print(f"  top regions: {top}")


if __name__ == "__main__":
    main()