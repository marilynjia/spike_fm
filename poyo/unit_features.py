"""Unit identity features, computed from each unit's spike times (no binning)
and, optionally, its mean waveform.

Spike features (NUM_SPIKE_FEATURES = 3 + ACG_BINS):
    log firing rate      log1p(spikes / s)
    ISI CV               std / mean of the inter-spike intervals (1 for a Poisson unit)
    refractory fraction  fraction of inter-spike intervals shorter than 2 ms
    autocorrelogram      log1p of the spike counts at lags 0-50 ms in 1 ms bins,
                         divided by what a Poisson unit with the same rate would
                         give (so a flat ACG is log1p(1) everywhere)

Waveform features (NUM_WAVEFORM_FEATURES = 4 + 64), from the peak channel of
``units.waveform`` (40 channels x 128 samples at 30 kHz; only convert_sessions.py --aws-root stores it):
    log amplitude        log1p of the peak-to-peak amplitude in microvolts
    trough-to-peak       ms from the trough to the following peak (narrow vs broad spikes)
    peak fraction        peak / (peak + |trough|)
    spatial spread       number of channels with at least half the peak-to-peak amplitude
    shape                peak-channel trace divided by its largest |value|, averaged
                         over pairs of samples (128 -> 64)

Features use the whole session.
"""

import numpy as np

ACG_BIN = 0.001  # s
ACG_BINS = 50
REFRACTORY = 0.002  # s
NUM_SPIKE_FEATURES = 3 + ACG_BINS

WAVEFORM_RATE = 30000  # Hz
NUM_WAVEFORM_FEATURES = 4 + 64


def _isi_stats(times):
    """ISI CV and the fraction of ISIs inside the refractory period."""
    isi = np.diff(times)
    if len(isi) < 2:
        return 0.0, 0.0
    cv = isi.std() / max(isi.mean(), 1e-9)
    refractory = (isi < REFRACTORY).mean()
    return cv, refractory


def _autocorrelogram(times, rate):
    """Spike counts at lags (0, ACG_BIN], (ACG_BIN, 2 ACG_BIN], ... relative to Poisson.

    For each spike, searchsorted counts the spikes up to `lag` later; summing over
    spikes and differencing across lags gives the count in each lag bin.
    """
    if len(times) < 2 or rate == 0:
        return np.zeros(ACG_BINS)
    edges = np.arange(ACG_BINS + 1) * ACG_BIN
    cumulative = np.array(
        [np.searchsorted(times, times + e, side="right").sum() for e in edges],
        dtype=np.float64,
    )
    counts = np.diff(cumulative)
    expected = len(times) * rate * ACG_BIN
    return counts / expected


def waveform_features(waveform):
    """Feature vector (NUM_WAVEFORM_FEATURES,) of one mean waveform (channels, samples)."""
    w = np.nan_to_num(waveform)  # channels past the probe edge are NaN
    ptp = w.max(1) - w.min(1)
    amplitude = ptp.max()
    if amplitude == 0:
        return np.zeros(NUM_WAVEFORM_FEATURES, dtype=np.float32)

    trace = w[ptp.argmax()]
    trough = trace.argmin()
    peak = trough + trace[trough:].argmax()
    trough_to_peak = (peak - trough) / WAVEFORM_RATE * 1e3
    peak_fraction = trace[peak] / (trace[peak] + abs(trace[trough]) + 1e-12)
    spread = (ptp >= 0.5 * amplitude).sum()
    shape = (trace / np.abs(trace).max()).reshape(-1, 2).mean(1)
    scalars = [np.log1p(amplitude * 1e6), trough_to_peak, peak_fraction, spread]
    return np.concatenate([scalars, shape]).astype(np.float32)


def compute_unit_features(times, duration, waveform=None):
    """Feature vector of one unit: spike features, then waveform features if given.

    Args:
        times: sorted spike times of the unit, in seconds.
        duration: length of the recording, in seconds.
        waveform: mean waveform (channels, samples), or None.
    """
    rate = len(times) / duration
    cv, refractory = _isi_stats(times)
    acg = _autocorrelogram(times, rate)
    parts = [[np.log1p(rate), cv, refractory], np.log1p(acg)]
    if waveform is not None:
        parts.append(waveform_features(waveform))
    return np.concatenate(parts).astype(np.float32)


def build_feature_table(dataset, vocab, waveforms=False):
    """Features of every unit in `dataset`, one row per word of `vocab`.

    Row vocab[unit_id] holds that unit's features, z-scored across units. Row 0
    is the padding word "NA" and stays zero.

    Args:
        dataset: a torch_brain spiking Dataset (e.g. IBLBWMDataset).
        vocab: dict unit_id -> row, i.e. POYO's ``model.unit_emb.vocab``.
        waveforms: also use ``units.waveform`` (files from convert_sessions.py --aws-root).
    """
    num_features = NUM_SPIKE_FEATURES + (NUM_WAVEFORM_FEATURES if waveforms else 0)
    table = np.zeros((len(vocab), num_features), dtype=np.float32)
    filled = np.zeros(len(vocab), dtype=bool)

    for rid in dataset.recording_ids:
        data = dataset.get_recording(rid)
        times = np.asarray(data.spikes.timestamps[:])
        unit_index = np.asarray(data.spikes.unit_index[:])
        domain = data.spikes.domain
        duration = float((np.asarray(domain.end[:]) - np.asarray(domain.start[:])).sum())
        unit_ids = data.units.id.tolist()

        wf = [None] * len(unit_ids)
        if waveforms:
            if not hasattr(data.units, "waveform"):
                raise ValueError(f"{rid} has no units.waveform; convert it with convert_sessions.py --aws-root")
            wf = np.asarray(data.units.waveform[:])

        # group spikes by unit; the stable sort keeps each unit's spikes in time order
        order = np.argsort(unit_index, kind="stable")
        bounds = np.searchsorted(unit_index[order], np.arange(len(unit_ids) + 1))
        for i, uid in enumerate(unit_ids):
            row = vocab[uid]
            unit_times = times[order[bounds[i] : bounds[i + 1]]]
            table[row] = compute_unit_features(unit_times, duration, wf[i])
            filled[row] = True

    mu = table[filled].mean(0)
    sd = table[filled].std(0) + 1e-6
    table[filled] = (table[filled] - mu) / sd
    return table
