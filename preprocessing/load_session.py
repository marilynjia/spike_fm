from pathlib import *
import numpy as np
import pandas as pd
import h5py


def read_session_h5(root, eid):
    """Read one session .h5 file.

    Output (dict):
        spike_times       (n_spikes,) float64, seconds
        spike_unit_index  (n_spikes,) int32, row of `units` that fired the spike
        units             DataFrame, one row per unit: unit_index, unit_uuid, cluster_id, probe
        waveforms         (n_units, 40, 128) float32, mean waveform of each unit
        trials            DataFrame, one row per trial: every array in trials/
        wheel_times       (n_samples,) float64, seconds
        wheel_position    (n_samples,) float64, radians
        attrs             dict of file attributes: eid, n_units, t_start_s, t_end_s, ...
    """
    session_path = Path(root) / f'{eid}.h5'
    if not session_path.exists():
        raise FileNotFoundError(f'{session_path} not found')

    with h5py.File(session_path, "r") as f:
        session = {}
        session['spike_times'] = f['spikes']['times'][:]
        session['spike_unit_index'] = f['spikes']['unit_index'][:]
        session['units'] = _make_df(f['units'])
        session['waveforms'] = f['waveforms']['mean'][:]
        session['trials'] = _make_df(f['trials'])
        session['wheel_times'] = f['wheel']['timestamps'][:]
        session['wheel_position'] = f['wheel']['position'][:]
        session['attrs'] = dict(f.attrs)

    return session


def _make_df(group):
    """Read a flat h5 group of equal-length 1D arrays into a DataFrame (one column per array)."""
    columns = {}
    for name, dataset in group.items():
        columns[name] = dataset[:]
    return pd.DataFrame(columns)


def load_session(root, eid, apply_ibl_qc=False):
    """
    """
    session = read_session_h5(root, eid)

    units = pd.read_parquet(Path(root) / ('units.parquet'), filters=[('eid', '==', eid)])
    units = units.sort_values('unit_index').reset_index(drop=True)

    h5_uuids = session['units']['unit_uuid'].str.decode('utf-8').to_numpy()
    if len(units) != len(h5_uuids):
        raise ValueError(f'{eid}: units.parquet has {len(units)} units, h5 has {len(h5_uuids)}')
    if not (units['unit_uuid'].to_numpy() == h5_uuids).all():
        raise ValueError(f'{eid}: unit_uuid order differs between units.parquet and h5')

    if apply_ibl_qc:
        good = (units['label'] == 1).to_numpy()
    else:
        good = np.ones(len(units), dtype=bool)

    old_to_new = np.full(len(good), -1)
    old_to_new[good] = np.arange(good.sum())

    spike_mask = good[session['spike_unit_index']]
    session['spike_times'] = session['spike_times'][spike_mask]
    session['spike_unit_index'] = old_to_new[session['spike_unit_index'][spike_mask]]
    session['waveforms'] = session['waveforms'][good]
    session['units'] = units[good].reset_index(drop=True)

    return session
