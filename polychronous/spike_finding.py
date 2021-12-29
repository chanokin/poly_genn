import numpy as np
from tqdm import tqdm
from polychronous.constants import PRE, POST, TS, IDS

def find_limit_indices(spikes, start_ms, end_ms, points_per_chunk, reverse=False):
    if reverse:
        return find_limit_indices_reverse(spikes, start_ms, end_ms, points_per_chunk)
    else:
        return find_limit_indices_forward(spikes, start_ms, end_ms, points_per_chunk)


def find_limit_indices_forward(spikes, start_ms, end_ms, points_per_chunk):
    n_times = spikes.shape[1]
    start_ms_idx = None
    end_ms_idx = None
    start_found = False
    end_found = False
    for start_idx in tqdm(range(0, n_times, points_per_chunk), desc="Finding spike indices"):
        end_idx = min(n_times, start_idx + points_per_chunk)
        spike_times = spikes[TS, start_idx:end_idx]

        if spike_times[-1] < start_ms:
            continue

        if not start_found:
            whr = np.where(spike_times >= start_ms)[0]
            if len(whr):
                start_found = True
                start_ms_idx = start_idx + np.min(whr)
            else:
                continue

        if start_found and not end_found:
            whr = np.where(spike_times > end_ms)[0]
            if len(whr):
                end_found = True
                end_ms_idx = start_idx + np.min(whr)

        if start_found and end_found:
            break

    return start_ms_idx, end_ms_idx


def find_limit_indices_reverse(spikes, start_ms, end_ms, points_per_chunk):
    n_times = spikes.shape[1]
    start_ms_idx = None
    end_ms_idx = None
    start_found = False
    end_found = False
    for end_idx in tqdm(range(n_times, 0, -points_per_chunk), desc="Finding spike indices"):
        start_idx = max(0, end_idx - points_per_chunk)

        spike_times = spikes[TS, start_idx:end_idx]
        min_t = np.min(spike_times)
        max_t = np.max(spike_times)

        if spike_times[0] > end_ms:
            continue

        if not end_found:
            whr = np.where(spike_times <= end_ms)[0]
            if len(whr):
                end_found = True
                end_ms_idx = start_idx + np.max(whr)
            else:
                continue

        if end_found and not start_found:
            whr = np.where(spike_times < start_ms)[0]
            if len(whr):
                start_found = True
                start_ms_idx = start_idx + np.max(whr)

        if start_found and end_found:
            break

    return start_ms_idx, end_ms_idx