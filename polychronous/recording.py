import gc
import os.path
from typing import Dict

import h5py
from polychronous.utils import is_e2e
from pygenn import SynapseGroup


def get_group(parent, group):
    if group in parent:
        return parent[group]
    else:
        return parent.create_group(group)


def init_spike_recordings(filename, spike_groups_dict, h5_mode="w"):
    with h5py.File(filename, h5_mode) as h5_file:
        for group in spike_groups_dict:
            pop_group = get_group(h5_file, group)
            pop_group.create_dataset("spikes", dtype="float32", shape=(3, 1),
                                     maxshape=(3, None), chunks=(3, 1))

        # def get_all(name):
        #     print(name)
        #
        # h5_file.visit(get_all)


def update_spike_recordings(filename, spike_groups_dict, epoch_idx, h5_mode="a"):
    with h5py.File(filename, h5_mode) as h5_file:
        spike_times = []
        neuron_ids = []

        for group in spike_groups_dict:
            h5_path = os.path.join("/", group, "spikes")
            dst = h5_file[h5_path]

            neuron_pop = spike_groups_dict[group]
            spike_times[:], neuron_ids[:] = neuron_pop.spike_recording_data
            rec_cols = len(spike_times)
            # mb_mem = int(np.round(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))
            # print(f"memory = {mb_mem}, n_spikes = {rec_cols} for {h5_path} at epoch {epoch_idx}")
            # print()

            h5_rows, h5_cols = dst.shape
            if h5_cols == 1:
                h5_cols = 0

            dst.resize((h5_rows, h5_cols + rec_cols))
            dst[0, h5_cols:] = spike_times
            dst[1, h5_cols:] = neuron_ids
            dst[2, h5_cols:] = epoch_idx

        del spike_times, neuron_ids

    gc.collect()


def get_weights(all_synapses: Dict[str, Dict[str, SynapseGroup]]):
    all_weights= {}
    for group_name in all_synapses:
        group_dict = {}
        for syn_name in all_synapses[group_name]:
            syn = all_synapses[group_name][syn_name]
            syn.pull_var_from_device("g")
            group_dict[syn_name] = syn.get_var_values("g").copy()

        all_weights[group_name] = group_dict

    return all_weights


def get_exc_to_exc_synapses(synapse_groups: Dict[str, SynapseGroup]):
    return {k: synapse_groups[k]
            for k in synapse_groups if is_e2e(k)}