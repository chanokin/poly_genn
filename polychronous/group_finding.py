import os
import gc
import h5py
import numpy as np

from polychronous.connectivity import sort_by_post, sort_by_pre, conn_to_matrix
from polychronous.constants import (
    PRE, POST, MIN_PRE_FOR_GROUP, TS, IDS,
    SPIKE_T_TOLERANCE_FOR_GROUP,
    MAX_CHAIN_LENGTH_FOR_GROUP
)
from polychronous.spike_finding import find_limit_indices
from polychronous.utils import make_triplets


def get_neurons_with_high_incoming_weights(threshold, weights, connectivity):
    neurons = set()
    for syn_name in weights:
        delay = float(syn_name.split('d')[1])

        # get pre := conns[0], post := conns[1] arrays for a given delay
        conns = connectivity[delay]

        # which of the weights in the connection are higher than threshold
        high_weight_indices = np.where(weights[syn_name] >= threshold)[0]

        # get post neurons ids
        neurons |= set(conns[1][high_weight_indices].tolist())

    return np.asarray(list(neurons))


def transform_connectivity(connectivity, post_ids, threshold=-np.inf):
    """bundle connectivity by post, then delay
        returns a dictionary {post: {delay: [pre_indices]}}
    """
    by_post = {nid: {} for nid in post_ids}
    for delay in connectivity:
        conns = connectivity[delay]
        for post in post_ids:
            list_indices_for_post = np.where(conns[POST] == post)
            by_post[post][delay] = conns[PRE][list_indices_for_post]

    return by_post


def build_group_chains(dt, start_neuron_ids, start_times, spikes,
                       conns_by_post, conns_by_pre, depth=0):
    chain = []
    possible = {}
    possible_set = set()
    for t, pre_id in zip(start_times, start_neuron_ids):
        post_ids = np.asarray(sorted(conns_by_pre[pre_id].keys()))
        if len(post_ids) == 0:
            continue

        delays = np.asarray([conns_by_pre[pre_id][_id][0][1] for _id in post_ids])
        post_times = t + delays + dt

        present_times, present_ids = which_spiked_at_times(post_times, post_ids, spikes)

        # if no post exist in proper time, this chain is cut off
        if len(present_ids) == 0:
            return chain

        print(f"n_post for {pre_id} = {len(present_ids)}")
        possible[pre_id] = (present_times, present_ids)
        possible_set |= set([x for x in zip(present_times.astype('int'), present_ids)])

    print(f"len(possible_set) = {len(possible_set)}")
    print(possible_set)


    return chain


def are_all_inputs_present(candidate_times, candidate_ids, spikes,
                           atol=SPIKE_T_TOLERANCE_FOR_GROUP):
    for t, nid in zip(candidate_times, candidate_ids):
        whr = np.where(np.isclose(spikes[TS], t, atol=atol))[0]
        if len(whr) == 0:
            return False

        nids_spiked_at_time_t = spikes[IDS, whr]
        if nid not in nids_spiked_at_time_t:
            return False

    return True


def which_spiked_at_times(candidate_times, candidate_ids, spikes,
                          atol=SPIKE_T_TOLERANCE_FOR_GROUP):
    times = []
    nids = []
    for t, nid in zip(candidate_times, candidate_ids):
        whr = np.where(np.isclose(spikes[TS], t, atol=atol))[0]
        if len(whr) == 0:
            continue
        ts = spikes[TS, whr]
        ids = spikes[IDS, whr]

        if nid in ids:
            times.append(t)
            nids.append(nid)

    return np.hstack(times), np.hstack(nids)


def find_groups(dt, start_time, max_delay, spikes_to_analyse,
                conns_by_post, conns_by_pre):
    """
        :param conns_by_post: {post: {pre: [(weight, delay)]}}
        :param conns_by_pre: {pre: {post: [(weight, delay)]}}
    """
    groups = {}
    min_time = start_time + max_delay

    indices_for_spikes_above_min_time = np.where(spikes_to_analyse[0] >= min_time)[0]
    sorted_indices = np.argsort(spikes_to_analyse[0, indices_for_spikes_above_min_time])
    indices_for_spikes_above_min_time = indices_for_spikes_above_min_time[sorted_indices]

    for ii, pivot_index in enumerate(indices_for_spikes_above_min_time):
        if ii == (len(indices_for_spikes_above_min_time) - 1):
            break

        pivot_time, pivot_id = spikes_to_analyse[0:2, pivot_index]
        back_spikes = spikes_to_analyse[0:2, :pivot_index].view()

        next_spike_index = indices_for_spikes_above_min_time[ii + 1]
        forward_spikes = spikes_to_analyse[0:2, pivot_index:].view()
        # if this post neuron is connected to fewer than 3 inputs, skip
        if len(conns_by_post[pivot_id]) < MIN_PRE_FOR_GROUP:
            continue

        pre_ids = np.asarray(sorted(conns_by_post[pivot_id].keys()))

        if len(pre_ids) == 0:
            continue

        delays = np.asarray([conns_by_post[pivot_id][pre_id][0][1] for pre_id in pre_ids])
        # make triplets over indices to use them for pre ids and delays
        triplets = make_triplets(np.arange(len(pre_ids)))
        for triplet in triplets:
            triplet = np.asarray(triplet)
            tri_delays = delays[triplet]
            tri_ids = pre_ids[triplet]
            tri_times = pivot_time - tri_delays - dt
            tri_present = are_all_inputs_present(tri_times, tri_ids, back_spikes)

            if not tri_present:
                continue

            text_ids = "{}_{}_{}".format(*tri_ids)
            grp = build_group_chains(dt, tri_ids, tri_times, forward_spikes,
                                     conns_by_post, conns_by_pre)

            if len(grp) == 0:
                continue

            groups[text_ids] = grp



    return groups


def find_groups_by_weights(experiment_filename, threshold, start_time):
    data = np.load(experiment_filename, mmap_mode=True, allow_pickle=True)

    sim_time = data['sim_time']
    half_sim_time = sim_time / 2

    dt = data['dt']
    n_exc = data['n_exc']
    max_delay = data['max_delay']

    weights = data['final_weights'].item()
    connectivity = data['conn_pairs'].item()

    conns_by_post = sort_by_post(weights, connectivity['e_to_e'], np.arange(n_exc), threshold)
    conns_by_pre = sort_by_pre(weights, connectivity['e_to_e'], np.arange(n_exc), threshold)

    weight_matrix = conn_to_matrix(n_exc, n_exc, connectivity['e_to_e'], weights)

    reverse_search = start_time > half_sim_time

    h5_file = h5py.File(data["spikes_filename"].item(), "r")
    exc_spikes = h5_file[os.path.join("exc", "spikes")]
    start_idx, end_idx = find_limit_indices(exc_spikes, start_time, sim_time,
                                            reverse=reverse_search)
    spikes_to_analyse = exc_spikes[:, start_idx:end_idx]

    h5_file.close()
    data.close()
    # try and remove objects from memory
    gc.collect()

    return find_groups(dt, start_time, max_delay, spikes_to_analyse,
                       conns_by_post, conns_by_pre)


def find_groups_by_activity(experiment_filename, threshold, start_time):
    data =  np.load(experiment_filename, mmap_mode=True, allow_pickle=True)

    sim_time = data['sim_time']
    dt = data['dt']
    n_exc = data['n_exc']
    max_delay = data['max_delay']
    post_start = start_time + max_delay + dt

    spikes = data['exc_spikes']
    weights = data['final_weights'].item()
    connectivity = data['conn_pairs'].item()

    weight_matrix = conn_to_matrix(n_exc, n_exc, connectivity['e_to_e'], weights)

    candidate_neurons = get_neurons_with_high_incoming_weights(
                            threshold, weights, connectivity['e_to_e'])

    test_spike_indices = np.where(spikes[_TS] >= start_time)[0]
    candidate_spikes = spikes[:, test_spike_indices]
    spike_neuron_ids = np.unique(candidate_spikes[_IDS])

    sorted_conns = transform_connectivity(connectivity['e_to_e'], spike_neuron_ids)

    starting_spikes = candidate_spikes[:, np.where(candidate_spikes[_TS] < post_start)[0]]

    raw_groups = {nid: {} for nid in spike_neuron_ids}
    for pivot in spike_neuron_ids:
        pivot_time_indices = np.where(
                                np.logical_and(candidate_spikes[_IDS] == pivot,
                                               candidate_spikes[_TS] >= post_start))[0]
        pivot_times = candidate_spikes[_TS][pivot_time_indices]
        pivot_conns = sorted_conns[pivot]
        all_pres = np.hstack([_pre_ids for _, _pre_ids in pivot_conns.items()])

        for pt in pivot_times:
            group_pres = []
            for delay in pivot_conns:
                conn_pres = pivot_conns[delay]
                spike_time = pt - delay - dt
                indices_for_spike_time = np.where(np.isclose(
                                                candidate_spikes[_TS], spike_time,
                                                # atol=1
                                            ))
                pre_ids = candidate_spikes[_IDS][indices_for_spike_time]

                if len(pre_ids) == 0:
                    # print(f"No pre spikes at {spike_time}, delay {delay} for post {pivot}")
                    continue

                pres_in_group = np.intersect1d(conn_pres, pre_ids)

                for pre in pres_in_group:
                    w = weight_matrix[int(pre), int(pivot)]
                    if w > threshold:
                        group_pres.append((pre, spike_time, w, delay))
                    # else:
                        # print(f"Pre found but weight too small {pre}, {pivot} = "
                        #       f"w {w:6.4f}, d {delay}")
            if len(group_pres):
                raw_groups[pivot][pt] = group_pres

    groups = []

    # slim_groups = {int(post): set([int(pre_t_d[0]) for pre_t_d in groups[post]])
    #                 for post in groups}

    return groups

