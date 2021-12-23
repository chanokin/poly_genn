import numpy as np
import itertools

_TS, _IDS = 0, 1
_PRE, _POST = 0, 1

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


def transform_connectivity(connectivity, post_ids):
    """bundle connectivity by post, then delay
        returns a dictionary {post: {delay: [pre_indices]}}
    """
    by_post = {nid: {} for nid in post_ids}
    for delay in connectivity:
        conns = connectivity[delay]
        for post in post_ids:
            list_indices_for_post = np.where(conns[1] == post)
            by_post[post][delay] = conns[_PRE][list_indices_for_post]

    return by_post

def conn_to_matrix(n_source, n_target, connectivity, weights):
    weight_matrix = np.zeros((n_source, n_target))
    for synapse_name in weights:
        delay = int(synapse_name.split('d')[1])
        conns = connectivity[delay]
        n_conns = len(conns[_PRE])
        for index in range(n_conns):
            row, col = conns[_PRE][index], conns[_POST][index]
            # weight_matrix[row, col] = max(weight_matrix[row, col], weights[index])
            weight_matrix[row, col] = weights[synapse_name][index]

    return weight_matrix


def make_triplets(neuron_ids):
    combinations = itertools.combinations(neuron_ids, 3)
    return combinations


def find_groups(experiment_filename, threshold, start_time):
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

