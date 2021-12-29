import numpy as np
from polychronous.constants import POST, PRE


def generate_exc_to_exc(conn_pairs, n_exc, min_delay, max_delay):
    """:param conn_pairs: a matrix containing pre-synaptic ids, each column
                          represents a post-synaptic neuron"""
    ### Excitatory to Excitatory
    delayed_pair_indices = np.where(conn_pairs[:, :n_exc] < n_exc)

    n_exc_to_exc = len(delayed_pair_indices[0])
    # generate as many random delays as exc-to-exc connections found
    # max is inclusive here
    delays = np.random.random_integers(min_delay, max_delay, n_exc_to_exc)

    # we have to use homogeneous delays in GeNN so we sort by delay to generate
    # one 'projection' for each delay 'slot'
    delayed_conns = {}
    for d in range(min_delay, max_delay+1):
        pair_ids_for_delay = np.where(delays == d)[0]
        # since conn_pairs contains pre_ids, we need to look for them via
        # delayed_pair_indices
        pre_ids = conn_pairs[delayed_pair_indices[0][pair_ids_for_delay],
                             delayed_pair_indices[1][pair_ids_for_delay]]

        # columns are already post ids, no need to 'parse' through conn_pairs
        post_ids = delayed_pair_indices[POST][pair_ids_for_delay]
        delayed_conns[d] = (pre_ids, post_ids)

    return delayed_conns


def generate_inh_to_inh(conn_pairs, n_exc):
    # get inhibitory pre (>= n_exc) for inhibitory posts ([:, n_exc:])
    pairs = np.where(conn_pairs[:, n_exc:] >= n_exc)

    # need to add n_exc to column because we reduced the search space before
    incoming_ids = conn_pairs[pairs[0], pairs[1] + n_exc]

    # remove n_exc so that we can map to separate inh/exc populations
    incoming_ids -= n_exc

    # because we'll keep excitatory and inhibitory populations separated, we just
    # don't add back the n_exc here
    i2i = (incoming_ids, pairs[1])

    return {1: i2i} # delay set to 1 always


def generate_exc_to_inh(conn_pairs, n_exc):
    # get excitatory pre (< n_exc) for inhibitory posts ([:, n_exc:])
    pairs = np.where(conn_pairs[:, n_exc:] < n_exc)

    # need to add n_exc to column because we reduced the search space before
    incoming_ids = conn_pairs[pairs[0], pairs[1] + n_exc]

    # because we'll keep excitatory and inhibitory populations separated, we just
    # don't add back the n_exc here
    e2i =  (incoming_ids, pairs[1])

    return {1: e2i} # delay set to 1 always


def generate_inh_to_exc(conn_pairs, n_exc):
    # get inhibitory pre (>= n_exc) for excitatory posts ([:, :n_exc])
    pairs = np.where(conn_pairs[:, :n_exc] >= n_exc)

    # incoming are all inhibitory here
    incoming_ids = conn_pairs[pairs[0], pairs[1]]
    # remove n_exc so that we can map to separate inh/exc populations
    incoming_ids -= n_exc

    i2e =  (incoming_ids, pairs[1])

    return {1: i2e} # delay set to 1 always



def generate_pairs_and_delays(conn_prob:float, n_exc:int, n_inh:int,
                              min_delay:int, max_delay:int, seed=-1):
    total = n_exc + n_inh
    n_incoming = int(conn_prob * total)
    if seed != -1:
        np.random.seed(seed)

    # each number here is a pre-synaptic neuron id, each column represents
    # a post-synaptic neuron (either exc or inh) NOTE: max is inclusive
    conn_pairs = np.empty((n_incoming, total), dtype='int')
    all_ids = np.arange(total)
    for post in range(total):
        conn_pairs[:, post] = np.random.choice(
                                        all_ids, size=n_incoming, replace=False)
        whr = np.where(conn_pairs[:, post] == post)[0]
        if len(whr):
            remaining = np.setdiff1d(all_ids, conn_pairs[:, post])
            conn_pairs[whr, post] = np.random.choice(
                                        remaining, size=len(whr), replace=False)

    conn_dict = {
        'original_pairs': conn_pairs,
        'e_to_e': generate_exc_to_exc(conn_pairs, n_exc, min_delay, max_delay),
        'e_to_i': generate_exc_to_inh(conn_pairs, n_exc),
        'i_to_e': generate_inh_to_exc(conn_pairs, n_exc),
        # 'i_to_i': generate_inh_to_inh(conn_pairs, n_exc),
    }
    # import matplotlib.pyplot as plt
    # k = 'e_to_e'
    # for d in conn_dict[k]:
    #     fig, ax = plt.subplots(2, 1, figsize=(5, 7))
    #     ax[0].hist(conn_dict[k][d][0])
    #     ax[0].set_title(f"pres for {k} delay {d}")
    #     ax[1].hist(conn_dict[k][d][1])
    #     ax[1].set_title(f"posts for {k} delay {d}")
    # plt.show()
    # import sys
    # sys.exit(0)
    return conn_dict


def get_weight_key_for_delay(delay, weights_delay):
    return [k for k in weights_delay
            if int(k.split("d")[1]) == delay][0]


def sort_by_post(weights, connectivity, post_ids, threshold):
    by_post = {p: {} for p in post_ids}

    for delay in connectivity:
        pairs = connectivity[delay]
        wkey = get_weight_key_for_delay(delay, weights)
        weights_for_delay = weights[wkey]

        for post in by_post:
            whr = np.where(
                    np.logical_and(
                        pairs[_POST] == post, weights_for_delay > threshold))

            for array_idx in whr[0]:
                pre_id = pairs[_PRE][array_idx]
                weight = weights_for_delay[array_idx]
                pre_list = by_post[post].get(pre_id, [])
                pre_list.append((weight, delay))
                by_post[post][pre_id] = pre_list

    return by_post


def sort_by_pre(weights, connectivity, pre_ids, threshold):
    by_pre = {p: {} for p in pre_ids}

    for delay in connectivity:
        pairs = connectivity[delay]
        wkey = get_weight_key_for_delay(delay, weights)
        weights_for_delay = weights[wkey]

        for pre in by_pre:
            whr = np.where(
                    np.logical_and(
                        pairs[_PRE] == pre, weights_for_delay > threshold))

            for array_idx in whr[0]:
                post_id = pairs[_POST][array_idx]
                weight = weights_for_delay[array_idx]
                pre_list = by_pre[pre].get(post_id, [])
                pre_list.append((weight, delay))
                by_pre[pre][post_id] = pre_list

    return by_pre


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