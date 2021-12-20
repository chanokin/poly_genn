import numpy as np

def generate_exc_to_exc(conn_pairs, n_exc, min_delay, max_delay):
    ### Excitatory to Excitatory
    delayed_pairs = np.where(conn_pairs[:, :n_exc] < n_exc)
    # max is inclusive here
    delays = np.random.random_integers(min_delay, max_delay, len(delayed_pairs[0]))

    # we have to use homogeneous delays in GeNN so we sort by delay to generate
    # one 'projection' for each delay 'slot'
    delayed_conns = {}
    for d in range(min_delay, max_delay+1):
        pair_ids_for_delay = np.where(delays == d)[0]
        incoming_ids = conn_pairs[delayed_pairs[0][pair_ids_for_delay],
                                  delayed_pairs[1][pair_ids_for_delay]]
        target_ids = delayed_pairs[1][pair_ids_for_delay]
        delayed_conns[d] = (incoming_ids, # pre
                            target_ids) # post
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
    conn_pairs = np.random.random_integers(
                    0, total-1, size=(n_incoming, total))

    conn_dict = {
        'e2e': generate_exc_to_exc(conn_pairs, n_exc, min_delay, max_delay),
        'e2i': generate_exc_to_inh(conn_pairs, n_exc),
        'i2e': generate_inh_to_exc(conn_pairs, n_exc),
        'i2i': generate_inh_to_inh(conn_pairs, n_exc),
    }

    return conn_dict
