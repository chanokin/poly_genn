import itertools


def make_triplets(neuron_ids):
    combinations = itertools.combinations(neuron_ids, 3)
    return combinations