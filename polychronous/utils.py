import itertools

def to_delay_step(delay_in_ms, dt):
    return max(0, int((delay_in_ms - 1.0) / dt))


def make_triplets(neuron_ids):
    combinations = itertools.combinations(neuron_ids, 3)
    return combinations


def is_e2e(k):
    splt = k.split("_to_")
    return (splt[0].startswith("exc") and splt[1].startswith("exc"))


def time_to_ms(hours, minutes, seconds):
    hours_to_seconds = 60.0 * 60.0
    minutes_to_seconds = 60.0
    seconds_to_ms = 1000.0

    return (hours * hours_to_seconds + minutes * minutes_to_seconds + seconds) * seconds_to_ms