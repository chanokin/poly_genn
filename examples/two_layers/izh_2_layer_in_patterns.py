import os.path
import os
import gc
import psutil
from copy import copy
import numpy as np
from matplotlib import pyplot as plt
from pygenn import genn_wrapper
from pygenn import genn_model
from pathlib import Path
import h5py
from tqdm import tqdm
from datetime import datetime
from polychronous.poisson_source import poisson_input_model
from polychronous.spike_source_array import spike_source_array
from polychronous.connectivity import (
    generate_pairs_and_delays,
    generate_inter_layer_conns
)
from polychronous.stdp_all_synapse import stdp_additive_all_model as stdp_synapse
# from polychronous.stdp_synapse import stdp_synapse
from polychronous.plotting import (
    plot_spikes, plot_weight_histograms, plot_rates
)
from polychronous.group_finding import find_groups

def g_adjust(g, dt):
    return g / dt

def remove_random_input(synapses):
    for s in synapses:
        s.vars["g"].view[:] = 0
        s.push_var_to_device("g")


def freeze_network(plastic_synapses):
    for ps in plastic_synapses:
        syn = plastic_synapses[ps]
        syn.vars["aPlus"].view[:] = 0
        syn.push_var_to_device("aPlus")

        syn.vars["aMinus"].view[:] = 0
        syn.push_var_to_device('aMinus')

def get_group(parent, group):
    if group in parent:
        return parent[group]
    else:
        return parent.create_group(group)


def init_spike_recordings(filename, spike_groups_dict, h5_mode="w"):
    with h5py.File(filename, h5_mode) as h5_file:
        for group in spike_groups:
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

        for group in spike_groups:
            h5_path = os.path.join("/", group, "spikes")
            dst = h5_file[h5_path]

            neuron_pop = spike_groups[group]
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


def time_to_ms(hours, minutes, seconds):
    hours_to_seconds = 60.0 * 60.0
    minutes_to_seconds = 60.0
    seconds_to_ms = 1000.0
    return (hours * hours_to_seconds + minutes * minutes_to_seconds + seconds) * seconds_to_ms

freeze_last_iteration = bool(1)
start_datetime = datetime.now().strftime("%d-%m-%Y__%H-%M")
print(start_datetime)
this_filename = Path(__file__).resolve().stem

# output_filename = f"experiment_{this_filename}_{start_datetime}.npz"
output_filename = f"experiment_{this_filename}.npz"
spikes_filename = f"spikes_for_{output_filename[:-4]}.h5"

np.random.seed(13)

do_all_conns = bool(1)
binding = bool(0)
n_total = 1000
n_exc = int(0.8 * n_total)
n_inh = n_total - n_exc
n_pat = 1000
min_delay = 1
max_delay = 20
conn_probability = 0.1

input_conns = generate_inter_layer_conns(conn_probability, n_pat, n_exc,
                                         min_delay, max_delay)

conn_pairs_0 = generate_pairs_and_delays(conn_probability, n_exc, n_inh,
                                         min_delay, max_delay)

conn_pairs_1 = generate_pairs_and_delays(conn_probability, n_exc, n_inh,
                                         min_delay, max_delay)

l0_to_l1_conns = generate_inter_layer_conns(conn_probability, n_exc, n_exc,
                                            min_delay, max_delay)

l1_to_l0_conns = generate_inter_layer_conns(conn_probability, n_exc, n_exc,
                                            min_delay, max_delay)

# Poisson
stim_init = {
    'rate': 1, # Hz
}

# dt = 0.1 # ms
dt = 1 # ms
sec_to_ms = 1000.

start_stdp_weight = g_adjust(6., dt)
inh_weight = g_adjust(-5., dt)
max_stdp_weight = g_adjust(8., dt)
in2pop_weight = g_adjust(0.01, dt)
pat2pop_weight = g_adjust(20.0, dt)

max_delay = 20 # ms
max_delay_step = int((max_delay + 1) / dt)
delay_steps_dist_params = {
    'min': 1.0, # ms
    'max': max_delay, # ms
}


pattern_size = int(n_pat * 0.1)
pattern_max_time = int(max_delay * 0.5)
pattern_period = 200
pattern_silence = pattern_period - pattern_max_time
n_epochs = 20#00
n_epoch_per_run = min(10, n_epochs)
n_pattern_repeat = 20
pattern_start_t = 10
n_patterns = 2
epoch_sim_time = n_pattern_repeat * n_patterns * pattern_period
patterns = {
    p: [np.random.randint(0, pattern_max_time + 1, pattern_size),
        np.random.choice(np.arange(n_pat), pattern_size, replace=False)]
    for p in range(n_patterns)
}

hours = 2#4
minutes = 10
seconds = 0
# seconds = 5
sim_time = time_to_ms(hours, minutes, seconds)
sim_time = epoch_sim_time * n_epochs
max_sim_time_per_run = 5 * 60 * sec_to_ms # run at most 1 minute at a time
max_sim_time_per_run = min(epoch_sim_time, max_sim_time_per_run)
max_steps_per_run = int(max_sim_time_per_run / dt)
sim_time_per_run = epoch_sim_time * n_epoch_per_run
steps_per_run = int(sim_time_per_run / dt)

sim_steps = int(sim_time / dt)

exc_params = {"a": 0.02, "b": 0.2, "c": -65, "d": 8} # RS

inh_params = {"a": 0.1, "b": 0.2, "c": -65, "d": 2} # FS

exc_synapse_init = {"g": start_stdp_weight,}

stdp_synapse_init = {
    "g": start_stdp_weight,
    "dg": 0.,
    # "aPlus": 0.1,
    # "aMinus": 0.12,
    "aPlus": g_adjust(0.1, dt),
    "aMinus": g_adjust(0.12, dt),
}
stdp_pre_init = {
    'preTrace': 0,
}

stdp_post_init = {
    'postTrace': 0,
}


inh_synapse_init = {"g": inh_weight,}


spike_times = [[] for _ in range(n_pat)]
for e in range(n_epoch_per_run):
    for p in patterns:
        for rep in range(n_pattern_repeat):
            for t, nid in zip(*patterns[p]):
                spike_times[nid].append(t + pattern_start_t)
            pattern_start_t += pattern_period

        # pattern_start_t += pattern_silence

# LIF neuron initial state
nrn_init = {"V": -65, "U": 0.2 * -65}


stdp_params = {
    "tauPlus": 20, "tauMinus": 20,
    "wMin": 0.0, "wMax": max_stdp_weight,
    "delayDecay": 0,
}
stdp_params["tauPlusDecay"] = np.exp(-dt/stdp_params["tauPlus"])
stdp_params["tauMinusDecay"] = np.exp(-dt/stdp_params["tauMinus"])

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Create model using single-precion and 1ms timesteps
model = genn_model.GeNNModel("float", this_filename)
model.dT = dt

start_spike = np.empty(shape=n_pat, dtype=np.uint32)
end_spike = np.empty(shape=n_pat, dtype=np.uint32)

cum_size = 0
for i, seq in enumerate(spike_times):
    start_spike[i] = cum_size
    cum_size += len(seq)
    end_spike[i] = cum_size

pat_ini = {
    "startSpike": start_spike,
    "endSpike": end_spike
}

pattern_pop = model.add_neuron_population(
                    "pattern_pop", n_pat, spike_source_array,
                    {}, pat_ini)
flat_spikes =  np.hstack([sts for sts in spike_times if len(sts)]).flatten()
pattern_pop.set_extra_global_param(
    "spikeTimes", flat_spikes
)

exc_pop_0 = model.add_neuron_population(
            "exc_0", n_exc, "Izhikevich", exc_params, nrn_init)

inh_pop_0 = model.add_neuron_population(
            "inh_0", n_inh, "Izhikevich", inh_params, nrn_init)

exc_pop_1 = model.add_neuron_population(
            "exc_1", n_exc, "Izhikevich", exc_params, nrn_init)

inh_pop_1 = model.add_neuron_population(
            "inh_1", n_inh, "Izhikevich", inh_params, nrn_init)


spike_groups = {
    # "input": stim,
    "pattern": pattern_pop,
    "exc_0": exc_pop_0,
    "inh_0": inh_pop_0,
    "exc_1": exc_pop_1,
    "inh_1": inh_pop_1,
}

for p in spike_groups:
    spike_groups[p].spike_recording_enabled = True


# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

net_conns_0 = {}
for conn_name in conn_pairs_0:
    if '_to_' not in conn_name:
        continue

    s, t = conn_name.split('_to_')
    source = exc_pop_0 if s == 'e' else inh_pop_0
    target = exc_pop_0 if t == 'e' else inh_pop_0
    if s == 'e':
        conn_g_init = stdp_synapse_init #if t == 'e' else exc_synapse_init
        pre_wu_init = stdp_pre_init #if t == 'e' else {}
        post_wu_init = stdp_post_init #if t == 'e' else {}
        synapse_type = stdp_synapse #if t == 'e' else 'StaticPulse'
        synapse_params = stdp_params #if t == 'e' else {}

    else:
        conn_g_init = inh_synapse_init
        pre_wu_init = {}
        post_wu_init = {}
        synapse_type = 'StaticPulse'
        synapse_params = {}

    for delay in conn_pairs_0[conn_name]:
        _synapse_params = copy(synapse_params)
        # if s == 'e' and t == 'e':
        if s == 'e':
            _synapse_params['delay'] = delay
            _synapse_params['delayDecay'] = np.exp(-delay/_synapse_params['tauPlus'])

        synapse_name = f"{source.name}_to_{target.name}_d{int(delay)}"
        print(f"setting up synapse {synapse_name}")
        delay_step = (delay if delay == genn_wrapper.NO_DELAY
                      else max(0, int((delay-1) / dt)))
        net_conns_0[synapse_name] =  model.add_synapse_population(
            synapse_name, "SPARSE_INDIVIDUALG", delay_step,
            source, target,
            synapse_type, _synapse_params, conn_g_init,
            pre_wu_init, post_wu_init,
            "DeltaCurr", {}, {},
        )
        pre_indices, post_indices = conn_pairs_0[conn_name][delay]
        net_conns_0[synapse_name].set_sparse_connections(
                                    pre_indices, post_indices)

net_conns_1 = {}
for conn_name in conn_pairs_1:
    if '_to_' not in conn_name:
        continue

    s, t = conn_name.split('_to_')
    source = exc_pop_1 if s == 'e' else inh_pop_1
    target = exc_pop_1 if t == 'e' else inh_pop_1
    if s == 'e':
        conn_g_init = stdp_synapse_init #if t == 'e' else exc_synapse_init
        pre_wu_init = stdp_pre_init #if t == 'e' else {}
        post_wu_init = stdp_post_init #if t == 'e' else {}
        synapse_type = stdp_synapse #if t == 'e' else 'StaticPulse'
        synapse_params = stdp_params #if t == 'e' else {}

    else:
        conn_g_init = inh_synapse_init
        pre_wu_init = {}
        post_wu_init = {}
        synapse_type = 'StaticPulse'
        synapse_params = {}

    for delay in conn_pairs_1[conn_name]:
        _synapse_params = copy(synapse_params)
        # if s == 'e' and t == 'e':
        if s == 'e':
            _synapse_params['delay'] = delay
            _synapse_params['delayDecay'] = np.exp(-delay/_synapse_params['tauPlus'])

        synapse_name = f"{source.name}_to_{target.name}_d{int(delay)}"
        print(f"setting up synapse {synapse_name}")
        delay_step = (delay if delay == genn_wrapper.NO_DELAY
                      else max(0, int((delay-1) / dt)))
        net_conns_1[synapse_name] =  model.add_synapse_population(
            synapse_name, "SPARSE_INDIVIDUALG", delay_step,
            source, target,
            synapse_type, _synapse_params, conn_g_init,
            pre_wu_init, post_wu_init,
            "DeltaCurr", {}, {},
        )
        pre_indices, post_indices = conn_pairs_1[conn_name][delay]
        net_conns_1[synapse_name].set_sparse_connections(
                                    pre_indices, post_indices)

e0_to_e1_conns = {}
conns_for_loop = l0_to_l1_conns
for delay in conns_for_loop:
    conn_g_init = stdp_synapse_init  # if t == 'e' else exc_synapse_init
    pre_wu_init = stdp_pre_init  # if t == 'e' else {}
    post_wu_init = stdp_post_init  # if t == 'e' else {}
    synapse_type = stdp_synapse  # if t == 'e' else 'StaticPulse'
    synapse_params = stdp_params  # if t == 'e' else {}
    source = exc_pop_0
    target = exc_pop_1
    _synapse_params = copy(synapse_params)
    _synapse_params['delay'] = delay
    _synapse_params['delayDecay'] = np.exp(-delay / _synapse_params['tauPlus'])

    synapse_name = f"{source.name}_to_{target.name}_d{int(delay)}"
    print(f"setting up synapse {synapse_name}")
    delay_step = (delay if delay == genn_wrapper.NO_DELAY
                  else max(0, int((delay - 1) / dt)))
    e0_to_e1_conns[synapse_name] = model.add_synapse_population(
        synapse_name, "SPARSE_INDIVIDUALG", delay_step,
        source, target,
        synapse_type, _synapse_params, conn_g_init,
        pre_wu_init, post_wu_init,
        "DeltaCurr", {}, {},
    )
    pre_indices, post_indices = conns_for_loop[delay]
    e0_to_e1_conns[synapse_name].set_sparse_connections(pre_indices, post_indices)

e1_to_e0_conns = {}
conns_for_loop = l1_to_l0_conns
for delay in conns_for_loop:
    conn_g_init = stdp_synapse_init  # if t == 'e' else exc_synapse_init
    pre_wu_init = stdp_pre_init  # if t == 'e' else {}
    post_wu_init = stdp_post_init  # if t == 'e' else {}
    synapse_type = stdp_synapse  # if t == 'e' else 'StaticPulse'
    synapse_params = stdp_params  # if t == 'e' else {}
    source = exc_pop_1
    target = exc_pop_0
    _synapse_params = copy(synapse_params)
    _synapse_params['delay'] = delay
    _synapse_params['delayDecay'] = np.exp(-delay / _synapse_params['tauPlus'])

    synapse_name = f"{source.name}_to_{target.name}_d{int(delay)}"
    print(f"setting up synapse {synapse_name}")
    delay_step = (delay if delay == genn_wrapper.NO_DELAY
                  else max(0, int((delay - 1) / dt)))
    e1_to_e0_conns[synapse_name] = model.add_synapse_population(
        synapse_name, "SPARSE_INDIVIDUALG", delay_step,
        source, target,
        synapse_type, _synapse_params, conn_g_init,
        pre_wu_init, post_wu_init,
        "DeltaCurr", {}, {},
    )
    pre_indices, post_indices = conns_for_loop[delay]
    e1_to_e0_conns[synapse_name].set_sparse_connections(pre_indices, post_indices)


pat2pop_conns = {}
for delay in input_conns:
    conn_g_init = stdp_synapse_init  # if t == 'e' else exc_synapse_init
    pre_wu_init = stdp_pre_init  # if t == 'e' else {}
    post_wu_init = stdp_post_init  # if t == 'e' else {}
    synapse_type = stdp_synapse  # if t == 'e' else 'StaticPulse'
    synapse_params = stdp_params  # if t == 'e' else {}
    source = pattern_pop
    target = exc_pop_0
    _synapse_params = copy(synapse_params)
    _synapse_params['delay'] = delay
    _synapse_params['delayDecay'] = np.exp(-delay / _synapse_params['tauPlus'])

    synapse_name = f"{source.name}_to_{target.name}_d{int(delay)}"
    print(f"setting up synapse {synapse_name}")
    delay_step = (delay if delay == genn_wrapper.NO_DELAY
                  else max(0, int((delay - 1) / dt)))
    pat2pop_conns[synapse_name] = model.add_synapse_population(
        synapse_name, "SPARSE_INDIVIDUALG", delay_step,
        source, target,
        synapse_type, _synapse_params, conn_g_init,
        pre_wu_init, post_wu_init,
        "DeltaCurr", {}, {},
    )
    pre_indices, post_indices = input_conns[delay]
    pat2pop_conns[synapse_name].set_sparse_connections(pre_indices, post_indices)

# Build and load model
model.build(force_rebuild=True)
model.load(num_recording_timesteps=steps_per_run)

# [net_conns[k].pull_connectivity_from_device()
#  for k in net_conns if 'e_to_e' in k]
[net_conns_0[k].pull_var_from_device('g')
 for k in net_conns_0 if 'e_to_e' in k]
initial_weights = {k: net_conns_0[k].get_var_values('g').copy()
                   for k in net_conns_0 if 'e_to_e' in k}

# **************************************************************************** #
# **************************************************************************** #
# **************************************************************************** #

h5_mode = "w"
if os.path.isfile(spikes_filename):
    os.remove(spikes_filename)

init_spike_recordings(spikes_filename, spike_groups)

weights = []
# Simulate model
for epoch_index in tqdm(range(0, n_epochs, n_epoch_per_run)):
    t_step = 0
    # if epoch_index >= (n_epochs - n_epoch_per_run):
    #     freeze_network({k: net_conns[k] for k in net_conns if 'e_to_e' in k})
    #     remove_random_input([in2exc, in2inh])

    while t_step < steps_per_run:
        model.step_time()
        t_step += 1

    model.pull_recording_buffers_from_device()

    update_spike_recordings(spikes_filename, spike_groups, epoch_index)

    pattern_pop.vars["startSpike"].view[:] = start_spike
    pattern_pop.push_var_to_device("startSpike")

    pattern_pop.extra_global_params["spikeTimes"].view[:] += sim_time_per_run
    pattern_pop.push_extra_global_param_to_device("spikeTimes")


[net_conns_0[k].pull_var_from_device('g')
 for k in net_conns_0 if 'e_to_e' in k]
final_weights_0 = {k: net_conns_0[k].get_var_values('g').copy()
                   for k in net_conns_0 if 'e_to_e' in k}

[net_conns_1[k].pull_var_from_device('g')
 for k in net_conns_1 if 'e_to_e' in k]
final_weights_1 = {k: net_conns_1[k].get_var_values('g').copy()
                   for k in net_conns_1 if 'e_to_e' in k}

[pat2pop_conns[k].pull_var_from_device('g') for k in pat2pop_conns]
final_pat2pop_weights = {k: pat2pop_conns[k].get_var_values('g').copy()
                         for k in pat2pop_conns}

[l0_to_l1_conns[k].pull_var_from_device('g') for k in l0_to_l1_conns]
final_l0_to_l1_weights = {k: l0_to_l1_conns[k].get_var_values('g').copy()
                          for k in l0_to_l1_conns}

[l1_to_l0_conns[k].pull_var_from_device('g') for k in l1_to_l0_conns]
final_l1_to_l0_weights = {k: l1_to_l0_conns[k].get_var_values('g').copy()
                          for k in l1_to_l0_conns}

final_weights = dict(
    final_weights_0=final_weights_0,
    final_weights_1=final_weights_1,
    final_pat2pop_weights=final_pat2pop_weights,
    final_l0_to_l1_weights=final_l0_to_l1_weights,
    final_l1_to_l0_weights=final_l1_to_l0_weights
)

experiment_data = dict(
    pattern_size = pattern_size,
    pattern_max_time = pattern_max_time,
    n_patterns = n_patterns,
    patterns = patterns,
    pattern_silence = pattern_period,
    n_pattern_repeat = n_pattern_repeat,
    pattern_start_t = pattern_start_t,
    spikes_filename=spikes_filename,
    initial_weights=initial_weights,
    final_weights=final_weights,
    final_pat2pop_weights=final_pat2pop_weights,
    n_exc=n_exc,
    n_inh=n_inh,
    exc_params=exc_params,
    inh_params=inh_params,
    min_delay=min_delay,
    max_delay=max_delay,
    conn_pairs=conn_pairs,
    conn_probability=conn_probability,
    sim_time=sim_time,
    dt=dt,
    stim_init=stim_init,
    stdp_params=stdp_params,
    exc_synapse_init=exc_synapse_init,
    inh_synapse_init=inh_synapse_init,
    min_weight=0,
    max_stdp_weight=max_stdp_weight,
)


np.savez_compressed(output_filename, **experiment_data)


spikes_file = h5py.File(spikes_filename, 'r')
pat_spikes = spikes_file[os.path.join("pattern", "spikes")]
exc_spikes = spikes_file[os.path.join("exc", "spikes")]
inh_spikes = spikes_file[os.path.join("inh", "spikes")]
n_pat_spikes = pat_spikes.shape

analysis_length = min(sim_time, 1000)
analysis_start = 0
analysis_end = analysis_start + analysis_length
plot_spikes(pat_spikes, exc_spikes, inh_spikes, n_exc, dt, sim_time,
            analysis_start, analysis_end, analysis_length)
plt.savefig("start_spikes.png", dpi=150)

analysis_length = min(sim_time, 1000)
analysis_start = sim_time - analysis_length
analysis_end = analysis_start + analysis_length
plot_spikes(pat_spikes, exc_spikes, inh_spikes, n_exc, dt, sim_time,
            analysis_start, analysis_end, analysis_length)
plt.savefig("start_spikes.png", dpi=150)

plot_weight_histograms(initial_weights, final_weights_0)
plt.savefig("weights_0_histograms.png", dpi=150)

plot_weight_histograms(initial_weights, final_weights_1)
plt.savefig("weights_1_histograms.png", dpi=150)

plot_weight_histograms(initial_weights, final_pat2pop_weights)
plt.savefig("input_weight_histograms.png", dpi=150)

plt.show()
