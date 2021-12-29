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
from polychronous.connectivity_generation import generate_pairs_and_delays
from polychronous.stdp_all_synapse import stdp_additive_all_model as stdp_synapse
# from polychronous.stdp_synapse import stdp_synapse
from polychronous.plotting import (
    plot_spikes, plot_weight_histograms, plot_rates
)
from polychronous.find_groups import find_groups

def freeze_network(plastic_synapses):
    for ps in plastic_synapses:
        syn = plastic_synapses[ps]
        syn.vars['aPlus'].view[:] = 0
        syn.push_var_to_device('aPlus')

        syn.vars['aMinus'].view[:] = 0
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

output_filename = f"experiment_{this_filename}_{start_datetime}.npz"
spikes_filename = f"spikes_for_{output_filename[:-4]}.h5"

np.random.seed(13)

do_all_conns = bool(1)
binding = bool(0)
n_total = 1000
n_exc = int(0.8 * n_total)
n_inh = n_total - n_exc
min_delay = 1
max_delay = 20
conn_probability = 0.1
conn_pairs = generate_pairs_and_delays(conn_probability, n_exc, n_inh,
                                       min_delay, max_delay)

# Poisson
stim_init = {
    'rate': 1, # Hz
}

dt = 1.0 # ms
sec_to_ms = 1000.


max_delay = 20 # ms
max_delay_step = int((max_delay + 1) / dt)
delay_steps_dist_params = {
    'min': 1.0, # ms
    'max': max_delay, # ms
}

pattern_size = int(n_exc * 0.1)
pattern_max_time = int(max_delay * 0.5)
pattern_silence = 200 - pattern_max_time
n_pattern_repeat = 6#000
pattern_start_t = 0
n_patterns = 2
patterns = {
    p: [np.random.randint(0, pattern_max_time + 1, pattern_size),
        np.random.choice(np.arange(n_exc), pattern_size, replace=False)]
    for p in range(n_patterns)
}

hours = 24
minutes = 10
seconds = 0
# seconds = 5
sim_time = time_to_ms(hours, minutes, seconds)
# sim_time = n_patterns * (pattern_silence + pattern_max_time) * n_pattern_repeat
max_sim_time_per_run = 5 * 60 * sec_to_ms # run at most 1 minute at a time
max_sim_time_per_run = min(sim_time, max_sim_time_per_run)
max_steps_per_run = int(max_sim_time_per_run / dt)

sim_steps = int(sim_time / dt)

exc_params = {"a": 0.02, "b": 0.2, "c": -65, "d": 8} # RS

inh_params = {"a": 0.1, "b": 0.2, "c": -65, "d": 2} # FS

exc_synapse_init = {"g": 6.,}

stdp_synapse_init = {
    "g": 6.,
    "dg": 0.,
    "aPlus": 0.1,
    "aMinus": 0.12,
}
stdp_pre_init = {
    'preTrace': 0,
}

stdp_post_init = {
    'postTrace': 0,
}

inh_synapse_init = {"g": -5.,}
max_weight = 10.
in2pop_weight = 20.
pat2pop_weight = 0.

spike_times = [[] for _ in range(n_exc)]
for rep in range(n_pattern_repeat):
    for p in patterns:
        for t, nid in zip(*patterns[p]):
            spike_times[nid].append(t + pattern_start_t)
        pattern_start_t +=  pattern_max_time + pattern_silence

    # pattern_start_t += pattern_silence

# LIF neuron initial state
nrn_init = {"V": -65, "U": 0.2 * -65}


stdp_params = {
    "tauPlus": 20, "tauMinus": 20,
    "wMin": 0.0, "wMax": max_weight,
    "delayDecay": 0,
}
stdp_params["tauPlusDecay"] = np.exp(-dt/stdp_params["tauPlus"])
stdp_params["tauMinusDecay"] = np.exp(-dt/stdp_params["tauMinus"])

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Create model using single-precion and 1ms timesteps
model = genn_model.GeNNModel("float", this_filename)
model.dT = dt

stim = model.add_neuron_population(
        "Stim", n_total, poisson_input_model,
        {'dt': dt}, stim_init)
stim.spike_recording_enabled = True



start_spike = np.empty(shape=n_exc, dtype=np.uint32)
end_spike = np.empty(shape=n_exc, dtype=np.uint32)

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
                    "pattern_pop", n_exc, spike_source_array,
                    {}, pat_ini)
flat_spikes =  np.hstack([sts for sts in spike_times if len(sts)]).flatten()
pattern_pop.set_extra_global_param(
    "spikeTimes", flat_spikes
)
pattern_pop.spike_recording_enabled = True


exc_pop = model.add_neuron_population(
            "exc", n_exc, "Izhikevich", exc_params, nrn_init)

exc_pop.spike_recording_enabled = True

inh_pop = model.add_neuron_population(
            "inh", n_inh, "Izhikevich", inh_params, nrn_init)

inh_pop.spike_recording_enabled = True


in2exc = model.add_synapse_population(
    "StimToExc", "SPARSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
    stim, exc_pop,
    "StaticPulse", {}, {'g': in2pop_weight}, {}, {},
    "DeltaCurr", {}, {},
)
in2exc.set_sparse_connections(np.arange(n_exc), np.arange(n_exc))

in2inh = model.add_synapse_population(
    "StimToInh", "SPARSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
    stim, inh_pop,
    "StaticPulse", {}, {'g': in2pop_weight}, {}, {},
    "DeltaCurr", {}, {},
)
in2inh.set_sparse_connections(np.arange(n_inh) + n_exc, np.arange(n_inh))

pat2pop = model.add_synapse_population(
    "PatToPop", "SPARSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
    pattern_pop, exc_pop,
    "StaticPulse", {}, {'g': pat2pop_weight}, {}, {},
    "DeltaCurr", {}, {},
    genn_model.init_connectivity("OneToOne", {})
)

net_conns = {}
if do_all_conns:
    for conn_name in conn_pairs:
        if '_to_' not in conn_name:
            continue

        s, t = conn_name.split('_to_')
        source = exc_pop if s == 'e' else inh_pop
        target = exc_pop if t == 'e' else inh_pop
        if s == 'e':
            conn_g_init = stdp_synapse_init if t == 'e' else exc_synapse_init
            pre_wu_init = stdp_pre_init if t == 'e' else {}
            post_wu_init = stdp_post_init if t == 'e' else {}

        else:
            conn_g_init = inh_synapse_init
            pre_wu_init = {}
            post_wu_init = {}

        synapse_type = stdp_synapse if s == 'e' and t == 'e' else 'StaticPulse'
        synapse_params = stdp_params if s == 'e' and t == 'e' else {}
        for delay in conn_pairs[conn_name]:
            _synapse_params = copy(synapse_params)
            if s == 'e' and t == 'e':
                _synapse_params['delay'] = delay
                _synapse_params['delayDecay'] = np.exp(-delay/_synapse_params['tauPlus'])

            synapse_name = f"{s}_to_{t}_d{delay}"
            net_conns[synapse_name] =  model.add_synapse_population(
                synapse_name, "SPARSE_INDIVIDUALG", delay,
                source, target,
                synapse_type, _synapse_params, conn_g_init,
                pre_wu_init, post_wu_init,
                "DeltaCurr", {}, {},
            )
            pre_indices, post_indices = conn_pairs[conn_name][delay]
            net_conns[synapse_name].set_sparse_connections(
                                        pre_indices, post_indices)

# Build and load model
model.build(force_rebuild=True)
model.load(num_recording_timesteps=max_steps_per_run)

# [net_conns[k].pull_connectivity_from_device()
#  for k in net_conns if 'e_to_e' in k]
[net_conns[k].pull_var_from_device('g')
 for k in net_conns if 'e_to_e' in k]
initial_weights = {k: net_conns[k].get_var_values('g').copy()
                   for k in net_conns if 'e_to_e' in k}



h5_mode = "w"
spike_groups = {
    "input": stim,
    "pattern": pattern_pop,
    "exc": exc_pop,
    "inh": inh_pop,
}
if os.path.isfile(spikes_filename):
    os.remove(spikes_filename)

init_spike_recordings(spikes_filename, spike_groups)

weights = []
# Simulate model
n_global_steps = int(np.ceil(sim_time / dt)) // max_steps_per_run
for global_step in tqdm(range(n_global_steps)):
    t_step = 0
    if global_step == (n_global_steps - 1):
        freeze_network({k: net_conns[k] for k in net_conns if 'e_to_e' in k})

    while t_step < max_steps_per_run:
        model.step_time()
        t_step += 1
    model.pull_recording_buffers_from_device()

    update_spike_recordings(spikes_filename, spike_groups, global_step)



# [net_conns[k].pull_connectivity_from_device()
#  for k in net_conns if 'e_to_e' in k]
[net_conns[k].pull_var_from_device('g')
 for k in net_conns if 'e_to_e' in k]
final_weights = {k: net_conns[k].get_var_values('g').copy()
                   for k in net_conns if 'e_to_e' in k}

experiment_data = dict(
    pattern_size = pattern_size,
    pattern_max_time = pattern_max_time,
    n_patterns = n_patterns,
    patterns = patterns,
    pattern_silence = pattern_silence,
    n_pattern_repeat = n_pattern_repeat,
    pattern_start_t = pattern_start_t,
    spikes_filename=spikes_filename,
    initial_weights=initial_weights,
    final_weights=final_weights,
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
    max_weight=max_weight,
)


np.savez_compressed(output_filename, **experiment_data)

# analysis_start = sim_time - 2 * 60 * sec_to_ms

# plot_spikes(pat_spikes, exc_spikes, inh_spikes,
#             n_exc, dt, analysis_start, sim_time, 1000)
# plot_spikes(stim_spikes, exc_spikes, inh_spikes,
#             n_exc, dt, analysis_start, sim_time, 1000)

plot_weight_histograms(initial_weights, final_weights)

# plot_rates(stim_spikes, exc_spikes, inh_spikes, n_exc, n_inh, sim_time)

plt.show()

