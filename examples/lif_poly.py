import numpy as np
from matplotlib import pyplot as plt
from pygenn import genn_wrapper
from pygenn import genn_model
from pathlib import Path
from polychronous.poisson_source import poisson_input_model
from polychronous.connectivity_generation import generate_pairs_and_delays
from polychronous.stdp_synapse import stdp_synapse
from polychronous.plotting import (
    plot_spikes, plot_weight_histograms, plot_rates
)
from polychronous.find_groups import find_groups

this_filename = Path(__file__).resolve().stem

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

seconds = 60. * 2
# seconds = 5
sec_to_ms = 1000.
sim_time = seconds * sec_to_ms # ms
dt = 0.1 # ms

sim_steps = int(sim_time / dt)

max_delay = 20 # ms
max_delay_step = int((max_delay + 1) / dt)
delay_steps_dist_params = {
    'min': 1.0, # ms
    'max': max_delay, # ms
}

# LIF neuron parameters
if binding:
    exc_params = {"C": 0.5, "TauM": 20.0, "Vrest": -74.0, "Vreset": -57.0,
                  "Vthresh": -53.0, "Ioffset": 0.0, "TauRefrac": 2.0}

    inh_params = {"C": 0.214, "TauM": 12.0, "Vrest": -82.0, "Vreset": -58.0,
                  "Vthresh": -53.0, "Ioffset": 0.0, "TauRefrac": 2.0}

    exc_synapse_init = {"g": 1./5.,}

    inh_synapse_init = {"g": -1./6.,}

else:
    exc_params = {"C": 0.1, "TauM": 20.0, "Vrest": 0.0, "Vreset": 0.0,
                  "Vthresh": 1.0, "Ioffset": 0.0, "TauRefrac": 2.0}

    inh_params = {"C": 0.1, "TauM": 10.0, "Vrest": 0.0, "Vreset": 0.0,
                  "Vthresh": 1.0, "Ioffset": 0.0, "TauRefrac": 2.0}

    exc_synapse_init = {"g": 1./5.,}

    inh_synapse_init = {"g": -1./6.,}


conductance_params = {
    'e_to_e': {'tau': 150, 'E': 0},
    'e_to_i': {'tau': 2, 'E': 0},
    'i_to_e': {'tau': 5, 'E': -70},
}

# LIF neuron initial state
nrn_init = {"V": 0.0, "RefracTime": 0.0}

max_weight = 1./3.
stdp_params = {
    "tauPlus": 20, "tauMinus": 20, "aPlus": 0.1,
    "aMinus": 0.12,  "wMin": 0.0, "wMax": max_weight
}
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Create model using single-precion and 1ms timesteps
model = genn_model.GeNNModel("float", this_filename)
model.dT = dt

stim = model.add_neuron_population(
        "Stim", n_total, poisson_input_model,
        {'dt': dt}, stim_init)
stim.spike_recording_enabled = True

exc_pop = model.add_neuron_population(
            "exc", n_exc, "LIF", exc_params, nrn_init)

exc_pop.spike_recording_enabled = True

inh_pop = model.add_neuron_population(
            "inh", n_inh, "LIF", inh_params, nrn_init)

inh_pop.spike_recording_enabled = True


in2exc = model.add_synapse_population(
    "StimToExc", "SPARSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
    stim, exc_pop,
    "StaticPulse", {}, {'g': 1.0}, {}, {},
    "DeltaCurr", {}, {},
)
in2exc.set_sparse_connections(np.arange(n_exc), np.arange(n_exc))

in2inh = model.add_synapse_population(
    "StimToInh", "SPARSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
    stim, inh_pop,
    "StaticPulse", {}, {'g': 1.0}, {}, {},
    "DeltaCurr", {}, {},
)
in2inh.set_sparse_connections(np.arange(n_inh) + n_exc, np.arange(n_inh))


net_conns = {}
if do_all_conns:
    for conn_name in conn_pairs:
        if '_to_' not in conn_name:
            continue

        s, t = conn_name.split('_to_')
        source = exc_pop if s == 'e' else inh_pop
        target = exc_pop if t == 'e' else inh_pop
        conn_g_init = exc_synapse_init if s == 'e' else inh_synapse_init
        synapse_type = stdp_synapse if s == 'e' and t == 'e' else 'StaticPulse'
        synapse_params = stdp_params if s == 'e' and t == 'e' else {}
        for delay in conn_pairs[conn_name]:
            synapse_name = f"{s}_to_{t}_d{delay}"
            net_conns[synapse_name] =  model.add_synapse_population(
                synapse_name, "SPARSE_INDIVIDUALG", delay,
                source, target,
                synapse_type, synapse_params, conn_g_init, {}, {},
                "DeltaCurr", {}, {},
            )
            pre_indices, post_indices = conn_pairs[conn_name][delay]
            net_conns[synapse_name].set_sparse_connections(
                                        pre_indices, post_indices)

# Build and load model
model.build(force_rebuild=True)
model.load(num_recording_timesteps=sim_steps)

# [net_conns[k].pull_connectivity_from_device()
#  for k in net_conns if 'e_to_e' in k]
[net_conns[k].pull_var_from_device('g')
 for k in net_conns if 'e_to_e' in k]
initial_weights = {k: net_conns[k].get_var_values('g').copy()
                   for k in net_conns if 'e_to_e' in k}


# Simulate model
while model.t < sim_time:
    model.step_time()

# [net_conns[k].pull_connectivity_from_device()
#  for k in net_conns if 'e_to_e' in k]
[net_conns[k].pull_var_from_device('g')
 for k in net_conns if 'e_to_e' in k]
final_weights = {k: net_conns[k].get_var_values('g').copy()
                   for k in net_conns if 'e_to_e' in k}

model.pull_recording_buffers_from_device()

stim_spikes = stim.spike_recording_data
exc_spikes = exc_pop.spike_recording_data
inh_spikes = inh_pop.spike_recording_data

experiment_data = dict(
    stim_spikes=stim_spikes,
    exc_spikes=exc_spikes,
    inh_spikes=inh_spikes,
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

filename = 'izh_polychronous_experiment.npz'
np.savez_compressed(filename, **experiment_data)
analysis_start = sim_time - 2 * sec_to_ms

plot_spikes(stim_spikes, exc_spikes, inh_spikes,
            n_exc, dt, analysis_start, sim_time, 10000)

plot_weight_histograms(initial_weights, final_weights)

plot_rates(stim_spikes, exc_spikes, inh_spikes, n_exc, n_inh, sim_time)

plt.show()

groups = find_groups(filename, max_weight * 0.666, analysis_start)
