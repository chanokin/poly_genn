import numpy as np
from matplotlib import pyplot as plt
from pygenn import genn_wrapper
from pygenn import genn_model
from pathlib import Path

this_filename = Path(__file__).resolve().stem

sim_time = 10.
dt = 0.1 # ms
sim_steps = int(sim_time / dt)
delays = np.array([1, 5, 7], dtype='int')*(1/dt)
synapse_init = {
    "g": 0.34,
    "d": delays,
}
# at least one ms greater!
max_delay_slots = int(8 / dt)

# LIF neuron parameters
lif_params = {"C": 0.1, "TauM": 20.0, "Vrest": 0.0, "Vreset": 0.0,
              "Vthresh": 1.0, "Ioffset": 0.0, "TauRefrac": 1.0}
# LIF neuron initial state
lif_init = {"V": 0.0, "RefracTime": 0.0}

# spike_times = np.array([7, 3, 1])
# stim_ini = {"startSpike": [0, 1, 2], "endSpike": [1, 2, 3]}
spike_times = np.array([
    # 1, 1, 1
    7, 3, 1
])
stim_ini = {"startSpike": [0, 1, 2], "endSpike": [1, 2, 3]}


# Create model using single-precion and 1ms timesteps
model = genn_model.GeNNModel("float", this_filename)
model.dT = dt

stim = model.add_neuron_population("Stim", 3, "SpikeSourceArray",
                                   {}, stim_ini)
stim.set_extra_global_param("spikeTimes", spike_times)
stim.spike_recording_enabled = True

output = model.add_neuron_population("target", 1, "LIF", lif_params, lif_init)
output.spike_recording_enabled = True

synapse = model.add_synapse_population(
    "PreStimToPop", "DENSE_INDIVIDUALG", 1,
    stim, output,
    "StaticPulseDendriticDelay", {}, synapse_init, {}, {},
    "DeltaCurr", {}, {},
)
# Set max dendritic delay and span type
synapse.pop.set_max_dendritic_delay_timesteps(max_delay_slots)

# Build and load model
model.build()
model.load(num_recording_timesteps=sim_steps)

v = np.empty((sim_steps, 1))
v_view = output.vars["V"].view
# Simulate model
while model.t < sim_time:
    model.step_time()
    output.pull_var_from_device("V")

    v[model.timestep - 1, :] = v_view[:]

model.pull_recording_buffers_from_device()

stim_spikes = stim.spike_recording_data
output_spikes = output.spike_recording_data

print(stim_spikes)
print(output_spikes)
fig, ax = plt.subplots(1, 1)
for t, nid in zip(*stim_spikes):
    plt.axvline(t/dt, color='tab:green')

for t, nid in zip(*output_spikes):
    plt.axvline(t/dt, color='tab:blue')

ax.plot(v)

ax.set_xlim(0, sim_steps)
# ax.set_ylim(-0.5, 2.5)
plt.grid()
plt.show()

