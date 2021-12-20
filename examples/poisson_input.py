import numpy as np
import matplotlib.pyplot as plt
from pygenn.genn_model import GeNNModel
from polychronous.poisson_source import poisson_input_model

# Create a single-precision GeNN model
model = GeNNModel("float", "pygenn", backend='SingleThreadedCPU')

# Set simulation timestep to 1ms
dt = 1.0
model.dT = dt

poisson_params = {
    'dt': dt
}
scale = 1000.0
seconds = 10.0
rates = np.array([70.0, 1.0, 30.0, 200.0])
# rates = (np.random.random((10, 20)) / scale).flatten()
poisson_init = {
    "rate": rates
}
pixels = model.add_neuron_population(
    "input", # unique name for population
    len(rates), # how many neurons in the population
    poisson_input_model,  # type of neuron (or source)
    poisson_params, # parameters for the type of neuron
    poisson_init # initialization for variables (given neuron type)
)


# Build and load model
model.build()
model.load()

# Simulate
spike_times = [[] for _ in range(len(rates))]
idx = 0
sim_time = 1000.0 * seconds
while model.t < sim_time:
    model.step_time()
    pixels.pull_current_spikes_from_device()

    for i in pixels.current_spikes:
        spike_times[i].append(idx * model.dT)
    idx += 1

prates = [(len(ts)) / seconds for ts in spike_times]

markers = ['o', 'd', 's', '^']
fig, ax = plt.subplots(1, 1, figsize=(15, 7))
for i, ts in enumerate(spike_times):
    ax.plot(ts, np.ones_like(ts) * i, linestyle='none', markersize=10,
            marker='|', label=f"{prates[i]:3.2f}", markeredgewidth=0.5,
            )


plt.legend()
ax.margins(0.01, 0.2)
plt.show()
