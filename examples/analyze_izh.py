import numpy as np
from matplotlib import pyplot as plt
from polychronous.plotting import (
    plot_spikes, plot_weight_histograms, plot_rates
)
from polychronous.find_groups import find_groups

filename = 'izh_polychronous_experiment.npz'

data =  np.load(filename, mmap_mode=True, allow_pickle=True)
sec_to_ms = 1000.

dt = data['dt']
sim_time = data['sim_time']
analysis_length = 5000
analysis_start = sim_time - analysis_length #200 * 4
n_exc = data['n_exc']
n_inh = data['n_inh']
max_weight = data['max_weight']
stim_spikes = data['stim_spikes']
pat_spikes = data['pat_spikes']
exc_spikes = data['exc_spikes']
inh_spikes = data['inh_spikes']
initial_weights = data['initial_weights'].item()
final_weights = data['final_weights'].item()

plot_spikes(None, exc_spikes, inh_spikes,
            n_exc, dt, 0, analysis_length, analysis_length)
plt.savefig("start_spikes.png", dpi=150)

mid_start = 1000 * 1000
plot_spikes(None, exc_spikes, inh_spikes,
            n_exc, dt, mid_start, mid_start + analysis_length, analysis_length)
plt.savefig("mid_spikes.png", dpi=150)


plot_spikes(None, exc_spikes, inh_spikes,
            n_exc, dt, analysis_start, analysis_start+analysis_length, analysis_length)
plt.savefig("end_spikes.png", dpi=150)

# plot_weight_histograms(initial_weights, final_weights)
# plt.savefig("weight_histograms.png", dpi=150)

# plot_rates(stim_spikes, exc_spikes, inh_spikes, n_exc, n_inh, sim_time)
# plt.savefig("rates.png", dpi=150)

plt.show()

groups = find_groups(filename, max_weight * 0.9, analysis_start)