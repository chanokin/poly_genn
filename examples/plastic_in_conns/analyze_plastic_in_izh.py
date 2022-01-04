import numpy as np
import h5py
import os
from matplotlib import pyplot as plt
from polychronous.plotting import (
    plot_spikes, plot_weight_histograms, plot_rates
)
from polychronous.group_finding import find_groups_by_weights

sec_to_ms = 1000.
filename = "experiment_izh_static_in_patterns"
filename = f"{filename}.npz"
data =  np.load(filename, mmap_mode=True, allow_pickle=True)

dt = data['dt']
sim_time = data['sim_time']
analysis_length = 1000
analysis_start = sim_time - analysis_length #200 * 4
n_exc = data['n_exc']
n_inh = data['n_inh']
max_weight = data['max_weight']
initial_weights = data['initial_weights'].item()
final_weights = data['final_weights'].item()

spikes_fname = data['spikes_filename'].item()
spikes_file = h5py.File(spikes_fname, 'r')
stim_spikes = spikes_file[os.path.join("input", "spikes")]
pat_spikes = spikes_file[os.path.join("pattern", "spikes")]
exc_spikes = spikes_file[os.path.join("exc", "spikes")]
inh_spikes = spikes_file[os.path.join("inh", "spikes")]
n_stim_spikes = stim_spikes.shape

plot_spikes(pat_spikes, exc_spikes, inh_spikes,
            n_exc, dt, sim_time, 0, analysis_length, analysis_length)
plt.savefig("start_spikes.png", dpi=150)
#
# mid_start = 1000 * 1000
# plot_spikes(None, exc_spikes, inh_spikes,
#             n_exc, dt, sim_time, mid_start, mid_start + analysis_length, analysis_length)
# plt.savefig("mid_spikes.png", dpi=150)
#
#
# plot_spikes(None, exc_spikes, inh_spikes,
#             n_exc, dt, sim_time, analysis_start, analysis_start+analysis_length,
#             analysis_length)
# plt.savefig("end_spikes.png", dpi=150)

# groups_by_weight = find_groups_by_weights(filename, max_weight * 0.95, analysis_start)

# plot_weight_histograms(initial_weights, final_weights)
# plt.savefig("weight_histograms.png", dpi=150)

plt.show()

print("end of analysis")
# groups = find_groups(filename, max_weight * 0.9, analysis_start)