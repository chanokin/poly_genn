import numpy as np
import h5py
import os
from matplotlib import pyplot as plt
from polychronous.plotting import (
    plot_spikes, plot_weight_histograms, plot_rates
)
from polychronous.group_finding import find_groups_by_weights

sec_to_ms = 1000.
out_dir = './max_w_10'
filename = "experiment_izh_plastic_in_patterns"
filename = f"{filename}.npz"
data =  np.load(os.path.join(out_dir, filename),
                mmap_mode=True, allow_pickle=True)

dt = data['dt']
sim_time = data['sim_time']
analysis_length = 400
analysis_start = sim_time - analysis_length #200 * 4
n_exc = data['n_exc']
n_inh = data['n_inh']
initial_weights = data['initial_weights'].item()
final_weights = data['final_weights'].item()
final_pat2pop_weights = data["final_pat2pop_weights"].item()

spikes_fname = data['spikes_filename'].item()
spikes_fname = os.path.join(out_dir, spikes_fname)
spikes_file = h5py.File(spikes_fname, 'r')
pat_spikes = spikes_file[os.path.join("pattern", "spikes")]
exc_spikes = spikes_file[os.path.join("exc", "spikes")]
inh_spikes = spikes_file[os.path.join("inh", "spikes")]

plot_spikes(pat_spikes, exc_spikes, inh_spikes,
            n_exc, dt, sim_time, 0, analysis_length, analysis_length)
plt.savefig(os.path.join(out_dir, "start_spikes.png"), dpi=150)
#
# mid_start = 1000 * 1000
# plot_spikes(None, exc_spikes, inh_spikes,
#             n_exc, dt, sim_time, mid_start, mid_start + analysis_length, analysis_length)
# plt.savefig("mid_spikes.png", dpi=150)
#
#
plot_spikes(pat_spikes, exc_spikes, inh_spikes,
            n_exc, dt, sim_time, analysis_start, analysis_start+analysis_length,
            analysis_length)
plt.savefig(os.path.join(out_dir, "end_spikes.png"), dpi=150)

# groups_by_weight = find_groups_by_weights(filename, max_weight * 0.95, analysis_start)

plot_weight_histograms(initial_weights, final_weights)
plt.savefig(os.path.join(out_dir, "weight_histograms.png"), dpi=150)

plot_weight_histograms(initial_weights, final_pat2pop_weights)
plt.savefig(os.path.join(out_dir, "input_weight_histograms.png"), dpi=150)

plt.show()

print("end of analysis")
# groups = find_groups(filename, max_weight * 0.9, analysis_start)