import numpy as np
import h5py
import os
from matplotlib import pyplot as plt
from polychronous.plotting import (
    plot_spikes, plot_weight_histograms, plot_rates
)
from polychronous.group_finding import find_groups_by_weights

sec_to_ms = 1000.
out_dir = '.'
filename = "experiment_izh_2_layer_in_patterns"
filename = f"{filename}.npz"
data =  np.load(os.path.join(out_dir, filename),
                mmap_mode=True, allow_pickle=True)

dt = data['dt']
sim_time = data['sim_time']
analysis_length = 200
analysis_start = sim_time - 20*analysis_length #200 * 4
n_exc = data['n_exc']
n_inh = data['n_inh']
initial_weights = data['initial_weights'].item()
final_weights = data['final_weights'].item()

spikes_fname = data['spikes_filename'].item()
spikes_fname = os.path.join(out_dir, spikes_fname)
spikes_file = h5py.File(spikes_fname, 'r')

def get_all(name):
    print(name)

spikes_file.visit(get_all)

# pat_spikes = spikes_file[os.path.join("pattern", "spikes")]
exc0_spikes = spikes_file[os.path.join("exc_0", "spikes")]
exc1_spikes = spikes_file[os.path.join("exc_1", "spikes")]
inh_0_spikes = spikes_file[os.path.join("inh_0", "spikes")]
inh_1_spikes = spikes_file[os.path.join("inh_1", "spikes")]

for i in range(20):
    start = analysis_length * i
    end = start + analysis_length
    plot_spikes(None, exc0_spikes, inh_0_spikes,
                n_exc, dt, sim_time, start, end, analysis_length)
    plt.savefig(os.path.join(out_dir, f"start_spikes_0_{i:06d}.png"), dpi=150)

    plot_spikes(None, exc1_spikes, inh_1_spikes,
                n_exc, dt, sim_time, start, end, analysis_length)
    plt.savefig(os.path.join(out_dir, f"start_spikes_1_{i:06d}.png"), dpi=150)

# mid_start = 1000 * 1000
# plot_spikes(None, exc_spikes, inh_spikes,
#             n_exc, dt, sim_time, mid_start, mid_start + analysis_length, analysis_length)
# plt.savefig("mid_spikes.png", dpi=150)
#
#
for i in range(20):
    start = analysis_start + i * analysis_length
    end = start + analysis_length
    plot_spikes(None, exc0_spikes, inh_0_spikes,
                n_exc, dt, sim_time, start, end,
                analysis_length)
    plt.savefig(os.path.join(out_dir, f"end_spikes_0_{i:06d}.png"), dpi=150)

    plot_spikes(None, exc1_spikes, inh_1_spikes,
                n_exc, dt, sim_time, start, end,
                analysis_length)
    plt.savefig(os.path.join(out_dir, f"end_spikes_1_{i:06d}.png"), dpi=150)


# groups_by_weight = find_groups_by_weights(filename, max_weight * 0.95, analysis_start)
for k in final_weights:
    title = f"{k.replace('_', ' ')} histograms"
    print(title)
    fw = final_weights[k]
    iw = initial_weights[k]
    plot_weight_histograms(iw, fw, title)
    plt.savefig(os.path.join(out_dir, f"{k}_histograms.png"), dpi=150)


plt.show()

print("end of analysis")
# groups = find_groups(filename, max_weight * 0.9, analysis_start)