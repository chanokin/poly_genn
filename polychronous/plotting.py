import numpy as np
from matplotlib import pyplot as plt


def plot_spikes(stim_spikes, exc_spikes, inh_spikes, n_exc,
                dt, sim_time_ms, ms_per_plot):

    stim_times = stim_spikes[0] #* (1. / dt)
    exc_times = exc_spikes[0] #* (1. / dt)
    inh_times = inh_spikes[0] #* (1. / dt)

    for start_ms in np.arange(0, sim_time_ms, ms_per_plot):
        end_ms = min(sim_time_ms, start_ms + ms_per_plot)
        fig, ax = plt.subplots(2, 1, figsize=(5, 7))

        whr = np.where(np.logical_and(start_ms <= stim_times,
                                      stim_times < end_ms))
        ax[0].plot(stim_times[whr], stim_spikes[1][whr],
                   color='tab:green',
                   marker='.', markeredgewidth=0.,
                   markersize=3, linestyle='none')
        ax[0].grid()

        whr = np.where(np.logical_and(start_ms <= exc_times,
                                      exc_times < end_ms))
        ax[1].plot(exc_times[whr], exc_spikes[1][whr],
                   color='tab:blue',
                   marker='.', markeredgewidth=0.,
                   markersize=3, linestyle='none')

        whr = np.where(np.logical_and(start_ms <= inh_times,
                                      inh_times < end_ms))
        ax[1].plot(inh_times[whr], inh_spikes[1][whr] + n_exc,
                   color='tab:red',
                   marker='.', markeredgewidth=0.,
                   markersize=3, linestyle='none')
        ax[1].grid()
