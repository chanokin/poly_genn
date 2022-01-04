import numpy as np
from matplotlib import pyplot as plt
from polychronous.spike_finding import find_limit_indices
from tqdm import tqdm
import sys


def plot_weight_histograms(initial_weights, final_weights):
    sys.stdout.write("Plotting weight histograms\n")
    sys.stdout.flush()

    fig, ax = plt.subplots(1, 2, figsize=(17, 5))
    ax[0].hist(np.hstack([initial_weights[k] for k in initial_weights]))
    ax[0].set_title("Initial weights")
    ax[0].set_xlabel("Weight value")

    ax[1].hist(np.hstack([final_weights[k] for k in final_weights]))
    ax[1].set_title('Final weights')
    ax[1].set_xlabel("Number of synapses")


def plot_spikes(stim_spikes, exc_spikes, inh_spikes, n_exc, dt,
                total_simulation_time, start_time_ms, end_time_ms,  ms_per_plot,
                points_per_chunk=5000000):

    half_sim_time = total_simulation_time / 2
    use_reverse_search = start_time_ms >= half_sim_time

    sys.stdout.write("Plotting spikes\n")
    sys.stdout.flush()

    if stim_spikes is not None:
        print("will plot stim also")
        start_stim_idx, end_stim_idx = find_limit_indices(
                                        stim_spikes, start_time_ms, end_time_ms,
                                        points_per_chunk, reverse=use_reverse_search)
        stim_times = stim_spikes[0, start_stim_idx:end_stim_idx] #* (1. / dt)
        stim_ids = stim_spikes[1, start_stim_idx:end_stim_idx]

    sys.stdout.write("\tExcitatory spikes\n")
    sys.stdout.flush()

    start_exc_idx, end_exc_idx = find_limit_indices(exc_spikes, start_time_ms,
                                                    end_time_ms, points_per_chunk,
                                                    reverse=use_reverse_search)
    exc_times = exc_spikes[0,start_exc_idx:end_exc_idx] #* (1. / dt)
    exc_ids = exc_spikes[1,start_exc_idx:end_exc_idx] #* (1. / dt)

    sys.stdout.write("\tInhibitory spikes\n")
    sys.stdout.flush()

    start_inh_idx, end_inh_idx = find_limit_indices(inh_spikes, start_time_ms,
                                                    end_time_ms, points_per_chunk,
                                                    reverse=use_reverse_search)
    inh_times = inh_spikes[0,start_inh_idx:end_inh_idx] #* (1. / dt)
    inh_ids = inh_spikes[1,start_inh_idx:end_inh_idx] #* (1. / dt)

    ms_to_s = 1.0/1000.0
    for start_ms in tqdm(np.arange(start_time_ms, end_time_ms, ms_per_plot)):
        end_ms = min(end_time_ms, start_ms + ms_per_plot)
        if stim_spikes is not None:
            fig, axs = plt.subplots(1, 2, figsize=(17, 5))
        else:
            fig, axs = plt.subplots(1, 1, figsize=(17, 5))

        plt.suptitle(f"from {start_ms * ms_to_s} to {end_ms * ms_to_s} [s]")

        if stim_spikes is not None:
            whr = np.where(np.logical_and(start_ms <= stim_times,
                                          stim_times < end_ms))
            axs[0].plot(stim_times[whr], stim_ids[whr],
                       color='tab:green',
                       marker='.', markeredgewidth=0.,
                       markersize=3, linestyle='none')
            axs[0].grid()
            axs[0].set_xlim(start_ms, end_ms)
            ticks = axs[0].get_xticks()
            axs[0].set_xticklabels([f"{x:6.2f}" for x in (ticks * ms_to_s)])

        ax = axs[1] if stim_spikes is not None else axs
        whr = np.where(np.logical_and(start_ms <= exc_times,
                                      exc_times < end_ms))
        ax.plot(exc_times[whr], exc_ids[whr],
                color='tab:blue',
                marker='.', markeredgewidth=0.,
                markersize=3, linestyle='none')

        whr = np.where(np.logical_and(start_ms <= inh_times,
                                      inh_times < end_ms))
        ax.plot(inh_times[whr], inh_ids[whr] + n_exc,
                color='tab:red',
                marker='.', markeredgewidth=0.,
                markersize=3, linestyle='none')
        ax.grid()
        ax.set_xlim(start_ms, end_ms)
        ticks = ax.get_xticks()
        ax.set_xticklabels([f"{x:6.2f}" for x in (ticks * ms_to_s)])


def plot_rates(stim_spikes, exc_spikes, inh_spikes, n_exc, n_inh, sim_time_ms,
               alpha=0.5):

    sys.stdout.write("Plotting rates\n")
    sys.stdout.flush()

    def plot_curve(ax, rates, color):
        im = ax.fill_between(
            np.arange(rates.shape[0]),
            np.min(rates, axis=1), np.max(rates, axis=1),
            color=color, linewidth=0, alpha=alpha)
        im = ax.plot(np.mean(rates, axis=1), color=color, markersize=0)
        return im

    n_total = n_inh + n_exc
    ms_per_plot = 1000
    stim_times = stim_spikes[0] #* (1. / dt)
    exc_times = exc_spikes[0] #* (1. / dt)
    inh_times = inh_spikes[0] #* (1. / dt)

    stim_rates = []
    exc_rates = []
    inh_rates = []
    for start_ms in tqdm(np.arange(0, sim_time_ms, ms_per_plot)):
        end_ms = min(sim_time_ms, start_ms + ms_per_plot)

        # stim
        whr = np.where(np.logical_and(start_ms <= stim_times,
                                      stim_times < end_ms))
        ids = stim_spikes[1][whr]
        times = stim_times[whr]
        stim_rates.append([len(np.where(ids == id)[0]) for id in range(n_total)])

        # exc
        whr = np.where(np.logical_and(start_ms <= exc_times,
                                      exc_times < end_ms))
        ids = exc_spikes[1][whr]
        times = exc_times[whr]
        exc_rates.append([len(np.where(ids == id)[0]) for id in range(n_exc)])

        # inh
        whr = np.where(np.logical_and(start_ms <= inh_times,
                                      inh_times < end_ms))
        ids = inh_spikes[1][whr]
        times = inh_times[whr]
        inh_rates.append([len(np.where(ids == id)[0]) for id in range(n_inh)])

    stim_rates = np.vstack(stim_rates)
    exc_rates = np.vstack(exc_rates)
    inh_rates = np.vstack(inh_rates)

    fig, ax = plt.subplots(1, 3, figsize=(17, 5), sharey=True)

    # im = ax[0].plot(stim_rates, marker='.', color='tab:green', alpha=alpha * 0.001,
    #                 linestyle='none', markersize=10, markeredgewidth=0)
    im = plot_curve(ax[0], stim_rates, 'tab:green')
    # im = ax[0].imshow(stim_rates.T, interpolation='nearest', aspect='auto')
    # divider = make_axes_locatable(ax[0])
    # cax = divider.new_vertical(size="5%", pad=0.7, pack_start=True)
    # fig.add_axes(cax)
    # fig.colorbar(im, cax=cax, orientation="horizontal")
    ax[0].grid()
    ax[0].set_ylabel('Hertz')
    ax[0].set_xlabel('Seconds')

    # im = ax[1].plot(exc_rates, marker='.', color='tab:blue', alpha=alpha,
    #                 linestyle='none', markersize=10, markeredgewidth=0)
    im = plot_curve(ax[1], exc_rates, 'tab:blue')
    # im = ax[0].imshow(stim_rates.T, interpolation='nearest', aspect='auto')
    # divider = make_axes_locatable(ax[0])
    # cax = divider.new_vertical(size="5%", pad=0.7, pack_start=True)
    # fig.add_axes(cax)
    # fig.colorbar(im, cax=cax, orientation="horizontal")
    ax[1].grid()
    ax[1].set_xlabel('Seconds')

    # im = ax[2].plot(inh_rates, marker='.', color='tab:red', alpha=alpha,
    #                 linestyle='none', markersize=10, markeredgewidth=0)
    im = plot_curve(ax[2], inh_rates, 'tab:red')
    # im = ax[0].imshow(stim_rates.T, interpolation='nearest', aspect='auto')
    # divider = make_axes_locatable(ax[0])
    # cax = divider.new_vertical(size="5%", pad=0.7, pack_start=True)
    # fig.add_axes(cax)
    # fig.colorbar(im, cax=cax, orientation="horizontal")
    ax[2].grid()
    ax[2].set_xlabel('Seconds')

