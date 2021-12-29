import h5py
from tqdm import tqdm

fname = "spikes_for_experiment_izh_patterns_28-12-2021__10-13.h5"

in_file = h5py.File(fname, "r")
out_file = h5py.File(f"tmp_{fname}", "w")

max_n_per_row = 1000000
for k in in_file:
    out_grp = out_file.create_group(k)
    for ds_name in in_file[k]:
        ds = in_file[k][ds_name]
        n_cols = ds.shape[1]
        out_ds = out_grp.create_dataset(ds_name, dtype="float32", shape=ds.shape,
                                        maxshape=ds.shape, chunks=(3, 1))
        for col_start in tqdm(range(0, n_cols, max_n_per_row)):
            col_end = min(n_cols, col_start + max_n_per_row)
            out_ds[:, col_start:col_end] = ds[:, col_start:col_end]



out_file.close()
in_file.close()

