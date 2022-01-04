import pickle
import numpy as np

with open("groups_file.pickle", "rb") as groups_file:
    grps = pickle.load(groups_file)
    # print(len(grps))
    count_small = 0
    count_groups = 0
    for index, k in enumerate(grps.keys()):
        group = grps[k]
        for chain in group:
            chain = np.squeeze(chain)
            chain_length = chain.shape[0]
            if chain_length < 3:
                count_small += 1
            else:
                count_groups += 1

        # for link in chain:
        print(k, count_groups, count_small)

        # print(k, len(grps[k]))
    print(count_groups, count_small)
