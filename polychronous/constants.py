from typing import Final
TS:Final = 0
IDS:Final = 1
PRE:Final = 0
POST:Final = 1
MAX_SPIKES_PER_SEARCH_STEP:Final = int(1e6)
MIN_PRE_FOR_GROUP:Final = 3
SPIKE_T_TOLERANCE_FOR_GROUP:Final = 1
# SPIKE_T_TOLERANCE_FOR_GROUP:Final = 1e-9
MAX_CHAIN_LENGTH_FOR_GROUP:Final = 1e-9
MIN_N_NEURONS_TO_FORWARD:Final = 1
MAX_PARALLEL_GROUP_FINDERS:Final = 8