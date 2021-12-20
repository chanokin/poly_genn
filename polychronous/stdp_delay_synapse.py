from pygenn import (genn_wrapper, genn_model)

# Additive STDP model with nearest neighbour spike pairing
stdp_delay_synapse = genn_model.create_custom_weight_update_class(
    "hebbian_stdp",
    param_names=["tauPlus", "tauMinus", "aPlus", "aMinus", "wMin", "wMax"],
    var_name_types=[
        ("g", "scalar"),
        ("d", "scalar"),
    ],

    # Code that gets called whenever a presynaptic spike arrives at the synapse
    sim_code=
    """
    $(addToInSynDelay, $(g), $(d));
    const scalar dt = $(t) - $(sT_post);
    if(dt > 0) {
        const scalar newWeight = $(g) - ($(aMinus) * exp(-dt / $(tauMinus)));
        $(g) = fmin($(wMax), fmax($(wMin), newWeight));
    }
    """,

    # Code that gets called whenever a back-propagated postsynaptic spike arrives at the synapse
    learn_post_code=
    """
    const scalar dt = $(t) - $(sT_pre);
    if(dt > 0) {
        const scalar newWeight = $(g) + ($(aPlus) * exp(-dt / $(tauPlus)));
        $(g) = fmin($(wMax), fmax($(wMin), newWeight));
    }
    """,

    is_pre_spike_time_required=True,
    is_post_spike_time_required=True
)
