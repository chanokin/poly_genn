from pygenn import (genn_wrapper, genn_model)

slow = 0
stdp_additive_all_model = genn_model.create_custom_weight_update_class(
    "stdp_additive_2",
    param_names=["tauPlus", "tauMinus", "wMin", "wMax",
                 "tauPlusDecay", "tauMinusDecay", "delayDecay", "delay"],
    var_name_types=[
        ("g", "scalar"),
        ("dg", "scalar"),
        ("aPlus", "scalar"),
        ("aMinus", "scalar"),
    ],
    pre_var_name_types=[("preTrace", "scalar")],
    post_var_name_types=[("postTrace", "scalar")],

    # at incoming pre spike
    sim_code=f"""
        $(addToInSyn, $(g));
        const scalar dt = $(t) - $(sT_post);
        if(dt > 0) {{
            if ({slow}) {{
                $(dg) -= ($(aMinus) * $(postTrace));
            }} else {{
                const scalar newWeight = $(g) - ($(aMinus) * $(postTrace));
                $(g) = fmin($(wMax), fmax($(wMin), newWeight));
            }}
        }}
        """,
    learn_post_code=f"""
        const scalar dt = $(t) - $(sT_pre);
        if(dt > 0) {{
            if ({slow}) {{
                $(dg) += ($(aPlus) * $(preTrace));
            }} else {{
                const scalar newWeight = $(g) + ($(aPlus) * $(preTrace));
                $(g) = fmin($(wMax), fmax($(wMin), newWeight));
            }}
        }}
        """,
    pre_spike_code="""
        $(preTrace) += 1.0;
        """,
    pre_dynamics_code="""
        $(preTrace) *= $(tauPlusDecay);
        """,
    post_spike_code="""
        $(postTrace) += 1.0;
        """,
    post_dynamics_code="""
        $(postTrace) *= $(tauMinusDecay);
        """,
    synapse_dynamics_code="""
        $(dg) *= 0.9; //// this should be approx e^(-1ms/1000ms)
        $(g) = fmin($(wMax), fmax($(wMin), $(g) + 0.01 + $(dg)));
    """ if slow else None,
    is_pre_spike_time_required=True,
    is_post_spike_time_required=True)