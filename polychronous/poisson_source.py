from pygenn.genn_model import (
    create_custom_neuron_class,
    create_dpf_class as dpf,
)
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY_DUPLICATE

poisson_input_model = create_custom_neuron_class(
    'poisson_input',
    var_name_types=[
        ('rate', 'scalar', VarAccess_READ_ONLY_DUPLICATE)
    ],
    param_names=[
        'dt'
    ],
    derived_params=[
        ('time_scale', dpf(lambda pars, dt: dt/1000.0)() )
    ],
    sim_code='''
    //printf("input = %f\\ttime = %f\\n", $(rate), $(t));
    const bool is_ms = ( $(t) - int($(t)) ) < DT;
    const bool spike = ( $(gennrand_uniform) >= exp(-fabs($(rate)) * $(time_scale)) ) && is_ms;
    ''',
    threshold_condition_code='$(rate) > 0.0 && spike',
    is_auto_refractory_required=False,
)