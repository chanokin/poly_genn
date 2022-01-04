from pygenn.genn_model import (
    create_custom_neuron_class,
    create_dpf_class as dpf,
)
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY_DUPLICATE

spike_source_array = create_custom_neuron_class(
    "spike_source_array",
    threshold_condition_code="$(startSpike) != $(endSpike) && $(t) >= $(spikeTimes)[$(startSpike)]",
    reset_code=(
        #"printf(\"start before %d\\n\", $(startSpike)); \n"
        "$(startSpike)++;\n"
        # "printf(\"start after %d\\n\", $(startSpike)); \n"
    ),
    var_name_types=[("startSpike", "unsigned int"), ("endSpike", "unsigned int")],
    extra_global_params=[("spikeTimes", "scalar*")],
    is_auto_refractory_required=False
)