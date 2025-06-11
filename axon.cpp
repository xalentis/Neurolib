#include "axon.h"
#include "synapse.h"

Axon::Axon(float len, float diam, bool myelinated, int max_syn)
    : length(len), diameter(diam), is_myelinated(myelinated), synapse_count(0), max_synapses(max_syn) {
    // calculate conduction velocity based on myelination and diameter
    if (is_myelinated) {
        conduction_velocity = 6.0f * diameter; // myelinated: 6*diameter m/s
    } else {
        conduction_velocity = 0.5f * diameter; // unmyelinated: much slower
    }
    
    output_synapses = new Synapse*[max_synapses];
    for (int i = 0; i < max_synapses; ++i) {
        output_synapses[i] = nullptr;
    }
}

bool Axon::add_output_synapse(Synapse* synapse) {
    if (synapse_count < max_synapses) {
        output_synapses[synapse_count] = synapse;
        synapse_count++;
        return true;
    }
    return false;
}

void Axon::propagate_action_potential(float amplitude) {
    for (int i = 0; i < synapse_count; ++i) {
        if (output_synapses[i] != nullptr) {
            output_synapses[i]->propagate_signal(amplitude);
        }
    }
}

Axon::~Axon() {
    delete[] output_synapses;
}