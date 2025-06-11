#include "dendrite.h"
#include "synapse.h"
#include "neuron.h"

Dendrite::Dendrite(float len, float diam, int spines, Neuron* parent) 
    : length(len), diameter(diam), spine_count(spines), membrane_potential(-70.0f),
      synapse_count(0), max_synapses(spines), is_active(false), parent_neuron(parent) {
    synapses = new Synapse*[max_synapses];
    for (int i = 0; i < max_synapses; ++i) {
        synapses[i] = nullptr;
    }
}

float Dendrite::integrate_synaptic_inputs() {
    float total_input = 0.0f;
    for (int i = 0; i < synapse_count; ++i) {
        if (synapses[i] != nullptr) {
            total_input += synapses[i]->get_synaptic_contribution();
        }
    }
    return total_input;
}

void Dendrite::update_membrane_potential() {
    float synaptic_input = integrate_synaptic_inputs();
    membrane_potential = -70.0f + synaptic_input;
    
    // simple decay toward resting potential
    if (membrane_potential > -70.0f) {
        membrane_potential -= 0.1f;
    }
    
    is_active = (membrane_potential >= -50.0f);
    
    // notify parent neuron if we have one
    if (parent_neuron != nullptr) {
        parent_neuron->update_and_check_spike();
    }
}

bool Dendrite::add_synapse(Synapse* synapse) {
    if (synapse_count < max_synapses) {
        synapses[synapse_count] = synapse;
        synapse_count++;
        return true;
    }
    return false;
}

bool Dendrite::remove_synapse(Synapse* synapse) {
    for (int i = 0; i < synapse_count; ++i) {
        if (synapses[i] == synapse) {
            for (int j = i; j < synapse_count - 1; ++j) {
                synapses[j] = synapses[j + 1];
            }
            synapse_count--;
            synapses[synapse_count] = nullptr;
            return true;
        }
    }
    return false;
}

Dendrite::~Dendrite() {
    delete[] synapses;
}