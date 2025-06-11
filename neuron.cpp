#include "neuron.h"
#include "dendrite.h"
#include "axon.h"
#include "synapse.h"

Neuron::Neuron(float soma_diam, int max_dend, bool excitatory, int type_id)
    : soma_diameter(soma_diam), membrane_potential(-70.0f), resting_potential(-70.0f),
      threshold_potential(-50.0f), is_spiking(false), refractory_period(0.0f),
      spike_amplitude(50.0f), dendrite_count(0), max_dendrites(max_dend),
      is_excitatory(excitatory), neuron_type_id(type_id) {
    
    dendrites = new Dendrite*[max_dendrites];
    for (int i = 0; i < max_dendrites; ++i) {
        dendrites[i] = nullptr;
    }
    
    // create default axon
    axon = new Axon();
}

bool Neuron::add_dendrite(Dendrite* dendrite) {
    if (dendrite_count < max_dendrites) {
        dendrites[dendrite_count] = dendrite;
        dendrite_count++;
        return true;
    }
    return false;
}

float Neuron::integrate_inputs() {
    float total_input = 0.0f;
    for (int i = 0; i < dendrite_count; ++i) {
        if (dendrites[i] != nullptr) {
            total_input += dendrites[i]->integrate_synaptic_inputs();
        }
    }
    return total_input;
}

bool Neuron::update_and_check_spike() {
    if (refractory_period > 0.0f) {
        refractory_period -= 1.0f; // decrease refractory period
        membrane_potential = resting_potential;
        is_spiking = false;
        return false;
    }
    
    float synaptic_input = integrate_inputs();
    membrane_potential = resting_potential + synaptic_input;
    
    // check for action potential threshold
    if (membrane_potential >= threshold_potential) {
        spike();
        return true;
    }
    
    // passive decay toward resting potential
    if (membrane_potential != resting_potential) {
        float decay_factor = 0.9f;
        membrane_potential = resting_potential + (membrane_potential - resting_potential) * decay_factor;
    }
    
    return false;
}

void Neuron::spike() {
    is_spiking = true;
    membrane_potential = spike_amplitude;
    refractory_period = 2.0f; // 2ms refractory period
    
    // propagate through axon
    if (axon != nullptr) {
        axon->propagate_action_potential(spike_amplitude);
    }
}

bool Neuron::connect_to_neuron(Neuron* target_neuron, int target_dendrite_idx, 
                               float synapse_weight, bool inhibitory) {
    if (target_neuron == nullptr || target_dendrite_idx >= target_neuron->dendrite_count) {
        return false;
    }
    
    // create synapse
    Synapse* new_synapse = new Synapse(synapse_weight, -50.0f, inhibitory);
    
    // connect axon to synapse
    if (axon->add_output_synapse(new_synapse)) {
        // connect synapse to target dendrite
        return new_synapse->connect_to_dendrite(target_neuron->dendrites[target_dendrite_idx]);
    }
    
    delete new_synapse;
    return false;
}

Neuron::~Neuron() {
    for (int i = 0; i < dendrite_count; ++i) {
        delete dendrites[i];
    }
    delete[] dendrites;
    delete axon;
}