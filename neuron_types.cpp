#include "neuron_types.h"
#include "dendrite.h"
#include "axon.h"

// pyramidal neuron
PyramidalNeuron::PyramidalNeuron() : Neuron(25.0f, 15, true, 1) {
    add_dendrite(new Dendrite(800.0f, 3.0f, 8000, this)); // apical dendrite
    add_dendrite(new Dendrite(400.0f, 2.0f, 3000, this)); // basal dendrite 1
    add_dendrite(new Dendrite(350.0f, 2.0f, 2500, this)); // basal dendrite 2
    add_dendrite(new Dendrite(300.0f, 1.8f, 2000, this)); // basal dendrite 3
}

// interneuron
Interneuron::Interneuron() : Neuron(15.0f, 8, false, 2) {
    spike_amplitude = 40.0f; // typically smaller spikes
    threshold_potential = -45.0f; // more excitable
    
    add_dendrite(new Dendrite(200.0f, 1.5f, 1000, this));
    add_dendrite(new Dendrite(180.0f, 1.5f, 800, this));
    add_dendrite(new Dendrite(160.0f, 1.4f, 600, this));
}

// purkinje cell
PurkinjeNeuron::PurkinjeNeuron() : Neuron(30.0f, 20, false, 3) {
    spike_amplitude = 60.0f; // Large spikes
    
    // purkinje cells have extensive dendritic trees
    for (int i = 0; i < 8; ++i) {
        add_dendrite(new Dendrite(600.0f + i*50.0f, 2.5f, 15000, this));
    }
}

// motor neuron
MotorNeuron::MotorNeuron() : Neuron(40.0f, 12, true, 4) {
    spike_amplitude = 70.0f; // strong spikes for muscle control
    
    // large dendrites for integrating many inputs
    for (int i = 0; i < 6; ++i) {
        add_dendrite(new Dendrite(500.0f, 4.0f, 5000, this));
    }
    
    // motor neurons have large, myelinated axons
    delete axon;
    axon = new Axon(100000.0f, 15.0f, true, 100); // very long axon to muscles
}

// sensory neuron
SensoryNeuron::SensoryNeuron() : Neuron(18.0f, 6, true, 5) {
    threshold_potential = -55.0f; // less excitable than interneurons
    
    // fewer, specialized dendrites
    add_dendrite(new Dendrite(250.0f, 2.0f, 1500, this));
    add_dendrite(new Dendrite(200.0f, 1.8f, 1200, this));
}