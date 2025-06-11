#ifndef NEURON_TYPES_H
#define NEURON_TYPES_H

#include "neuron.h"

// Pyramidal Neuron - most common excitatory neuron type
class PyramidalNeuron : public Neuron {
public:
    explicit PyramidalNeuron();
};

// Interneuron - inhibitory neuron type
class Interneuron : public Neuron {
public:
    explicit Interneuron();
};

// Purkinje Cell - found in cerebellum
class PurkinjeNeuron : public Neuron {
public:
    explicit PurkinjeNeuron();
};

// Motor Neuron - controls muscle movement
class MotorNeuron : public Neuron {
public:
    explicit MotorNeuron();
};

// Sensory Neuron - processes sensory input
class SensoryNeuron : public Neuron {
public:
    explicit SensoryNeuron();
};

#endif // NEURON_TYPES_H