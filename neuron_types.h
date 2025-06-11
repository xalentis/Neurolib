#ifndef NEURON_TYPES_H
#define NEURON_TYPES_H

#include "neuron.h"

// pyramidal Neuron - most common excitatory neuron type
class PyramidalNeuron : public Neuron {
public:
    explicit PyramidalNeuron();
};

// interneuron - inhibitory neuron type
class Interneuron : public Neuron {
public:
    explicit Interneuron();
};

// purkinje Cell - found in cerebellum
class PurkinjeNeuron : public Neuron {
public:
    explicit PurkinjeNeuron();
};

// motor neuron - controls muscle movement
class MotorNeuron : public Neuron {
public:
    explicit MotorNeuron();
};

// sensory Neuron - processes sensory input
class SensoryNeuron : public Neuron {
public:
    explicit SensoryNeuron();
};

#endif