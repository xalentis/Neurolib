#ifndef NEURON_H
#define NEURON_H

class Dendrite;
class Axon;

// base neuron class
class Neuron {
protected:
    float soma_diameter;        // cell body diameter in micrometers
    float membrane_potential;   // current membrane potential in mV
    float resting_potential;    // resting membrane potential in mV
    float threshold_potential;  // action potential threshold in mV
    bool is_spiking;            // currently generating action potential?
    float refractory_period;    // time until next spike possible (ms)
    float spike_amplitude;      // action potential amplitude in mV
    
    Dendrite** dendrites;       // array of dendrites
    int dendrite_count;
    int max_dendrites;
    
    Axon* axon;                 // aingle axon (most neurons have one)
    
    // neuron type classification
    bool is_excitatory;         // true for excitatory, false for inhibitory
    int neuron_type_id;         // specific neuron subtype identifier
    
public:
    explicit Neuron(float soma_diam = 20.0f, int max_dend = 10, bool excitatory = true, int type_id = 0);
    
    bool add_dendrite(Dendrite* dendrite);
    virtual float integrate_inputs();
    virtual bool update_and_check_spike();
    virtual void spike();
    bool connect_to_neuron(Neuron* target_neuron, int target_dendrite_idx = 0, 
                          float synapse_weight = 1.0f, bool inhibitory = false);
    
    inline float get_membrane_potential() const { return membrane_potential; }
    inline bool get_is_spiking() const { return is_spiking; }
    inline bool get_is_excitatory() const { return is_excitatory; }
    inline int get_neuron_type_id() const { return neuron_type_id; }
    inline int get_dendrite_count() const { return dendrite_count; }
    inline Axon* get_axon() const { return axon; }
    inline float get_soma_diameter() const { return soma_diameter; }
    inline float get_threshold_potential() const { return threshold_potential; }
    inline float get_spike_amplitude() const { return spike_amplitude; }
    
    virtual ~Neuron();
};

#endif