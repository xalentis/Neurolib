#ifndef SYNAPSE_H
#define SYNAPSE_H

class Dendrite;

class Synapse {
private:
    float weight;           // synaptic strength in mV (EPSP/IPSP amplitude)
    float threshold;        // activation threshold in mV
    bool is_inhibitory;     // true for inhibitory, false for excitatory
    Dendrite** connections; // array of connected dendrites
    int connection_count;   // number of connected dendrites
    int max_connections;    // maximum allowed connections
    
public:
    // constructor with biologically plausable default values
    explicit Synapse(float w = 1.0f, float t = -50.0f, bool inhibit = false, int max_conn = 1);
    
    // model synaptic transmission
    inline float get_synaptic_contribution() const {
        return is_inhibitory ? -weight : weight;
    }
    
    // check if transmission occurs given membrane potential
    inline bool transmit(float membrane_potential) const {
        float new_potential = membrane_potential + get_synaptic_contribution();
        return new_potential >= threshold;
    }
    
    bool connect_to_dendrite(Dendrite* dendrite);
    bool disconnect_from_dendrite(Dendrite* dendrite);
    void disconnect_all();
    void propagate_signal(float signal_strength);
    
    // weight management - clamp to realistic EPSP/IPSP range
    inline void adjust_weight(float delta) {
        weight += delta;
        if (weight > 10.0f) weight = 10.0f;
        else if (weight < 0.1f) weight = 0.1f;
    }
    
    inline float get_weight() const { return weight; }
    inline float get_threshold() const { return threshold; }
    inline bool is_inhibitory_synapse() const { return is_inhibitory; }
    inline int get_connection_count() const { return connection_count; }
    
    virtual ~Synapse();
};

#endif