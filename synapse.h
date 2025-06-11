#ifndef SYNAPSE_H
#define SYNAPSE_H

// Forward declaration
class Dendrite;

class Synapse {
private:
    float weight;           // Synaptic strength in mV (EPSP/IPSP amplitude)
    float threshold;        // Activation threshold in mV
    bool is_inhibitory;     // True for inhibitory, false for excitatory
    Dendrite** connections; // Array of connected dendrites
    int connection_count;   // Number of connected dendrites
    int max_connections;    // Maximum allowed connections
    
public:
    // Constructor with biologically realistic default values
    explicit Synapse(float w = 1.0f, float t = -50.0f, bool inhibit = false, int max_conn = 1);
    
    // Core functionality - models synaptic transmission
    inline float get_synaptic_contribution() const {
        return is_inhibitory ? -weight : weight;
    }
    
    // Check if transmission occurs given membrane potential
    inline bool transmit(float membrane_potential) const {
        float new_potential = membrane_potential + get_synaptic_contribution();
        return new_potential >= threshold;
    }
    
    // Connection management
    bool connect_to_dendrite(Dendrite* dendrite);
    bool disconnect_from_dendrite(Dendrite* dendrite);
    void disconnect_all();
    
    // Signal propagation to connected dendrites
    void propagate_signal(float signal_strength);
    
    // Weight management - clamp to realistic EPSP/IPSP range
    inline void adjust_weight(float delta) {
        weight += delta;
        if (weight > 10.0f) weight = 10.0f;
        else if (weight < 0.1f) weight = 0.1f;
    }
    
    // Accessors
    inline float get_weight() const { return weight; }
    inline float get_threshold() const { return threshold; }
    inline bool is_inhibitory_synapse() const { return is_inhibitory; }
    inline int get_connection_count() const { return connection_count; }
    
    virtual ~Synapse();
};

#endif // SYNAPSE_H