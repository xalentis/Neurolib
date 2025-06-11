#ifndef AXON_H
#define AXON_H

class Synapse;

class Axon {
private:
    float length;           // Axon length in micrometers
    float diameter;         // Axon diameter in micrometers
    bool is_myelinated;     // Whether axon is myelinated
    float conduction_velocity; // m/s
    Synapse** output_synapses; // Synapses this axon connects to
    int synapse_count;
    int max_synapses;
    
public:
    explicit Axon(float len = 10000.0f, float diam = 1.0f, bool myelinated = true, int max_syn = 1000);
    
    // Add output synapse
    bool add_output_synapse(Synapse* synapse);
    
    // Propagate action potential through all output synapses
    void propagate_action_potential(float amplitude = 50.0f);
    
    // Accessors
    inline float get_conduction_velocity() const { return conduction_velocity; }
    inline bool get_is_myelinated() const { return is_myelinated; }
    inline int get_synapse_count() const { return synapse_count; }
    inline float get_length() const { return length; }
    inline float get_diameter() const { return diameter; }
    
    virtual ~Axon();
};

#endif // AXON_H