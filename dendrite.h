#ifndef DENDRITE_H
#define DENDRITE_H

class Synapse;
class Neuron;

class Dendrite {
private:
    float length;           // Length in micrometers
    float diameter;         // Diameter in micrometers
    int spine_count;        // Number of dendritic spines
    float membrane_potential; // Current membrane potential in mV
    Synapse** synapses;     // Array of connected synapses
    int synapse_count;      // Current number of synapses
    int max_synapses;       // Maximum synapses (based on spine count)
    bool is_active;         // Whether dendrite is currently active
    Neuron* parent_neuron;  // Reference to parent neuron
    
public:
    explicit Dendrite(float len = 300.0f, float diam = 2.0f, int spines = 5000, Neuron* parent = nullptr);
    
    // Synaptic integration - sum all synaptic inputs
    float integrate_synaptic_inputs();
    
    // Update membrane potential and notify parent neuron
    void update_membrane_potential();
    
    // Synapse management
    bool add_synapse(Synapse* synapse);
    bool remove_synapse(Synapse* synapse);
    
    // Accessors
    inline float get_membrane_potential() const { return membrane_potential; }
    inline bool get_is_active() const { return is_active; }
    inline int get_synapse_count() const { return synapse_count; }
    inline Neuron* get_parent_neuron() const { return parent_neuron; }
    inline float get_length() const { return length; }
    inline float get_diameter() const { return diameter; }
    inline int get_spine_count() const { return spine_count; }
    
    // Calculate surface area for synaptic density calculations
    inline float get_surface_area() const {
        const float PI = 3.14159f;
        return PI * diameter * length; // Cylindrical approximation
    }
    
    // Get synaptic density (synapses per unit area)
    inline float get_synaptic_density() const {
        return static_cast<float>(synapse_count) / get_surface_area();
    }
    
    virtual ~Dendrite();
};

#endif // DENDRITE_H