#ifndef DENDRITE_H
#define DENDRITE_H

class Synapse;
class Neuron;

class Dendrite {
private:
    float length;             // length in micrometers
    float diameter;           // diameter in micrometers
    int spine_count;          // number of dendritic spines
    float membrane_potential; // current membrane potential in mV
    Synapse** synapses;       // array of connected synapses
    int synapse_count;        // current number of synapses
    int max_synapses;         // maximum synapses (based on spine count)
    bool is_active;           // whether dendrite is currently active
    Neuron* parent_neuron;    // reference to parent neuron
    
public:
    explicit Dendrite(float len = 300.0f, float diam = 2.0f, int spines = 5000, Neuron* parent = nullptr);
    
    // synaptic integration - sum all synaptic inputs
    float integrate_synaptic_inputs();
    
    void update_membrane_potential();
    bool add_synapse(Synapse* synapse);
    bool remove_synapse(Synapse* synapse);
    
    inline float get_membrane_potential() const { return membrane_potential; }
    inline bool get_is_active() const { return is_active; }
    inline int get_synapse_count() const { return synapse_count; }
    inline Neuron* get_parent_neuron() const { return parent_neuron; }
    inline float get_length() const { return length; }
    inline float get_diameter() const { return diameter; }
    inline int get_spine_count() const { return spine_count; }
    
    // calculate surface area for synaptic density calculations
    inline float get_surface_area() const {
        const float PI = 3.14159f;
        return PI * diameter * length; // cylindrical approximation
    }
    
    // get synaptic density (synapses per unit area)
    inline float get_synaptic_density() const {
        return static_cast<float>(synapse_count) / get_surface_area();
    }
    
    virtual ~Dendrite();
};

#endif