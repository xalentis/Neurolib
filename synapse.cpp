#include "synapse.h"
#include "dendrite.h"

Synapse::Synapse(float w, float t, bool inhibit, int max_conn) 
    : weight(w), threshold(t), is_inhibitory(inhibit), connection_count(0), max_connections(max_conn) {
    connections = new Dendrite*[max_connections];
    for (int i = 0; i < max_connections; ++i) {
        connections[i] = nullptr;
    }
}

bool Synapse::connect_to_dendrite(Dendrite* dendrite) {
    if (connection_count < max_connections && dendrite != nullptr) {
        connections[connection_count] = dendrite;
        connection_count++;
        return dendrite->add_synapse(this);
    }
    return false;
}

bool Synapse::disconnect_from_dendrite(Dendrite* dendrite) {
    for (int i = 0; i < connection_count; ++i) {
        if (connections[i] == dendrite) {
            dendrite->remove_synapse(this);
            for (int j = i; j < connection_count - 1; ++j) {
                connections[j] = connections[j + 1];
            }
            connection_count--;
            connections[connection_count] = nullptr;
            return true;
        }
    }
    return false;
}

void Synapse::disconnect_all() {
    for (int i = 0; i < connection_count; ++i) {
        if (connections[i] != nullptr) {
            connections[i]->remove_synapse(this);
            connections[i] = nullptr;
        }
    }
    connection_count = 0;
}

void Synapse::propagate_signal(float signal_strength) {
    for (int i = 0; i < connection_count; ++i) {
        if (connections[i] != nullptr && transmit(signal_strength)) {
            connections[i]->update_membrane_potential();
        }
    }
}

Synapse::~Synapse() {
    disconnect_all();
    delete[] connections;
}