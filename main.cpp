#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <vector>
#include <iomanip>
#include "neuron_types.h"
#include "dendrite.h"
#include "axon.h"
#include "synapse.h"

// Structure to store simulation data for visualization
struct SimulationData {
    std::vector<std::vector<float>> membrane_potentials; // [timestep][neuron_id]
    std::vector<std::pair<int, int>> spike_events;       // [(timestep, neuron_id)]
    std::vector<int> spikes_per_timestep;                // spike count per timestep
    std::vector<float> network_activity;                 // average membrane potential per timestep
    int total_timesteps;
    int total_spikes;
};

// Global simulation data
SimulationData sim_data;

// Utility function to create random connections with higher weights
void create_random_connections(Neuron** neurons, int neuron_count, int connection_density = 5) {
    for (int i = 0; i < neuron_count; ++i) {
        for (int j = 0; j < connection_density; ++j) {
            int target = rand() % neuron_count;
            if (target != i && neurons[target]->get_dendrite_count() > 0) {
                int target_dendrite = rand() % neurons[target]->get_dendrite_count();
                // Increased synaptic weights for stronger connections
                float weight = 1.5f + static_cast<float>(rand()) / RAND_MAX * 3.0f; // 1.5-4.5mV (higher than original)
                bool inhibitory = !neurons[i]->get_is_excitatory() || (rand() % 8 == 0); // Reduced inhibition to 12.5%
                
                neurons[i]->connect_to_neuron(neurons[target], target_dendrite, weight, inhibitory);
            }
        }
    }
}

// Function to apply background noise/spontaneous activity
void apply_background_activity(Neuron** neurons, int neuron_count, float noise_probability = 0.3f) {
    for (int i = 0; i < neuron_count; ++i) {
        if (static_cast<float>(rand()) / RAND_MAX < noise_probability) {
            // Add small random depolarization to simulate background activity
            float noise_amplitude = 2.0f + static_cast<float>(rand()) / RAND_MAX * 8.0f; // 2-10mV
            // This would need a method in your neuron class to add current injection
            // For now, we'll use a probabilistic spike trigger with increased chance
            if (static_cast<float>(rand()) / RAND_MAX < 0.25f) { // Increased from 10% to 25% chance for background spike
                neurons[i]->spike();
            }
        }
    }
}

// Function to collect membrane potential data
void collect_membrane_data(Neuron** neurons, int neuron_count, int timestep) {
    std::vector<float> potentials;
    float total_potential = 0.0f;
    
    for (int i = 0; i < neuron_count; ++i) {
        float potential = neurons[i]->get_membrane_potential();
        potentials.push_back(potential);
        total_potential += potential;
    }
    
    sim_data.membrane_potentials.push_back(potentials);
    sim_data.network_activity.push_back(total_potential / neuron_count);
}

// Function to record spike events
void record_spike_event(int timestep, int neuron_id) {
    sim_data.spike_events.push_back(std::make_pair(timestep, neuron_id));
}

// Function to generate CSV file for membrane potentials
void generate_membrane_potential_csv(const std::string& filename, int neuron_count) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing." << std::endl;
        return;
    }
    
    // Write header
    file << "Timestep";
    for (int i = 0; i < neuron_count; ++i) {
        file << ",Neuron_" << i;
    }
    file << ",Network_Average" << std::endl;
    
    // Write data
    for (size_t t = 0; t < sim_data.membrane_potentials.size(); ++t) {
        file << t;
        for (int i = 0; i < neuron_count; ++i) {
            file << "," << std::fixed << std::setprecision(2) << sim_data.membrane_potentials[t][i];
        }
        file << "," << std::fixed << std::setprecision(2) << sim_data.network_activity[t] << std::endl;
    }
    
    file.close();
    std::cout << "Membrane potential data saved to: " << filename << std::endl;
}

// Function to generate CSV file for spike events (raster plot data)
void generate_spike_raster_csv(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing." << std::endl;
        return;
    }
    
    // Write header
    file << "Timestep,Neuron_ID,Spike" << std::endl;
    
    // Write spike events
    for (const auto& spike : sim_data.spike_events) {
        file << spike.first << "," << spike.second << ",1" << std::endl;
    }
    
    file.close();
    std::cout << "Spike raster data saved to: " << filename << std::endl;
}

// Function to generate activity summary CSV
void generate_activity_summary_csv(const std::string& filename, int neuron_count) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing." << std::endl;
        return;
    }
    
    // Write header
    file << "Timestep,Total_Spikes,Network_Activity" << std::endl;
    
    // Count spikes per timestep
    std::vector<int> spikes_per_step(sim_data.total_timesteps, 0);
    for (const auto& spike : sim_data.spike_events) {
        if (spike.first < sim_data.total_timesteps) {
            spikes_per_step[spike.first]++;
        }
    }
    
    // Write data
    for (int t = 0; t < sim_data.total_timesteps && t < static_cast<int>(sim_data.network_activity.size()); ++t) {
        file << t << "," << spikes_per_step[t] << "," 
             << std::fixed << std::setprecision(2) << sim_data.network_activity[t] << std::endl;
    }
    
    file.close();
    std::cout << "Activity summary data saved to: " << filename << std::endl;
}

// Function to generate Python plotting script
void generate_python_plotting_script(const std::string& filename, int neuron_count) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing." << std::endl;
        return;
    }
    
    file << "#!/usr/bin/env python3\n";
    file << "import pandas as pd\n";
    file << "import matplotlib.pyplot as plt\n";
    file << "import numpy as np\n";
    file << "from matplotlib.patches import Rectangle\n\n";
    
    file << "# Load data\n";
    file << "membrane_data = pd.read_csv('membrane_potentials.csv')\n";
    file << "spike_data = pd.read_csv('spike_raster.csv')\n";
    file << "activity_data = pd.read_csv('activity_summary.csv')\n\n";
    
    file << "# Create figure with subplots\n";
    file << "fig, axes = plt.subplots(2, 2, figsize=(15, 12))\n";
    file << "fig.suptitle('Enhanced Human Neuron Network Simulation Results', fontsize=16, fontweight='bold')\n\n";
    
    file << "# Plot 1: Membrane Potentials Over Time\n";
    file << "ax1 = axes[0, 0]\n";
    file << "colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43']\n";
    file << "for i in range(min(" << neuron_count << ", len(colors))):\n";
    file << "    ax1.plot(membrane_data['Timestep'], membrane_data[f'Neuron_{i}'], \n";
    file << "             color=colors[i], linewidth=1.5, alpha=0.8, label=f'Neuron {i}')\n";
    file << "ax1.axhline(y=-55, color='red', linestyle='--', alpha=0.7, label='Spike Threshold')\n";
    file << "ax1.set_xlabel('Timestep')\n";
    file << "ax1.set_ylabel('Membrane Potential (mV)')\n";
    file << "ax1.set_title('Membrane Potential Traces')\n";
    file << "ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')\n";
    file << "ax1.grid(True, alpha=0.3)\n\n";
    
    file << "# Plot 2: Spike Raster Plot\n";
    file << "ax2 = axes[0, 1]\n";
    file << "if not spike_data.empty:\n";
    file << "    for neuron_id in range(" << neuron_count << "):\n";
    file << "        neuron_spikes = spike_data[spike_data['Neuron_ID'] == neuron_id]\n";
    file << "        ax2.scatter(neuron_spikes['Timestep'], neuron_spikes['Neuron_ID'], \n";
    file << "                   color=colors[neuron_id % len(colors)], s=20, alpha=0.8)\n";
    file << "ax2.set_xlabel('Timestep')\n";
    file << "ax2.set_ylabel('Neuron ID')\n";
    file << "ax2.set_title('Spike Raster Plot')\n";
    file << "ax2.set_yticks(range(" << neuron_count << "))\n";
    file << "ax2.grid(True, alpha=0.3)\n\n";
    
    file << "# Plot 3: Network Activity Over Time\n";
    file << "ax3 = axes[1, 0]\n";
    file << "ax3.plot(activity_data['Timestep'], activity_data['Network_Activity'], \n";
    file << "         color='purple', linewidth=2, label='Average Membrane Potential')\n";
    file << "ax3.set_xlabel('Timestep')\n";
    file << "ax3.set_ylabel('Average Membrane Potential (mV)')\n";
    file << "ax3.set_title('Network Activity')\n";
    file << "ax3.legend()\n";
    file << "ax3.grid(True, alpha=0.3)\n\n";
    
    file << "# Plot 4: Spike Count Histogram\n";
    file << "ax4 = axes[1, 1]\n";
    file << "ax4.bar(activity_data['Timestep'], activity_data['Total_Spikes'], \n";
    file << "        color='orange', alpha=0.7, width=0.8)\n";
    file << "ax4.set_xlabel('Timestep')\n";
    file << "ax4.set_ylabel('Number of Spikes')\n";
    file << "ax4.set_title('Spike Count per Timestep')\n";
    file << "ax4.grid(True, alpha=0.3)\n\n";
    
    file << "# Add simulation statistics as text\n";
    file << "total_spikes = len(spike_data)\n";
    file << "total_timesteps = len(activity_data)\n";
    file << "spike_rate = total_spikes / total_timesteps if total_timesteps > 0 else 0\n";
    file << "stats_text = f'Total Spikes: {total_spikes}\\nTotal Timesteps: {total_timesteps}\\nSpike Rate: {spike_rate:.2f} spikes/timestep'\n";
    file << "fig.text(0.02, 0.02, stats_text, fontsize=10, \n";
    file << "         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))\n\n";
    
    file << "plt.tight_layout()\n";
    file << "plt.savefig('neural_network_simulation.png', dpi=300, bbox_inches='tight')\n";
    file << "plt.show()\n";
    file << "print('Visualization saved as neural_network_simulation.png')\n";
    
    file.close();
    std::cout << "Python plotting script saved to: " << filename << std::endl;
}

// Function to generate MATLAB plotting script
void generate_matlab_plotting_script(const std::string& filename, int neuron_count) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing." << std::endl;
        return;
    }
    
    file << "% MATLAB script to visualize neural network simulation data\n";
    file << "clear; clc; close all;\n\n";
    
    file << "% Load data\n";
    file << "membrane_data = readtable('membrane_potentials.csv');\n";
    file << "spike_data = readtable('spike_raster.csv');\n";
    file << "activity_data = readtable('activity_summary.csv');\n\n";
    
    file << "% Create figure\n";
    file << "figure('Position', [100, 100, 1200, 900]);\n";
    file << "sgtitle('Enhanced Human Neuron Network Simulation Results', 'FontSize', 16, 'FontWeight', 'bold');\n\n";
    
    file << "% Plot 1: Membrane Potentials\n";
    file << "subplot(2, 2, 1);\n";
    file << "colors = lines(" << neuron_count << ");\n";
    file << "hold on;\n";
    file << "for i = 1:min(" << neuron_count << ", width(membrane_data)-2)\n";
    file << "    plot(membrane_data.Timestep, membrane_data{:, i+1}, 'Color', colors(i,:), 'LineWidth', 1.5);\n";
    file << "end\n";
    file << "yline(-55, '--r', 'Spike Threshold', 'LineWidth', 1.5);\n";
    file << "xlabel('Timestep');\n";
    file << "ylabel('Membrane Potential (mV)');\n";
    file << "title('Membrane Potential Traces');\n";
    file << "legend('Location', 'eastoutside');\n";
    file << "grid on;\n\n";
    
    file << "% Plot 2: Spike Raster\n";
    file << "subplot(2, 2, 2);\n";
    file << "if height(spike_data) > 0\n";
    file << "    scatter(spike_data.Timestep, spike_data.Neuron_ID, 30, 'filled', 'MarkerFaceAlpha', 0.7);\n";
    file << "end\n";
    file << "xlabel('Timestep');\n";
    file << "ylabel('Neuron ID');\n";
    file << "title('Spike Raster Plot');\n";
    file << "yticks(0:" << neuron_count-1 << ");\n";
    file << "grid on;\n\n";
    
    file << "% Plot 3: Network Activity\n";
    file << "subplot(2, 2, 3);\n";
    file << "plot(activity_data.Timestep, activity_data.Network_Activity, 'b-', 'LineWidth', 2);\n";
    file << "xlabel('Timestep');\n";
    file << "ylabel('Average Membrane Potential (mV)');\n";
    file << "title('Network Activity');\n";
    file << "grid on;\n\n";
    
    file << "% Plot 4: Spike Count\n";
    file << "subplot(2, 2, 4);\n";
    file << "bar(activity_data.Timestep, activity_data.Total_Spikes, 'FaceColor', [1 0.5 0], 'FaceAlpha', 0.7);\n";
    file << "xlabel('Timestep');\n";
    file << "ylabel('Number of Spikes');\n";
    file << "title('Spike Count per Timestep');\n";
    file << "grid on;\n\n";
    
    file << "% Save figure\n";
    file << "saveas(gcf, 'neural_network_simulation_matlab.png');\n";
    file << "fprintf('Visualization saved as neural_network_simulation_matlab.png\\n');\n";
    
    file.close();
    std::cout << "MATLAB plotting script saved to: " << filename << std::endl;
}

int main() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    
    std::cout << "=== Enhanced Human Neuron Network Simulation with Visualization ===" << std::endl;
    std::cout << "Creating neural network with increased spiking probability..." << std::endl;
    
    const int NEURON_COUNT = 10;
    Neuron* neurons[NEURON_COUNT];
    
    // Initialize simulation data
    sim_data.total_timesteps = 0;
    sim_data.total_spikes = 0;
    
    // Create diverse neuron types with bias toward excitatory neurons
    neurons[0] = new PyramidalNeuron();
    neurons[1] = new PyramidalNeuron();
    neurons[2] = new PyramidalNeuron();
    neurons[3] = new PyramidalNeuron(); // Added extra pyramidal (excitatory)
    neurons[4] = new Interneuron();     // Reduced inhibitory neurons
    neurons[5] = new PurkinjeNeuron();
    neurons[6] = new MotorNeuron();
    neurons[7] = new MotorNeuron();
    neurons[8] = new SensoryNeuron();
    neurons[9] = new SensoryNeuron();
    
    std::cout << "Created " << NEURON_COUNT << " neurons with excitatory bias:" << std::endl;
    std::cout << "- 4 Pyramidal neurons (excitatory, cortical)" << std::endl;
    std::cout << "- 1 Interneuron (inhibitory, local processing)" << std::endl;
    std::cout << "- 1 Purkinje neuron (cerebellar, complex dendrites)" << std::endl;
    std::cout << "- 2 Motor neurons (large, muscle control)" << std::endl;
    std::cout << "- 2 Sensory neurons (input processing)" << std::endl;
    
    // Create denser connections with stronger weights
    std::cout << "\nCreating dense synaptic connections with stronger weights..." << std::endl;
    create_random_connections(neurons, NEURON_COUNT, 6); // Increased connection density
    
    // Display connection information
    for (int i = 0; i < NEURON_COUNT; ++i) {
        std::cout << "Neuron " << i << " (type " << neurons[i]->get_neuron_type_id() 
                  << "): " << neurons[i]->get_axon()->get_synapse_count() 
                  << " outgoing synapses, " << neurons[i]->get_dendrite_count() 
                  << " dendrites" << std::endl;
    }
    
    std::cout << "\n=== Simulating Enhanced Neural Activity with Data Collection ===" << std::endl;
    std::cout << "Running simulation until at least 5 spikes occur..." << std::endl;
    
    int timestep = 0;
    int total_spikes = 0;
    const int MIN_SPIKES_REQUIRED = 5;
    const int MAX_TIMESTEPS = 10000; // Safety limit to prevent infinite loops
    
    // Run simulation until we get at least 5 spikes or hit max timesteps
    while (total_spikes < MIN_SPIKES_REQUIRED && timestep < MAX_TIMESTEPS) {
        std::cout << "\n--- Timestep " << timestep + 1 << " ---" << std::endl;
        
        // Collect membrane potential data before updates
        collect_membrane_data(neurons, NEURON_COUNT, timestep);
        
        // Strategy 1: More frequent direct stimulation (every 2nd timestep for higher probability)
        if (timestep % 2 == 0) { // Increased frequency from every 3rd to every 2nd
            int num_stimulated = 1 + rand() % 4; // Stimulate 1-4 neurons (increased from 1-3)
            for (int s = 0; s < num_stimulated; ++s) {
                int stimulated_neuron = rand() % NEURON_COUNT;
                std::cout << "Directly stimulating neuron " << stimulated_neuron << std::endl;
                neurons[stimulated_neuron]->spike();
            }
        }
        
        // Strategy 2: Apply background activity/noise (increased probability)
        apply_background_activity(neurons, NEURON_COUNT, 0.6f); // Increased from 40% to 60%
        
        // Strategy 3: More frequent burst stimulation
        if (timestep % 8 == 0) { // Every 8th timestep instead of specific ones
            std::cout << "Applying burst stimulation to multiple neurons!" << std::endl;
            for (int burst = 0; burst < 5; ++burst) { // Increased burst size
                int burst_neuron = rand() % NEURON_COUNT;
                neurons[burst_neuron]->spike();
            }
        }
        
        // Strategy 4: Progressive stimulation intensity
        if (timestep > 20 && total_spikes < 2) {
            std::cout << "Low spike activity detected - increasing stimulation intensity!" << std::endl;
            // Force stimulate more neurons
            for (int extra = 0; extra < 3; ++extra) {
                int extra_neuron = rand() % NEURON_COUNT;
                neurons[extra_neuron]->spike();
            }
        }
        
        // Update all neurons and check for spikes
        int spike_count = 0;
        for (int i = 0; i < NEURON_COUNT; ++i) {
            if (neurons[i]->update_and_check_spike()) {
                std::cout << "Neuron " << i << " (type " << neurons[i]->get_neuron_type_id() 
                          << ") SPIKED! Membrane potential: " 
                          << neurons[i]->get_membrane_potential() << "mV" << std::endl;
                spike_count++;
                total_spikes++;
                
                // Record spike event for visualization
                record_spike_event(timestep, i);
            }
        }
        
        if (spike_count == 0) {
            std::cout << "No spikes this timestep" << std::endl;
        } else {
            std::cout << "Timestep spikes: " << spike_count << " | Total spikes so far: " << total_spikes << std::endl;
        }
        
        // Show membrane potentials of first few neurons
        std::cout << "Membrane potentials: ";
        for (int i = 0; i < 5 && i < NEURON_COUNT; ++i) {
            std::cout << "N" << i << ":" << neurons[i]->get_membrane_potential() << "mV ";
        }
        std::cout << std::endl;
        
        // Add cascade detection
        if (spike_count >= 3) {
            std::cout << "*** SPIKE CASCADE DETECTED! Multiple neurons firing together ***" << std::endl;
        }
        
        // Progress indicator
        if (timestep % 10 == 0 && timestep > 0) {
            std::cout << "=== Progress: " << timestep << " timesteps completed, " 
                      << total_spikes << "/" << MIN_SPIKES_REQUIRED << " spikes achieved ===" << std::endl;
        }
        
        timestep++;
    }
    
    // Store final simulation parameters
    sim_data.total_timesteps = timestep;
    sim_data.total_spikes = total_spikes;
    
    // Final results
    std::cout << "\n=== SIMULATION RESULTS ===" << std::endl;
    if (total_spikes >= MIN_SPIKES_REQUIRED) {
        std::cout << "SUCCESS! Achieved " << total_spikes << " spikes in " << timestep << " timesteps!" << std::endl;
        std::cout << "Average spike rate: " << (float)total_spikes / timestep << " spikes per timestep" << std::endl;
    } else {
        std::cout << "Simulation reached maximum timesteps (" << MAX_TIMESTEPS << ") with " << total_spikes << " spikes." << std::endl;
        std::cout << "Consider further increasing stimulation parameters." << std::endl;
    }
    
    // Generate visualization data files
    std::cout << "\n=== GENERATING VISUALIZATION DATA ===" << std::endl;
    generate_membrane_potential_csv("membrane_potentials.csv", NEURON_COUNT);
    generate_spike_raster_csv("spike_raster.csv");
    generate_activity_summary_csv("activity_summary.csv", NEURON_COUNT);
    
    // Generate plotting scripts
    std::cout << "\n=== GENERATING PLOTTING SCRIPTS ===" << std::endl;
    generate_python_plotting_script("plot_simulation.py", NEURON_COUNT);
    generate_matlab_plotting_script("plot_simulation.m", NEURON_COUNT);
    
    std::cout << "\n=== VISUALIZATION INSTRUCTIONS ===" << std::endl;
    std::cout << "To generate plots, run one of the following:" << std::endl;
    std::cout << "1. Python: python3 plot_simulation.py" << std::endl;
    std::cout << "2. MATLAB: run plot_simulation.m" << std::endl;
    std::cout << "\nRequired dependencies:" << std::endl;
    std::cout << "- Python: matplotlib, pandas, numpy" << std::endl;
    std::cout << "- MATLAB: base installation" << std::endl;
    
    std::cout << "\n=== Enhanced Simulation Complete ===" << std::endl;
    std::cout << "Neural activity successfully demonstrated with enhanced spiking through:" << std::endl;
    std::cout << "- Stronger synaptic weights (1.5-4.5mV vs original 0.5-2.5mV)" << std::endl;
    std::cout << "- Increased connection density (6 vs original 3-4 connections per neuron)" << std::endl;
    std::cout << "- Reduced inhibitory connections (12.5% vs 20%)" << std::endl;
    std::cout << "- More excitatory neurons (4 pyramidal vs 3)" << std::endl;
    std::cout << "- Adaptive stimulation (every 2nd timestep + 60% background activity)" << std::endl;
    std::cout << "- Enhanced background noise (25% spike probability vs 10%)" << std::endl;
    std::cout << "- Progressive intensity scaling based on activity levels" << std::endl;
    std::cout << "- Guaranteed minimum spike threshold with safety limits" << std::endl;
    std::cout << "- Complete data collection and visualization generation" << std::endl;
    
    // Cleanup
    for (int i = 0; i < NEURON_COUNT; ++i) {
        delete neurons[i];
    }
    
    return 0;
}