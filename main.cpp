#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <sstream> 
#include "neuron_types.h"
#include "dendrite.h"
#include "axon.h"
#include "synapse.h"


// structure to store simulation data for visualization
struct SimulationData {
    std::vector<std::vector<float>> membrane_potentials; // [timestep][neuron_id]
    std::vector<std::pair<int, int>> spike_events;       // [(timestep, neuron_id)]
    std::vector<int> spikes_per_timestep;                // spike count per timestep
    std::vector<float> network_activity;                 // average membrane potential per timestep
    int total_timesteps;
    int total_spikes;
};

// create random connections with higher weights
void create_random_connections(Neuron** neurons, int neuron_count, int connection_density = 5) {
    for (int i = 0; i < neuron_count; ++i) {
        for (int j = 0; j < connection_density; ++j) {
            int target = rand() % neuron_count;
            if (target != i && neurons[target]->get_dendrite_count() > 0) {
                int target_dendrite = rand() % neurons[target]->get_dendrite_count();
                float weight = 1.5f + static_cast<float>(rand()) / RAND_MAX * 3.0f; // 1.5-4.5mV
                bool inhibitory = !neurons[i]->get_is_excitatory() || (rand() % 8 == 0);
                
                neurons[i]->connect_to_neuron(neurons[target], target_dendrite, weight, inhibitory);
            }
        }
    }
}

// structure to hold stability metrics
struct StabilityMetrics {
    float coefficient_of_variation;   // cv of spike intervals
    float burst_coefficient;          // measure of bursting activity
    float synchrony_index;            // network synchronization
    float entropy;                    // shannon entropy of spike patterns
    float lyapunov_exponent;          // chaos/stability measure
    float homeostatic_deviation;      // deviation from target activity
    float network_coherence;          // phase coherence across neurons
    float critical_branching_ratio;   // near-critical dynamics indicator
};

// simulation data
SimulationData sim_data;

// coefficient of variation for spike timing
float calculate_spike_timing_cv(const std::vector<std::pair<int, int>>& spike_events) {
    if (spike_events.size() < 2) return 0.0f;
    
    // calculate inter-spike intervals for each neuron
    std::vector<float> all_intervals;
    
    for (int neuron_id = 0; neuron_id < 10; ++neuron_id) {
        std::vector<int> neuron_spikes;
        for (const auto& spike : spike_events) {
            if (spike.second == neuron_id) {
                neuron_spikes.push_back(spike.first);
            }
        }
        
        // calculate intervals for this neuron
        for (size_t i = 1; i < neuron_spikes.size(); ++i) {
            all_intervals.push_back(neuron_spikes[i] - neuron_spikes[i-1]);
        }
    }
    
    if (all_intervals.empty()) return 0.0f;
    
    // calculate mean and std dev
    float mean = std::accumulate(all_intervals.begin(), all_intervals.end(), 0.0f) / all_intervals.size();
    float variance = 0.0f;
    for (float interval : all_intervals) {
        variance += (interval - mean) * (interval - mean);
    }
    variance /= all_intervals.size();
    float std_dev = std::sqrt(variance);
    
    return (mean > 0) ? std_dev / mean : 0.0f;
}

// calculate burst coefficient
float calculate_burst_coefficient(const std::vector<std::pair<int, int>>& spike_events, int total_timesteps) {
    if (spike_events.empty()) return 0.0f;
    
    // create spike count per timestep
    std::vector<int> spikes_per_timestep(total_timesteps, 0);
    for (const auto& spike : spike_events) {
        if (spike.first < total_timesteps) {
            spikes_per_timestep[spike.first]++;
        }
    }
    
    // calculate burst events (timesteps with >1 spike)
    int burst_timesteps = 0;
    int total_spikes = 0;
    for (int count : spikes_per_timestep) {
        if (count > 1) burst_timesteps++;
        total_spikes += count;
    }
    
    if (total_spikes == 0) return 0.0f;
    return static_cast<float>(burst_timesteps) / total_timesteps;
}

// calculate network synchrony index
float calculate_synchrony_index(const std::vector<std::vector<float>>& membrane_potentials) {
    if (membrane_potentials.empty() || membrane_potentials[0].empty()) return 0.0f;
    
    int timesteps = membrane_potentials.size();
    int neurons = membrane_potentials[0].size();
    
    // Calculate pairwise correlations
    float total_correlation = 0.0f;
    int pairs = 0;
    
    for (int i = 0; i < neurons; ++i) {
        for (int j = i + 1; j < neurons; ++j) {
            // Calculate correlation between neuron i and j
            float mean_i = 0.0f, mean_j = 0.0f;
            for (int t = 0; t < timesteps; ++t) {
                mean_i += membrane_potentials[t][i];
                mean_j += membrane_potentials[t][j];
            }
            mean_i /= timesteps;
            mean_j /= timesteps;
            
            float numerator = 0.0f, denom_i = 0.0f, denom_j = 0.0f;
            for (int t = 0; t < timesteps; ++t) {
                float diff_i = membrane_potentials[t][i] - mean_i;
                float diff_j = membrane_potentials[t][j] - mean_j;
                numerator += diff_i * diff_j;
                denom_i += diff_i * diff_i;
                denom_j += diff_j * diff_j;
            }
            
            if (denom_i > 0 && denom_j > 0) {
                float correlation = numerator / std::sqrt(denom_i * denom_j);
                total_correlation += std::abs(correlation);
                pairs++;
            }
        }
    }
    
    return (pairs > 0) ? total_correlation / pairs : 0.0f;
}

// calculate shannon entropy of network activity patterns
float calculate_network_entropy(const std::vector<float>& network_activity) {
    if (network_activity.empty()) return 0.0f;
    
    // discretize activity levels into bins
    const int NUM_BINS = 10;
    float min_activity = *std::min_element(network_activity.begin(), network_activity.end());
    float max_activity = *std::max_element(network_activity.begin(), network_activity.end());
    
    if (max_activity <= min_activity) return 0.0f;
    
    std::vector<int> bins(NUM_BINS, 0);
    float bin_width = (max_activity - min_activity) / NUM_BINS;
    
    // count occurrences in each bin
    for (float activity : network_activity) {
        int bin = static_cast<int>((activity - min_activity) / bin_width);
        if (bin >= NUM_BINS) bin = NUM_BINS - 1;
        bins[bin]++;
    }
    
    // calculate shannon entropy
    float entropy = 0.0f;
    int total_samples = network_activity.size();
    for (int count : bins) {
        if (count > 0) {
            float probability = static_cast<float>(count) / total_samples;
            entropy -= probability * std::log2(probability);
        }
    }
    
    return entropy;
}

// calculate homeostatic deviation
float calculate_homeostatic_deviation(const std::vector<float>& network_activity, float target_activity = -65.0f) {
    if (network_activity.empty()) return 0.0f;
    
    float mean_activity = std::accumulate(network_activity.begin(), network_activity.end(), 0.0f) / network_activity.size();
    return std::abs(mean_activity - target_activity);
}

// calculate network coherence (phase synchronization measure)
float calculate_network_coherence(const std::vector<std::vector<float>>& membrane_potentials) {
    if (membrane_potentials.empty() || membrane_potentials[0].empty()) return 0.0f;
    
    int timesteps = membrane_potentials.size();
    int neurons = membrane_potentials[0].size();
    
    // variance of mean network activity
    std::vector<float> mean_at_timestep(timesteps);
    for (int t = 0; t < timesteps; ++t) {
        float sum = 0.0f;
        for (int n = 0; n < neurons; ++n) {
            sum += membrane_potentials[t][n];
        }
        mean_at_timestep[t] = sum / neurons;
    }
    
    // calculate variance of network mean
    float overall_mean = std::accumulate(mean_at_timestep.begin(), mean_at_timestep.end(), 0.0f) / timesteps;
    float variance = 0.0f;
    for (float value : mean_at_timestep) {
        variance += (value - overall_mean) * (value - overall_mean);
    }
    variance /= timesteps;
    
    // higher variance indicates less coherence, so return inverse measure
    return 1.0f / (1.0f + variance);
}

// estimate critical branching ratio (avalanche dynamics)
float calculate_critical_branching_ratio(const std::vector<std::pair<int, int>>& spike_events, int total_timesteps) {
    if (spike_events.empty()) return 0.0f;
    
    // count spikes per timestep
    std::vector<int> spikes_per_timestep(total_timesteps, 0);
    for (const auto& spike : spike_events) {
        if (spike.first < total_timesteps) {
            spikes_per_timestep[spike.first]++;
        }
    }
    
    // find avalanche sizes
    std::vector<int> avalanche_sizes;
    int current_avalanche = 0;
    
    for (int count : spikes_per_timestep) {
        if (count > 0) {
            current_avalanche += count;
        } else if (current_avalanche > 0) {
            avalanche_sizes.push_back(current_avalanche);
            current_avalanche = 0;
        }
    }
    
    if (current_avalanche > 0) {
        avalanche_sizes.push_back(current_avalanche);
    }
    
    if (avalanche_sizes.empty()) return 0.0f;
    
    // calculate branching ratio
    float total_size = std::accumulate(avalanche_sizes.begin(), avalanche_sizes.end(), 0);
    return total_size / avalanche_sizes.size();
}

// main function to calculate all stability metrics
StabilityMetrics calculate_stability_metrics(const SimulationData& sim_data) {
    StabilityMetrics metrics;
    
    metrics.coefficient_of_variation = calculate_spike_timing_cv(sim_data.spike_events);
    metrics.burst_coefficient = calculate_burst_coefficient(sim_data.spike_events, sim_data.total_timesteps);
    metrics.synchrony_index = calculate_synchrony_index(sim_data.membrane_potentials);
    metrics.entropy = calculate_network_entropy(sim_data.network_activity);
    metrics.homeostatic_deviation = calculate_homeostatic_deviation(sim_data.network_activity);
    metrics.network_coherence = calculate_network_coherence(sim_data.membrane_potentials);
    metrics.critical_branching_ratio = calculate_critical_branching_ratio(sim_data.spike_events, sim_data.total_timesteps);
    
    return metrics;
}

// stability metrics
void print_stability_report(const StabilityMetrics& metrics) {
    std::cout << "\n=== NETWORK STABILITY ANALYSIS ===" << std::endl;
    std::cout << "Coefficient of Variation: " << std::fixed << std::setprecision(3) << metrics.coefficient_of_variation << std::endl;
    std::cout << "  (Lower = more regular firing, Higher = more irregular)" << std::endl;
    
    std::cout << "Burst Coefficient: " << metrics.burst_coefficient << std::endl;
    std::cout << "  (0 = no bursts, 1 = all activity is bursts)" << std::endl;
    
    std::cout << "Synchrony Index: " << metrics.synchrony_index << std::endl;
    std::cout << "  (0 = no synchrony, 1 = perfect synchrony)" << std::endl;
    
    std::cout << "Network Entropy: " << metrics.entropy << std::endl;
    std::cout << "  (Higher = more diverse activity patterns)" << std::endl;
    
    std::cout << "Homeostatic Deviation: " << metrics.homeostatic_deviation << std::endl;
    std::cout << "  (Distance from target resting potential)" << std::endl;
    
    std::cout << "Network Coherence: " << metrics.network_coherence << std::endl;
    std::cout << "  (Higher = more coherent network oscillations)" << std::endl;
    
    std::cout << "Critical Branching Ratio: " << metrics.critical_branching_ratio << std::endl;
    std::cout << "  (Near 1.0 = critical dynamics, optimal information processing)" << std::endl;
    
    // stability assessment
    std::cout << "\n=== STABILITY ASSESSMENT ===" << std::endl;
    if (metrics.coefficient_of_variation > 2.0f) {
        std::cout << "HIGH IRREGULARITY: Spike timing is very irregular" << std::endl;
    }
    if (metrics.synchrony_index > 0.8f) {
        std::cout << "HYPERSYNCHRONY: Network may be prone to seizure-like activity" << std::endl;
    }
    if (metrics.homeostatic_deviation > 15.0f) {
        std::cout << "HOMEOSTATIC IMBALANCE: Network activity far from healthy baseline" << std::endl;
    }
    if (metrics.critical_branching_ratio < 0.5f || metrics.critical_branching_ratio > 2.0f) {
        std::cout << "NON-CRITICAL DYNAMICS: Network may be sub-optimal for information processing" << std::endl;
    }
    if (metrics.entropy < 1.0f) {
        std::cout << "LOW COMPLEXITY: Network activity patterns are overly simple" << std::endl;
    }
}

// metabolic dysfunction parameters
struct MetabolicCondition {
    std::string name;
    float glucose_level;              // mg/dL (normal: 70-100)
    float atp_efficiency;             // 0.0-1.0 (normal: 1.0)
    float ion_pump_function;          // 0.0-1.0 (normal: 1.0)
    float neurotransmitter_synthesis; // 0.0-1.0 (normal: 1.0)
    float membrane_integrity;         // 0.0-1.0 (normal: 1.0)
    float oxidative_stress;           // 0.0-5.0 (normal: 0.5)
    bool progressive;                 // whether condition worsens over time
    int onset_timestep;               // when dysfunction begins
};

// define various metabolic dysfunctions
MetabolicCondition create_hypoglycemia() {
    return {
        "Severe Hypoglycemia",
        35.0f,    // critically low glucose
        0.3f,     // severely reduced ATP production
        0.4f,     // impaired ion pumps
        0.5f,     // reduced neurotransmitter synthesis
        0.8f,     // mild membrane damage
        2.5f,     // elevated oxidative stress
        true,     // progressive worsening
        1000      // onset after 1000 timesteps
    };
}

MetabolicCondition create_diabetes_ketoacidosis() {
    return {
        "Diabetic Ketoacidosis",
        350.0f,   // extremely high glucose
        0.6f,     // moderately reduced ATP efficiency
        0.3f,     // severely impaired ion pumps
        0.4f,     // impaired neurotransmitter function
        0.6f,     // membrane damage from ketones
        3.5f,     // high oxidative stress
        true,     // progressive
        800       // earlier onset
    };
}

MetabolicCondition create_hypoxia() {
    return {
        "Cerebral Hypoxia",
        85.0f,    // normal glucose but no oxygen
        0.1f,     // severely reduced ATP (anaerobic metabolism)
        0.2f,     // critical ion pump failure
        0.3f,     // impaired synthesis
        0.5f,     // membrane damage
        4.0f,     // very high oxidative stress
        true,     // rapidly progressive
        500       // early onset
    };
}

MetabolicCondition create_mitochondrial_dysfunction() {
    return {
        "Mitochondrial Dysfunction",
        90.0f,    // normal glucose
        0.4f,     // poor ATP production
        0.6f,     // moderately impaired pumps
        0.7f,     // mildly reduced synthesis
        0.7f,     // mild membrane issues
        3.0f,     // high oxidative stress
        false,    // stable dysfunction
        200       // early onset
    };
}

// apply metabolic dysfunction to a neuron
void apply_metabolic_dysfunction(Neuron* neuron, const MetabolicCondition& condition, int current_timestep) {
    if (current_timestep < condition.onset_timestep) return;
    
    // calculate severity based on time since onset and progression
    float time_factor = 1.0f;
    if (condition.progressive) {
        time_factor = 1.0f + (current_timestep - condition.onset_timestep) * 0.001f; // gradual worsening
        time_factor = std::min(time_factor, 3.0f); // cap at 3x severity
    }
    
    // modify neuron properties based on metabolic state
    // we'll simulate effects through external manipulation
    
    // atp depletion affects ion pumps - make neurons less able to maintain resting potential
    float atp_factor = condition.atp_efficiency * condition.ion_pump_function / time_factor;
    
    // simulate ion pump failure by gradually depolarizing neurons
    if (atp_factor < 0.5f) {
        // force membrane potential toward 0 (depolarization block)
        std::cout << "Metabolic dysfunction causing depolarization in neuron (ATP factor: " 
                  << atp_factor << ")" << std::endl;
    }
    
    // oxidative stress damages membrane integrity
    if (condition.oxidative_stress > 2.0f) {
        // increase membrane leak - would make neurons more excitable initially
        std::cout << "Oxidative stress detected (level: " << condition.oxidative_stress << ")" << std::endl;
    }
}

// collect membrane potential data
void collect_membrane_data(Neuron** neurons, int neuron_count) {
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

// record spike events
void record_spike_event(int timestep, int neuron_id) {
    sim_data.spike_events.push_back(std::make_pair(timestep, neuron_id));
}

// simulation with metabolic dysfunction
void run_metabolic_dysfunction_simulation(MetabolicCondition condition, int max_timesteps = 3000) {
    std::cout << "\n=== METABOLIC DYSFUNCTION SIMULATION ===" << std::endl;
    std::cout << "Condition: " << condition.name << std::endl;
    std::cout << "Glucose Level: " << condition.glucose_level << " mg/dL (normal: 70-100)" << std::endl;
    std::cout << "ATP Efficiency: " << (condition.atp_efficiency * 100) << "% (normal: 100%)" << std::endl;
    std::cout << "Ion Pump Function: " << (condition.ion_pump_function * 100) << "% (normal: 100%)" << std::endl;
    std::cout << "Dysfunction Onset: Timestep " << condition.onset_timestep << std::endl;
    
    const int NEURON_COUNT = 10;
    Neuron* neurons[NEURON_COUNT];
    
    neurons[0] = new PyramidalNeuron();
    neurons[1] = new PyramidalNeuron();
    neurons[2] = new PyramidalNeuron();
    neurons[3] = new PyramidalNeuron();
    neurons[4] = new Interneuron();
    neurons[5] = new PurkinjeNeuron();
    neurons[6] = new MotorNeuron();
    neurons[7] = new MotorNeuron();
    neurons[8] = new SensoryNeuron();
    neurons[9] = new SensoryNeuron();
    
    create_random_connections(neurons, NEURON_COUNT, 6);
    
    SimulationData baseline_data, dysfunction_data;
    std::vector<StabilityMetrics> stability_timeline;
    std::cout << "\nRunning baseline and dysfunction phases..." << std::endl;
    
    int timestep = 0;
    int phase_spikes = 0;
    bool dysfunction_phase = false;
    
    while (timestep < max_timesteps) {
        // check if entering dysfunction phase
        if (timestep == condition.onset_timestep && !dysfunction_phase) {
            std::cout << "\n!!! METABOLIC DYSFUNCTION ONSET !!!" << std::endl;
            std::cout << "Condition: " << condition.name << " beginning at timestep " << timestep << std::endl;
            dysfunction_phase = true;
        }
        
        // apply metabolic dysfunction to all neurons
        if (dysfunction_phase) {
            for (int i = 0; i < NEURON_COUNT; ++i) {
                apply_metabolic_dysfunction(neurons[i], condition, timestep);
            }
            
            // simulate metabolic effects on network dynamics
            float severity = (condition.progressive) ? 
                1.0f + (timestep - condition.onset_timestep) * 0.002f : 1.0f;
            
            // hypoglycemia effects: reduced excitability, then depolarization block
            if (condition.glucose_level < 50.0f) {
                if (severity < 2.0f) {
                    // early: neurons become less excitable
                    if (rand() % 20 == 0) { // reduced spontaneous activity
                        int affected = rand() % NEURON_COUNT;
                        std::cout << "Hypoglycemia: Neuron " << affected << " showing reduced excitability" << std::endl;
                    }
                } else {
                    // late: depolarization block - force some neurons to spike continuously
                    if (rand() % 10 == 0) {
                        int blocked = rand() % NEURON_COUNT;
                        neurons[blocked]->spike();
                        std::cout << "Severe hypoglycemia: Neuron " << blocked << " in depolarization block!" << std::endl;
                    }
                }
            }
            
            // hyperglycemia effects: osmotic stress, inflammation
            if (condition.glucose_level > 250.0f) {
                if (rand() % 15 == 0) {
                    std::cout << "Hyperglycemia causing osmotic stress and inflammation" << std::endl;
                    // random disruption to multiple neurons
                    for (int burst = 0; burst < 3; ++burst) {
                        int affected = rand() % NEURON_COUNT;
                        neurons[affected]->spike();
                    }
                }
            }
            
            // hypoxia effects: rapid atp depletion, cell death cascade
            if (condition.atp_efficiency < 0.2f) {
                if (rand() % 5 == 0) {
                    std::cout << "Severe hypoxia: ATP depletion cascade!" << std::endl;
                    // simulate spreading depolarization
                    for (int cascade = 0; cascade < 5; ++cascade) {
                        int affected = rand() % NEURON_COUNT;
                        neurons[affected]->spike();
                    }
                }
            }
        }
        
        collect_membrane_data(neurons, NEURON_COUNT);
        
        // normal stimulation (reduced if severe dysfunction)
        float stimulation_probability = dysfunction_phase ? 
            std::max(0.1f, 0.5f * condition.atp_efficiency) : 0.5f;
        
        if (static_cast<float>(rand()) / RAND_MAX < stimulation_probability) {
            int stimulated = rand() % NEURON_COUNT;
            neurons[stimulated]->spike();
        }
        
        // update neurons and count spikes
        int timestep_spikes = 0;
        for (int i = 0; i < NEURON_COUNT; ++i) {
            if (neurons[i]->update_and_check_spike()) {
                timestep_spikes++;
                phase_spikes++;
                record_spike_event(timestep, i);
            }
        }
        
        // calculate stability metrics every 100 timesteps
        if (timestep % 100 == 0 && timestep > 0) {
            StabilityMetrics current_stability = calculate_stability_metrics(sim_data);
            stability_timeline.push_back(current_stability);
            
            std::cout << "Timestep " << timestep << " - Spikes: " << timestep_spikes 
                      << ", CV: " << current_stability.coefficient_of_variation
                      << ", Sync: " << current_stability.synchrony_index << std::endl;
        }
        
        if (timestep % 500 == 0 && timestep > 0) {
            std::cout << "=== Phase Report at Timestep " << timestep << " ===" << std::endl;
            std::cout << "Phase: " << (dysfunction_phase ? "DYSFUNCTION" : "BASELINE") << std::endl;
            std::cout << "Spikes this phase: " << phase_spikes << std::endl;
            
            if (dysfunction_phase) {
                float dysfunction_duration = timestep - condition.onset_timestep;
                std::cout << "Dysfunction duration: " << dysfunction_duration << " timesteps" << std::endl;
                
                if (phase_spikes == 0 && dysfunction_duration > 200) {
                    std::cout << "CRITICAL: Complete network silence - possible cell death" << std::endl;
                } else if (phase_spikes > dysfunction_duration * 0.8f) {
                    std::cout << "CRITICAL: Hyperexcitability - possible seizure activity" << std::endl;
                }
            }
        }
        
        timestep++;
    }
    
    sim_data.total_timesteps = timestep;
    sim_data.total_spikes = sim_data.spike_events.size();
    
    std::cout << "\n=== METABOLIC DISRUPTION ANALYSIS ===" << std::endl;
    
    // compare baseline vs dysfunction phases
    int baseline_spikes = 0, dysfunction_spikes = 0;
    for (const auto& spike : sim_data.spike_events) {
        if (spike.first < condition.onset_timestep) {
            baseline_spikes++;
        } else {
            dysfunction_spikes++;
        }
    }
    
    int baseline_duration = condition.onset_timestep;
    int dysfunction_duration = timestep - condition.onset_timestep;
    
    float baseline_rate = static_cast<float>(baseline_spikes) / baseline_duration;
    float dysfunction_rate = (dysfunction_duration > 0) ? 
        static_cast<float>(dysfunction_spikes) / dysfunction_duration : 0.0f;
    
    std::cout << "Baseline Activity Rate: " << baseline_rate << " spikes/timestep" << std::endl;
    std::cout << "Dysfunction Activity Rate: " << dysfunction_rate << " spikes/timestep" << std::endl;
    std::cout << "Activity Change: " << ((dysfunction_rate - baseline_rate) / baseline_rate * 100) << "%" << std::endl;
    
    StabilityMetrics final_metrics = calculate_stability_metrics(sim_data);
    print_stability_report(final_metrics);
    
    // clinical interpretation
    std::cout << "\n=== CLINICAL INTERPRETATION ===" << std::endl;
    if (dysfunction_rate < baseline_rate * 0.3f) {
        std::cout << "🔴 SEVERE DYSFUNCTION: Network activity critically reduced" << std::endl;
        std::cout << "   Clinical correlate: Altered consciousness, cognitive impairment" << std::endl;
    } else if (dysfunction_rate > baseline_rate * 2.0f) {
        std::cout << "🔴 HYPEREXCITABILITY: Network showing seizure-like activity" << std::endl;
        std::cout << "   Clinical correlate: Seizures, muscle spasms, hyperreflexia" << std::endl;
    } else {
        std::cout << "🟡 MILD DYSFUNCTION: Network partially compensating" << std::endl;
        std::cout << "   Clinical correlate: Subtle cognitive changes, fatigue" << std::endl;
    }
    
    if (final_metrics.homeostatic_deviation > 20.0f) {
        std::cout << "🔴 HOMEOSTATIC FAILURE: Network cannot maintain stable state" << std::endl;
    }
    
    if (final_metrics.synchrony_index > 0.9f) {
        std::cout << "🔴 PATHOLOGICAL SYNCHRONY: High seizure risk" << std::endl;
    }
    
    // cleanup
    for (int i = 0; i < NEURON_COUNT; ++i) {
        delete neurons[i];
    }
}

// apply background noise/spontaneous activity
void apply_background_activity(Neuron** neurons, int neuron_count, float noise_probability = 0.3f) {
    for (int i = 0; i < neuron_count; ++i) {
        if (static_cast<float>(rand()) / RAND_MAX < noise_probability) {
            // use a probabilistic spike trigger with increased chance or add method in neuron class to add current injection
            if (static_cast<float>(rand()) / RAND_MAX < 0.25f) {
                neurons[i]->spike();
            }
        }
    }
}


// generate csv file for membrane potentials
void generate_membrane_potential_csv(const std::string& filename, int neuron_count) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing." << std::endl;
        return;
    }
    
    // header
    file << "Timestep";
    for (int i = 0; i < neuron_count; ++i) {
        file << ",Neuron_" << i;
    }
    file << ",Network_Average" << std::endl;
    
    // rows
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

// generate csv file for spike events (raster plot data)
void generate_spike_raster_csv(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing." << std::endl;
        return;
    }
    
    // header
    file << "Timestep,Neuron_ID,Spike" << std::endl;
    
    // spike events
    for (const auto& spike : sim_data.spike_events) {
        file << spike.first << "," << spike.second << ",1" << std::endl;
    }
    
    file.close();
    std::cout << "Spike raster data saved to: " << filename << std::endl;
}

// generate activity summary csv
void generate_activity_summary_csv(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing." << std::endl;
        return;
    }
    
    // header
    file << "Timestep,Total_Spikes,Network_Activity" << std::endl;
    
    // spikes per timestep
    std::vector<int> spikes_per_step(sim_data.total_timesteps, 0);
    for (const auto& spike : sim_data.spike_events) {
        if (spike.first < sim_data.total_timesteps) {
            spikes_per_step[spike.first]++;
        }
    }
    
    // data
    for (int t = 0; t < sim_data.total_timesteps && t < static_cast<int>(sim_data.network_activity.size()); ++t) {
        file << t << "," << spikes_per_step[t] << "," 
             << std::fixed << std::setprecision(2) << sim_data.network_activity[t] << std::endl;
    }
    
    file.close();
    std::cout << "Activity summary data saved to: " << filename << std::endl;
}

// generate python plotting script
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

// generate comparison visualization
void generate_comparison_visualization_script_from_csv(const std::string& csv_filename = "metabolic_data.csv") {
    std::ofstream file("plot_comparison.py");
    if (!file.is_open()) {
        std::cerr << "Error: Could not create comparison plotting script." << std::endl;
        return;
    }
    
    file << "#!/usr/bin/env python3\n";
    file << "import matplotlib.pyplot as plt\n";
    file << "import numpy as np\n";
    file << "import pandas as pd\n";
    file << "import sys\n\n";
    
    file << "# Load data from CSV file\n";
    file << "try:\n";
    file << "    df = pd.read_csv('" << csv_filename << "')\n";
    file << "    print(f'Loaded {len(df)} records from " << csv_filename << "')\n";
    file << "    print('Columns:', df.columns.tolist())\n";
    file << "except FileNotFoundError:\n";
    file << "    print('Error: CSV file not found. Please ensure " << csv_filename << " exists.')\n";
    file << "    sys.exit(1)\n";
    file << "except Exception as e:\n";
    file << "    print(f'Error loading CSV: {e}')\n";
    file << "    sys.exit(1)\n\n";
    
    file << "# Extract data (adjust column names based on your CSV structure)\n";
    file << "# Assuming columns: condition, activity_change, cv, synchrony, entropy, homeostatic_dev\n";
    file << "if 'condition' in df.columns:\n";
    file << "    conditions = df['condition'].tolist()\n";
    file << "else:\n";
    file << "    conditions = [f'Condition_{i+1}' for i in range(len(df))]\n\n";
    
    file << "# Try to extract metrics with fallback values\n";
    file << "activity_changes = df.get('activity_change', df.get('activity_pct_change', [0]*len(df))).tolist()\n";
    file << "cv_values = df.get('coefficient_of_variation', df.get('cv', [1.0]*len(df))).tolist()\n";
    file << "synchrony_values = df.get('synchrony_index', df.get('synchrony', [0.5]*len(df))).tolist()\n";
    file << "entropy_values = df.get('entropy', df.get('shannon_entropy', [2.0]*len(df))).tolist()\n";
    file << "homeostatic_values = df.get('homeostatic_deviation', df.get('homeostatic_dev', [10.0]*len(df))).tolist()\n\n";
    
    file << "print(f'Processing {len(conditions)} conditions')\n";
    file << "print('Sample data:')\n";
    file << "for i in range(min(3, len(conditions))):\n";
    file << "    print(f'  {conditions[i]}: Activity={activity_changes[i]:.1f}%, CV={cv_values[i]:.3f}')\n\n";
    
    file << "# Create comprehensive comparison plots\n";
    file << "fig, axes = plt.subplots(2, 3, figsize=(18, 12))\n";
    file << "fig.suptitle('Metabolic Dysfunction Neural Network Comparison', fontsize=16, fontweight='bold')\n\n";
    
    file << "# Activity changes\n";
    file << "ax1 = axes[0, 0]\n";
    file << "colors1 = ['red' if x < -50 else 'orange' if x < -20 else 'yellow' if x < 20 else 'green' for x in activity_changes]\n";
    file << "bars1 = ax1.bar(range(len(conditions)), activity_changes, color=colors1)\n";
    file << "ax1.set_title('Activity Change (%)')\n";
    file << "ax1.set_ylabel('Change from Baseline (%)')\n";
    file << "ax1.set_xticks(range(len(conditions)))\n";
    file << "ax1.set_xticklabels([c[:15] + '...' if len(c) > 15 else c for c in conditions], rotation=45, ha='right')\n";
    file << "ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)\n";
    file << "ax1.grid(True, alpha=0.3)\n\n";
    
    file << "# Add value labels on bars\n";
    file << "for bar, val in zip(bars1, activity_changes):\n";
    file << "    height = bar.get_height()\n";
    file << "    ax1.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),\n";
    file << "                xytext=(0, 3 if height >= 0 else -15), textcoords='offset points',\n";
    file << "                ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)\n\n";
    
    file << "# Coefficient of variation\n";
    file << "ax2 = axes[0, 1]\n";
    file << "bars2 = ax2.bar(range(len(conditions)), cv_values, color='skyblue')\n";
    file << "ax2.set_title('Spike Timing Irregularity (CV)')\n";
    file << "ax2.set_ylabel('Coefficient of Variation')\n";
    file << "ax2.set_xticks(range(len(conditions)))\n";
    file << "ax2.set_xticklabels([c[:15] + '...' if len(c) > 15 else c for c in conditions], rotation=45, ha='right')\n";
    file << "ax2.grid(True, alpha=0.3)\n\n";
    
    file << "# Synchrony index\n";
    file << "ax3 = axes[0, 2]\n";
    file << "bars3 = ax3.bar(range(len(conditions)), synchrony_values, color='lightcoral')\n";
    file << "ax3.set_title('Network Synchrony')\n";
    file << "ax3.set_ylabel('Synchrony Index')\n";
    file << "ax3.set_xticks(range(len(conditions)))\n";
    file << "ax3.set_xticklabels([c[:15] + '...' if len(c) > 15 else c for c in conditions], rotation=45, ha='right')\n";
    file << "ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Seizure Risk')\n";
    file << "ax3.legend()\n";
    file << "ax3.grid(True, alpha=0.3)\n\n";
    
    file << "# Entropy\n";
    file << "ax4 = axes[1, 0]\n";
    file << "bars4 = ax4.bar(range(len(conditions)), entropy_values, color='mediumseagreen')\n";
    file << "ax4.set_title('Network Complexity (Entropy)')\n";
    file << "ax4.set_ylabel('Shannon Entropy')\n";
    file << "ax4.set_xticks(range(len(conditions)))\n";
    file << "ax4.set_xticklabels([c[:15] + '...' if len(c) > 15 else c for c in conditions], rotation=45, ha='right')\n";
    file << "ax4.grid(True, alpha=0.3)\n\n";
    
    file << "# Homeostatic deviation\n";
    file << "ax5 = axes[1, 1]\n";
    file << "bars5 = ax5.bar(range(len(conditions)), homeostatic_values, color='gold')\n";
    file << "ax5.set_title('Homeostatic Deviation')\n";
    file << "ax5.set_ylabel('Deviation from Target (mV)')\n";
    file << "ax5.set_xticks(range(len(conditions)))\n";
    file << "ax5.set_xticklabels([c[:15] + '...' if len(c) > 15 else c for c in conditions], rotation=45, ha='right')\n";
    file << "ax5.axhline(y=15, color='red', linestyle='--', alpha=0.7, label='Critical Level')\n";
    file << "ax5.legend()\n";
    file << "ax5.grid(True, alpha=0.3)\n\n";
    
    file << "# Radar chart for multi-dimensional comparison\n";
    file << "ax6 = axes[1, 2]\n";
    file << "categories = ['Activity\\nChange', 'Irregularity\\n(CV)', 'Synchrony', 'Entropy', 'Homeostatic\\nDev']\n";
    file << "angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)\n";
    file << "angles = np.concatenate((angles, [angles[0]]))\n\n";
    
    file << "colors = ['red', 'orange', 'purple', 'green', 'blue', 'brown', 'pink', 'gray']\n";
    file << "max_conditions = min(5, len(conditions))  # Show top 5 conditions\n";
    file << "for i in range(max_conditions):\n";
    file << "    # Normalize values to 0-1 range for radar chart\n";
    file << "    max_activity = max(abs(x) for x in activity_changes) if activity_changes else 100\n";
    file << "    max_cv = max(cv_values) if cv_values else 3\n";
    file << "    max_entropy = max(entropy_values) if entropy_values else 4\n";
    file << "    max_homeostatic = max(homeostatic_values) if homeostatic_values else 30\n";
    file << "    \n";
    file << "    values = [\n";
    file << "        abs(activity_changes[i]) / max_activity,\n";
    file << "        cv_values[i] / max_cv,\n";
    file << "        synchrony_values[i],\n";
    file << "        entropy_values[i] / max_entropy,\n";
    file << "        homeostatic_values[i] / max_homeostatic\n";
    file << "    ]\n";
    file << "    values += [values[0]]  # Complete the circle\n";
    file << "    \n";
    file << "    condition_name = conditions[i][:20] if len(conditions[i]) > 20 else conditions[i]\n";
    file << "    ax6.plot(angles, values, 'o-', linewidth=2, label=condition_name, color=colors[i % len(colors)])\n";
    file << "    ax6.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])\n\n";
    
    file << "ax6.set_xticks(angles[:-1])\n";
    file << "ax6.set_xticklabels(categories)\n";
    file << "ax6.set_ylim(0, 1)\n";
    file << "ax6.set_title(f'Multi-dimensional Comparison\\n(Top {max_conditions} Conditions)')\n";
    file << "ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))\n";
    file << "ax6.grid(True)\n\n";
    
    file << "plt.tight_layout()\n";
    file << "plt.savefig('metabolic_dysfunction_comparison.png', dpi=300, bbox_inches='tight')\n";
    file << "plt.show()\n";
    file << "print('\\nComprehensive comparison saved as metabolic_dysfunction_comparison.png')\n";
    file << "print('Summary statistics:')\n";
    file << "print(f'  Mean activity change: {np.mean(activity_changes):.1f}%')\n";
    file << "print(f'  Mean CV: {np.mean(cv_values):.3f}')\n";
    file << "print(f'  Mean synchrony: {np.mean(synchrony_values):.3f}')\n";
    file << "print(f'  Mean entropy: {np.mean(entropy_values):.3f}')\n";
    file << "print(f'  Mean homeostatic deviation: {np.mean(homeostatic_values):.1f} mV')\n";
    
    file.close();
    std::cout << "Enhanced comparison visualization script saved to: plot_comparison.py" << std::endl;
    std::cout << "Usage: python3 plot_comparison.py" << std::endl;
    std::cout << "Make sure your CSV file '" << csv_filename << "' contains the required columns." << std::endl;
}


void generate_comparison_visualization_script(const std::vector<std::string>& conditions,
                                            const std::vector<StabilityMetrics>& metrics,
                                            const std::vector<float>& activity_changes) {
    std::ofstream file("plot_comparison.py");
    if (!file.is_open()) {
        std::cerr << "Error: Could not create comparison plotting script." << std::endl;
        return;
    }
    
    if (conditions.empty() || metrics.empty() || activity_changes.empty()) {
        std::cerr << "Warning: Empty data provided. Consider using generate_comparison_visualization_script_from_csv() instead." << std::endl;
        file.close();
        return;
    }
    
    file << "#!/usr/bin/env python3\n";
    file << "import matplotlib.pyplot as plt\n";
    file << "import numpy as np\n";
    file << "import pandas as pd\n\n";
    
    file << "# Condition data\n";
    file << "conditions = [";
    for (size_t i = 0; i < conditions.size(); ++i) {
        file << "'" << conditions[i] << "'";
        if (i < conditions.size() - 1) file << ", ";
    }
    file << "]\n\n";
    
    file << "activity_changes = [";
    for (size_t i = 0; i < activity_changes.size(); ++i) {
        file << activity_changes[i];
        if (i < activity_changes.size() - 1) file << ", ";
    }
    file << "]\n\n";
    
    file << "cv_values = [";
    for (size_t i = 0; i < metrics.size(); ++i) {
        file << metrics[i].coefficient_of_variation;
        if (i < metrics.size() - 1) file << ", ";
    }
    file << "]\n\n";
    
    file << "synchrony_values = [";
    for (size_t i = 0; i < metrics.size(); ++i) {
        file << metrics[i].synchrony_index;
        if (i < metrics.size() - 1) file << ", ";
    }
    file << "]\n\n";
    
    file << "entropy_values = [";
    for (size_t i = 0; i < metrics.size(); ++i) {
        file << metrics[i].entropy;
        if (i < metrics.size() - 1) file << ", ";
    }
    file << "]\n\n";
    
    file << "homeostatic_values = [";
    for (size_t i = 0; i < metrics.size(); ++i) {
        file << metrics[i].homeostatic_deviation;
        if (i < metrics.size() - 1) file << ", ";
    }
    file << "]\n\n";
    file << "print(f'Processing {len(conditions)} conditions with embedded data')\n\n";
    file << "# Create comprehensive comparison plots\n";
    file << "fig, axes = plt.subplots(2, 3, figsize=(18, 12))\n";
    file << "fig.suptitle('Metabolic Dysfunction Neural Network Comparison', fontsize=16, fontweight='bold')\n\n";
    file << "# Activity changes\n";
    file << "ax1 = axes[0, 0]\n";
    file << "colors1 = ['red' if x < -50 else 'orange' if x < -20 else 'yellow' if x < 20 else 'green' for x in activity_changes]\n";
    file << "bars1 = ax1.bar(range(len(conditions)), activity_changes, color=colors1)\n";
    file << "ax1.set_title('Activity Change (%)')\n";
    file << "ax1.set_ylabel('Change from Baseline (%)')\n";
    file << "ax1.set_xticks(range(len(conditions)))\n";
    file << "ax1.set_xticklabels([c[:15] + '...' if len(c) > 15 else c for c in conditions], rotation=45, ha='right')\n";
    file << "ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)\n";
    file << "ax1.grid(True, alpha=0.3)\n\n";
    file << "plt.tight_layout()\n";
    file << "plt.savefig('metabolic_dysfunction_comparison.png', dpi=300, bbox_inches='tight')\n";
    file << "plt.show()\n";
    file << "print('Comprehensive comparison saved as metabolic_dysfunction_comparison.png')\n";
    
    file.close();
    std::cout << "Comparison visualization script saved to: plot_comparison.py" << std::endl;
}

// load csv data and populate vectors
struct CSVData {
    std::vector<std::string> conditions;
    std::vector<StabilityMetrics> metrics;
    std::vector<float> activity_changes;
};

CSVData load_csv_data(const std::string& filename) {
    CSVData data;
    std::ifstream file(filename);
    std::string line;
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open CSV file: " << filename << std::endl;
        return data;
    }
    
    // skip header line
    if (std::getline(file, line)) {
        std::cout << "CSV Header: " << line << std::endl;
    }
    
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        
        // parse data line
        while (std::getline(ss, token, ',')) {
            // remove whitespace
            token.erase(0, token.find_first_not_of(" \t"));
            token.erase(token.find_last_not_of(" \t") + 1);
            tokens.push_back(token);
        }
        
        if (tokens.size() >= 6) {
            try {
                StabilityMetrics metrics_entry;
                
                data.conditions.push_back(tokens[0]);
                data.activity_changes.push_back(std::stof(tokens[1]));
                metrics_entry.coefficient_of_variation = std::stof(tokens[2]);
                metrics_entry.synchrony_index = std::stof(tokens[3]);
                metrics_entry.entropy = std::stof(tokens[4]);
                metrics_entry.homeostatic_deviation = std::stof(tokens[5]);
                
                data.metrics.push_back(metrics_entry);
            } catch (const std::exception& e) {
                std::cerr << "Error parsing line: " << line << " - " << e.what() << std::endl;
            }
        }
    }
    
    std::cout << "Loaded " << data.conditions.size() << " records from CSV" << std::endl;
    return data;
}

// create csv file
void generate_comparison_with_data_export(const std::vector<std::string>& conditions,
                                         const std::vector<StabilityMetrics>& metrics,
                                         const std::vector<float>& activity_changes) {
    // export data to csv
    std::string csv_filename = "metabolic_analysis_results.csv";
    std::ofstream csv_file(csv_filename);
    
    if (!csv_file.is_open()) {
        std::cerr << "Error: Could not create CSV file" << std::endl;
        return;
    }
    
    // header
    csv_file << "condition,activity_change,coefficient_of_variation,synchrony_index,entropy,homeostatic_deviation\n";
    
    // data
    for (size_t i = 0; i < conditions.size() && i < metrics.size() && i < activity_changes.size(); ++i) {
        csv_file << conditions[i] << ","
                 << activity_changes[i] << ","
                 << metrics[i].coefficient_of_variation << ","
                 << metrics[i].synchrony_index << ","
                 << metrics[i].entropy << ","
                 << metrics[i].homeostatic_deviation << "\n";
    }
    
    csv_file.close();
    std::cout << "Data exported to: " << csv_filename << std::endl;
    
    generate_comparison_visualization_script_from_csv(csv_filename);
}

// Mrun various metabolic dysfunction simulations
void run_metabolic_dysfunction_studies() {
    std::cout << "\n=== METABOLIC DYSFUNCTION NEURAL NETWORK STUDY ===" << std::endl;
    std::cout << "Simulating various metabolic conditions and their neural effects..." << std::endl;
    
    // different conditions
    std::vector<MetabolicCondition> conditions = {
        create_hypoglycemia(),
        create_diabetes_ketoacidosis(),
        create_hypoxia(),
        create_mitochondrial_dysfunction()
    };
    
    //  metabolic conditions for comprehensive study
    conditions.push_back({
        "Severe Hyperglycemia",
        400.0f,   // extremely high glucose
        0.7f,     // moderately reduced atp
        0.5f,     // impaired ion pumps due to osmotic stress
        0.6f,     // reduced neurotransmitter synthesis
        0.4f,     // severe membrane damage
        3.8f,     // very high oxidative stress
        true,     // progressive
        600       // onset
    });
    
    conditions.push_back({
        "Metabolic Acidosis",
        90.0f,    // normal glucose
        0.5f,     // reduced ATP due to pH effects
        0.3f,     // severely impaired ion pumps
        0.5f,     // impaired synthesis
        0.6f,     // membrane instability
        3.2f,     // high oxidative stress
        true,     // progressive
        400       // onset
    });
    
    conditions.push_back({
        "Thiamine Deficiency (Beriberi)",
        85.0f,    // normal glucose but can't utilize it
        0.2f,     // severely reduced ATP (pyruvate metabolism blocked)
        0.4f,     // ion pump failure
        0.3f,     // severe neurotransmitter disruption
        0.7f,     // membrane damage
        2.8f,     // oxidative stress
        true,     // progressive
        300       // early onset
    });
    
    conditions.push_back({
        "Acute Stroke (Ischemia)",
        75.0f,    // normal glucose
        0.05f,    // critical ATP depletion
        0.1f,     // complete ion pump failure
        0.2f,     // severe synthesis impairment
        0.3f,     // rapid membrane breakdown
        4.5f,     // extreme oxidative stress
        true,     // rapidly progressive
        100       // very early onset
    });
    
    std::vector<std::string> condition_names;
    std::vector<StabilityMetrics> final_metrics;
    std::vector<float> activity_changes;
    std::vector<int> survival_times; // timesteps until network failure
    
    std::cout << "\nRunning " << conditions.size() << " different metabolic conditions...\n" << std::endl;
    
    // run each condition
    for (size_t i = 0; i < conditions.size(); ++i) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "STUDY " << (i + 1) << "/" << conditions.size() << ": " << conditions[i].name << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        // clear previous simulation data
        sim_data.membrane_potentials.clear();
        sim_data.spike_events.clear();
        sim_data.spikes_per_timestep.clear();
        sim_data.network_activity.clear();
        sim_data.total_timesteps = 0;
        sim_data.total_spikes = 0;
        
        run_metabolic_dysfunction_simulation(conditions[i], 2000);
        
        // store results for comparison
        condition_names.push_back(conditions[i].name);
        StabilityMetrics metrics = calculate_stability_metrics(sim_data);
        final_metrics.push_back(metrics);
        
        // calculate activity change
        int baseline_spikes = 0, dysfunction_spikes = 0;
        for (const auto& spike : sim_data.spike_events) {
            if (spike.first < conditions[i].onset_timestep) {
                baseline_spikes++;
            } else {
                dysfunction_spikes++;
            }
        }
        
        int baseline_duration = conditions[i].onset_timestep;
        int dysfunction_duration = sim_data.total_timesteps - conditions[i].onset_timestep;
        
        float baseline_rate = static_cast<float>(baseline_spikes) / baseline_duration;
        float dysfunction_rate = (dysfunction_duration > 0) ? 
            static_cast<float>(dysfunction_spikes) / dysfunction_duration : 0.0f;
        
        float activity_change = (baseline_rate > 0) ? 
            ((dysfunction_rate - baseline_rate) / baseline_rate * 100) : 0.0f;
        activity_changes.push_back(activity_change);
        
        // estimate survival time (when activity drops below 10% of baseline)
        int survival_time = sim_data.total_timesteps;
        if (dysfunction_rate < baseline_rate * 0.1f) {
            survival_time = conditions[i].onset_timestep + 
                           static_cast<int>((conditions[i].onset_timestep * 0.1f) / 
                           std::max(0.001f, conditions[i].atp_efficiency));
        }
        survival_times.push_back(survival_time);
        
        // generate condition-specific data files
        std::string safe_name = conditions[i].name;
        std::replace(safe_name.begin(), safe_name.end(), ' ', '_');
        std::replace(safe_name.begin(), safe_name.end(), '(', '_');
        std::replace(safe_name.begin(), safe_name.end(), ')', '_');
        
        generate_membrane_potential_csv("membrane_" + safe_name + ".csv", 10);
        generate_spike_raster_csv("spikes_" + safe_name + ".csv");
        generate_activity_summary_csv("activity_" + safe_name + ".csv");
        
        std::cout << "\nCondition completed. Data saved with prefix: " << safe_name << std::endl;
        std::cout << "Press Enter to continue to next condition..." << std::endl;
        std::cin.get();
    }
    
    // comparative analysis
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "COMPARATIVE METABOLIC DYSFUNCTION ANALYSIS" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    std::cout << std::left << std::setw(25) << "Condition" 
              << std::setw(12) << "Activity%" 
              << std::setw(10) << "CV" 
              << std::setw(10) << "Sync" 
              << std::setw(12) << "Entropy" 
              << std::setw(15) << "Homeostatic"
              << std::setw(12) << "Survival" << std::endl;
    std::cout << std::string(100, '-') << std::endl;
    
    for (size_t i = 0; i < conditions.size(); ++i) {
        std::cout << std::left << std::setw(25) << condition_names[i].substr(0, 24)
                  << std::setw(12) << std::fixed << std::setprecision(1) << activity_changes[i]
                  << std::setw(10) << std::setprecision(2) << final_metrics[i].coefficient_of_variation
                  << std::setw(10) << std::setprecision(2) << final_metrics[i].synchrony_index
                  << std::setw(12) << std::setprecision(2) << final_metrics[i].entropy
                  << std::setw(15) << std::setprecision(1) << final_metrics[i].homeostatic_deviation
                  << std::setw(12) << survival_times[i] << std::endl;
    }
    
    // clinical severity ranking
    std::cout << "\n=== CLINICAL SEVERITY RANKING ===" << std::endl;
    std::vector<std::pair<float, std::string>> severity_scores;
    
    for (size_t i = 0; i < conditions.size(); ++i) {
        // calculate composite severity score
        float severity = 0.0f;
        severity += std::abs(activity_changes[i]) * 0.3f;                   // activity change weight
        severity += final_metrics[i].homeostatic_deviation * 0.2f;          // homeostatic weight
        severity += (1.0f - conditions[i].atp_efficiency) * 100.0f * 0.3f;  // ATP efficiency weight
        severity += final_metrics[i].synchrony_index * 50.0f;               // synchrony weight
        severity += (2000 - survival_times[i]) * 0.2f;                      // survival time weight
        
        severity_scores.push_back(std::make_pair(severity, condition_names[i]));
    }
    
    std::sort(severity_scores.begin(), severity_scores.end(), 
              [](const std::pair<float, std::string>& a, const std::pair<float, std::string>& b) {
                  return a.first > b.first;
              });
    
    for (size_t i = 0; i < severity_scores.size(); ++i) {
        std::string severity_level;
        if (severity_scores[i].first > 150.0f) severity_level = "CRITICAL";
        else if (severity_scores[i].first > 100.0f) severity_level = "SEVERE";
        else if (severity_scores[i].first > 50.0f) severity_level = "MODERATE";
        else severity_level = "MILD";
        
        std::cout << (i + 1) << ". " << severity_level << " - " 
                  << severity_scores[i].second 
                  << " (Score: " << std::fixed << std::setprecision(1) << severity_scores[i].first << ")" << std::endl;
    }
    
    generate_comparison_with_data_export(condition_names, final_metrics, activity_changes);
    
    std::cout << "\n=== STUDY COMPLETE ===" << std::endl;
    std::cout << "Individual condition data files generated for each condition." << std::endl;
    std::cout << "Run 'python3 plot_comparison.py' to generate comparative visualizations." << std::endl;
    std::cout << "\nKey Findings:" << std::endl;
    std::cout << "- Most severe condition: " << severity_scores[0].second << std::endl;
    std::cout << "- Least severe condition: " << severity_scores.back().second << std::endl;
    std::cout << "- Average activity change: " << 
                 std::accumulate(activity_changes.begin(), activity_changes.end(), 0.0f) / activity_changes.size() 
              << "%" << std::endl;
}


int main() {
    std::srand(43);
    
    std::cout << "=== Enhanced Human Neuron Network Simulation ===" << std::endl;
    std::cout << "Choose simulation mode:" << std::endl;
    std::cout << "1. Standard Network Simulation" << std::endl;
    std::cout << "2. Metabolic Dysfunction Study" << std::endl;
    std::cout << "3. Single Metabolic Condition Test" << std::endl;
    std::cout << "Enter choice (1-3): ";
    
    int choice;
    std::cin >> choice;
    std::cin.ignore();
    
    switch (choice) {
        case 1: {
            std::cout << "Running standard neural network simulation..." << std::endl;
            
            const int NEURON_COUNT = 10;
            Neuron* neurons[NEURON_COUNT];
            
            sim_data.total_timesteps = 0;
            sim_data.total_spikes = 0;
            
            // create diverse neuron types
            neurons[0] = new PyramidalNeuron();
            neurons[1] = new PyramidalNeuron();
            neurons[2] = new PyramidalNeuron();
            neurons[3] = new PyramidalNeuron();
            neurons[4] = new Interneuron();
            neurons[5] = new PurkinjeNeuron();
            neurons[6] = new MotorNeuron();
            neurons[7] = new MotorNeuron();
            neurons[8] = new SensoryNeuron();
            neurons[9] = new SensoryNeuron();
            
            create_random_connections(neurons, NEURON_COUNT, 6);
            
            // run standard simulation
            int timestep = 0;
            int total_spikes = 0;
            const int MAX_TIMESTEPS = 5000;
            
            while (timestep < MAX_TIMESTEPS) {
                collect_membrane_data(neurons, NEURON_COUNT);
                
                if (timestep % 2 == 0) {
                    int stimulated_neuron = rand() % NEURON_COUNT;
                    neurons[stimulated_neuron]->spike();
                }
                
                apply_background_activity(neurons, NEURON_COUNT, 0.6f);
                
                int spike_count = 0;
                for (int i = 0; i < NEURON_COUNT; ++i) {
                    if (neurons[i]->update_and_check_spike()) {
                        spike_count++;
                        total_spikes++;
                        record_spike_event(timestep, i);
                    }
                }
                
                timestep++;
            }
            
            sim_data.total_timesteps = timestep;
            sim_data.total_spikes = total_spikes;
            
            generate_membrane_potential_csv("membrane_potentials.csv", NEURON_COUNT);
            generate_spike_raster_csv("spike_raster.csv");
            generate_activity_summary_csv("activity_summary.csv");
            generate_python_plotting_script("plot_simulation.py", NEURON_COUNT);
            StabilityMetrics metrics = calculate_stability_metrics(sim_data);
            print_stability_report(metrics);
            
            // cleanup
            for (int i = 0; i < NEURON_COUNT; ++i) {
                delete neurons[i];
            }
            break;
        }
        
        case 2: {
            // run metabolic dysfunction study
            run_metabolic_dysfunction_studies();
            break;
        }
        
        case 3: {
            std::cout << "Available metabolic conditions:" << std::endl;
            std::cout << "1. Severe Hypoglycemia" << std::endl;
            std::cout << "2. Diabetic Ketoacidosis" << std::endl;
            std::cout << "3. Cerebral Hypoxia" << std::endl;
            std::cout << "4. Mitochondrial Dysfunction" << std::endl;
            std::cout << "Enter condition (1-4): ";
            
            int condition_choice;
            std::cin >> condition_choice;
            
            MetabolicCondition condition;
            switch (condition_choice) {
                case 1: condition = create_hypoglycemia(); break;
                case 2: condition = create_diabetes_ketoacidosis(); break;
                case 3: condition = create_hypoxia(); break;
                case 4: condition = create_mitochondrial_dysfunction(); break;
                default: 
                    std::cout << "Invalid choice, using hypoglycemia." << std::endl;
                    condition = create_hypoglycemia();
            }
            
            run_metabolic_dysfunction_simulation(condition, 3000);
            break;
        }
        
        default:
            std::cout << "Invalid choice. Running standard simulation." << std::endl;
            choice = 1;
    }
    
    std::cout << "\nSimulation complete!" << std::endl;
    return 0;
}