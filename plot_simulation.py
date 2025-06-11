#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Load data
membrane_data = pd.read_csv('membrane_potentials.csv')
spike_data = pd.read_csv('spike_raster.csv')
activity_data = pd.read_csv('activity_summary.csv')

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Enhanced Human Neuron Network Simulation Results', fontsize=16, fontweight='bold')

# Plot 1: Membrane Potentials Over Time
ax1 = axes[0, 0]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43']
for i in range(min(10, len(colors))):
    ax1.plot(membrane_data['Timestep'], membrane_data[f'Neuron_{i}'], 
             color=colors[i], linewidth=1.5, alpha=0.8, label=f'Neuron {i}')
ax1.axhline(y=-55, color='red', linestyle='--', alpha=0.7, label='Spike Threshold')
ax1.set_xlabel('Timestep')
ax1.set_ylabel('Membrane Potential (mV)')
ax1.set_title('Membrane Potential Traces')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)

# Plot 2: Spike Raster Plot
ax2 = axes[0, 1]
if not spike_data.empty:
    for neuron_id in range(10):
        neuron_spikes = spike_data[spike_data['Neuron_ID'] == neuron_id]
        ax2.scatter(neuron_spikes['Timestep'], neuron_spikes['Neuron_ID'], 
                   color=colors[neuron_id % len(colors)], s=20, alpha=0.8)
ax2.set_xlabel('Timestep')
ax2.set_ylabel('Neuron ID')
ax2.set_title('Spike Raster Plot')
ax2.set_yticks(range(10))
ax2.grid(True, alpha=0.3)

# Plot 3: Network Activity Over Time
ax3 = axes[1, 0]
ax3.plot(activity_data['Timestep'], activity_data['Network_Activity'], 
         color='purple', linewidth=2, label='Average Membrane Potential')
ax3.set_xlabel('Timestep')
ax3.set_ylabel('Average Membrane Potential (mV)')
ax3.set_title('Network Activity')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Spike Count Histogram
ax4 = axes[1, 1]
ax4.bar(activity_data['Timestep'], activity_data['Total_Spikes'], 
        color='orange', alpha=0.7, width=0.8)
ax4.set_xlabel('Timestep')
ax4.set_ylabel('Number of Spikes')
ax4.set_title('Spike Count per Timestep')
ax4.grid(True, alpha=0.3)

# Add simulation statistics as text
total_spikes = len(spike_data)
total_timesteps = len(activity_data)
spike_rate = total_spikes / total_timesteps if total_timesteps > 0 else 0
stats_text = f'Total Spikes: {total_spikes}\nTotal Timesteps: {total_timesteps}\nSpike Rate: {spike_rate:.2f} spikes/timestep'
fig.text(0.02, 0.02, stats_text, fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

plt.tight_layout()
plt.savefig('neural_network_simulation.png', dpi=300, bbox_inches='tight')
plt.show()
print('Visualization saved as neural_network_simulation.png')
