#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 10:06:15 2025

@author: Mónica Sagastuy-Breña
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
file_path = "HVDC_flows_test.csv"
df = pd.read_csv(file_path)

# Clean column names to identify directions
forward_cols = [col for col in df.columns if "-" in col and "_year_" in col and df[col].notna().any()]
reverse_cols = [col for col in forward_cols if col.split("_year_")[0].split("-")[0] != col.split("_year_")[0].split("-")[1]]
forward_cols = [col for col in forward_cols if col not in reverse_cols]

# Convert to NumPy arrays for plotting
forward_data = df[forward_cols].fillna(0).to_numpy()
reverse_data = df[reverse_cols].fillna(0).to_numpy()

# Time axis
time = np.arange(forward_data.shape[0])

# Compute total forward and reverse flows over time (summing across years)
total_forward = np.sum(forward_data, axis=1)
total_reverse = np.sum(reverse_data, axis=1)

# For mirrored plot, reverse is plotted as negative
mirrored_reverse = -total_reverse

# Plot A: Mirrored plot with positive y-ticks only
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(time, total_forward, label="Forward Flow", color="blue")
ax.plot(time, mirrored_reverse, label="Reverse Flow", color="orange")
ax.set_title("HVDC Flows (Mirrored Directional Plot)")
ax.set_xlabel("Hour")
ax.set_ylabel("Flow")
ax.legend()
ax.spines['bottom'].set_position('zero')
ax.set_yticks(np.linspace(-max(total_reverse.max(), total_forward.max()),
                          max(total_reverse.max(), total_forward.max()), 9))
ax.set_yticklabels([f"{abs(int(tick))}" for tick in ax.get_yticks()])
ax.grid(True)

# Plot B: Same axis, color-coded
fig, ax2 = plt.subplots(figsize=(15, 5))
ax2.plot(time, total_forward, label="Forward Flow", color="blue")
ax2.plot(time, total_reverse, label="Reverse Flow", color="orange")
ax2.set_title("HVDC Flows (Overlaid Directional Plot)")
ax2.set_xlabel("Hour")
ax2.set_ylabel("Flow")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()



def plot_hourly_flows_from_csv(csv_path, period=1):
    df = pd.read_csv(csv_path)
    
    # Extract years from column names
    forward_cols = [col for col in df.columns if col.startswith("0-2")]
    reverse_cols = [col for col in df.columns if col.startswith("2-0")]

    # Sort by year number
    get_year = lambda name: int(name.split("_year_")[1])
    forward_cols = sorted(forward_cols, key=get_year)
    reverse_cols = sorted(reverse_cols, key=get_year)

    # Concatenate flows
    forward_flows = np.concatenate([df[col].values for col in forward_cols])
    reverse_flows = np.concatenate([df[col].values for col in reverse_cols])

    total_hours = len(forward_flows)
    num_years = len(forward_cols)
    time_hours = np.arange(0, total_hours * period, period)

    # Plot 1: Forward positive, reverse negative
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(time_hours, forward_flows, label="0 → 2 Flow", color='blue')
    ax.plot(time_hours, -reverse_flows, label="2 → 0 Flow (shown as negative)", color='red')
    ax.axhline(0, color='black', linewidth=0.8)

    hours_per_year = len(forward_flows) // num_years
    year_ticks = [i * hours_per_year for i in range(num_years + 1)]
    year_labels = [f"Year {get_year(forward_cols[0]) + i}" for i in range(num_years + 1)]
    ax.set_xticks(year_ticks)
    ax.set_xticklabels(year_labels)

    ax.set_ylabel("Flow (MW) – Absolute")
    ax.set_xlabel("Time (hours)")
    ax.set_title("Hourly HVDC Flows (Positive = 0→2, Negative = 2→0)")
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 2: Both series positive, separate colors
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(time_hours, forward_flows, label="0 → 2 Flow", color='blue')
    ax.plot(time_hours, reverse_flows, label="2 → 0 Flow", color='red')

    ax.set_xticks(year_ticks)
    ax.set_xticklabels(year_labels)

    ax.set_ylabel("Flow (MW)")
    ax.set_xlabel("Time (hours)")
    ax.set_title("Hourly HVDC Flows by Direction")
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
plot_hourly_flows_from_csv("HVDC_flows_test.csv")
    

