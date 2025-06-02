#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 15:43:35 2025

@author: Mónica Sagastuy-Breña
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load your dataset
df = pd.read_csv("MX_capacity_params.csv")
df = df[df['year'] >= 2012]

# S-curve with floor cost
def bounded_s_curve(year, C_min, C_max, k, t0):
    return C_min + (C_max - C_min) / (1 + np.exp(-k * (year - t0)))

# WIND data
wind_valid = df['wind_installed_costs_BUSD/GW'] > 0
years_wind = df['year'][wind_valid].values
costs_wind = df['wind_installed_costs_BUSD/GW'][wind_valid].values

# PV data
pv_valid = df['pv_installed_costs_BUSD/GW'] > 0.01
years_pv = df['year'][pv_valid].values
costs_pv = df['pv_installed_costs_BUSD/GW'][pv_valid].values

# Reasonable floor costs (in BUSD/GW)
C_min_wind = 0.670   # Example floor from IRENA/NREL
C_min_pv = 0.430     # Example floor from IEA/NREL

# Initial guesses: [C_max, k, t0]
p0_wind = [max(costs_wind), -0.1, np.median(years_wind)]
p0_pv = [max(costs_pv), -0.1, np.median(years_pv)]

# Fit the models
params_wind, _ = curve_fit(lambda x, C_max, k, t0: bounded_s_curve(x, C_min_wind, C_max, k, t0),
                           years_wind, costs_wind, p0=p0_wind, maxfev=5000)

params_pv, _ = curve_fit(lambda x, C_max, k, t0: bounded_s_curve(x, C_min_pv, C_max, k, t0),
                         years_pv, costs_pv, p0=p0_pv, maxfev=5000)

# Future projection
future_years = np.arange(2012, 2056)
predicted_wind = bounded_s_curve(future_years, C_min_wind, *params_wind)
predicted_pv = bounded_s_curve(future_years, C_min_pv, *params_pv)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(years_wind, costs_wind, 'o', color='blue', label='Historical Wind')
plt.plot(future_years, predicted_wind, '-', color='blue', label='Projected Wind (bounded)')
plt.plot(years_pv, costs_pv, 'x', color='orange', label='Historical PV')
plt.plot(future_years, predicted_pv, '-', color='orange', label='Projected PV (bounded)')

plt.xlabel('Year')
plt.ylabel('Installed Cost (BUSD/GW)')
plt.title('Bounded S-Curve Projections for Wind and PV in Mexico (2012–2055)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --- Extract and save 2025–2050 projections ---
export_years = np.arange(2025, 2056)
export_pv = bounded_s_curve(export_years, C_min_pv, *params_pv)
export_wind = bounded_s_curve(export_years, C_min_wind, *params_wind)

# Build DataFrame
export_df = pd.DataFrame([export_pv, export_wind], columns=export_years)
export_df.index = ['PV_cost_BUSD_per_GW', 'Wind_cost_BUSD_per_GW']

# Save to CSV
export_df.to_csv("projected_costs_2025_2055.csv")
print("Exported projected costs to 'projected_costs_2025_2055.csv'")