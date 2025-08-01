#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 06:27:57 2025

@author: Mónica Sagastuy-Breña
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the data (no headers)
gef = pd.read_csv('gef_per_year.csv', header=None)
gef = gef * 1000 # kgCO2/kWh

lcoe = pd.read_csv('lcoe_per_year.csv', header=None)
lcoe = lcoe * 1000 # USD / kWh
# Define years based on length
start_year = 1  # change this if your series starts from a different year
years = pd.Series(range(start_year, start_year + len(gef)))

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(years, gef, label='Grid Intensity (kgCO₂/kWh)', color='tab:green', marker='o')
plt.plot(years, lcoe, label='LCOE (USD/kWh)', color='tab:blue', marker='s')

plt.xlabel('Year')
plt.ylabel('Value')
# plt.title('GEF and LCOE Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("gef_lcoe_17280.png", dpi=300)
plt.show()
