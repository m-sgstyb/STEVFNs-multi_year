#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 08:09:04 2025

@author: Mónica Sagastuy-Breña
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Let's also illustrate a collaboration scenario.
# Suppose we have three countries (A, B, C) with different baselines.
# For the example, we'll just generate two additional baselines by scaling and slightly reshaping the provided one.
# In your real case, replace A_base, B_base, C_base with your actual arrays (same length).

def linear_scaling_to_cumulative_budget(baseline, budget_fraction=0.60):
    """
    Produce a monotone linear scaling schedule m_t in [0,1] that multiplies the baseline each year:
      path_t = baseline_t * m_t

    - baseline: 1D array-like of emissions (e.g. MtCO2e per year).
    - budget_fraction: desired share of baseline cumulative emissions (e.g. 0.60 for 60%).

    Returns:
      path  -> adjusted emissions trajectory (numpy array)
      m     -> the multipliers m_t applied to each year
    """
    b = np.array(baseline, dtype=float)
    T = len(b)
    # target cumulative (budget)
    B_base = b.sum()
    B_target = budget_fraction * B_base
    
    # multipliers: m_t = 1 - (1 - m_T) * (t/(T-1))
    t = np.arange(T)
    denom = (t/(T-1) * b).sum()
    if denom == 0:
        # fallback if baseline is weird (all zero or concentrated)
        m_T = B_target / B_base if B_base > 0 else 0.0
    else:
        alpha = (B_base - B_target) / denom
        m_T = 1 - alpha
    
    # clamp to [0,1]
    m_T = max(0.0, min(1.0, m_T))
    m = 1 - (1 - m_T) * (t/(T-1))
    return b * m, m


def apply_common_reduction_factor(country_baselines, factors):
    """
    Apply the SAME annual reduction factor m_t to each country's baseline.

    - country_baselines: list of arrays (same length), one per country.
    - factors: array-like of multipliers m_t, length T.

    Returns:
      reduced  -> list of arrays (one per country, baseline * m_t)
      total    -> coalition total (sum of reduced across countries)
    """
    factors = np.array(factors, dtype=float)
    reduced = [np.array(b, dtype=float) * factors for b in country_baselines]
    total = np.sum(np.vstack(reduced), axis=0)
    return reduced, total


def linear_path_from_start(baseline, end_fraction=0.5):
    """
    Linear reduction from E0 down to end_fraction * E0 by last year.
    baseline: array of baseline emissions (only baseline[0] is used).
    """
    E0 = baseline[0]
    T = len(baseline)
    E_end = end_fraction * E0
    return np.linspace(E0, E_end, T)

def s_curve_from_start(baseline, end_fraction=0.5, steepness=0.3, t_mid=None):
    """
    Fast-then-slow logistic reduction from E0 down to end_fraction * E0 by last year.
    baseline: array of baseline emissions (only baseline[0] is used).
    """
    E0 = baseline[0]
    T = len(baseline)
    if t_mid is None:
        t_mid = (T-1)/2
    E_end = end_fraction * E0
    t = np.arange(T)
    # logistic from 0→1, then scaled
    L = 1 / (1 + np.exp(-steepness*(t - t_mid)))
    # rescale so at t=0 → 1, at t=T-1 → 0
    L = (L[-1] - L) / (L[-1] - L[0])
    return E_end + (E0 - E_end) * L


# A_base = np.array([124.0, 107.0, 105.0, 106.0, 108.0, 103.0, 106.0, 109.0, 111.0, 114.0,
#           117.0, 120.0, 122.0, 124.0, 127.0, 128.0, 127.0, 127.0, 125.0, 123.0,
#           112.0, 108.0, 96.6, 96.5, 90.0, 79.0, 72.0, 64.9, 58.2, 57.7])

# B_base = np.array([4.52, 4.57, 4.39, 4.59, 4.83, 4.51, 4.67, 4.79, 4.94, 4.99, 5.42, 5.2,
#           5.24, 5.04, 5.4, 5.32, 5.05, 5.0, 5.1, 4.99, 4.99, 4.87, 4.56, 4.64,
#           4.43, 4.59, 4.47, 4.2, 4.03, 3.78])   # CHL

# C_base = np.array([5560.0, 5580.0, 5570.0, 5610.0, 5680.0, 5750.0, 5810.0, 5950.0, 5940.0,
#           5970.0, 5980.0, 6010.0, 5980.0, 5990.0, 6010.0, 6110.0, 6130.0, 6150.0,
#           6030.0, 6010.0, 6020.0, 6010.0, 6030.0, 6020.0, 6010.0, 6000.0, 5990.0,
#           5970.0, 5960.0, 5950.0]) # WECC


A_base = np.array([144.0, 146.0, 153.0, 153.0, 158.0, 163.0, 167.0, 171.0, 175.0, 184.0,
                   185.0, 192.0, 195.0, 203.0, 206.0, 216.0, 219.0, 223.0, 226.0, 235.0,
                   241.0, 240.0, 248.0, 252.0, 259.0, 259.0, 270.0, 269.0, 276.0, 279.0])

B_base = np.array([23.9, 24.2, 24.5, 24.5, 25.5, 27.3, 27.0, 27.5, 27.8, 30.1, 30.6, 32.8,
                   33.0, 34.5, 35.2, 37.4, 40.7, 41.9, 43.9, 45.0, 46.1, 46.1, 48.3, 48.0,
                   48.5, 48.4, 50.2, 50.1, 51.2, 51.7])   # CHL

C_base = np.array([5450.0, 5500.0, 5410.0, 5440.0, 5620.0, 5700.0, 5710.0, 5890.0, 5790.0,
                   5870.0, 5940.0, 5980.0, 5980.0, 5920.0, 5930.0, 6080.0, 6250.0, 6220.0,
                   6280.0, 6180.0, 6230.0, 6270.0, 6300.0, 6230.0, 6180.0, 6160.0, 6200.0,
                   6230.0, 6180.0, 6200.0]) # WECC

years = np.arange(len(A_base),dtype=int)
# ========== Linear reduction but from whole profile so follows shape roughly in reduction =========

# Coalition target method 1: choose a *coalition* carbon budget = 65% of coalition baseline cumulative.
coalition_baseline = A_base + B_base + C_base
# coalition_budget_path, coalition_m = linear_scaling_to_cumulative_budget(coalition_baseline, budget_fraction=0.65)

# Apply the *same* annual reduction factor m_t (derived at coalition level) back to each country baseline.
# (A_path, B_path, C_path), coalition_total = apply_common_reduction_factor([A_base, B_base, C_base], coalition_m)

# ======= Linear path from start year =======
# A_path = linear_path_from_start(A_base, end_fraction=0.5)
# B_path = linear_path_from_start(B_base, end_fraction=0.5)
# C_path = linear_path_from_start(C_base, end_fraction=0.5)

# coalition_total = linear_path_from_start(coalition_baseline, end_fraction=0.5)


# ========= S-curve reduction from start
A = s_curve_from_start(A_base, end_fraction=0.7, t_mid=25)
B = s_curve_from_start(B_base, end_fraction=0.7,t_mid=25)
C = s_curve_from_start(C_base, end_fraction=0.5,t_mid=25)

coalition_total = s_curve_from_start(coalition_baseline, end_fraction=0.5)

A_path = np.zeros(30)
B_path = np.zeros(30)
C_path = np.zeros(30)
coalition = np.zeros(30)
for counter in range(len(A_path)):
    A_path[counter] += np.round(A[counter],decimals=3)
    B_path[counter] += np.round(B[counter],decimals=3)
    C_path[counter] += np.round(C[counter],decimals=3)
    coalition[counter] += np.round(coalition_total[counter],decimals=3)
    
    
    
# Organize results
coal_df = pd.DataFrame({
    "YearIndex": years,
    "A_baseline": A_base,
    "B_baseline": B_base,
    "C_baseline": C_base,
    "A_path": A_path,
    "B_path": B_path,
    "C_path": C_path,
    "Coalition_total_baseline": coalition_baseline,
    "Coalition_total_path": coalition,
})

coal_csv = "collaboration_profiles_scurve_50.csv"
coal_df.T.to_csv(coal_csv, index=True)

# display_dataframe_to_user("Collaboration scenario (example)", coal_df.round(3))

# Plot coalition baseline vs coalition path
plt.figure()
plt.plot(years, coalition_baseline, label="Coalition baseline")
plt.plot(years, coalition_total, label="Coalition path (budget 65%)")
plt.legend()
plt.xlabel("Year")
plt.ylabel("MtCO2e")
# plt.title("Coalition: baseline vs budget-constrained path")
plt.tight_layout()
plt.show()


plt.figure()
plt.plot(years, A_base, label="Mexico emissions baseline")
plt.plot(years, A_path, label="Mexico emissions reduction pathway")
plt.legend()
plt.xlabel("Year")
plt.ylabel("MtCO2e")
# plt.title("Coalition: baseline vs budget-constrained path")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(years, B_base, label="Chile emissions baseline")
plt.plot(years, B_path, label="Chile emissions reduction pathway")
plt.legend()
plt.xlabel("Year")
plt.ylabel("MtCO2e")
# plt.title("Coalition: baseline vs budget-constrained path")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(years, C_base, label="WECC emissions baseline")
plt.plot(years, C_path, label="WECC emissions reduction pathway")
plt.legend()
plt.xlabel("Year")
plt.ylabel("MtCO2e")
# plt.title("Coalition: baseline vs budget-constrained path")
plt.tight_layout()
plt.show()

coal_csv
