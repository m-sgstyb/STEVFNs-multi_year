#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 20:03:33 2024

@author: Mónica Sagastuy-Breña
"""

import pandas as pd
import numpy as np

def update_existing_RE_capacity(my_network, tech_lim, tech_existing):
    """
    Calculates the updated existing capacity for a given technology after an iteration.
    Value should be input into Assets/tech_existing/parameters.csv
    
    Parameters
    ----------
    my_network : STEVFNs network
        Network built after running my_network.build function in main.py.
    tech_lim : str
        Asset name for the limited capacity technology (e.g., 'RE_PV_Openfield_Lim').
    tech_existing : str
        Asset name for the existing capacity technology (e.g., 'RE_PV_Existing').

    Returns
    -------
    FLOAT
        Updated value for installed capacity for the next iteration.
    """
    new_capacity = 0

    # Find the new capacity to add if it exceeds the threshold
    for asset in my_network.assets:
        if asset.asset_name == tech_lim and asset.asset_size() >= 5e-4:  # 0.5 MW threshold
            new_capacity = asset.asset_size()

    # Update the existing capacity with the new capacity
    for asset in my_network.assets:
        if asset.asset_name == tech_existing:
            asset.parameters_df['existing_capacity'] += new_capacity
            return asset.parameters_df['existing_capacity']
        
def update_FF_existing_cap(my_network):
    '''
    Updates existing fossil power plant capacity. Value should be input to
    Assets/PP_CO2_Existing/parameters.csv
    

    Parameters
    ----------
    my_network : STEVFNs NETWORK
        Network built after running my_network.build function in main.py.

    Returns
    -------
    new_capacity : FLOAT
        New capacity for existing FF. As emissions constraints increase, this value should
        decrease per iteration

    '''
    for asset in my_network.assets:
        if asset.asset_name == 'PP_CO2':
            new_capacity = asset.asset_size()
    return new_capacity


def update_RE_installable_cap(my_network, t, input_file, tech_lim, tech_existing):
    '''
    With the logistic function, finds capacity for technology at time t.
    Uses the new existing capacity value from previous iteration to update the 
    remaining installable capacity in next time period
    Value for tech_lim should be input into Assets/tech_lim/parameters.csv

    Parameters
    ----------
    my_network : STEVFNs NETWORK
        Network built after running my_network.build function in main.py
    t : INT
        End year of the iteration period, e.g. if running from 2020-2030, t is 2030
        Finds the total installed capacity at this point given by the logistic adoption
        curve
    input_file : PATH
        To a csv file containing logistic function parameters for countries and
        technologies
    tech_lim : str
        Asset name for the limited capacity technology (e.g., 'RE_PV_Openfield_Lim').
    tech_existing : str
        Asset name for the existing capacity technology (e.g., 'RE_PV_Existing').

    Returns
    -------
    dict
        A dictionary with country-technology keys and installable capacities as values.

    '''
    log_params_df = pd.read_csv(input_file)
    
    installable_capacities = {}

    for _, row in log_params_df.iterrows():
        country = row['Country']
        technology = row['Technology']
        t = row['Year']
        L = row['L']
        t_alpha = row['t_alpha']
        t_beta = row['t_beta']
        alpha = row['alpha']
        beta = row['beta']

        # Calculate the logistic function parameters
        k = (np.log((1 - beta) / beta) - np.log((1 - alpha) / alpha)) / (t_alpha - t_beta)
        t0 = t_alpha + (np.log((1 - alpha) / alpha)) / k

        # Calculate total installable capacity at year t
        total_cap = L / (1 + np.exp(-k * (t - t0)))

        # Update the installable capacity by subtracting the installed capacity
        installed_cap = update_existing_RE_capacity(my_network, tech_lim, tech_existing)
        installable_cap = total_cap - installed_cap

        # Store the result in the dictionary with a composite key to ID country and tech
        key = f"{country}_{technology}"
        installable_capacities[key] = max(0, installable_cap)  # Ensure non-negative capacity

    return installable_capacities


