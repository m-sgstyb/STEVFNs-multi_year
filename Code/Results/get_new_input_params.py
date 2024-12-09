#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 20:03:33 2024

@author: Mónica Sagastuy-Breña
"""

import pandas as pd
import numpy as np
import os

def update_existing_RE_capacity(my_network, tech_lim, tech_existing,
                                assets_folder, iteration_year, case_study_name):
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
    asset_folder : path
        Path to the Assets folder in the Code for STEVFNs
    iteration_year : str
        String of the end year being modelled. i.e. if the period 2020-2030 is being modelled,
        iteration_year would be '2030'

    Returns
    -------
    CSV FILE
        Updated paramters.csv file saved in the asset folder for next run
    """
    new_capacity = 0

    # Find the new capacity to add if it exceeds the threshold
    for asset in my_network.assets:
        if asset.asset_name == tech_lim and asset.asset_size() >= 5e-4:  # 0.5 MW threshold
            new_capacity = asset.asset_size()

    # Update the existing capacity with the new capacity
    for asset in my_network.assets:
        if asset.asset_name == tech_existing:
            previous_existing = asset.asset_size()
            # Need to find path to asset's parameters.csv to update there
            asset_folder = os.path.join(assets_folder, tech_existing)
            df = pd.read_csv(os.path.join(asset_folder, 'parameters.csv'))
            
            # Find row that has description column value equal to the iteration year

            id_row = df.index[df['iteration'] == iteration_year and df['location_name'] == case_study_name].tolist()
            # print("ID_ROW", id_row)
            df.at[id_row[0], 'existing_capacity'] = previous_existing + new_capacity    
            
    return df.to_csv(os.path.join(asset_folder, 'parameters.csv'), index=False)
        
def update_FF_existing_cap(my_network, assets_folder, iteration_year):
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
            
    
            asset_folder = os.path.join(assets_folder, 'PP_CO2')
            df = pd.read_csv(os.path.join(asset_folder, 'parameters.csv'))
            id_row = df.index[df['iteration'] == iteration_year].tolist()
            df.at[id_row[0], 'existing_capacity'] = previous_existing + new_capacity   
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


