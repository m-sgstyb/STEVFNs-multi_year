#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 20:03:33 2024

@author: Mónica Sagastuy-Breña
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from Code.Plotting import DPhil_Plotting
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
    assets_folder : path
        Path to the "Assets" folder in STEVFNS > Code
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
            
            # Find row for country and iteration year to be updated
            id_row = df.index[df['iteration'] == iteration_year and df['location_name'] == case_study_name].tolist()
            df.at[id_row[0], 'existing_capacity'] = previous_existing + new_capacity    
            
    return df.to_csv(os.path.join(asset_folder, 'parameters.csv'), index=False)
        
def update_FF_existing_cap(my_network, assets_folder, iteration_year, case_study_name):
    '''
    Updates existing fossil power plant capacity, should be decreasing

    Parameters
    ----------
    my_network : STEVFNs NETWORK
        Network built after running my_network.build function in main.py.
    assets_folder : path
        Path to the Assets folder in the Code for STEVFNs
    iteration_year : str
        String of the end year being modelled. i.e. if the period 2020-2030 is being modelled,
        iteration_year would be '2030'
    case_study_name : str
        Case study name which is defined at the start of main.py.

    Returns
    -------
    CSV FILE
        Updated paramters.csv file saved in the asset folder for next run

    '''
    for asset in my_network.assets:
        if asset.asset_name == 'PP_CO2' and asset.asset_size() <= 5e-4:
            continue
            
        if asset.asset_name == 'PP_CO2_Existing':
            new_existing = asset.asset_size()
            asset_folder = os.path.join(assets_folder, 'PP_CO2_Existing')
            df = pd.read_csv(os.path.join(asset_folder, 'parameters.csv'))
            id_row = df.index[df['iteration'] == iteration_year and df['location_name'] == case_study_name].tolist()
            df.at[id_row[0], 'existing_capacity'] = new_existing  
    return df.to_csv(os.path.join(asset_folder, 'parameters.csv', index=False))

def get_30y_opt_capacities(case_study_folder, tech_lim, assets_folder):
    # Find 30y case study folder
    ref_case_study_dir = f'{case_study_folder}_30y'
    results_filename = os.path.join(ref_case_study_dir, 'Results', 'results.csv')
    df = pd.read_csv(results_filename)
    country_id = os.path.basename(case_study_folder)
    id_row = df.index[df['technology_name'] == f'{tech_lim}_[{country_id}]'].tolist()
    opt_capacity = df.at[id_row[0], 'technology_size']
    
    return opt_capacity

def get_max_growth_rate(case_study_folder, tech_lim, assets_folder,
                  goal_capacity, goal_year, historical_data_file):
    '''
    

    Parameters
    ----------
    case_study_folder : TYPE
        DESCRIPTION.
    tech_lim : TYPE
        DESCRIPTION.
    assets_folder : TYPE
        DESCRIPTION.
    goal_capacity : TYPE
        DESCRIPTION.
    goal_year : TYPE
        DESCRIPTION.
    historical_data_file : TYPE
        DESCRIPTION.
     : TYPE
        DESCRIPTION.

    Returns
    -------
    r_max : TYPE
        DESCRIPTION.

    '''
    K_fit, r_fit, t0_fit, years, capacities = DPhil_Plotting.fit_s_curves(case_study_folder, tech_lim, assets_folder,
                      goal_capacity, goal_year, historical_data_file)
    r_max = (r_fit * K_fit) / 4
    return r_max

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


