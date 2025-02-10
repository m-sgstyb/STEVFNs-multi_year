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
    case_study_name : str
        Case study name defined to run main.py

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
            id_row = df.index[(df['iteration'] == iteration_year) & (df['location_name'] == case_study_name)].tolist()
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
    '''
    

    Parameters
    ----------
    case_study_folder : path
        Path to the case study folder that is running.
    tech_lim : str
        Asset name for the limited capacity technology (e.g., 'RE_PV_Openfield_Lim').
    assets_folder : path
        Path to the Assets folder in the Code for STEVFNs

    Returns
    -------
    opt_capacity : float
        Value for the final optimal capacity after 30-year modeling.

    '''
    # Find 30y case study folder
    ref_case_study_dir = f'{case_study_folder}_30y'
    results_filename = os.path.join(ref_case_study_dir, 'Results', 'results.csv')
    df = pd.read_csv(results_filename)
    country_id = os.path.basename(case_study_folder)
    id_row = df.index[df['technology_name'] == f'{tech_lim}_[{country_id}]'].tolist()
    opt_capacity = df.at[id_row[0], 'technology_size']
    
    return opt_capacity

def get_max_growth_rate(case_study_folder, tech_lim, assets_folder,
                        goal_year, historical_data_file):
    '''
    

    Parameters
    ----------
    case_study_folder : path
        Path to the case study folder that is running.
    tech_lim : str
        Asset name for the limited capacity technology (e.g., 'RE_PV_Openfield_Lim').
    assets_folder : path
        Path to the Assets folder in the Code for STEVFNs.
    goal_year : int
        Goal year to end the modeling, e.g. 2060.
    historical_data_file : path
        Path to the file where the historical capacity data is stored.

    Returns
    -------
    r_max : float
        Maximum rate of adoption obtained from S-curve fitting.

    '''
    goal_capacity = get_30y_opt_capacities(case_study_folder, tech_lim, assets_folder)
    
    K_fit, r_fit, t0_fit, years, capacities = DPhil_Plotting.fit_s_curves(case_study_folder, tech_lim, assets_folder,
                      goal_capacity, goal_year, historical_data_file)
    r_max = (r_fit * K_fit) / 4
    
    return r_max

def get_scenario_year(scenario_folder):
    '''
    Parameters
    ----------
    scenario_folder : path
         Path to the scenario folder that is running inside the for loop in main.py

    Returns
    -------
    scenario_year : int
        Value for the scenario year (start of control horizon), e.g. 2030.

    '''
    scenario_name = os.path.basename(scenario_folder)
    scenario_year = int(scenario_name.split('_')[1])
    return scenario_year

def get_prev_existing_capacity(scenario_folder, assets_folder, tech_existing,
                               case_study_name):
    '''
    

    Parameters
    ----------
    scenario_folder : path
         Path to the scenario folder that is running inside the for loop in main.py
    assets_folder : path
        Path to the "Assets" folder in STEVFNS > Code
    tech_existing : str
        Asset name for the existing capacity technology (e.g., 'RE_PV_Existing').
    case_study_name : str
        Case study name defined to run main.py

    Returns
    -------
    Start year of current scenario and previous existing capacity

    '''
    scenario_year = get_scenario_year(scenario_folder)
    asset_folder = os.path.join(assets_folder, tech_existing)
    df = pd.read_csv(os.path.join(asset_folder, 'parameters.csv'))
    # Get existing capacity for previous modeled period
    id_row = df.index[(df['iteration'] == str(scenario_year)) & (df['location_name'] == case_study_name)]
    previous_existing = df.at[int(id_row[0]-1), 'existing_capacity']
    
    return scenario_year, previous_existing
    
def update_RE_installable_cap(my_network, case_study_folder, scenario_folder, tech_lim,
                              tech_existing, assets_folder, goal_year, historical_data_file,
                              t_i=2025, t_f=2060, dt=5):
    '''
    

    Parameters
    ----------
    my_network : STEVFNs network
        Network built after running my_network.build function in main.py.
    case_study_folder : path
        Path to the case study folder that is running.
    scenario_folder : path
         Path to the scenario folder that is running inside the for loop in main.py
    tech_lim : str
        Asset name for the limited capacity technology (e.g., 'RE_PV_Openfield_Lim').
    tech_existing : str
        Asset name for the existing capacity technology (e.g., 'RE_PV_Existing').
    assets_folder : path
        Path to the "Assets" folder in STEVFNS > Code
    goal_year : int
        Goal year to end the modeling, e.g. 2060.
    historical_data_file : path
        Path to the file where the historical capacity data is stored.
    t_i : int, optional
        start year of the whole period being modeled. The default is 2025.
    t_f : int, optional
        End year of the whole period being modeled. The default is 2060.
    dt : int, optional
        Number of years per iteration sub-period to be modeled. The default is 5.

    Returns
    -------
    Updated CSV file with calculated min and max capacity bounds

    '''
    case_study_name = os.path.basename(case_study_folder)
    
    r_max = get_max_growth_rate(case_study_folder, tech_lim, assets_folder,
                                goal_year, historical_data_file)
    
    scenario_year, previous_existing = get_prev_existing_capacity(scenario_folder,
                                                                  assets_folder,
                                                                  tech_existing,
                                                                  case_study_name)
    
    k_goal = get_30y_opt_capacities(case_study_folder, tech_lim, assets_folder)
    existing_asset_folder = os.path.join(assets_folder, tech_existing)
    existing_params = pd.read_csv(os.path.join(existing_asset_folder, 'parameters.csv'))
    
    lim_asset_folder = os.path.join(assets_folder, tech_lim)
    lim_params = pd.read_csv(os.path.join(lim_asset_folder, 'parameters.csv'))
    
    # Find prev iteration existing capacity
    id_row = existing_params.index[(existing_params['iteration'] == str(scenario_year)) & (existing_params['location_name'] == case_study_name)].tolist()
    k_existing = existing_params.at[int(id_row[0]-1), 'existing_capacity']
    
    if scenario_year == 2030:
        k_opt = 0
    else: 
        for asset in my_network.assets:
            if asset.asset_name == tech_lim:
                k_opt = asset.asset_size()
    
    years_left = goal_year - scenario_year

    k_existing += k_opt
    
    k_max = min(
         k_goal - k_existing,
         (r_max * dt)
    )

    k_min = max((k_goal - (r_max * years_left)),
                k_existing) - k_existing
    
    
    id_row = lim_params.index[(lim_params['iteration'] == str(scenario_year)) & (lim_params['location_name'] == case_study_name)].tolist()
    lim_params.at[int(id_row[0]+1), 'minimum_size'] = k_min
    lim_params.at[int(id_row[0]+1), 'maximum_size'] = k_max
    
    
    return lim_params.to_csv(os.path.join(lim_asset_folder, 'parameters.csv'), index=False)


