#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 17:38:43 2021

@author: aniqahsan
Adapted by:
    @author: Mónica Sagastuy-Breña 2025
"""

import pandas as pd
import numpy as np
import time
import os
import cvxpy as cp
from collections import defaultdict
import matplotlib.pyplot as plt


from Code.Network.Network import Network_STEVFNs
from Code.Plotting import DPhil_Plotting
from Code.Results import Results

#### Define Input Files ####
case_study_name = "MEX_30y_MY"
# case_study_name = "MEX_30y_MY_no_CO2_budget"
# case_study_name = "toy_wind_PP"


base_folder = os.path.dirname(__file__)
data_folder = os.path.join(base_folder, "Data")
case_study_folder = os.path.join(data_folder, "Case_Study", case_study_name)
results_folder = os.path.join(case_study_folder, "Results")
if not os.path.exists(results_folder):
    os.mkdir(results_folder)
# scenario_folders_list = [x[0] for x in os.walk(case_study_folder)][1:]
# network_structure_filename = os.path.join(case_study_folder, "Network_Structure.csv")

# Get list of scenario folders, excluding the Results folder
scenario_folders_list = [x[0] for x in os.walk(case_study_folder)][1:]
scenario_folders_list = [f for f in scenario_folders_list if os.path.basename(f) != "Results"]
network_structure_filename = os.path.join(case_study_folder, "Network_Structure.csv")

### Read Input Files ###

network_structure_df = pd.read_csv(network_structure_filename)

### Build Network ###
start_time = time.time()

my_network = Network_STEVFNs()
my_network.build(network_structure_df)

end_time = time.time()
print("Time taken to build network = ", end_time - start_time, "s")
total_df = pd.DataFrame()
total_df_1 = pd.DataFrame()

for counter1 in range(len(scenario_folders_list)):
# for counter1 in range(1):
    # Read Input Files ###
    scenario_folder = scenario_folders_list[-1-counter1]
    asset_parameters_filename = os.path.join(scenario_folder, "Asset_Parameters.csv")
    location_parameters_filename = os.path.join(scenario_folder, "Location_Parameters.csv")
    system_parameters_filename = os.path.join(scenario_folder, "System_Parameters.csv")
    
    asset_parameters_df = pd.read_csv(asset_parameters_filename)
    location_parameters_df = pd.read_csv(location_parameters_filename)
    system_parameters_df = pd.read_csv(system_parameters_filename)
    
    ### Update Network Parameters ###
    start_time = time.time()
    
    my_network.update(location_parameters_df, asset_parameters_df, system_parameters_df)
    my_network.scenario_name = os.path.basename(scenario_folder)
    
    end_time = time.time()
    print("Time taken to update network = ", end_time - start_time, "s")
    
    ### Run Simulation ###
    start_time = time.time()
    
    # my_network.solve_problem()
    # my_network.problem.solve(solver = cp.CLARABEL, max_iter=10000, ignore_dpp=False, verbose=True)
    my_network.problem.solve(solver = cp.MOSEK, ignore_dpp=True, verbose=True)
    
    end_time = time.time()

    ### Plot Results ############
    print("Scenario: ", my_network.scenario_name)
    print("Time taken to solve problem = ", end_time - start_time, "s")
    print(my_network.problem.solution.status)
    if my_network.problem.value == float("inf"):
        continue
    print("Total cost to satisfy all demand = ", my_network.problem.value, " Billion USD")
    # print("Total emissions = ", my_network.assets[0].asset_size(), "MtCO2e")

if my_network.problem.value != float("inf"):
      
    yearly_path = os.path.join(case_study_folder, "all_flows_yearly.csv")
    Results.save_yearly_flows_to_csv(my_network, yearly_path)
    
    DPhil_Plotting.plot_yearly_flows(my_network, results_folder)
    DPhil_Plotting.plot_yearly_flows_stacked(my_network, results_folder)


DPhil_Plotting.get_install_pathways(my_network.assets[1], results_folder, tech_name="PV")
DPhil_Plotting.get_install_pathways(my_network.assets[2], results_folder, tech_name="Wind")
DPhil_Plotting.get_dual_install_pathways(my_network.assets[1], my_network.assets[2], results_folder, "PV", "Wind")
