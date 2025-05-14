#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 17:38:43 2021

@author: aniqahsan
"""

import pandas as pd
import numpy as np
import time
import os
import cvxpy as cp


from Code.Network.Network import Network_STEVFNs
from Code.Plotting import DPhil_Plotting
from Code.Results import Results


#### Define Input Files ####
case_study_name = "test_multi_year"


base_folder = os.path.dirname(__file__)
data_folder = os.path.join(base_folder, "Data")
case_study_folder = os.path.join(data_folder, "Case_Study", case_study_name)
scenario_folders_list = [x[0] for x in os.walk(case_study_folder)][1:]
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
    my_network.problem.solve(solver = cp.CLARABEL, max_iter=10000, ignore_dpp=False, verbose=False)
    
    # my_network.problem.solve(solver = cp.ECOS, warm_start=True, max_iters=10000, feastol=1e-5, reltol=1e-5, abstol=1e-5, ignore_dpp=True, verbose=False)
    # my_network.problem.solve(solver = cp.SCS, warm_start=True, max_iters=10000, ignore_dpp=True, verbose=False)
    end_time = time.time()

    
    ### Plot Results ############
    print("Scenario: ", my_network.scenario_name)
    print("Time taken to solve problem = ", end_time - start_time, "s")
    print(my_network.problem.solution.status)
    if my_network.problem.value == float("inf"):
        continue
    print("Total cost to satisfy all demand = ", my_network.problem.value, " Billion USD")
    print("Total emissions = ", my_network.assets[0].asset_size(), "MtCO2e")
    # DPhil_Plotting.plot_all(my_network)
    # DPhil_Plotting.plot_asset_sizes(my_network)
    # DPhil_Plotting.plot_asset_costs(my_network)
    
       
demand_flows = my_network.assets[2].get_plot_data()
pp_flows = my_network.assets[1].get_plot_data()
re_flows = my_network.assets[3].get_plot_data()
cf_profile = my_network.assets[3].gen_profile.value


# Combine the arrays as columns
combined_flows = np.column_stack((demand_flows, pp_flows, re_flows, cf_profile))

# Define header names for columns (optional)
header = "demand_flows,pp_flows,re_flows,cf_profile"

# Save to a single CSV file
np.savetxt(
    os.path.join(case_study_folder, "all_flows.csv"),
    combined_flows,
    delimiter=",",
    header=header,
    comments=''  # Avoids '#' at the beginning of the header line
)
   