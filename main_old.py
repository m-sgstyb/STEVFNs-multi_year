#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 17:38:43 2021
@author: aniqahsan

Last Adapted on 19 Nov 2024
@author: Mónica Sagastuy-Breña
"""

import pandas as pd
import time
import os
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

from Code.Network.Network import Network_STEVFNs
from Code.Plotting import DPhil_Plotting
from Code.Plotting import mitigation_plots
from Code.Results import Results
from Code.Results import get_new_input_params

#### Define Input Files ####
case_study_name = "test_multi_year"
# case_study_name = "MEX_30y"
# case_study_name = "CHL"
# case_study_name = "USA_WECC"

# case_study_name = "MEX-CHL_Autarky"
# case_study_name = "MEX-CHL_Collab"

# case_study_name = "USA_WECC-CHL_Autarky"
# case_study_name = "USA_WECC-CHL_Collab"

base_folder = os.path.dirname(__file__)
code_folder = os.path.join(base_folder, "Code")
assets_folder = os.path.join(code_folder, "Assets")
data_folder = os.path.join(base_folder, "Data")
case_study_folder = os.path.join(data_folder, "Case_Study", case_study_name)
results_folder = os.path.join(case_study_folder, "Results")
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

scenario_folders_list = []
for folder in os.listdir(case_study_folder):
    full_path = os.path.join(case_study_folder, folder)
    # Check if the item is a folder and starts with 'scenario_' to add to scenario_folders_list
    if os.path.isdir(full_path) and folder.startswith('scenario_'):
        scenario_folders_list.append(full_path)
    
historical_data_file = os.path.join(case_study_folder, f're_historic_data_{case_study_name}.csv')
network_structure_filename = os.path.join(case_study_folder, "Network_Structure.csv")

#### Define Output Files ####
rounded_results_filename = os.path.join(results_folder, "results_rounded.csv")
results_filename = os.path.join(results_folder, "results.csv")
# flows_filename = os.path.join(results_folder, "flows.csv")
# curtailment_filename = os.path.join(results_folder, "curtailment.csv")

### Read Network Structure ###
network_structure_df = pd.read_csv(network_structure_filename)

### Build Network ###
start_time = time.time()
my_network = Network_STEVFNs()
my_network.build(network_structure_df)

build_time = time.time()
print("Time taken to build network = ", build_time - start_time, "s")
# Define empty dataframes for results
total_df = pd.DataFrame()
total_df_1 = pd.DataFrame()

# Sort scenario folders list so it starts with 2030 upwards
scenario_folders_list.sort()

# for scenario in range(len(scenario_folders_list)):
for scenario in range(1):
    # Read Input Files ###
    scenario_folder = scenario_folders_list[scenario]
    # Get year value out of the scenario path to replace results later
    scenario_year = scenario_folder[-4:] # STRING of scenario year
    
    asset_parameters_filename = os.path.join(scenario_folder, "Asset_Parameters.csv")
    location_parameters_filename = os.path.join(scenario_folder, "Location_Parameters.csv")
    system_parameters_filename = os.path.join(scenario_folder, "System_Parameters.csv")
    
    asset_parameters_df = pd.read_csv(asset_parameters_filename)
    location_parameters_df = pd.read_csv(location_parameters_filename)
    system_parameters_df = pd.read_csv(system_parameters_filename)
    
    ### Update Network Parameters ###
    update_time = time.time()
    
    my_network.update(location_parameters_df, asset_parameters_df, system_parameters_df)
    my_network.scenario_name = os.path.basename(scenario_folder)
    
    updated_time = time.time()
    print("Time taken to update network = ", updated_time - update_time, "s")
    
    ### Run Simulation ###
    solve_time = time.time()
    my_network.problem.solve(solver = cp.MOSEK, verbose=False)
    # my_network.solve_problem() # Default solver is CLARABEL with max_iter=10000
    solved_time = time.time()

    # Print some results
    print("Scenario: ", my_network.scenario_name)
    print("Time taken to solve problem = ", solved_time - solve_time, "s")
    print(my_network.problem.solution.status)
    
    # Avoid breaking the optimisation if a scenario does not converge
    if my_network.problem.value == float("inf"):
        print("problem value inf")
        continue
    print("Total cost to satisfy all demand = ", my_network.problem.value, " Billion USD")
    print("Total emissions = ", my_network.assets[0].asset_size(), "MtCO2e")
    # DPhil_Plotting.plot_asset_sizes(my_network)
    DPhil_Plotting.plot_asset_sizes_stacked(my_network,
                                            location_parameters_df,
                                            save_path=os.path.join(results_folder, "asset_sizes", f"{my_network.scenario_name}_nocurtailment.png"))
    
    # Save results for asset flows and total data per scenario
    results_df = Results.get_total_data(my_network, location_parameters_df, asset_parameters_df)
    results_rounded_df = Results.get_total_data_rounded(my_network, location_parameters_df, asset_parameters_df)
    
    
    
    if scenario == 0:
        total_results = results_df
        total_results_rounded = results_rounded_df
    else:
        # Concatenate next scenario's results into one dataframe
        total_results = pd.concat([total_results, results_df], ignore_index=True)
        total_results_rounded = pd.concat([total_results_rounded, results_rounded_df], ignore_index=True)
    
    # This works for all PV existing in MEX case study, updates well. Will need to add a condition to meet
    # Location column AND description column do be able to differenciate with other case studies
    pv_lim = 'RE_PV_Openfield_Lim'
    pv_existing = 'RE_PV_Existing' 
    wind_lim = 'RE_WIND_Onshore_Lim'
    wind_existing = 'RE_WIND_Existing'
    
    # print("RMAX Wind", get_new_input_params.get_max_growth_rate(case_study_folder, wind_lim, assets_folder, 2055, historical_data_file))
    # for tech in techs:
    # get_new_input_params.update_existing_RE_capacity(my_network,
    #                                                  pv_lim,
    #                                                  pv_existing,
    #                                                  assets_folder,
    #                                                  scenario_year,
    #                                                  case_study_name)
    
    # get_new_input_params.update_existing_RE_capacity(my_network,
    #                                                  wind_lim, 
    #                                                  wind_existing,
    #                                                  assets_folder,
    #                                                  scenario_year,
    #                                                  case_study_name)
    
    # get_new_input_params.get_prev_existing_capacity(scenario_folder,
    #                                                 assets_folder,
    #                                                 pv_existing,
    #                                                 case_study_name)
    
    # get_new_input_params.get_prev_existing_capacity(scenario_folder,
    #                                                 assets_folder,
    #                                                 wind_existing,
    #                                                 case_study_name)
    
    # get_new_input_params.update_RE_installable_cap(my_network,
    #                                                case_study_folder,
    #                                                scenario_folder,
    #                                                pv_lim,
    #                                                pv_existing,
    #                                                assets_folder,
    #                                                2055,
    #                                                historical_data_file)
    
    # get_new_input_params.update_RE_installable_cap(my_network,
    #                                                case_study_folder,
    #                                                scenario_folder,
    #                                                wind_lim,
    #                                                wind_existing,
    #                                                assets_folder,
    #                                                2055,
    #                                                historical_data_file)

        
    # flows_df = Results.export_aut_flows(my_network)
    flows_df = Results.export_collab_flows(my_network, location_parameters_df)
    
    curtailment = Results.calculate_curtailment_aut(my_network)
    # curtailment = Results.calculate_curtailment_collab(my_network, location_parameters_df)
    
    flows_filename = os.path.join(results_folder, f"flows_{my_network.scenario_name}.csv")
    curtailment_filename = os.path.join(results_folder, f"curtailment_{my_network.scenario_name}.csv")
    curtailment.to_csv(curtailment_filename, index=False, header=True)
    flows_df.to_csv(flows_filename, index=False, header=True)
    
    
    
    

# # Save total_data for all scenarios into a single csv file
total_results.to_csv(results_filename, index=False, header=True)
total_results_rounded.to_csv(rounded_results_filename, index=False, header=True)
# Save flows into a separate file
end_time = time.time()
print("Time taken to build, update and solve:", end_time - start_time, "s")

#%% Plotting

# plot_filename = os.path.join(results_folder, "emissions-reduction.png")
# total_data_filename = os.path.join(data_folder, "Case_Study", "MEX", "Results", "results.csv")
# countries=["MX"]
# mitigation_plots.dpacc_subplots(total_data_filename, total_data_filename, plot_filename,
#                 "MEX", countries)


## Manual plotting 
# categories = []
# values = []

# # # Iterate through assets, starting from i = 1 to skip i = 0
# for i in range(1, len(my_network.assets)):  
#     category = my_network.assets[i].asset_name
#     value = my_network.assets[i].asset_size()
    
#     # Only add the category and value if the value is >= 1e-6
#     if value >= 1e-3:
#         categories.append(category)
#         values.append(value)

# # Normalize colors to the number of bars
# num_bars = len(values)
# colors = plt.cm.viridis(np.linspace(0, 1, num_bars))  # Change 'viridis' to any preferred colormap

# # Plot all bars together after the loop ends, applying the color map
# plt.figure(figsize=(12, 8))
# plt.bar(categories, values, color=colors)
# plt.title("Asset Sizes")
# plt.xlabel("Asset Name")
# plt.ylabel("Asset Size")

# # Show the combined plot
# plt.xticks(rotation=45)  # Rotate x-axis labels if needed for better readability
# plt.tight_layout()       # Adjust layout to fit x-axis labels
# plt.show()
