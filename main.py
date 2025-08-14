#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 17:38:43 2021

@author: aniqahsan
Adapted by:
    @author: Mónica Sagastuy-Breña 2025
"""

import pandas as pd
import time
import os
import cvxpy as cp
from Code.Network.Network import Network_STEVFNs
from Code.Plotting import DPhil_Plotting
from Code.Results import Results


#### Define Input Files ####
# sample_sizes = [51840, 69120, 131760]
#sample_sizes = [4320, 8640, 17280, 69120]
# sample_sizes = [51840, 69120] # Need emissions from each scenario to compare
sample_sizes = [4320]

# case_study_name = "MEX_34560"
# case_study_name = "three_countries_no_emissions_constraint4320"
case_study_name = "three_country_Autarky"
# case_study_name = "MEX-CHL_Collab"

for sample in sample_sizes:
    
    # case_study_name = f"MEX_30y_MY_{sample}"
    base_folder = os.path.dirname(__file__)
    data_folder = os.path.join(base_folder, "Data")
    case_study_folder = os.path.join(data_folder, "Case_Study", case_study_name)
    
    # Get list of scenario folders, excluding the Results folder
    scenario_folders_list = [x[0] for x in os.walk(case_study_folder)][1:]
    scenario_folders_list = [f for f in scenario_folders_list if os.path.basename(f) != "Results"] # filter out the results folder
    network_structure_filename = os.path.join(case_study_folder, "Network_Structure.csv")
    
    ### Read Network Structure file ###
    network_structure_df = pd.read_csv(network_structure_filename)
    print("Read Network_Structure.csv")
    ### Build Network ###
    runtimes = []
    start_time_0 = time.time()
    my_network = Network_STEVFNs()
    print("===================== Building Network =====================")
    try:
        my_network.build(network_structure_df)
    except Exception as e:
        print(f"Build failed: {e}")
        print("Failed at time:", (time.time() - start_time_0)/ 60, "min")
    
    
    end_time = time.time()
    build_time = end_time - start_time_0
    print("Time taken to build network = ", build_time, "s")
    total_df = pd.DataFrame()
    total_df_1 = pd.DataFrame()
    
    emissions_dict = {}
    
    for counter1 in range(len(scenario_folders_list)):
    # for counter1 in range(1):
        # Read Input Files ###
        scenario_folder = scenario_folders_list[-1-counter1]
        my_network.scenario_name = os.path.basename(scenario_folder)
        print(f"===================== Starting Scenario {my_network.scenario_name} =====================")
        results_folder = os.path.join(scenario_folder, "Results")
        if not os.path.exists(results_folder):
            os.mkdir(results_folder)
        asset_parameters_filename = os.path.join(scenario_folder, "Asset_Parameters.csv")
        location_parameters_filename = os.path.join(scenario_folder, "Location_Parameters.csv")
        system_parameters_filename = os.path.join(scenario_folder, "System_Parameters.csv")
        
        asset_parameters_df = pd.read_csv(asset_parameters_filename)
        location_parameters_df = pd.read_csv(location_parameters_filename)
        system_parameters_df = pd.read_csv(system_parameters_filename)
        
        ### Update Network Parameters ###
        start_time = time.time()
        
        my_network.update(location_parameters_df, asset_parameters_df, system_parameters_df)
        
        end_time = time.time()
        update_time = end_time - start_time
        print("===================== Updating Network =====================")
        print("Time taken to update network = ", update_time, "s")
        
        ### Run Simulation ###
        start_time = time.time()
        # my_network.problem.solve(solver = cp.CLARABEL, max_iter=10000, ignore_dpp=False, verbose=True)
        my_network.problem.solve(solver = cp.MOSEK, ignore_dpp=True, verbose=True)
        end_time = time.time()
        full_time = end_time - start_time_0
        solve_time = end_time - start_time
        runtimes.append({
                    "Case_study": case_study_name,
                    "scenario": my_network.scenario_name,
                    "build_time_s": build_time if counter1 == 0 else "",  # Only log once
                    "build_time_min": build_time / 60 if counter1 == 0 else "",  # Only log once
                    "update_time_s": update_time,
                    "update_time_min": update_time / 60,
                    "solve_time_s": solve_time,
                    "solve_time_min": solve_time / 60,
                    "full_time_s": full_time,
                    "full_time_min": full_time / 60,
                    })
    
    
        ######## Get and save Results ############
        print("====================== Results =========================")
        print("Scenario: ", my_network.scenario_name)
        print("Time taken to build, update and solve problem = ", full_time, "s")
        print(my_network.problem.solution.status)
        if my_network.problem.value == float("inf"):
            continue
        print("Total cost to satisfy all demand = ", my_network.problem.value, " Billion USD")
        for asset in my_network.assets:
            if asset.asset_name == "CO2_Budget_MY":
                print("Total emissions = ", [float(f"{i:.3g}") for i in asset.asset_size()], "MtCO2e")    
        
        emissions_reduction = my_network.assets[0].asset_size()
        scenario_name = my_network.scenario_name
        emissions_dict[scenario_name] = emissions_reduction
        sampled_days = int((network_structure_df["End_Time"][0] / 24) / 30)
        simulation_factor = 365 / sampled_days
        if my_network.problem.value != float("inf"):
              
            yearly_path = os.path.join(case_study_folder, f"all_flows_yearly_{scenario_name}.csv")

            if case_study_name.endswith("_Autarky"):
                DPhil_Plotting.plot_yearly_flows_stacked_by_location(my_network, case_study_name, 
                                                                     location_parameters_df, results_folder)
                DPhil_Plotting.plot_dual_install_pathways_all_locations(my_network, network_structure_df, "RE_PV_MY", "RE_WIND_MY", results_folder,
                                                             tech_name_1="PV", tech_name_2="Wind")
                time_series_df, summary_df = Results.export_multi_country_scenario_results(my_network, network_structure_df, scenario_name, simulation_factor)
                time_series_df.to_csv(os.path.join(results_folder, "time_series_results.csv"))
            elif case_study_name.endswith("_Collab"):
                Results.save_yearly_flows_to_csv_multiloc(my_network, location_parameters_df, yearly_path)
                DPhil_Plotting.plot_yearly_flows_stacked_by_location(my_network, case_study_name,
                                                                     location_parameters_df, results_folder)
                DPhil_Plotting.plot_dual_install_pathways_all_locations(my_network, network_structure_df, "RE_PV_MY", "RE_WIND_MY", results_folder,
                                                             tech_name_1="PV", tech_name_2="Wind")
                time_series_df, summary_df = Results.export_multi_country_scenario_results(my_network, network_structure_df, scenario_name, simulation_factor)
                time_series_df.to_csv(os.path.join(results_folder, "summary_results.csv"))
            else:
                Results.save_yearly_flows_to_csv(my_network, yearly_path)
                # DPhil_Plotting.plot_yearly_flows(my_network, results_folder)
                DPhil_Plotting.plot_yearly_flows_stacked(my_network, results_folder)
                DPhil_Plotting.get_dual_install_pathways(my_network.assets[1], my_network.assets[2], results_folder, "PV", "Wind")
        
                time_series_df, summary_df = Results.export_scenario_results(my_network, scenario_name)
                time_series_df.to_csv(os.path.join(results_folder, "time_series_results.csv"))
                summary_df.to_csv(os.path.join(results_folder, "summary_results.csv"))
        
                save_path_lcoe = os.path.join(results_folder, "lcoe_per_year.csv")
                save_path_gef = os.path.join(results_folder, "gef_per_year.csv")
                lcoe = Results.get_lcoe_per_year(my_network, save_path_lcoe)
                grid_emissions_factor = Results.get_grid_intensity(my_network, save_path_gef)
            
                emissions_df = pd.DataFrame(emissions_dict)
                # Save to CSV
                emissions_df.to_csv(os.path.join(scenario_folder, "all_scenarios_emissions.csv"), index=False)
    print("Saved emissions reductions to 'all_scenarios_emissions.csv'")
    runtimes_df = pd.DataFrame(runtimes)
    runtimes_df.to_csv(os.path.join(case_study_folder, "runtimes.csv"), index=False)
    print("Saved scenario runtimes to 'runtimes.csv'")
