#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 10:49:50 2023

@author: Mónica Sagastuy Breña

The functions in this script are hard-coded to export modelling results within the
case studies in the author's DPhil work. They may serve as guidance on how to 
obtain results from the model but changes to the network structure may require editing
of these functions.

Functions here serve to export asset data such as cost and sizes, along with total
emissions, and the power flows per hour for a given case study/scenario. 

They need to be called upon in main.py
"""

import pandas as pd
import numpy as np


def export_results(my_network):
    '''
    This function exports a DataFrame with asset sizes and costs, total throughout
    the project lifetime. 
    
    Parameters
    ----------
    my_network : STEVFNs Network
        Network object created based on the assets in a network structure, timesteps
        and other parameters defined in STEVFNs.

    Returns
    -------
    costs_df : DataFrame with cost and size results collated

    '''
    sizes_df = pd.DataFrame()
    costs_df = pd.DataFrame()
    number_assets = len(my_network.assets)
    
    
    for asset in range(1, number_assets):

        name = my_network.assets[asset].asset_name
        
        ### Exceptions in formatting or extracting results per type of asset ###
        if name == 'BESS' or name == 'NH3_Storage':
            loc1 = my_network.assets[asset].asset_structure["Location_1"]
            costs_df.insert(0, f'{name}_{loc1}_G$', [my_network.assets[asset].cost.value])
            sizes_df.insert(0, f'{name}_{loc1}_GWh', [my_network.assets[asset].asset_size()])        
            
        elif name == 'RE_PV_Rooftop_Lim' or name == 'RE_PV_Openfield_Lim' or name == 'RE_WIND_Onshore_Lim' \
            or name == 'RE_WIND_Offshore_Lim' or name == 'RE_WIND_Onshore_Existing' or name == 'RE_PV_Openfield_Exising':
            loc1 = my_network.assets[asset].target_node_location
            costs_df.insert(0, f'{name}_{loc1}_G$', [my_network.assets[asset].cost.value])
            sizes_df.insert(0, f'{name}_{loc1}_GW', [my_network.assets[asset].asset_size()])

        elif name == 'EL_Demand' or name == 'HTH_Demand':
            loc1 = my_network.assets[asset].node_location
            costs_df.insert(0, f'{name}_{loc1}_G$', [my_network.assets[asset].cost.value])
            sizes_df.insert(0, f'{name}_{loc1}_GWh', [my_network.assets[asset].asset_size()])
            
        elif name == 'EL_Transport':
            loc1 = my_network.assets[asset].asset_structure["Location_1"]
            loc2 = my_network.assets[asset].asset_structure["Location_2"]
            costs_df.insert(0, f'{name}_{loc1}-{loc2}_G$', [my_network.assets[asset].cost.value])
            sizes_df.insert(0, f'{name}_{loc1}-{loc2}_GW', [my_network.assets[asset].asset_size()])
        elif name == 'NH3_Transport':
            loc1 = my_network.assets[asset].asset_structure["Location_1"]
            loc2 = my_network.assets[asset].asset_structure["Location_2"]
            costs_df.insert(0, f'{name}_{loc1}-{loc2}_G$', [my_network.assets[asset].cost.value])
            sizes_df.insert(0, f'{name}_{loc1}-{loc2}_kt_NH3', [my_network.assets[asset].asset_size()])
        
        ### The rest of the assets, in general ###
        else:
            loc1 = my_network.assets[asset].asset_structure["Location_1"]
            costs_df.insert(0, f'{name}_{loc1}_G$', [my_network.assets[asset].cost.value])
            sizes_df.insert(0, f'{name}_{loc1}_GW', [my_network.assets[asset].asset_size()])
    
    costs_df.insert(0, 'Total_System_Cost', [my_network.problem.value])
    sizes_df.insert(0, 'CO2_Budget_GgCO2',  [my_network.assets[0].asset_size()])    
    
    ## hardcoded for better format output 
    index = list(range(number_assets))
    costs_df = costs_df.T
    costs_df['Number'] = index
    costs_df['Asset Name'] = costs_df.index
    costs_df = costs_df.set_index(costs_df['Number'])
    costs_df = costs_df.drop('Number', axis=1)
    costs_df = costs_df.rename(columns={0: "Costs"})
    
    sizes_df = sizes_df.T
    sizes_df['Number'] = index
    sizes_df['Asset Name'] = sizes_df.index
    sizes_df = sizes_df.set_index(sizes_df['Number'])
    sizes_df = sizes_df.drop('Number', axis=1)
    sizes_df = sizes_df.rename(columns={0: "Sizes"})
    
    costs_df = pd.concat([costs_df, sizes_df], axis=1)
    
    return costs_df

def get_total_data_rounded(my_network, location_parameters_df, asset_parameters_df):
    '''
    This function exports data for the case study being run for assets per location
    including asset size, asset cost and the country(ies) total emissions per scenario
    
    It needs to be run within the scenario loop and concatenated to another dataframe
    to export all results in a case study together.

    Parameters
    ----------
    my_network : STEVFNs Network
        Network object created based on the assets in a network structure, timesteps
        and other parameters defined in STEVFNs.
    location_parameters_df : DataFrame
        Obtained by reading Location_Parameters.csv file in a case study, coded in main.
    asset_parameters_df : DataFrame
        Obtained by reading Asset_Parameters.csv file in a case study, coded in main.

    Returns
    -------
    total_data_df : DataFrame
        Results compiled for sets of countries in a case study (either autarky or
        collaboration) for size, cost, and emissions. Rounds the values to one decimal

    '''
    location_names = list(location_parameters_df["location_name"])
    loc_names_set_list = list(set(asset_parameters_df["Location_1"]).union(set(asset_parameters_df["Location_2"])))
    loc_names_list = ["",]*4
    for counter1 in range(len(loc_names_set_list)):
        loc_names_list[counter1] = location_names[loc_names_set_list[counter1]]
    
    total_data_columns = ["country_1",
                  "country_2",
                  "country_3",
                  "country_4",
                  "collaboration_emissions",
                  "technology_cost",
                  "technology_size",
                  "technology_name",]
    total_data_df = pd.DataFrame(columns = total_data_columns)
    
    # Hardcoded CO2_Budget Asset always in Network Structure as asset 0
    collaboration_emissions =  my_network.assets[0].asset_size()
    loc_names_set = set()
    
    storage_assets = ['BESS', 'NH3_Storage']
    renewable_assets = ['RE_PV_Rooftop_Lim', 'RE_PV_Openfield_Lim', 'RE_WIND_Onshore_Lim',
                        'RE_WIND_Offshore_Lim', 'RE_WIND_Onshore_Existing', 'RE_PV_Openfield_Exising']
    demand_assets = ['EL_Demand', 'HTH_Demand']
    transport_assets= ['EL_Transport', 'NH3_Transport']
    
    for counter1 in range(1,len(my_network.assets)):
        asset = my_network.assets[counter1]
        name = asset.asset_name
        
        ### Exceptions in formatting or extracting results per type of asset ###
        if name in storage_assets:
            loc1 = asset.asset_structure["Location_1"]
            loc_name = location_names[loc1]
            loc_names_set.add(loc_name)
            technology_name = name + r"_[" + loc_name + r"]"
            technology_cost = asset.cost.value
            technology_size = asset.asset_size()
            
        elif name in renewable_assets:
            loc1 = asset.target_node_location
            loc_name = location_names[loc1]
            loc_names_set.add(loc_name)
            technology_name = name + r"_[" + loc_name + r"]"
            technology_cost =  asset.cost.value
            technology_size = asset.asset_size()

        elif name in demand_assets:
            loc1 = asset.node_location
            loc_name = location_names[loc1]
            loc_names_set.add(loc_name)
            technology_name = name + r"_[" + loc_name + r"]"
            technology_cost = asset.cost.value
            technology_size = asset.asset_size()
            
        elif name in transport_assets:
            loc1 = asset.asset_structure["Location_1"]
            loc2 = asset.asset_structure["Location_2"]
            loc_name_1 = location_names[loc1]
            loc_name_2 = location_names[loc2]
            loc_names_set.add(loc_name_1)
            loc_names_set.add(loc_name_1)
            technology_name = name + r"_[" + loc_name_1 + r"-" + loc_name_2 + r"]"
            technology_cost = asset.cost.value
            technology_size = asset.asset_size()
        ### The rest of the assets, in general ###
        else:
            loc1 = asset.asset_structure["Location_1"]
            loc_name = location_names[loc1]
            loc_names_set.add(loc_name)
            technology_name = name + r"_[" + loc_name + r"]"
            technology_cost = asset.cost.value
            technology_size = asset.asset_size()
        
        N = np.ceil(my_network.system_parameters_df.loc["project_life", "value"]/8760) #number of years for the project
        t_df = pd.DataFrame({"country_1": [loc_names_list[0]],
                             "country_2": [loc_names_list[1]],
                             "country_3": [loc_names_list[2]],
                             "country_4": [loc_names_list[3]],
                             "collaboration_emissions_MtCO2e/y": [round(collaboration_emissions/N, 1)],# Number is annualized, number is converted from ktCO2e to MtCO2e
                             "technology_cost_G$/y": [round(technology_cost/N, 1)],# Number is annualized
                             "technology_size": [technology_size],
                             "technology_name": [technology_name],  
            })
        total_data_df = pd.concat([total_data_df, t_df], ignore_index=True)
    return total_data_df


def get_total_data(my_network, location_parameters_df, asset_parameters_df):
    '''
    This function exports data for the case study being run for assets per location
    including asset size, asset cost and the country(ies) total emissions per scenario
    
    It needs to be run within the scenario loop and concatenated to another dataframe
    to export all results in a case study together.
    
    Parameters
    ----------
    my_network : STEVFNs Network
        Network object created based on the assets in a network structure, timesteps
        and other parameters defined in STEVFNs.
    location_parameters_df : DataFrame
        Obtained by reading Location_Parameters.csv file in a case study, coded in main.
    asset_parameters_df : DataFrame
        Obtained by reading Asset_Parameters.csv file in a case study, coded in main.
    
    Returns
    -------
    total_data_df : DataFrame
        Results compiled for sets of countries in a case study (either autarky or
        collaboration) for size, cost, and emissions. Does NOT round values
    '''
    location_names = list(location_parameters_df["location_name"])
    loc_names_set_list = list(set(asset_parameters_df["Location_1"]).union(set(asset_parameters_df["Location_2"])))
    loc_names_list = ["",]*4
    for counter1 in range(len(loc_names_set_list)):
        loc_names_list[counter1] = location_names[loc_names_set_list[counter1]]
    
    total_data_columns = ["country_1",
                  "country_2",
                  "country_3",
                  "country_4",
                  "collaboration_emissions",
                  "technology_cost",
                  "technology_name",]
    total_data_df = pd.DataFrame(columns = total_data_columns)
    
    collaboration_emissions =  my_network.assets[0].asset_size()
    loc_names_set = set()
    
    # Define asset types to distinguish formatting
    storage_assets = ['BESS', 'NH3_Storage']
    renewable_assets = ['RE_PV_Rooftop_Lim', 'RE_PV_Openfield_Lim', 'RE_WIND_Onshore_Lim',
                        'RE_WIND_Offshore_Lim', 'RE_WIND_Onshore_Existing', 'RE_PV_Openfield_Exising']
    demand_assets = ['EL_Demand', 'HTH_Demand']
    transport_assets= ['EL_Transport', 'NH3_Transport']
    
    for counter1 in range(1,len(my_network.assets)):
        asset = my_network.assets[counter1]
        name = asset.asset_name
        
        ### Exceptions in formatting or extracting results per type of asset ###
        if name in storage_assets:
            loc1 = asset.asset_structure["Location_1"]
            loc_name = location_names[loc1]
            loc_names_set.add(loc_name)
            technology_name = name + r"_[" + loc_name + r"]"
            technology_cost = asset.cost.value
            technology_size = asset.asset_size()
            
        elif name in renewable_assets:
            loc1 = asset.target_node_location
            loc_name = location_names[loc1]
            loc_names_set.add(loc_name)
            technology_name = name + r"_[" + loc_name + r"]"
            technology_cost =  asset.cost.value
            technology_size = asset.asset_size()

        elif name in demand_assets:
            loc1 = asset.node_location
            loc_name = location_names[loc1]
            loc_names_set.add(loc_name)
            technology_name = name + r"_[" + loc_name + r"]"
            technology_cost = asset.cost.value
            technology_size = asset.asset_size()
            
        elif name in transport_assets:
            loc1 = asset.asset_structure["Location_1"]
            loc2 = asset.asset_structure["Location_2"]
            loc_name_1 = location_names[loc1]
            loc_name_2 = location_names[loc2]
            loc_names_set.add(loc_name_1)
            loc_names_set.add(loc_name_1)
            technology_name = name + r"_[" + loc_name_1 + r"-" + loc_name_2 + r"]"
            technology_cost = asset.cost.value
            technology_size = asset.asset_size()
        
        ### The rest of the assets, in general ###
        else:
            loc1 = asset.asset_structure["Location_1"]
            loc_name = location_names[loc1]
            loc_names_set.add(loc_name)
            technology_name = name + r"_[" + loc_name + r"]"
            technology_cost = asset.cost.value
            technology_size = asset.asset_size()
        
        N = np.ceil(my_network.system_parameters_df.loc["project_life", "value"]/8760) #number of years for the project
        t_df = pd.DataFrame({"country_1": [loc_names_list[0]],
                             "country_2": [loc_names_list[1]],
                             "country_3": [loc_names_list[2]],
                             "country_4": [loc_names_list[3]],
                             "collaboration_emissions_MtCO2e/y": [collaboration_emissions/N],# Number is annualized, number is converted from ktCO2e to MtCO2e
                             "technology_cost_G$/y": [technology_cost/N],# Number is annualized
                             "technology_size": [technology_size],
                             "technology_name": [technology_name],
            })
        total_data_df = pd.concat([total_data_df, t_df], ignore_index=True)
    return total_data_df

def export_aut_flows(my_network):
    '''
    This function exports the flows of the network in autarky, (single-country)
    case studies. 
    
    Parameters
    ----------
    my_network : STEVFNs network
        Full network after running a given system

    Returns
    -------
    Dataframe of flows for Autarky Case Study
    '''
    # Initialize an empty list to hold DataFrames for each asset
    data_frames = []

    # Iterate over each asset and collect flow data
    for i in range(1, len(my_network.assets)):
        asset = my_network.assets[i]
        name = asset.asset_name
        
        if name == 'EL_Demand_UM':
            demand_data = asset.assets_dictionary['Net_EL_Demand'].flows.value
            df = pd.DataFrame({"Net_demand": demand_data})
            data_frames.append(df)
        
        elif name == 'EL_Demand':
            demand_data = asset.flows.value
            df = pd.DataFrame({"Net_demand": demand_data})
            data_frames.append(df)
        
        elif name == 'BESS':
            BESS_ch = asset.assets_dictionary['Charging'].flows.value
            BESS_disch = asset.assets_dictionary['Discharging'].flows.value
            df = pd.DataFrame({"BESS_Charging": BESS_ch, "BESS_Discharging": BESS_disch})
            data_frames.append(df)
        
        elif name in ['PP_CO2', 'PP_CO2_Existing']:
            pp_data = asset.flows.value
            df = pd.DataFrame({"PP_total": pp_data})
            data_frames.append(df)
        
        elif name in ['RE_PV_Existing', 'RE_PV_Openfield_Lim']:
            pv_data = asset.get_plot_data()
            df = pd.DataFrame({"PV_total": pv_data})
            data_frames.append(df)
        
        elif name in ['RE_Wind_Existing', 'RE_WIND_Onshore_Lim']:
            wind_data = asset.get_plot_data()
            df = pd.DataFrame({"Wind_total": wind_data})
            data_frames.append(df)

    flows_df = pd.concat(data_frames, axis=1)
    
    return flows_df
    
def export_collab_flows(my_network, location_parameters_df):
    '''
    This function exports the flows of the network in autarky and collaboration
    forms for multiple country configuration case studies.
    
    Parameters
    ----------
    my_network : STEVFNs network
        Full network after running a given system
    location_parameters_df : DataFrame
        From Location_Parameters.csv in a scenario folder
    Returns
    -------
    Dataframe of flows for Collab Case Studies
    '''
    # Extract timesteps being run
    timesteps = my_network.system_structure_properties["simulated_timesteps"]

    # Initialize an empty list to hold individual DataFrames for each asset and location
    data_frames = []

    # Extract hourly arrays of flows for each asset
    for i in range(1, len(my_network.assets)):
        asset = my_network.assets[i]
        name = asset.asset_name
        
        loc1 = asset.node_location if 'node_location' in dir(asset) \
            else asset.asset_structure.get("Location_1")
        if loc1 not in location_parameters_df.index:
            continue  # Skip if the location is not in the location parameters DataFrame

        loc_name = location_parameters_df.loc[loc1, "location_name"]

        if name == 'EL_Demand_UM' or name == 'EL_Demand':
            demand_name = f"{name}_[{loc_name}]"
            demand_data = asset.assets_dictionary['Net_EL_Demand'].flows.value \
                if name == 'EL_Demand_UM' else asset.flows.value
            df = pd.DataFrame({demand_name: demand_data})
            data_frames.append(df)

        elif name == 'BESS':
            bess_ch_name = f"{name}ch_[{loc_name}]"
            bess_disch_name = f"{name}disch_[{loc_name}]"
            BESS_ch = asset.assets_dictionary['Charging'].flows.value
            BESS_disch = asset.assets_dictionary['Discharging'].flows.value
            df = pd.DataFrame({bess_ch_name: BESS_ch, bess_disch_name: BESS_disch})
            data_frames.append(df)

        elif name in ['PP_CO2', 'PP_CO2_Existing']:
            pp_name = f"{name}_[{loc_name}]"
            pp_data = asset.flows.value
            df = pd.DataFrame({pp_name: pp_data})
            data_frames.append(df)

        elif name in ['RE_PV_Exiting', 'RE_PV_Openfield_Lim']:
            pv_name = f"{name}_[{loc_name}]"
            pv_data = asset.get_plot_data()
            df = pd.DataFrame({pv_name: pv_data})
            data_frames.append(df)

        elif name in ['RE_Wind_Existing', 'RE_WIND_Onshore_Lim']:
            wind_name = f"{name}_[{loc_name}]"
            wind_data = asset.get_plot_data()
            df = pd.DataFrame({wind_name: wind_data})
            data_frames.append(df)

        elif name == 'EL_Transport':
            loc2 = asset.asset_structure["Location_2"]
            loc_name_2 = location_parameters_df.loc[loc2, "location_name"]
            transport_out_name = f"{name}_[{loc_name}-{loc_name_2}]"
            transport_in_name = f"{name}_[{loc_name_2}-{loc_name}]"
            HVDC_out = asset.flows.value[:timesteps]
            HVDC_in = asset.flows.value[timesteps:(2*timesteps)]
            df = pd.DataFrame({transport_out_name: HVDC_out, transport_in_name: HVDC_in})
            data_frames.append(df)

    # Concatenate all data frames along columns to form a single output DataFrame
    flows_df = pd.concat(data_frames, axis=1)

    return flows_df

def calculate_curtailment_aut(my_network):
    '''
    Parameters
    ----------
    my_network : STEVFNs network
        Full network after running a given system

    Returns
    -------
    DataFrame
        Dataframe of curtailment data, hourly
    '''
    # Extract timesteps in network structure
    timesteps = my_network.system_structure_properties["simulated_timesteps"]

    total_generation = np.zeros(timesteps)
    total_demand = np.zeros(timesteps)
    total_outflows = np.zeros(timesteps)
    total_storage_discharge = np.zeros(timesteps)
    pv_generation = np.zeros(timesteps)
    wind_generation = np.zeros(timesteps)

    for i in range(1, len(my_network.assets)):
        asset = my_network.assets[i]
        name = asset.asset_name

        if name in ['RE_PV_Exiting', 'RE_PV_Openfield_Lim']:
            pv_data = asset.get_plot_data() if 'get_plot_data' in dir(asset) else asset.flows.value
            pv_generation += pv_data
            total_generation += pv_data

        elif name in ['RE_Wind_Existing', 'RE_WIND_Onshore_Lim']:
            wind_data = asset.get_plot_data() if 'get_plot_data' in dir(asset) else asset.flows.value
            wind_generation += wind_data
            total_generation += wind_data

        elif name in ['PP_CO2', 'PP_CO2_Existing']:
            generation_data = asset.flows.value
            total_generation += generation_data

        elif name == 'EL_Demand_UM' or name == 'EL_Demand':
            demand_data = asset.assets_dictionary['Net_EL_Demand'].flows.value if name == 'EL_Demand_UM' else asset.flows.value
            total_demand += demand_data

        elif name == 'BESS':
            BESS_charging = asset.assets_dictionary['Charging'].flows.value
            BESS_discharging = asset.assets_dictionary['Discharging'].flows.value
            total_outflows += BESS_charging
            total_storage_discharge += BESS_discharging

    # Calculate curtailment as the excess generation at each hour
    curtailment = (total_generation + total_storage_discharge) - (total_demand + total_outflows)
    curtailment[curtailment < 0] = 0  # Ensure curtailment is non-negative

    # Determine curtailment contributions from PV and Wind, assuming PV is curtailed first
    pv_curtailment = np.minimum(curtailment, pv_generation)
    wind_curtailment = np.minimum(curtailment - pv_curtailment, wind_generation)

    # Create a DataFrame for curtailment
    curtailment_df = pd.DataFrame({
        "Total_Curtailment": curtailment,
        "PV_Curtailment": pv_curtailment,
        "Wind_Curtailment": wind_curtailment
    })

    return curtailment_df

def calculate_curtailment_collab(my_network):
    '''
    Parameters
    ----------
    my_network : STEVFNs network
        Full network after running a given system in collaboration

    Returns
    -------
    DataFrame
        Dataframe of curtailment data, hourly
    '''
    # Extract timesteps in network structure
    timesteps = my_network.system_structure_properties["simulated_timesteps"]

    total_generation = np.zeros(timesteps)
    total_demand = np.zeros(timesteps)
    total_outflows = np.zeros(timesteps)
    total_storage_discharge = np.zeros(timesteps)
    pv_generation = np.zeros(timesteps)
    wind_generation = np.zeros(timesteps)
    hvdc_in = np.zeros(timesteps)
    hvdc_out = np.zeros(timesteps)

    for i in range(1, len(my_network.assets)):
        asset = my_network.assets[i]
        name = asset.asset_name

        if name in ['RE_PV_Exiting', 'RE_PV_Openfield_Lim']:
            pv_data = asset.get_plot_data() if 'get_plot_data' in dir(asset) else asset.flows.value
            pv_generation += pv_data
            total_generation += pv_data

        elif name in ['RE_Wind_Existing', 'RE_WIND_Onshore_Lim']:
            wind_data = asset.get_plot_data() if 'get_plot_data' in dir(asset) else asset.flows.value
            wind_generation += wind_data
            total_generation += wind_data

        elif name in ['PP_CO2', 'PP_CO2_Existing']:
            generation_data = asset.flows.value
            total_generation += generation_data

        elif name == 'EL_Demand_UM' or name == 'EL_Demand':
            demand_data = asset.assets_dictionary['Net_EL_Demand'].flows.value if name == 'EL_Demand_UM' else asset.flows.value
            total_demand += demand_data

        elif name == 'BESS':
            BESS_charging = asset.assets_dictionary['Charging'].flows.value
            BESS_discharging = asset.assets_dictionary['Discharging'].flows.value
            total_outflows += BESS_charging
            total_storage_discharge += BESS_discharging

        elif name == 'EL_Transport':
            HVDC_out = asset.flows.value[:timesteps]
            total_outflows += HVDC_out

    
    # Calculate curtailment as the excess generation at each hour
    curtailment = (total_generation + total_storage_discharge) - (total_demand + total_outflows)
    curtailment[curtailment < 0] = 0  # Ensure curtailment is non-negative

    # Determine curtailment contributions from PV and Wind, assuming PV is curtailed first
    pv_curtailment = np.minimum(curtailment, pv_generation)
    wind_curtailment = np.minimum(curtailment - pv_curtailment, wind_generation)

    # Create a DataFrame for curtailment
    curtailment_df = pd.DataFrame({
        "Total_Curtailment": curtailment,
        "PV_Curtailment": pv_curtailment,
        "Wind_Curtailment": wind_curtailment
    })

    return curtailment_df



