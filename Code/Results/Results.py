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


def export_scenario_results(my_network, scenario_name):
    print("========= Exporting summary scenario results ========")
    num_years = my_network.assets[0].num_years
    years = list(range(1, num_years + 1))
    print(f"Expected number of years: {num_years}")
    discount_rate = float(my_network.system_parameters_df.loc["discount_rate", "value"])
    discount_factors = 1 / ((1 + discount_rate) ** np.arange(num_years))
    # Initialize base dictionary
    data = {}

    def safe_assign(key, value):
        try:
            if hasattr(value, '__len__') and not isinstance(value, str):
                val_len = len(value)
                print(f"Assigning '{key}' with length {val_len}")
                if val_len != num_years:
                    print(f"  ❗ Length mismatch for '{key}': got {val_len}, expected {num_years}")
            else:
                print(f"Assigning '{key}' (non-list-like or scalar)")
            data[key] = value
        except Exception as e:
            print(f"❌ Error assigning '{key}': {e}")
            raise

    safe_assign("year", years)
    safe_assign("scenario", [scenario_name] * num_years)
    safe_assign("Annual_Emissions", [0] * num_years)
    safe_assign("Peak_Annual_Demand", [0] * num_years)
    safe_assign("Total_Annual_Demand", [0] * num_years)
    safe_assign("Discounted_Annual_Demand", [0] * num_years)
    safe_assign("Peak_Annual_Fossil_Gen_GWp", [0] * num_years)
    safe_assign("Total_Annual_Fossil_Gen_GWh", [0] * num_years)
    safe_assign("Discounted_Annual_Fossil_Gen_GWh", [0] * num_years)
    safe_assign("Annual_NPV_OPEX_PP_BUSD", [0] * num_years)

    for asset in my_network.assets[1:]:
        name = asset.asset_name
        print(f"\nProcessing asset: {name}")

        if hasattr(asset, "peak_demand"):
            safe_assign("Peak_Annual_Demand", asset.peak_demand())
            safe_assign("Total_Annual_Demand", asset.asset_size())
            safe_assign("Discounted_Annual_Demand", asset.asset_size() * discount_factors)

        if hasattr(asset, "get_yearly_emissions"):
            safe_assign("Annual_Emissions", asset.get_yearly_emissions())
            safe_assign("Peak_Annual_Fossil_Gen_GWp", asset.peak_generation())
            flows = asset.get_yearly_flows()[:num_years]
            summed_flows = [sum(year) for year in flows]
            safe_assign("Total_Annual_Fossil_Gen_GWh", summed_flows)
            discounted_flows = summed_flows * discount_factors
            safe_assign("Discounted_Annual_Fossil_Gen_GWh", discounted_flows.tolist())
            safe_assign("Annual_NPV_OPEX_PP_BUSD", asset.get_yearly_usage_costs())

        if hasattr(asset, "cumulative_new_installed"):
            flows = getattr(asset.flows, "value", asset.flows)
            cumulative_installed = asset.cumulative_new_installed.value
            existing_capacity_val = asset.conversion_fun_params["existing_capacity"].value

            payments_M = getattr(asset, "payments_M", None)
            if payments_M is not None:
                payments = getattr(payments_M, "value", payments_M)
                annual_payments = np.sum(payments, axis=1) if payments is not None else [0] * num_years
            else:
                annual_payments = [0] * num_years

            safe_assign(f"{name}_new_annual_installed_GWp", flows.tolist())
            safe_assign(f"{name}_total_capacity_GWp", (cumulative_installed + existing_capacity_val).tolist())
            safe_assign(f"{name}_annual_payments_BUSD", annual_payments)
            annual_gen = np.sum(asset.get_yearly_flows(), axis=1)
            safe_assign(f"{name}_annual_generation_GWh", annual_gen)
            safe_assign(f"{name}_discounted_annual_gen_GWh", annual_gen * discount_factors)

    print("\n✅ All data lengths checked. Creating DataFrame...")
    time_series_df = pd.DataFrame(data)

    # Cost summary
    system_cost = my_network.problem.value
    cost_summary = []
    for asset in my_network.assets[1:]:
        cost_val = getattr(asset.cost, "value", asset.cost)
        cost_summary.append({
            "scenario": scenario_name,
            "total_system_cost": system_cost,
            "asset_name": asset.asset_name,
            "asset_cost": cost_val,
        })

    summary_df = pd.DataFrame(cost_summary)

    return time_series_df, summary_df


def export_multi_country_scenario_results(my_network, network_structure_df, scenario_name, simulation_factor):
    import numpy as np
    import pandas as pd

    print("========= Exporting multi-country scenario results ========")
    num_years = my_network.assets[0].num_years
    years = list(range(1, num_years + 1))
    discount_rate = float(my_network.system_parameters_df.loc["discount_rate", "value"])
    discount_factors = 1 / ((1 + discount_rate) ** np.arange(num_years))

    data = {}

    def safe_assign(key, value):
        try:
            if hasattr(value, '__len__') and not isinstance(value, str):
                val_len = len(value)
                print(f"Assigning '{key}' with length {val_len}")
                if val_len != num_years:
                    print(f"  ❗ Length mismatch for '{key}': got {val_len}, expected {num_years}")
            else:
                print(f"Assigning '{key}' (non-list-like or scalar)")
            data[key] = value
        except Exception as e:
            print(f"❌ Error assigning '{key}': {e}")
            raise

    # def safe_add(key, values):
    #     if key not in data:
    #         data[key] = list(values)
    #     else:
    #         data[key] = [old + new for old, new in zip(data[key], values)]

    # Base info
    safe_assign("year", years)
    safe_assign("scenario", [scenario_name] * num_years)

    # Detect locations
    target_locs = sorted({getattr(a, "target_node_location") for a in my_network.assets[1:] if hasattr(a, "target_node_location")})
    demand_locs = sorted({getattr(a, "node_location") for a in my_network.assets[1:] if hasattr(a, "node_location")})

    # Init per-location columns
    for loc in target_locs:
        safe_assign(f"Fossil_Gen_GWh_loc{loc}", [0] * num_years)
        safe_assign(f"Discounted_Fossil_Gen_GWh_loc{loc}", [0] * num_years)
        safe_assign(f"Annual_Emissions_loc{loc}", [0] * num_years)
    for loc in demand_locs:
        safe_assign(f"Total_Annual_Demand_loc{loc}", [0] * num_years)
        safe_assign(f"Discounted_Annual_Demand_loc{loc}", [0] * num_years)

    # Process assets
    for asset in my_network.assets[1:]:
        name = asset.asset_name

        # Demand assets
        if hasattr(asset, "peak_demand") and hasattr(asset, "node_location"):
            loc = asset.node_location
            total_annual_demand = [d * simulation_factor for d in asset.asset_size()]
            safe_assign(f"Total_Annual_Demand_loc{loc}", total_annual_demand)
            discounted_demand = total_annual_demand * discount_factors
            safe_assign(f"Discounted_Annual_Demand_loc{loc}", discounted_demand)

        # Generation assets
        if hasattr(asset, "get_yearly_emissions") and hasattr(asset, "target_node_location"):
            loc = asset.target_node_location
            yearly_emissions = asset.get_yearly_emissions()
            safe_assign(f"Annual_Emissions_loc{loc}", yearly_emissions)

            flows = asset.get_yearly_flows()[:num_years]
            summed_flows = [(sum(year) * simulation_factor) for year in flows]
            safe_assign(f"Fossil_Gen_GWh_loc{loc}", summed_flows)
            discounted_flows = summed_flows * discount_factors
            safe_assign(f"Discounted_Fossil_Gen_GWh_loc{loc}", discounted_flows)
            safe_assign(f"Annual_NPV_OPEX_PP_BUSD_loc{loc}", asset.get_yearly_usage_costs())

        # Capacity, payments, generation per asset
        if hasattr(asset, "cumulative_new_installed"):
            loc = getattr(asset, "target_node_location", getattr(asset, "node_location", "NA"))
            flows = np.array(getattr(asset.flows, "value", asset.flows), dtype=float)
            cumulative_installed = np.array(asset.cumulative_new_installed.value, dtype=float)
            existing_capacity_val = np.array(asset.conversion_fun_params["existing_capacity"].value, dtype=float)

            payments_M = getattr(asset, "payments_M", None)
            if payments_M is not None:
                payments = getattr(payments_M, "value", payments_M)
                annual_payments = np.sum(payments, axis=1) if payments is not None else [0] * num_years
            else:
                annual_payments = [0] * num_years

            safe_assign(f"{name}_new_annual_installed_GWp_loc{loc}", flows.tolist())
            safe_assign(f"{name}_total_capacity_GWp_loc{loc}", (cumulative_installed + existing_capacity_val).tolist())
            safe_assign(f"{name}_annual_payments_BUSD_loc{loc}", annual_payments)

            annual_gen = np.sum(asset.get_yearly_flows(), axis=1)
            annual_gen = [g * simulation_factor for g in annual_gen]
            safe_assign(f"{name}_annual_generation_GWh_loc{loc}", annual_gen)
            safe_assign(f"{name}_discounted_annual_gen_GWh_loc{loc}", annual_gen * discount_factors)

    # Create dataframe
    time_series_df = pd.DataFrame(data)
    # Get all locations from network_structure_df
    locations = sorted(network_structure_df["Location_1"].unique())
    
    # Find all generation columns dynamically
    gen_cols = [col for col in time_series_df.columns if "_discounted_annual_gen_GWh_loc" in col or "Discounted_Fossil_Gen_GWh_loc" in col]
    
    # Payments and OPEX columns
    payment_cols = [
        col
        for loc in locations
        for col in time_series_df.columns
        if col.endswith(f"_annual_payments_BUSD_loc{loc}")
    ]

    opex_cols = [
        col
        for loc in locations
        for col in time_series_df.columns
        if col == f"Annual_NPV_OPEX_PP_BUSD_loc{loc}"
    ]
    
    demand_cols = [
        col
        for loc in locations
        for col in time_series_df.columns
        if col == f"Discounted_Annual_Demand_loc{loc}"
    ]
    # Find generation columns
    gen_cols = [
        col
        for loc in locations
        for col in time_series_df.columns
        if col.endswith(f"_discounted_annual_gen_GWh_loc{loc}")
        or col == f"Discounted_Fossil_Gen_GWh_loc{loc}"
    ]
    
    # Totals
    total_gen = time_series_df[gen_cols].sum(axis=1)
    total_payments = time_series_df[payment_cols].sum(axis=1)
    total_opex = time_series_df[opex_cols].sum(axis=1)
    total_demand = time_series_df[demand_cols].sum(axis=1)
    
    # LCOE
    time_series_df["System_LCOE_USD_per_kWh"] = (
        (total_payments + total_opex) / total_gen
    ) * 1000
    
    # LCUE
    time_series_df["System_LCUE_USD_per_kWh"] = (
        (total_payments + total_opex) / total_demand
    ) * 1000

    # Cost summary
    system_cost = my_network.problem.value
    cost_summary = []
    for asset in my_network.assets[1:]:
        cost_val = getattr(asset.cost, "value", asset.cost)
        cost_summary.append({
            "scenario": scenario_name,
            "total_system_cost": system_cost,
            "asset_name": asset.asset_name,
            "asset_cost": cost_val,
        })
    summary_df = pd.DataFrame(cost_summary)

    return time_series_df, summary_df




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
            
        elif name == 'RE_PV_MY' or name == 'RE_WIND_MY':
            loc1 = my_network.assets[asset].target_node_location
            costs_df.insert(0, f'{name}_{loc1}_G$', [my_network.assets[asset].cost.value])
            sizes_df.insert(0, f'{name}_{loc1}_GWp', [my_network.assets[asset].asset_size()])

        elif name == 'EL_Demand':
            loc1 = my_network.assets[asset].node_location
            sizes_df.insert(0, f'{name}_{loc1}_GWp', [my_network.assets[asset].asset_size()])
            
        elif name == 'EL_Transport':
            loc1 = my_network.assets[asset].asset_structure["Location_1"]
            loc2 = my_network.assets[asset].asset_structure["Location_2"]
            costs_df.insert(0, f'{name}_{loc1}-{loc2}_G$', [my_network.assets[asset].cost.value])
            sizes_df.insert(0, f'{name}_{loc1}-{loc2}_GWp', [my_network.assets[asset].asset_size()])
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
    loc_names_list = ["",]*3
    for counter1 in range(len(loc_names_set_list)):
        loc_names_list[counter1] = location_names[loc_names_set_list[counter1]]
    
    total_data_columns = ["country_1",
                  "country_2",
                  "country_3",
                  "collaboration_emissions",
                  "technology_cost",
                  "technology_name",]
    total_data_df = pd.DataFrame(columns = total_data_columns)
    
    collaboration_emissions =  my_network.assets[0].asset_size() # this is now a list of self.num_years elements
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
            technology_size = asset.asset_size() # this is now a list of self.num_years elements

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

def calculate_curtailment_collab(my_network, location_parameters_df):
    '''
    Parameters
    ----------
    my_network : STEVFNs network
        Full network after running a given system in collaboration.
    location_parameters_df : DataFrame
        A DataFrame with location parameters, indexed by location IDs.

    Returns
    -------
    DataFrame
        Dataframe of curtailment data by location, hourly.
    '''
    # Extract timesteps in network structure
    timesteps = my_network.system_structure_properties["simulated_timesteps"]
    
    # Initialize dictionaries to store results for each location
    curtailment_data = {}
    
    # Process assets by location
    for i in range(1, len(my_network.assets)):
        asset = my_network.assets[i]
        name = asset.asset_name

        # Determine the location of the asset
        loc1 = asset.node_location if 'node_location' in dir(asset) \
            else asset.asset_structure.get("Location_1")
        if loc1 not in location_parameters_df.index:
            continue  # Skip if the location is not in the location parameters DataFrame

        loc_name = location_parameters_df.loc[loc1, "location_name"]

        # Initialize location-specific data structure if not already present
        if loc_name not in curtailment_data:
            curtailment_data[loc_name] = {
                "Total_Generation": np.zeros(timesteps),
                "Total_Demand": np.zeros(timesteps),
                "Total_Outflows": np.zeros(timesteps),
                "Total_Storage_Discharge": np.zeros(timesteps),
                "PV_Generation": np.zeros(timesteps),
                "Wind_Generation": np.zeros(timesteps),
                "HVDC_In": np.zeros(timesteps),
            }

        # Fetch generation or consumption data
        if name in ['RE_PV_Exiting', 'RE_PV_Openfield_Lim']:
            pv_data = asset.get_plot_data() if 'get_plot_data' in dir(asset) else asset.flows.value
            curtailment_data[loc_name]["PV_Generation"] += pv_data
            curtailment_data[loc_name]["Total_Generation"] += pv_data

        elif name in ['RE_Wind_Existing', 'RE_WIND_Onshore_Lim']:
            wind_data = asset.get_plot_data() if 'get_plot_data' in dir(asset) else asset.flows.value
            curtailment_data[loc_name]["Wind_Generation"] += wind_data
            curtailment_data[loc_name]["Total_Generation"] += wind_data

        elif name in ['PP_CO2', 'PP_CO2_Existing']:
            generation_data = asset.flows.value
            curtailment_data[loc_name]["Total_Generation"] += generation_data

        elif name == 'EL_Demand_UM' or name == 'EL_Demand':
            demand_data = asset.assets_dictionary['Net_EL_Demand'].flows.value if name == 'EL_Demand_UM' else asset.flows.value
            curtailment_data[loc_name]["Total_Demand"] += demand_data

        elif name == 'BESS':
            BESS_charging = asset.assets_dictionary['Charging'].flows.value
            BESS_discharging = asset.assets_dictionary['Discharging'].flows.value
            curtailment_data[loc_name]["Total_Outflows"] += BESS_charging
            curtailment_data[loc_name]["Total_Storage_Discharge"] += BESS_discharging

        elif name == 'EL_Transport':
            HVDC_out = asset.flows.value[:timesteps]
            HVDC_in = asset.flows.value[timesteps:timesteps*2]
            curtailment_data[loc_name]["Total_Outflows"] += HVDC_out
            curtailment_data[loc_name]["HVDC_In"] += HVDC_in

    # Calculate curtailment for each location
    results = []
    for loc_name, data in curtailment_data.items():
        # Calculate curtailment as excess generation
        curtailment = (data["Total_Generation"] + data["Total_Storage_Discharge"] + data["HVDC_In"]) - \
                      (data["Total_Demand"] + data["Total_Outflows"])
        curtailment[curtailment < 0] = 0  # Ensure non-negative curtailment

        # Determine curtailment contributions from PV and Wind
        pv_curtailment = np.minimum(curtailment, data["PV_Generation"])
        wind_curtailment = np.minimum(curtailment - pv_curtailment, data["Wind_Generation"])

        # Store results in a structured format
        results.append(pd.DataFrame({
            "Location": loc_name,
            "Hour": np.arange(timesteps),
            "Total_Curtailment": curtailment,
            "PV_Curtailment": pv_curtailment,
            "Wind_Curtailment": wind_curtailment
        }))

    # Combine results into a single DataFrame
    curtailment_df = pd.concat(results, ignore_index=True)

    return curtailment_df


def save_yearly_flows_to_csv(network, output_path):
    """
    Saves all asset flows split by year into a CSV file.
    Each column is labeled as assetname_yN.
    Handles different year lengths by padding with NaN.
    """
    flow_data = {}
    
    for asset in network.assets:
        if not hasattr(asset, "get_yearly_flows"):
            continue  # Skip assets without flow chunk method

        try:
            yearly_chunks = asset.get_yearly_flows()
        except Exception as e:
            print(f"[Skip] {asset.asset_name}: {e}")
            continue

        for year_idx, flow_array in enumerate(yearly_chunks):
            col_name = f"{asset.asset_name}_y{year_idx+1}"
            flow_data[col_name] = np.array(flow_array).flatten()

    # Determine max column length for padding
    max_len = max(len(arr) for arr in flow_data.values())

    # Pad all arrays with np.nan to equal length
    for key in flow_data:
        padded = np.full(max_len, np.nan)
        padded[:len(flow_data[key])] = flow_data[key]
        flow_data[key] = padded

    # Create DataFrame and save
    df = pd.DataFrame(flow_data)
    df.to_csv(output_path, index=False)
    print(f"[✓] Yearly flows saved to {output_path}")
    
def save_yearly_flows_to_csv_multiloc(network, location_parameters_df, output_path):
    """
    Saves all asset flows split by year into a CSV file.
    - HVDC transport assets are labeled as HVDC LocA-LocB_yN.
    - Other assets use their asset name.
    - Handles different year lengths by padding with NaN.
    """
    flow_data = {}

    for asset in network.assets:
        if not hasattr(asset, "get_yearly_flows"):
            continue  # Skip assets without flow chunk method

        name = asset.asset_name.lower()

        try:
            yearly_chunks = asset.get_yearly_flows()
        except Exception as e:
            print(f"[Skip] {asset.asset_name}: {e}")
            continue

        # HVDC / transport case
        if "el_transport" in name:
            df = yearly_chunks  # assuming it's a DataFrame with col names like '0-1_year_5'
            for col in df.columns:
                try:
                    real_year = int(col.split("_year_")[1])
                except Exception:
                    continue

                direction = col.split("_year_")[0]
                try:
                    source_id, target_id = map(int, direction.split("-"))
                except Exception as e:
                    print(f"[Skip] Could not parse source/target from '{direction}': {e}")
                    continue

                source_loc_name = location_parameters_df.iloc[source_id]["location_name"]
                target_loc_name = location_parameters_df.iloc[target_id]["location_name"]

                col_name = f"HVDC {source_loc_name}-{target_loc_name}_y{real_year}"
                flow_data[col_name] = np.array(df[col]).flatten()

        else:
            # Non-transport asset: location info optional
            for year_idx, flow_array in enumerate(yearly_chunks):
                col_name = f"{asset.asset_name}_y{year_idx+1}"
                flow_data[col_name] = np.array(flow_array).flatten()

    # Determine max column length for padding
    if not flow_data:
        print("No flows found to save.")
        return

    max_len = max(len(arr) for arr in flow_data.values())

    # Pad all arrays with NaN to equal length
    for key in flow_data:
        padded = np.full(max_len, np.nan)
        padded[:len(flow_data[key])] = flow_data[key]
        flow_data[key] = padded

    # Create DataFrame and save
    df_out = pd.DataFrame(flow_data)
    df_out.to_csv(output_path, index=False)
    print(f"[✓] Yearly flows saved to {output_path}")


def get_lcoe_per_year(network, output_path=None):
    """
    Returns a DataFrame with annual total discounted energy, cost, and LCOE (USD/MWh).
    Optionally saves this data as CSV.
    """
    print("======== Saving LCOE per year calculation =========")
    
    discount_rate = float(network.system_parameters_df.loc["discount_rate", "value"])
    num_years = network.assets[0].num_years

    total_gen_energy = np.zeros(num_years)
    total_discounted_energy = np.zeros(num_years) # total generated energy
    total_discounted_demand = np.zeros(num_years) # total utilised energy
    total_discounted_cost = np.zeros(num_years)

    years = np.arange(num_years)
    discount_factors = 1 / ((1 + discount_rate) ** years)

    for asset in network.assets:
        try:
            # Determine cost and apply discounting
            if hasattr(asset, "get_yearly_usage_costs"):
                cost = asset.get_yearly_usage_costs()
            elif asset.asset_name != "EL_Demand_MY":
                cost = asset.yearly_payments.value
            else:
                cost = None

            if cost is not None:
                total_discounted_cost += cost

            if asset.asset_name != "EL_Demand_MY":
                generation_per_year = np.sum(asset.get_yearly_flows()[:30], axis=1)
                # sampled_days = int((asset.number_of_edges / 24) / asset.num_years)
                # simulation_factor = 365 / sampled_days
                # generation_per_year *= simulation_factor

                discounted_generation = generation_per_year * discount_factors
                total_discounted_energy += discounted_generation

                total_gen_energy += generation_per_year
            else:
                demand_per_year = np.sum(asset.get_yearly_flows()[:30], axis=1)
                sampled_days = int((asset.number_of_edges / 24) / asset.num_years)
                simulation_factor = 365 / sampled_days
                demand_per_year *= simulation_factor
                
                discounted_demand = demand_per_year * discount_factors
                total_discounted_demand += discounted_demand

        except Exception as e:
            print(f"[Skip] {asset.asset_name}: {e}")
            continue

    # Compute LCOE (in USD/MWh)
    lcoe_per_year = np.divide(
        total_discounted_cost,
        total_discounted_energy,
        out=np.zeros_like(total_discounted_cost),
        where=total_discounted_energy != 0,
    ) * 1e3  # BUSD/GWh → USD/kWh
    
    # Compute LCUE (in USD/MWh)
    lcue_per_year = np.divide(
        total_discounted_cost,
        total_discounted_demand,
        out=np.zeros_like(total_discounted_cost),
        where=total_discounted_demand != 0,
    ) * 1e3  # BUSD/GWh → USD/kWh

    # Create DataFrame
    df = pd.DataFrame({
        "total_discounted_energy_GWh": total_discounted_energy,
        "total_discounted_cost_BUSD": total_discounted_cost,
        "total_discounted_demand_GWh": total_discounted_demand,
        "lcoe_USD/kWh": lcoe_per_year,
        "lcue_USD/kWh": lcue_per_year
    })

    # Save if needed
    if output_path:
        df.to_csv(output_path, index=False)

    return df


def get_grid_intensity(network, output_path=None):
    """
    Saves the annual grid intensity
    """
    num_years = network.assets[0].num_years

    total_gen_energy = np.zeros(num_years)
    emissions_per_year = np.zeros(num_years)
    for asset in network.assets:
        try:
            if asset.asset_name != "EL_Demand_MY":
                generation_per_year = np.sum(asset.get_yearly_flows()[:30], axis=1)
            if asset.asset_name == "PP_CO2_MY":
                emissions_per_year = asset.get_yearly_emissions()[:30]
                
            total_gen_energy += generation_per_year
            grid_intensity_per_year = emissions_per_year / total_gen_energy #MtCO2/GWh
        except Exception as e:
            print(f"[Skip] {asset.asset_name}: {e}")
            continue


    # Optionally save or return
    if output_path:    
        np.savetxt(output_path, grid_intensity_per_year, delimiter=",")
    return grid_intensity_per_year
    

            
    

    