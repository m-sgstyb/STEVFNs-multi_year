#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 08:41:51 2024

@author: Mónica Sagastuy-Breña

Script to export Flows data from Energy Assets
These functions are currently hard-coded for assets in the case studies of author's
DPhil Thesis. 

They may serve as guidance on how to extract energy flows for different types of
assets, but some assets have them as different attributes so adding a new asset
into the Network_Structure of a model requires updating this script accordingly.

"""
import pandas as pd 
import numpy as np

def export_AUT_Flows(my_network):
    '''
    Parameters
    ----------
    my_network : STEVFNs network
        Full network after running a given system

    Returns
    -------
    Dataframe of flows for Autarky Case Study
    '''
    # Initialize arrays with correct shape and 1D structure
    pv = np.zeros(my_network.system_structure_properties["simulated_timesteps"])
    wind = np.zeros(my_network.system_structure_properties["simulated_timesteps"])
    pp = np.zeros(my_network.system_structure_properties["simulated_timesteps"])
    demand = np.zeros(my_network.system_structure_properties["simulated_timesteps"])
    BESS_ch = np.zeros(my_network.system_structure_properties["simulated_timesteps"])
    BESS_disch = np.zeros(my_network.system_structure_properties["simulated_timesteps"])
    
    # Extract hourly arrays of flows for each asset
    for i in range(1, len(my_network.assets)):
        asset = my_network.assets[i]
        name = asset.asset_name
        
        if name == 'EL_Demand_UM':
            demand = asset.assets_dictionary['Net_EL_Demand'].flows.value
        elif name == 'EL_Demand':
            demand = asset.flows.value
        elif name == 'BESS':
            BESS_ch = asset.assets_dictionary['Charging'].flows.value
            BESS_disch = asset.assets_dictionary['Discharging'].flows.value
        elif name in ['PP_CO2', 'PP_CO2_Existing']:
            pp += asset.flows.value
        elif name in ['RE_PV_Existing', 'RE_PV_Openfield_Lim']:
            pv += asset.get_plot_data()
        elif name in ['RE_Wind_Existing', 'RE_WIND_Onshore_Lim']:
            wind += asset.get_plot_data()

    flows = pd.DataFrame({
        "Net_demand": demand,
        "BESS_Charging": BESS_ch,
        "BESS_Discharging": BESS_disch,
        "PP": pp,
        "PV": pv,
        "Wind": wind,
    })
    
    return flows
    
def export_Xlinks_Flows(my_network):

    flows = pd.DataFrame()
    demand_0 = pd.DataFrame(my_network.assets[3].assets_dictionary['Net_EL_Demand'].flows.value, columns = ["EL_Demand_UM_GB"])
    demand_1 = pd.DataFrame(my_network.assets[7].assets_dictionary['Net_EL_Demand'].flows.value, columns = ["EL_Demand_UM_MA"])

    BESS_ch_0 = pd.DataFrame(my_network.assets[2].assets_dictionary['Charging'].flows.value, columns = ["BESS_Charging_GB"])
    BESS_disch_0 = pd.DataFrame(my_network.assets[2].assets_dictionary['Discharging'].flows.value, columns = ["BESS_Discharging_GB"])
    BESS_ch_1 = pd.DataFrame(my_network.assets[6].assets_dictionary['Charging'].flows.value, columns = ["BESS_Charging_MA"])
    BESS_disch_1 = pd.DataFrame(my_network.assets[6].assets_dictionary['Discharging'].flows.value, columns = ["BESS_Discharging_MA"])
    
    PV_0 = pd.DataFrame(my_network.assets[0].get_plot_data(), columns = ['PV_GB'])
    Wind_0 = pd.DataFrame(my_network.assets[1].get_plot_data(), columns = ['Wind_GB'])
    PV_1 = pd.DataFrame(my_network.assets[4].get_plot_data(), columns = ['PV_MA'])
    Wind_1 = pd.DataFrame(my_network.assets[5].get_plot_data(), columns = ['Wind_MA'])
    
    HVDC_out = pd.DataFrame(my_network.assets[8].flows.value[0:8760], columns = ['HVDC_GB-MA'])
    HVDC_in = pd.DataFrame(my_network.assets[8].flows.value[8760:17520], columns = ['HVDC_MA-GB'])

    flows = pd.concat ([flows, demand_0, demand_1, BESS_ch_0, BESS_disch_0, BESS_ch_1, BESS_disch_1,
    PV_0, Wind_0, PV_1, Wind_1, HVDC_out, HVDC_in], axis = 1)
    return(flows)

def export_XlinksEXT_Flows(my_network):
    
    flows = pd.DataFrame()
    demand_0 = pd.DataFrame(my_network.assets[3].assets_dictionary['Net_EL_Demand'].flows.value, columns = ["EL_Demand_UM_GB"])
    demand_1 = pd.DataFrame(my_network.assets[7].assets_dictionary['Net_EL_Demand'].flows.value, columns = ["EL_Demand_UM_MA"])
    demand_2 = pd.DataFrame(my_network.assets[11].assets_dictionary['Net_EL_Demand'].flows.value, columns = ["EL_Demand_UM_ZA"])
    
    BESS_ch_0 = pd.DataFrame(my_network.assets[2].assets_dictionary['Charging'].flows.value, columns = ["BESS_Charging_GB"])
    BESS_disch_0 = pd.DataFrame(my_network.assets[2].assets_dictionary['Discharging'].flows.value, columns = ["BESS_Discharging_GB"])
    BESS_ch_1 = pd.DataFrame(my_network.assets[6].assets_dictionary['Charging'].flows.value, columns = ["BESS_Charging_MA"])
    BESS_disch_1 = pd.DataFrame(my_network.assets[6].assets_dictionary['Discharging'].flows.value, columns = ["BESS_Discharging_MA"])
    BESS_ch_2 = pd.DataFrame(my_network.assets[10].assets_dictionary['Charging'].flows.value, columns = ["BESS_Charging_ZA"])
    BESS_disch_2 = pd.DataFrame(my_network.assets[10].assets_dictionary['Discharging'].flows.value, columns = ["BESS_Discharging_ZA"])
    
    PV_0 = pd.DataFrame(my_network.assets[0].get_plot_data(), columns = ['PV_GB'])
    Wind_0 = pd.DataFrame(my_network.assets[1].get_plot_data(), columns = ['Wind_GB'])
    PV_1 = pd.DataFrame(my_network.assets[4].get_plot_data(), columns = ['PV_MA'])
    Wind_1 = pd.DataFrame(my_network.assets[5].get_plot_data(), columns = ['Wind_MA'])
    PV_2 = pd.DataFrame(my_network.assets[8].get_plot_data(), columns = ['PV_ZA'])
    Wind_2 = pd.DataFrame(my_network.assets[9].get_plot_data(), columns = ['Wind_ZA'])
    
    HVDC0_out = pd.DataFrame(my_network.assets[12].flows.value[0:8760], columns = ['HVDC_GB-MA'])
    HVDC0_in = pd.DataFrame(my_network.assets[12].flows.value[8760:17520], columns = ['HVDC_MA-GB'])
    HVDC1_out = pd.DataFrame(my_network.assets[13].flows.value[0:8760], columns = ['HVDC_MA-ZA'])
    HVDC1_in = pd.DataFrame(my_network.assets[13].flows.value[8760:17520], columns = ['HVDC_ZA-MA'])
    
    flows = pd.concat ([flows, demand_0, demand_1, demand_2, BESS_ch_0, BESS_disch_0, BESS_ch_1, BESS_disch_1,
                        BESS_ch_2, BESS_disch_2, PV_0, Wind_0, PV_1, Wind_1, PV_2, Wind_2, HVDC0_out, HVDC0_in,
                        HVDC1_out, HVDC1_in], axis = 1)
    return(flows)

    
    
def export_AUT_costs_sizes(my_network):

    costs_Sizes = pd.DataFrame()

    ### Unmet demand asset
    costs_Sizes.insert(0, 'EL_Demand_UM_GWp', [my_network.assets[3].asset_size()])
    costs_Sizes.insert(0, 'EL_Demand_UM_G$', [my_network.assets[3].cost.value])
    
    ### BESS ###
    costs_Sizes.insert(0, 'BESS_GWh', [my_network.assets[2].asset_size()])
    costs_Sizes.insert(0, 'BESS_G$', [my_network.assets[2].cost.value])

    ### RE ###
    costs_Sizes.insert(0, 'PV_GWp', [my_network.assets[0].asset_size()])
    costs_Sizes.insert(0, 'PV_G$', [my_network.assets[0].cost.value])
    costs_Sizes.insert(0, 'Wind_GWp', [my_network.assets[1].asset_size()])
    costs_Sizes.insert(0, 'Wind_G$', [my_network.assets[1].cost.value])
    
    return(costs_Sizes)
    
def export_Xlinks_costs_sizes(my_network):

    costs_Sizes = pd.DataFrame()

    ### Unmet demand asset
    costs_Sizes.insert(0, 'GB_EL_Demand_UM_GWp', [my_network.assets[3].asset_size()])
    costs_Sizes.insert(0, 'GB_EL_Demand_UM_G$', [my_network.assets[3].cost.value])
    costs_Sizes.insert(0, 'MA_EL_Demand_UM_GWp', [my_network.assets[7].asset_size()])
    costs_Sizes.insert(0, 'MA_EL_Demand_UM_G$', [my_network.assets[7].cost.value])
    
    ### BESS ###
    costs_Sizes.insert(0, 'GB_BESS_GWh', [my_network.assets[2].asset_size()])
    costs_Sizes.insert(0, 'GB_BESS_G$', [my_network.assets[2].cost.value])
    costs_Sizes.insert(0, 'MA_BESS_GWh', [my_network.assets[6].asset_size()])
    costs_Sizes.insert(0, 'MA_BESS_G$', [my_network.assets[6].cost.value])

    ### RE ###
    costs_Sizes.insert(0, 'GB_PV_GWp', [my_network.assets[0].asset_size()])
    costs_Sizes.insert(0, 'GB_PV_G$', [my_network.assets[0].cost.value])
    costs_Sizes.insert(0, 'GB_Wind_GWp', [my_network.assets[1].asset_size()])
    costs_Sizes.insert(0, 'GB_Wind_G$', [my_network.assets[1].cost.value])
    costs_Sizes.insert(0, 'MA_PV_GWp', [my_network.assets[4].asset_size()])
    costs_Sizes.insert(0, 'MA_PV_G$', [my_network.assets[4].cost.value])
    costs_Sizes.insert(0, 'MA_Wind_GWp', [my_network.assets[5].asset_size()])
    costs_Sizes.insert(0, 'MA_Wind_G$', [my_network.assets[5].cost.value])
    
    ### HVDC ###
    costs_Sizes.insert(0, 'GB-MA_HVDC_GWp', [my_network.assets[8].asset_size()])
    costs_Sizes.insert(0, 'GB-MA_HVDC_G$', [my_network.assets[8].cost.value])
    
    return(costs_Sizes)
        
def export_XlinksEXT_costs_sizes(my_network):
    costs_Sizes = pd.DataFrame()

    ### Unmet demand asset
    costs_Sizes.insert(0, 'GB_EL_Demand_UM_GWp', [my_network.assets[3].asset_size()])
    costs_Sizes.insert(0, 'GB_EL_Demand_UM_G$', [my_network.assets[3].cost.value])
    costs_Sizes.insert(0, 'MA_EL_Demand_UM_GWp', [my_network.assets[7].asset_size()])
    costs_Sizes.insert(0, 'MA_EL_Demand_UM_G$', [my_network.assets[7].cost.value])
    costs_Sizes.insert(0, 'ZA_EL_Demand_UM_GWp', [my_network.assets[11].asset_size()])
    costs_Sizes.insert(0, 'ZA_EL_Demand_UM_G$', [my_network.assets[11].cost.value])
    
    ### BESS ###
    costs_Sizes.insert(0, 'GB_BESS_GWh', [my_network.assets[2].asset_size()])
    costs_Sizes.insert(0, 'GB_BESS_G$', [my_network.assets[2].cost.value])
    costs_Sizes.insert(0, 'MA_BESS_GWh', [my_network.assets[6].asset_size()])
    costs_Sizes.insert(0, 'MA_BESS_G$', [my_network.assets[6].cost.value])
    costs_Sizes.insert(0, 'ZA_BESS_GWh', [my_network.assets[10].asset_size()])
    costs_Sizes.insert(0, 'ZA_BESS_G$', [my_network.assets[10].cost.value])

    ### RE ###
    costs_Sizes.insert(0, 'GB_PV_GWp', [my_network.assets[0].asset_size()])
    costs_Sizes.insert(0, 'GB_PV_G$', [my_network.assets[0].cost.value])
    costs_Sizes.insert(0, 'GB_Wind_GWp', [my_network.assets[1].asset_size()])
    costs_Sizes.insert(0, 'GB_Wind_G$', [my_network.assets[1].cost.value])
    costs_Sizes.insert(0, 'MA_PV_GWp', [my_network.assets[4].asset_size()])
    costs_Sizes.insert(0, 'MA_PV_G$', [my_network.assets[4].cost.value])
    costs_Sizes.insert(0, 'MA_Wind_GWp', [my_network.assets[5].asset_size()])
    costs_Sizes.insert(0, 'MA_Wind_G$', [my_network.assets[5].cost.value])
    costs_Sizes.insert(0, 'ZA_PV_GWp', [my_network.assets[8].asset_size()])
    costs_Sizes.insert(0, 'ZA_PV_G$', [my_network.assets[8].cost.value])
    costs_Sizes.insert(0, 'ZA_Wind_GWp', [my_network.assets[9].asset_size()])
    costs_Sizes.insert(0, 'ZA_Wind_G$', [my_network.assets[9].cost.value])
    
    ### HVDC ###
    costs_Sizes.insert(0, 'GB-MA_HVDC_GWp', [my_network.assets[12].asset_size()])
    costs_Sizes.insert(0, 'GB-MA_HVDC_G$', [my_network.assets[12].cost.value])
    costs_Sizes.insert(0, 'MA-ZA_HVDC_GWp', [my_network.assets[13].asset_size()])
    costs_Sizes.insert(0, 'MA-ZA_HVDC_G$', [my_network.assets[13].cost.value])
    
    return(costs_Sizes)
    
    
    
    
    
    
    