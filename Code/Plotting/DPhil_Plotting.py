# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 17:41:01 2022

@author: aniq_
"""

# from __init__.py import *
import numpy as np
import matplotlib.pyplot as plt
from ..Plotting import bar_chart_artist, stackplot_artist, twin_line_artist


def plot_asset_sizes(my_network, bar_width = 1.0, bar_spacing = 3.0):
    # Plots the size of assets in the system #
    
    # Find maximum asset size so that we can remove assets that are too small, i.e. size zero.
    og_df = my_network.system_structure_df.copy()
    asset_sizes_array = np.zeros(og_df.shape[0])
    for counter1 in range(len(asset_sizes_array)):
        asset_sizes_array[counter1] = my_network.assets[counter1].asset_size()
    og_df["Asset_Size"] = asset_sizes_array
    max_asset_size = np.max(asset_sizes_array)
    
    # Set minimum asset size to plot
    min_asset_size = max_asset_size * 1E-3
    
    # Remove all assets that are too small
    con1 = og_df["Asset_Size"] >= min_asset_size
    og_df = og_df[con1]
    
    # Remove CO2 Budget asset in plot to not skew barchart
    con2 = og_df['Asset_Class'] != 'CO2_Budget'
    og_df = og_df[con2]
    
    # initialize bar data dictionary for plotting assets of a system#
    bar_data_dict = dict()
    asset_class_list = np.sort(og_df["Asset_Class"].unique())
    for counter1 in range(len(asset_class_list)):
        bar_data = dict({
            "x" : [],
            "height" : [],
            })
        bar_data_dict.update({
            asset_class_list[counter1] : bar_data
            })
    # Initialize x ticks dictionary
    x_ticks_data_dict = dict({
        "ticks" : [],
        "labels" : []
        })
    
    #fill bar data dictionary for assets at a location i.e. loc_1 = loc_2
    loc_1_array = np.sort(og_df["Location_1"].unique())
    x_current = 0.0
    
    for counter1 in range(len(loc_1_array)):
        loc_1 = loc_1_array[counter1]
        loc_2 = loc_1
        con1 = og_df["Location_1"] == loc_1
        t_df1 = og_df[con1]
        con2 = t_df1["Location_2"] == loc_2
        t_df2 = t_df1[con2]
        x_tick_0 = x_current
        for counter2 in range(t_df2.shape[0]):
            asset_data = t_df2.iloc[counter2]
            #add size of asset in bar_data
            asset_number = asset_data["Asset_Number"]
            asset_size = my_network.assets[asset_number].asset_size()
            # check if asset is too small
            if asset_size < min_asset_size:
                continue
            bar_data_dict[asset_data["Asset_Class"]]["height"] += [asset_size]
            #add x location of asset in bar_data
            bar_data_dict[asset_data["Asset_Class"]]["x"] += [x_current + bar_width/2]
            #move to next asset
            x_current += bar_width
        #check if any asset was added to that location pair
        if x_current == x_tick_0:
            continue
        #add entry to x_ticks
        x_ticks_data_dict["labels"] += ["(" + str(loc_1) + ")"]
        x_ticks_data_dict["ticks"] += [(x_tick_0 + x_current)/2]
        #move to next location
        x_current += bar_spacing
    
    
    #fill bar data dictionary for assets between locations
    
    for counter1 in range(len(loc_1_array)):
        loc_1 = loc_1_array[counter1]
        con1 = og_df["Location_1"] == loc_1
        t_df1 = og_df[con1]
        loc_2_array = np.sort(t_df1["Location_2"].unique())
        for counter2 in range(len(loc_2_array)):
            loc_2 = loc_2_array[counter2]
            #check if asset is between locations
            if loc_2 == loc_1:
                continue
            con2 = t_df1["Location_2"] == loc_2
            t_df2 = t_df1[con2]
            x_tick_0 = x_current
            for counter3 in range(t_df2.shape[0]):
                asset_data = t_df2.iloc[counter3]
                #add size of asset in bar_data
                asset_number = asset_data["Asset_Number"]
                asset_size = my_network.assets[asset_number].asset_size()
                # check if asset is too small
                if asset_size < min_asset_size:
                    continue
                bar_data_dict[asset_data["Asset_Class"]]["height"] += [asset_size]
                #add x location of asset in bar_data
                bar_data_dict[asset_data["Asset_Class"]]["x"] += [x_current + bar_width/2]
                #move to next asset
                x_current += bar_width
            #check if any asset was added to that location pair
            if x_current == x_tick_0:
                continue
            #add entry to x_ticks
            x_ticks_data_dict["labels"] += ["(" + str(loc_1) + "," + str(loc_2) + ")"]
            x_ticks_data_dict["ticks"] += [(x_tick_0 + x_current)/2]
            #move to next location
            x_current += bar_spacing
    
    #Make a bar chart artist and plot
    my_artist = bar_chart_artist()
    my_artist.bar_data_dict = bar_data_dict
    my_artist.x_ticks_data_dict = x_ticks_data_dict
    my_artist.ylabel = "Asset Size (GWh)"
    my_artist.title = "Size of Assets in the System by Location and Location Pair \n Scenario: " + my_network.scenario_name
    my_artist.plot(bar_width = bar_width, bar_spacing = bar_spacing)
    return

def plot_asset_sizes_stacked(my_network, location_parameters_df, save_path=None):
    og_df = my_network.system_structure_df.copy()
    asset_sizes_array = np.array([my_network.assets[counter].asset_size() for counter in range(len(og_df))])
    og_df["Asset_Size"] = asset_sizes_array
    max_asset_size = np.max(asset_sizes_array)
    min_asset_size = max_asset_size * 1E-3

    og_df = og_df[og_df["Asset_Size"] >= min_asset_size]
    og_df = og_df[og_df['Asset_Class'] != 'CO2_Budget']
    asset_class_list = np.sort(og_df["Asset_Class"].unique())

    loc_1 = og_df["Location_1"].unique()
    loc_name = location_parameters_df.loc[loc_1[0]]['location_name']

    bars = []  # Collect all bars for the legend
    pv_colors = ['#f35b04', '#f18701']
    wind_colors = ['#126782', '#58B4D1']
    pp_colors = ['#8d99ae']
    bess_colors = ['#226f54', '#87c38f']
    hvdc_color = ['#5e548e']
    
    # PV Capacity
    if "RE_PV_Existing" in asset_class_list:
        existing_pv = float(og_df.query("Asset_Class == 'RE_PV_Existing'")['Asset_Size'].iloc[0])
        bars.append(plt.bar("Total PV", existing_pv, color=pv_colors[0], label="PV Existing"))

        if "RE_PV_Openfield_Lim" in asset_class_list:
            new_pv = float(og_df.query("Asset_Class == 'RE_PV_Openfield_Lim'")['Asset_Size'].iloc[0])
            bars.append(plt.bar("Total PV", new_pv, bottom=existing_pv, color=pv_colors[1], label="PV New"))

    # Wind Capacity
    if "RE_WIND_Existing" in asset_class_list:
        existing_wind = float(og_df.query("Asset_Class == 'RE_WIND_Existing'")['Asset_Size'].iloc[0])
        bars.append(plt.bar("Total Wind", existing_wind, color=wind_colors[0], label="Wind Existing"))

        if "RE_WIND_Onshore_Lim" in asset_class_list:
            new_wind = float(og_df.query("Asset_Class == 'RE_WIND_Onshore_Lim'")['Asset_Size'].iloc[0])
            bars.append(plt.bar("Total Wind", new_wind, bottom=existing_wind, color=wind_colors[1], label="Wind New"))

    # Fossil Generation
    if "PP_CO2_Existing" in asset_class_list:
        existing_fossil = float(og_df.query("Asset_Class == 'PP_CO2_Existing'")['Asset_Size'].iloc[0])
        bars.append(plt.bar("Fossil Gen.", existing_fossil, color=pp_colors[0], label="Fossil Existing"))
        
    if "BESS" in asset_class_list:
        new_bess = float(og_df.query("Asset_Class == 'BESS'")['Asset_Size'].iloc[0])
        bars.append(plt.bar("Total BESS", new_bess, color=bess_colors[1], label="BESS New"))
        
        if "BESS_Existing" in asset_class_list:
            existing_bess = float(og_df.query("Asset_Class == 'BESS_Existing'")['Asset_Size'].iloc[0])
            bars.append(plt.bar("Total BESS", existing_bess, color=bess_colors[0], label="BESS Existing"))
    
    if "EL_Transport" in asset_class_list:
        hvdc_cable = float(og_df.query("Asset_Class == 'EL_Transport'")['Asset_Size'].iloc[0])
        bars.append(plt.bar("EL_Transport", hvdc_cable, color=hvdc_color, label="HVDC Cable"))
    
    
    plt.xlabel(loc_name)
    plt.ylabel("Asset Size (GWp)")
    plt.title("Asset Sizes " + my_network.scenario_name)

    # Use only unique labels in the legend to avoid duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys())
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()
    
    return 


def plot_asset_costs(my_network, bar_width = 1.0, bar_spacing = 3.0):
    # Plots the cost of assets in the system #
    
    # Find maximum asset size so that we can remove assets that are too small, i.e. size zero.
    og_df = my_network.system_structure_df.copy()
    asset_costs_array = np.zeros(og_df.shape[0])
    for counter1 in range(len(asset_costs_array)):
        asset_costs_array[counter1] = my_network.assets[counter1].cost.value
    og_df["Asset_Cost"] = asset_costs_array
    max_asset_cost = np.max(asset_costs_array)
    # Set minimum asset size to plot
    min_asset_cost = max_asset_cost * 1E-3
    # Remove all assets that are too small
    con1 = og_df["Asset_Cost"] >= min_asset_cost
    og_df = og_df[con1]
    
    # initialize bar data dictionary for plotting assets of a system#
    bar_data_dict = dict()
    asset_class_list = np.sort(og_df["Asset_Class"].unique())
    for counter1 in range(len(asset_class_list)):
        bar_data = dict({
            "x" : [],
            "height" : [],
            })
        bar_data_dict.update({
            asset_class_list[counter1] : bar_data
            })
    # Initialize x ticks dictionary
    x_ticks_data_dict = dict({
        "ticks" : [],
        "labels" : []
        })
    
    #fill bar data dictionary for assets at a location i.e. loc_1 = loc_2
    loc_1_array = np.sort(og_df["Location_1"].unique())
    x_current = 0.0
    
    for counter1 in range(len(loc_1_array)):
        loc_1 = loc_1_array[counter1]
        loc_2 = loc_1
        con1 = og_df["Location_1"] == loc_1
        t_df1 = og_df[con1]
        con2 = t_df1["Location_2"] == loc_2
        t_df2 = t_df1[con2]
        x_tick_0 = x_current
        for counter2 in range(t_df2.shape[0]):
            asset_data = t_df2.iloc[counter2]
            #add size of asset in bar_data
            asset_number = asset_data["Asset_Number"]
            asset_cost = my_network.assets[asset_number].cost.value
            # check if asset is too small
            if asset_cost < min_asset_cost:
                continue
            bar_data_dict[asset_data["Asset_Class"]]["height"] += [asset_cost]
            #add x location of asset in bar_data
            bar_data_dict[asset_data["Asset_Class"]]["x"] += [x_current + bar_width/2]
            #move to next asset
            x_current += bar_width
        #check if any asset was added to that location pair
        if x_current == x_tick_0:
            continue
        #add entry to x_ticks
        x_ticks_data_dict["labels"] += ["(" + str(loc_1) + ")"]
        x_ticks_data_dict["ticks"] += [(x_tick_0 + x_current)/2]
        #move to next location
        x_current += bar_spacing
    
    
    #fill bar data dictionary for assets between locations
    
    for counter1 in range(len(loc_1_array)):
        loc_1 = loc_1_array[counter1]
        con1 = og_df["Location_1"] == loc_1
        t_df1 = og_df[con1]
        loc_2_array = np.sort(t_df1["Location_2"].unique())
        for counter2 in range(len(loc_2_array)):
            loc_2 = loc_2_array[counter2]
            #check if asset is between locations
            if loc_2 == loc_1:
                continue
            con2 = t_df1["Location_2"] == loc_2
            t_df2 = t_df1[con2]
            x_tick_0 = x_current
            for counter3 in range(t_df2.shape[0]):
                asset_data = t_df2.iloc[counter3]
                #add size of asset in bar_data
                asset_number = asset_data["Asset_Number"]
                asset_cost = my_network.assets[asset_number].cost.value
                # check if asset is too small
                if asset_cost < min_asset_cost:
                    continue
                bar_data_dict[asset_data["Asset_Class"]]["height"] += [asset_cost]
                #add x location of asset in bar_data
                bar_data_dict[asset_data["Asset_Class"]]["x"] += [x_current + bar_width/2]
                #move to next asset
                x_current += bar_width
            #check if any asset was added to that location pair
            if x_current == x_tick_0:
                continue
            #add entry to x_ticks
            x_ticks_data_dict["labels"] += ["(" + str(loc_1) + "," + str(loc_2) + ")"]
            x_ticks_data_dict["ticks"] += [(x_tick_0 + x_current)/2]
            #move to next location
            x_current += bar_spacing
    
    #Make a bar chart artist and plot
    my_artist = bar_chart_artist()
    my_artist.bar_data_dict = bar_data_dict
    my_artist.x_ticks_data_dict = x_ticks_data_dict
    my_artist.ylabel = "Asset Cost (Billion USD)"
    my_artist.title = "Cost of Assets in the System by Location and Location Pair \n Scenario: " + my_network.scenario_name
    my_artist.text_data = {"x": 0.12, "y": 0.5, "s": "Total Cost = " + f"{my_network.cost.value: .5}" + " Bil USD"}
    my_artist.plot(bar_width = bar_width, bar_spacing = bar_spacing)
    return



