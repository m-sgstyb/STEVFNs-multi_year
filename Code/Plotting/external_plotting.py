#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 18:37:28 2025

@author: Mónica Sagastuy-Breña
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_gef_and_emissions(annual_flows_filename, output_folder):
    """
    Gets the emissions annually and plots total;
    calculates grid emissions factor and plots that

    Parameters
    ----------
    my_network : TYPE
        DESCRIPTION.
    output_folder : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    annual_flows_df = pd.read_csv(annual_flows_filename)
    # Add a checker to see if there is more than one PP_CO2_MY in columns of annual_flows_df for collabs
    
    emissions = annual_flows_df["Annual_Emissions"]
    
    lcoe = annual_flows_df["lcoe"]
    total_gen = annual_flows_df["Annual_generation"]
    
    gef = emissions / total_gen
    
    # In collabs, there has to be generation per location for GEF
    # Plot two subplots one on top of the other
    # 1. Top one has emissions (MtCO2e) on one Y axis and GEF (MtCO2e/GWh) on a secondary Y axis
    # 2. Bottom one has the LCOE value
    # Have labels (a) and (b) on the left hand top corner, outside of the grid of the plot
    

    
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # STEVFNs folder
cs_folder = os.path.join(root_dir, "Data", "Case_Study")
case_study_name = "single_country_scurve_red_50"
scenario = "WECC"
cs_results_dir = os.path.join(cs_folder, "scenario", "Results")


