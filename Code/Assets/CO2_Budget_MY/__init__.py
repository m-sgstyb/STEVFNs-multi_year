#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 17:55:37 2025

@author: Mónica Sagastuy-Breña
Based on
CO2_Budget_Asset by @author: aniqahsan
"""

import os
import numpy as np
import cvxpy as cp
from ..Base_Assets import Asset_STEVFNs
from ...Network import Edge_STEVFNs


class CO2_Budget_MY_Asset(Asset_STEVFNs):
    """Class of Renewable Energy Sources """
    asset_name = "CO2_Budget_MY"
    source_node_type = "NULL"
    source_node_time = 0
    target_node_type = "CO2_Budget"
    target_node_time = 0
    period = 1
    transport_time = 0
    
    @staticmethod
    def conversion_fun(flows, params):
        return params["maximum_budget"]
    
    def __init__(self):
        super().__init__()
        self.conversion_fun_params = {"maximum_budget": cp.Parameter(nonneg=True)}
        return
        
    
    def define_structure(self, asset_structure):
        self.source_node_location = 0
        self.source_node_times = np.array([self.source_node_time])
        self.target_node_location = 0
        self.number_of_edges = 1
        self.target_node_times = np.array([self.target_node_time])
        self.flows = cp.Constant(np.zeros(self.number_of_edges))
        self.num_years = int(self.network.system_parameters_df.loc["control_horizon", "value"] / 8760)
        return
    
    def process_csv_values(self,values):
        """Method converts a comma-separated string to a NumPy array of floats or returns
        the original numeric values in an array.
        For any single-year modeling, all asset types in column must be float to be read
        as float. If one asset type is string, all will be read as string
        """
        if isinstance(values, str):
            return np.array([float(x) for x in values.split(",")], dtype=float)
        elif isinstance(values, float):
            return values
    
    #Added function for param update
    def _update_parameters(self):
        """Updates asset conversion parameters """
    
        # Update conversion function parameters
        for parameter_name, parameter in self.conversion_fun_params.items():
            parameter.value = self.process_csv_values(self.parameters_df[parameter_name])
        return
    
    def get_plot_data(self):
        return self.flows.value
    
    
    def component_size(self):
        # Returns size of component (i.e. asset) #
        return (self.edges[0].target_node.net_output_flows + self.conversion_fun_params["maximum_budget"]).value
    
    def get_input_emissions_list(self):
        # Get all edges that have emissions from power plant.
        # Note: this only works for a system where there is a single emitting asset in the network
        # Hard-coded for my thesis. MSB
        self.emissions_edges = []
        base_edge = self.edges[0]
        for edge in base_edge.target_node.input_edges:
            edge_flow = edge.extract_flow()
            if edge_flow.sign == 'NONPOSITIVE':
                self.emissions_edges.append(edge_flow.value)
        print(sum(self.emissions_edges))
        return self.emissions_edges
            
    def _get_year_change_indices(self):
        timesteps = len(self.get_input_emissions_list())
        num_years = self.num_years
        total_length = 8760 * num_years  # total length of full-resolution data
        set_size = self.parameters_df["set_size"]
        set_number = self.parameters_df["set_number"]
        n_sets = int(np.ceil(timesteps / set_size))
        gap = int(total_length / (n_sets * set_size)) * set_size
        offset = set_size * set_number
    
        self.year_change_indices = []
        last_year = -1  # initialize to a value that will never match first year
    
        for counter1 in range(n_sets):
            old_loc_0 = offset + gap * counter1
            new_loc_0 = set_size * counter1
    
            current_year = old_loc_0 // 8760  # get artificial year index
            if current_year != last_year:
                self.year_change_indices.append(new_loc_0)
                last_year = current_year

        return self.year_change_indices

    