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
        # Constraint form: flows + maximum_budget <= 0, where flows are negative values
        # and maximum_budget is forced nonneg as CO2 budget
        return params["maximum_budget"] - flows
    
    
    def __init__(self):
        super().__init__()
        self.conversion_fun_params = {"maximum_budget": cp.Parameter(nonneg=True)}
        return
    
    def define_structure(self, asset_structure):
        self.source_node_location = 0
        self.source_node_type = "NULL"
        self.source_node_times = np.array([self.source_node_time])
        self.target_node_location = 0
        self.number_of_edges = 1
        self.target_node_times = np.array([self.target_node_time])
        self.num_years = int(self.network.system_parameters_df.loc["control_horizon", "value"] / 8760)
        self.flows = cp.hstack([cp.Constant(0) for _ in range(self.num_years)])
        self.conversion_fun_params = {"maximum_budget": cp.Parameter(shape=(self.num_years,), nonneg=True)}
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
        
    def build_edge(self):
        source_node_type = "NULL"
        source_node_location = self.source_node_location
        source_node_time = 0
        target_node_type = self.target_node_type
        target_node_location = self.target_node_location
        target_node_time = self.target_node_time
        self.budget_edge = Edge_STEVFNs()
        self.edges += [self.budget_edge]
        if source_node_type != "NULL":
            self.budget_edge.attach_source_node(self.network.extract_node(
                source_node_location, source_node_type, source_node_time))
        if target_node_type != "NULL":
            self.budget_edge.attach_target_node(self.network.extract_node(
                target_node_location, target_node_type, target_node_time))
        self.budget_edge.flow = self.flows
        self.budget_edge.conversion_fun = self.conversion_fun
        self.budget_edge.conversion_fun_params = self.conversion_fun_params
        return

    def build_edges(self):
        self.edges = []
        self.build_edge()
    
    #Added function for param update
    def _update_parameters(self):
        """Updates asset conversion parameters """
        # Update conversion function parameters
        for parameter_name, parameter in self.conversion_fun_params.items():
            parameter.value = self.process_csv_values(self.parameters_df[parameter_name])
        self._update_emissions_flow_vector()
        return
    
    def get_plot_data(self):
        return self.flows.value
    
    def component_size(self):
        return self.flows.value # list of emissions per year
            
    def get_input_emissions_expressions(self):
        self.emissions_expressions = []
        co2_node = self.edges[0].target_node  # get CO2_Budget node
        for edge in co2_node.input_edges:
            if edge is self.budget_edge:
                continue  # skip our own budget edge
            self.emissions_expressions.append(edge.extract_flow())
        return self.emissions_expressions
    
    def _get_year_change_indices(self):
        timesteps = len(self.get_input_emissions_expressions())
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
    
    def _update_emissions_flow_vector(self):
        emissions_expr_list = self.get_input_emissions_expressions()  # list of expressions
        year_indices = self._get_year_change_indices()
        year_indices.append(len(emissions_expr_list))  # now valid
    
        yearly_emissions = []
        for start, end in zip(year_indices[:-1], year_indices[1:]):
            year_total = cp.sum(emissions_expr_list[start:end])
            yearly_emissions.append(year_total)
    
        self.flows = cp.hstack(yearly_emissions)
        return self.flows
    

    