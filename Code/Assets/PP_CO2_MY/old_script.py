#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 13:25:54 2021

@author: aniqahsan
"""

import numpy as np
import cvxpy as cp
from ..Base_Assets import Asset_STEVFNs
from ...Network import Edge_STEVFNs




class PP_CO2_MY_Asset(Asset_STEVFNs):
    """Class of Conventional Generators"""
    asset_name = "PP_CO2_MY"
    source_node_type = "NULL"
    target_node_type = "EL"
    target_node_type_2 = "CO2_Budget_MY"
    target_node_location_2 = 0
    target_node_time_2 = 0
    target_node_type_3 = "PP_CO2_MY" # For maximum constraint of asset size
    target_node_location_3 = 0
    target_node_time_3 = 0
    period = 1
    transport_time = 0
    
    @staticmethod
    def cost_fun(flows, params):
        sizing_constant = params["sizing_constant"]
        usage_constant_1 = params["usage_constant_1"]
        # usage_constant_2 = params["usage_constant_2"]
        # return (sizing_constant * cp.max(flows) +
        #         usage_constant_1 * cp.sum(flows) +
        #         usage_constant_2 * cp.sum(cp.power(flows,2)))
        return (cp.sum(sizing_constant @ cp.max(flows, axis=1)) +
                cp.sum(usage_constant_1 @ cp.sum(flows, axis=1))
                )
    
    @staticmethod
    def conversion_fun_2(flows, params):
        '''Conversion for edge to constrain emissions by a budget set in CO2_Budget_Asset'''
        CO2_emissions_factor = params["CO2_emissions_factor"]
    
        # Sum emissions over years → get total per edge
        emissions_per_year = -CO2_emissions_factor * cp.sum(flows, axis=0)  # Shape: (edges,)
    
        return emissions_per_year
        
        # CO2_emissions_factor = params["CO2_emissions_factor"]
        # return -CO2_emissions_factor * flows
    
    @staticmethod
    def conversion_fun_3(flows, params):
        '''
        Conversion for edge that constrains the size of PP_CO2 asset
        '''
        asset_size_per_year = cp.max(flows, axis=1)
        return params["maximum_size"] - asset_size_per_year
    
    def __init__(self):
        super().__init__()
        self.year_change_indices = [0] # Added for multi-year modeling
        self.cost_fun_params = {"sizing_constant": cp.Parameter(nonneg=True),
                          "usage_constant_1": cp.Parameter(nonneg=True),
                          # "usage_constant_2": cp.Parameter(nonneg=True)
                          }
        self.conversion_fun_params_2 = {"CO2_emissions_factor": cp.Parameter(nonneg=True)}
        self.conversion_fun_params_3 = {"maximum_size": cp.Parameter(nonneg=True)}
        return
    
    def define_structure(self, asset_structure):
        self.asset_structure = asset_structure
        self.num_years = int(self.network.system_parameters_df.loc["control_horizon", "value"] / 8760)
        self.source_node_location = asset_structure["Location_1"]
        self.source_node_times = np.arange(asset_structure["Start_Time"] + self.transport_time, 
                                           asset_structure["End_Time"] + self.transport_time, 
                                           self.period)
        self.target_node_location = asset_structure["Location_2"]
        self.target_node_times = np.arange(asset_structure["Start_Time"] + self.transport_time, 
                                           asset_structure["End_Time"] + self.transport_time, 
                                           self.period)
        self.number_of_edges = len(self.source_node_times)
        self.cost_fun_params = {"sizing_constant": cp.Parameter(shape=(self.num_years,),
                                                                nonneg=True),
                                "usage_constant_1": cp.Parameter(shape=(self.num_years,),
                                                                 nonneg=True),}
        self.conversion_fun_params_3 = {"maximum_size": cp.Parameter(shape=(self.num_years,),
                                                                     nonneg=True)}
        #EDITED: matrix of flows defined flows as a 2D variable: (num_years, timesteps_per_year)
        self.flows = cp.Variable(shape=(self.num_years, self.number_of_edges), nonneg = True)
        
        return
    
    def build_edge(self, edge_number):
        source_node_time = self.source_node_times[edge_number]
        target_node_time = self.target_node_times[edge_number]
        new_edge = Edge_STEVFNs()
        self.edges += [new_edge]
        if self.source_node_type != "NULL":
            new_edge.attach_source_node(self.network.extract_node(
                self.source_node_location, self.source_node_type, source_node_time))
        if self.target_node_type != "NULL":
            new_edge.attach_target_node(self.network.extract_node(
                self.target_node_location, self.target_node_type, target_node_time))
        # new_edge.flow = self.flows[edge_number] # Used for single-year version
        new_edge.flow = self.flows[:, edge_number] # New shape of power flows is 2D
        new_edge.conversion_fun = self.conversion_fun
        new_edge.conversion_fun_params = self.conversion_fun_params
        return
        
    def build_edges(self):
        super().build_edges()
        for counter1 in range(self.number_of_edges):
            self.build_edge_2(counter1)
            # self.build_edge_3(counter1)
        return
    
    def build_edge_2(self, edge_number):
        source_node_type = self.source_node_type
        source_node_location = self.source_node_location
        source_node_time = self.source_node_times[edge_number]
        target_node_type = self.target_node_type_2
        target_node_location = self.target_node_location_2
        target_node_time = self.target_node_time_2
        
        new_edge = Edge_STEVFNs()
        self.edges += [new_edge]
        if source_node_type != "NULL":
            new_edge.attach_source_node(self.network.extract_node(
                source_node_location, source_node_type, source_node_time))
        if target_node_type != "NULL":
            new_edge.attach_target_node(self.network.extract_node(
                target_node_location, target_node_type, target_node_time))
        new_edge.flow = self.flows[:, edge_number]  # New shape of power flows is 2D
        new_edge.conversion_fun = self.conversion_fun_2
        new_edge.conversion_fun_params = self.conversion_fun_params_2
        return
    
    def build_edge_3(self, edge_number):
        source_node_type = self.source_node_type
        source_node_location = self.source_node_location
        source_node_time = self.source_node_times[edge_number]
        target_node_type = self.target_node_type_3
        target_node_location = self.target_node_location_3
        target_node_time = self.target_node_time_3
        
        new_edge = Edge_STEVFNs()
        self.edges += [new_edge]
        if source_node_type != "NULL":
            new_edge.attach_source_node(self.network.extract_node(
                source_node_location, source_node_type, source_node_time))
        if target_node_type != "NULL":
            new_edge.attach_target_node(self.network.extract_node(
                target_node_location, target_node_type, target_node_time))
        new_edge.flow = cp.max(self.flows, axis=1) # Capacity of assets at all years, length num_years
        new_edge.conversion_fun = self.conversion_fun_3
        new_edge.conversion_fun_params = self.conversion_fun_params_3
        return
    
    def process_csv_values(self,values):
        """Method converts a comma-separated string to a NumPy array of floats or returns
        the original numeric values in an array."""
        if isinstance(values, str):
            return np.array([float(x) for x in values.split(",")], dtype=float)
        return np.array(values, dtype=float)  # Ensure it's always a NumPy array
    
    def _update_sizing_constant(self):
        """Updates the sizing constant with NPV discounting applied year-by-year."""
    
        # Convert discount rate to float
        discount_rate = float(self.network.system_parameters_df.loc["discount_rate", "value"])
        # Extract and process the original sizing_constant from the parameters DataFrame
        original_sizing_constant = self.process_csv_values(self.parameters_df["sizing_constant"])
        # Ensure the array length matches num_years
        if len(original_sizing_constant) != self.num_years:
            raise ValueError("Mismatch between num_years and sizing_constant length")
    
        # Apply NPV discounting: Keep the first value unchanged, discount the rest
        adjusted_sizing_constant = original_sizing_constant / np.array([(1 + discount_rate) ** t for t in range(self.num_years)])
        # Store updated values
        self.cost_fun_params["sizing_constant"].value = adjusted_sizing_constant
    
    def _update_usage_constants(self):
        """Updates the usage constants with NPV discounting applied year-by-year."""

        # Convert discount rate to float
        discount_rate = float(self.network.system_parameters_df.loc["discount_rate", "value"])
        
        # Extract the simulation factor
        simulation_factor = 8760 / self.network.system_structure_properties["simulated_timesteps"]
     
        # Extract and process the original usage constants from the parameters DataFrame
        original_usage_constant_1 = self.process_csv_values(self.parameters_df["usage_constant_1"])
        # original_usage_constant_2 = self.process_csv_values(self.parameters_df["usage_constant_2"])
     
        # Ensure the array lengths match num_years
        if len(original_usage_constant_1) != self.num_years:
            raise ValueError("Mismatch between num_years and usage_constant_1 length")
     
        # Apply NPV discounting: Keep the first value unchanged, discount the rest year-by-year
        adjusted_usage_constant_1 = original_usage_constant_1 / np.array([(1 + discount_rate) ** t for t in range(self.num_years)])
     
        # Apply simulation factor (accounting for time resolution, e.g., hourly)
        adjusted_usage_constant_1 *= simulation_factor
     
        # Store updated values in the cost function parameters
        self.cost_fun_params["usage_constant_1"].value = adjusted_usage_constant_1
        # self.cost_fun_params["usage_constant_2"].value = adjusted_usage_constant_2
        return
    
    def _update_co2_emissions_factor(self):
        simulation_factor = 8760/self.network.system_structure_properties["simulated_timesteps"]
        N = np.ceil(self.network.system_parameters_df.loc["project_life", "value"]/8760)
        self.conversion_fun_params_2["CO2_emissions_factor"].value = (self.conversion_fun_params_2["CO2_emissions_factor"].value * 
                                                                      simulation_factor * N)
        return
        
    def _update_parameters(self):
        # super()._update_parameters()
        for parameter_name, parameter in self.cost_fun_params.items():
            parameter.value = self.process_csv_values(self.parameters_df[parameter_name])
        
        for parameter_name, parameter in self.conversion_fun_params_2.items():
            parameter.value = self.parameters_df[parameter_name]
            
        for parameter_name, parameter in self.conversion_fun_params_3.items():
            parameter.value = self.process_csv_values(self.parameters_df[parameter_name])
    
        #Update cost parameters based on NPV#
        self._update_sizing_constant()
        self._update_usage_constants()
        self._update_co2_emissions_factor()
        return
    
    def size(self):
        if self.flows.value is None:
            return None  # Problem hasn't been solved yet
        # Max flow per year (column-wise max)
        size_per_year = self.flows.value.max(axis=1)
        return size_per_year
    
    def get_asset_sizes(self):
        # Returns the size of the asset as a dict #
        asset_size = self.size()
        asset_identity = self.asset_name + r"_location_" + str(self.node_location)
        return {asset_identity: asset_size}