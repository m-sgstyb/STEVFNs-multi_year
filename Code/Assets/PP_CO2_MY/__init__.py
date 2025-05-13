#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 08:49:26 2025

@author: Mónica Sagastuy-Breña
"""

import numpy as np
import cvxpy as cp
from ..Base_Assets import Asset_STEVFNs
from ...Network import Edge_STEVFNs


class PP_CO2_MY_Asset(Asset_STEVFNs):
    """Class of Conventional Generators in multi-year modelling"""
    asset_name = "PP_CO2_MY"
    source_node_type = "NULL"
    target_node_type = "EL"
    target_node_type_2 = "CO2_Budget"
    target_node_location_2 = 0
    target_node_time_2 = 0
    source_node_type_3 = "NULL"
    target_node_type_3 = "PP_CO2"
    period = 1
    transport_time = 0
    
    @staticmethod
    def cost_fun(flows, params):
        usage_constant_1 = params["usage_constant_1"]
        return usage_constant_1 * cp.sum(flows)
    
    @staticmethod
    def conversion_fun_2(flows, params):
        CO2_emissions_factor = params["CO2_emissions_factor"]
        return -CO2_emissions_factor * flows
    
    @staticmethod
    def conversion_fun_3(flows, params):
        existing_capacity = params["existing_capacity"]
        return existing_capacity - cp.max(flows)
    
    def __init__(self):
        super().__init__()
        self.cost_fun_params = {"usage_constant_1": cp.Parameter(nonneg=True),}
        self.conversion_fun_params_2 = {"CO2_emissions_factor": cp.Parameter(nonneg=True)}
        self.conversion_fun_params_3 = {"existing_capacity": cp.Parameter(nonneg=True)}
        self.year_change_indices = [0]
        return
    
    def define_structure(self, asset_structure):
        self.asset_structure = asset_structure
        self.source_node_location = asset_structure["Location_1"]
        self.source_node_times = np.arange(asset_structure["Start_Time"] + self.transport_time, 
                                           asset_structure["End_Time"] + self.transport_time, 
                                           self.period)
        self.target_node_location = asset_structure["Location_2"]
        self.target_node_times = np.arange(asset_structure["Start_Time"] + self.transport_time, 
                                           asset_structure["End_Time"] + self.transport_time, 
                                           self.period)
        self.source_node_location_3 = "NULL"
        self.target_node_location_3 = asset_structure["Location_1"]
        
        self.number_of_edges = len(self.source_node_times)
        self.flows = cp.Variable(self.number_of_edges, nonneg = True)
        self.num_years = int(self.network.system_parameters_df.loc["control_horizon", "value"] / 8760)
        return
        
    def build_edges(self):
        super().build_edges()
        for counter1 in range(self.number_of_edges):
            self.build_edge_2(counter1)
        self.build_edge_3()
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
        new_edge.flow = self.flows[edge_number]
        new_edge.conversion_fun = self.conversion_fun_2
        new_edge.conversion_fun_params = self.conversion_fun_params_2
        return
    
    def build_edge_3(self):
        source_node_type = "NULL"
        source_node_location = self.source_node_location_3
        source_node_time = 0
        target_node_type = self.target_node_type_3
        target_node_location = self.target_node_location_3
        target_node_time = 0
        
        new_edge = Edge_STEVFNs()
        self.edges += [new_edge]
        if source_node_type != "NULL":
            new_edge.attach_source_node(self.network.extract_node(
                source_node_location, source_node_type, source_node_time))
        if target_node_type != "NULL":
            new_edge.attach_target_node(self.network.extract_node(
                target_node_location, target_node_type, target_node_time))
        new_edge.flow = self.flows
        new_edge.conversion_fun = self.conversion_fun_3
        new_edge.conversion_fun_params = self.conversion_fun_params_3
        return

    def _get_year_change_indices(self):
        timesteps = self.number_of_edges
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
       
    
    def _update_usage_constants(self):
        simulation_factor = 8760/self.network.system_structure_properties["simulated_timesteps"]
        N = np.ceil(self.network.system_parameters_df.loc["project_life", "value"]/8760)
        r = (1 + self.network.system_parameters_df.loc["discount_rate", "value"])**-1
        NPV_factor = (1-r**N)/(1-r)
        self.cost_fun_params["usage_constant_1"].value = (self.cost_fun_params["usage_constant_1"].value * 
                                                        NPV_factor * simulation_factor)
        return
    
    def _update_co2_emissions_factor(self):
        simulation_factor = 8760/self.network.system_structure_properties["simulated_timesteps"]
        N = np.ceil(self.network.system_parameters_df.loc["project_life", "value"]/8760)
        self.conversion_fun_params_2["CO2_emissions_factor"].value = (self.conversion_fun_params_2["CO2_emissions_factor"].value * 
                                                                      simulation_factor * N)
        return
        
    def _update_parameters(self):
        super()._update_parameters()
        for parameter_name, parameter in self.conversion_fun_params_2.items():
            parameter.value = self.parameters_df[parameter_name]
        for parameter_name, parameter in self.conversion_fun_params_3.items():
            parameter.value = self.parameters_df[parameter_name]
        #Update cost parameters based on NPV#
        # self._update_sizing_constant()
        self._update_usage_constants()
        self._update_co2_emissions_factor()
        return
    
    def size(self):
        #Function gets peak generation from powerplant at each year does not work yet
        size_array = []
        for index in range(1,len(self.year_change_indices)):
            if index == 1:
                size = cp.max(self.flows[:,self.year_change_indices[index]].value)
                print(size)
            elif index > 1 & index < len(self.year_change_indices):
                # start = self.year_change_indices[index-1] + 1
                # end = 
                # size = cp.max(self.flows[self.year_change_indices[index-1] + 1:self.year_change_indices[index]].value)
                print(size)
            else:
                size = cp.max(self.flows[self.year_change_indices[index] + 1,:].value)
                print(size)
            
            size_array.append(size)
        return size_array
    
    def get_asset_sizes(self):
        # Returns the size of the asset as a dict #
        asset_size = self.size()
        asset_identity = self.asset_name + r"_location_" + str(self.node_location)
        return {asset_identity: asset_size}