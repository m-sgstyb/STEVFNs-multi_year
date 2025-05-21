#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 08:49:26 2025

@author: Mónica Sagastuy-Breña
Based on PP_CO2_Asset by:
@author: aniqahsan
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
        usage_constant_1 = params["usage_constant_1"]  # shape: (n_timesteps,)
        return cp.sum(cp.multiply(usage_constant_1, flows))  # scalar
    
    # @staticmethod
    # def conversion_fun(flows, params):
    #     return flows
    
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
        self.flows = cp.Variable(self.number_of_edges, nonneg = True, name=f"flows_{self.asset_name}")
        self.num_years = int(self.network.system_parameters_df.loc["control_horizon", "value"] / 8760)
        self.cost_fun_params = {"usage_constant_1": cp.Parameter(shape=(self.number_of_edges,),nonneg=True),}
        return
        
    def build_edges(self):
        super().build_edges()
        # for counter1 in range(self.number_of_edges):
        #     self.build_edge_2(counter1)
        self.build_emissions_aggregate_edge()
        self.build_edge_3()
        return
    
    # def build_edge(self, edge_number):
    #     source_node_time = self.source_node_times[edge_number]
    #     target_node_time = self.target_node_times[edge_number]
    #     new_edge = Edge_STEVFNs()
    #     self.edges += [new_edge]
    #     if self.source_node_type != "NULL":
    #         new_edge.attach_source_node(self.network.extract_node(
    #             self.source_node_location, self.source_node_type, source_node_time))
    #     if self.target_node_type != "NULL":
    #         new_edge.attach_target_node(self.network.extract_node(
    #             self.target_node_location, self.target_node_type, target_node_time))
    #     new_edge.flow = self.flows[edge_number]
    #     new_edge.conversion_fun = self.conversion_fun
    #     new_edge.conversion_fun_params = self.conversion_fun_params
    #     return
    
    def build_emissions_aggregate_edge(self):
        source_node_type = self.source_node_type
        source_node_location = self.source_node_location
        source_node_time = 0
        target_node_type = self.target_node_type_2
        target_node_location = self.target_node_location_2
        target_node_time = self.target_node_time_2
    
        new_edge = Edge_STEVFNs()
        self.edges += [new_edge]
    
        if source_node_type != "NULL":
            new_edge.attach_source_node(
                self.network.extract_node(source_node_location, source_node_type, source_node_time))
    
        if target_node_type != "NULL":
            new_edge.attach_target_node(
                self.network.extract_node(target_node_location, target_node_type, target_node_time))
        # Create yearly sums from self.flows
        year_indices = self._get_year_change_indices()
        year_indices.append(self.number_of_edges)  # ensure full coverage
    
        yearly_sums = []
        for start, end in zip(year_indices[:-1], year_indices[1:]):
            yearly_sums.append(cp.sum(self.flows[start:end]))
        # Set the yearly profile (cvxpy Expression vector)
        new_edge.flow = cp.hstack(yearly_sums)
        # Apply CO2 emissions factor conversion wiht flows
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
        set_size = 24
        set_number = 0
        # set_size = self.parameters_df["set_size"]
        # set_number = self.parameters_df["set_number"]
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
        """
        Updates usage cost parameters to a full-length vector matching flow resolution.
        Supports scalar or yearly values, applies NPV and simulation scaling.
        """
        # Compute simulation length and NPV adjustment
        simulation_factor = 8760 / self.network.system_structure_properties["simulated_timesteps"]
        discount_rate = self.network.system_parameters_df.loc["discount_rate", "value"]
        project_life_hours = self.network.system_parameters_df.loc["project_life", "value"]
        num_years = int(np.ceil(project_life_hours / 8760))
    
        # Parse CSV-style input first
        raw_input = self.parameters_df["usage_constant_1"]
        raw_costs = self.process_csv_values(raw_input)
    
        if raw_costs.size == 1:
            raw_costs = np.full(num_years, raw_costs[0])
        elif raw_costs.size != num_years:
            raise ValueError(f"Expected {num_years} yearly usage cost values, got {raw_costs.size}")
    
        # Apply NPV discounting and simulation scaling
        discount_factors = (1 / (1 + discount_rate)) ** np.arange(num_years)
        yearly_costs = raw_costs * discount_factors * simulation_factor
        # Expand to full-length vector over number_of_edges to broadcast properly with flows
        year_indices = self._get_year_change_indices() + [self.number_of_edges]
        expanded_costs = np.zeros(self.number_of_edges)
    
        for i, (start, end) in enumerate(zip(year_indices[:-1], year_indices[1:])):
            expanded_costs[start:end] = yearly_costs[i]
    
        # Assign value to pre-defined Parameter for static cost_fun
        self.cost_fun_params["usage_constant_1"].value = expanded_costs

    
    def process_csv_values(self,values):
        """Method converts a comma-separated string to a NumPy array of floats or returns
        the original numeric values in an array."""
        if isinstance(values, str):
            return np.array([float(x) for x in values.split(",")], dtype=float)
        return np.array(values, dtype=float)  # Ensure it's always a NumPy array

        
    def _update_parameters(self):
        for parameter_name, parameter in self.conversion_fun_params_2.items():
            parameter.value = self.parameters_df[parameter_name]
        for parameter_name, parameter in self.conversion_fun_params_3.items():
            parameter.value = self.parameters_df[parameter_name]
        #Update cost parameters based on NP, simulation sample, and expand for broadcasting asset cost#
        self._update_usage_constants()
        return
    
    def peak_generation_per_year(self):
        # Method gets the peak generation at each year
        year_change_indices = self._get_year_change_indices()
        peak_gen_list = []
    
        # Add the end of the flow array to the list to ensure complete range
        year_change_indices.append(self.number_of_edges)
    
        for i in range(len(year_change_indices) - 1):
            start = year_change_indices[i]
            end = year_change_indices[i + 1]
            peak_gen = cp.max(self.flows[start:end].value)
            peak_gen_list.append(peak_gen.value)

        return np.array(peak_gen_list)
    
    def get_asset_sizes(self):
        # Returns the size of the asset as a dict #
        asset_size = self.size()
        asset_identity = self.asset_name + r"_location_" + str(self.node_location)
        return {asset_identity: asset_size}
    
    def get_emissions_from_PP(self):
        # debugging and testing method for results
        annual_emissions = np.zeros(shape=(5,))
        for edge in self.edges:
            if edge.target_node.node_type == 'CO2_Budget':
                annual_emissions += (edge.flow.value * self.conversion_fun_params_2["CO2_emissions_factor"].value)
                
        return annual_emissions
    
    def get_yearly_flows(self):
        """
        Returns a list of flow slices split by each year using year_change_indices.
        """
        # Ensure indices are available
        if not hasattr(self, "year_change_indices"):
            if hasattr(self, "_get_year_change_indices"):
                self._get_year_change_indices()
            else:
                raise AttributeError("Asset has no year_change_indices or method to compute them.")
                
        flows_full = self.flows.value
    
        # Guard against None or unexpected shape
        if flows_full is None:
            raise ValueError("Flow values not assigned yet.")
        
        if not isinstance(flows_full, np.ndarray):
            flows_full = np.array(flows_full)
    
        # Final slicing using year_change_indices
        year_indices = list(self.year_change_indices) + [len(flows_full)]
        yearly_flows = [flows_full[start:end] for start, end in zip(year_indices[:-1], year_indices[1:])]
        
        return yearly_flows
                