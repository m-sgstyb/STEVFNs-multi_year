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
    
    @staticmethod
    def conversion_fun_2(flows, params):
        # Conversion function for emissions
        CO2_emissions_factor = params["CO2_emissions_factor"]
        return -CO2_emissions_factor * flows
    
    @staticmethod
    def conversion_fun_3(flows, params):
        # Conversion function to limit by existing capacity
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
        self.cost_fun_params = {"usage_constant_1": cp.Parameter(shape=(self.number_of_edges,),nonneg=True,
                                                                 name=f"usage_cost_{self.asset_name}"),}
        self.conversion_fun_params_2 = {"CO2_emissions_factor": cp.Parameter(nonneg=True,
                                                                             name=f"emissions_factor_{self.asset_name}")}
        return
        
    def build_edges(self):
        super().build_edges()
        for year in range(self.num_years):
            self.build_emissions_edges_per_year(year)
        self.build_edge_3()
        return
    
    def build_emissions_edges_per_year(self, year_number):
        """Build one emissions edge per year that sums all hourly emissions for that year."""
        source_node_type = self.source_node_type
        target_node_type = self.target_node_type_2
        source_node_location = self.source_node_location
        target_node_location = 0  # Assuming global co2 budget in location 0
        # target_node_location = source_node_location # assuming individual CO2 budgets in each country when collaborating
    
        source_node_time = year_number
        target_node_time = year_number
    
        # Get year index bounds
        year_change_indices = self._get_year_change_indices()
        year_change_indices.append(self.number_of_edges)
    
        start = year_change_indices[year_number]
        end = year_change_indices[year_number + 1]
    
        # Get all hourly flows for the year
        yearly_flows = self.flows[start:end]
    
        # Compute total emissions
        yearly_emissions = self.conversion_fun_2(yearly_flows, self.conversion_fun_params_2)
        yearly_emissions_sum = cp.sum(yearly_emissions)
        
        # Scale emissions from sampled hours to full year
        hours_per_day = 24
        n_years = self.num_years
        sampled_days = int((self.number_of_edges / hours_per_day) / n_years) # (sampled hours / hours per day) / project life
        emission_scaling_factor = 365 / sampled_days
        yearly_emissions_sum *= emission_scaling_factor
    
        # Create and connect the edge
        edge = Edge_STEVFNs()
        self.edges.append(edge)
    
        if source_node_type != "NULL":
            edge.attach_source_node(
                self.network.extract_node(source_node_location, source_node_type, source_node_time)
            )
    
        if target_node_type != "NULL":
            edge.attach_target_node(
                self.network.extract_node(target_node_location, target_node_type, target_node_time)
            )
        edge.flow = yearly_emissions_sum # scaled emissions to full year from sampled size

    def build_edge_3(self):
        '''Build edge to limit hourly peak generation'''
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
        hours_per_day = 24
        num_years = self.num_years
        days_per_year = int((self.number_of_edges / hours_per_day) / num_years)
        hours_per_year = days_per_year * hours_per_day
        self.year_change_indices = [i * hours_per_year for i in range(num_years)]
        return self.year_change_indices
    
    def _update_usage_constants(self):
        """
        Updates usage cost parameters to a full-length vector matching flow resolution.
        Supports scalar or yearly values, applies NPV and simulation scaling.
        """
        # Compute simulation length and NPV adjustment
        sampled_days = int((self.number_of_edges / 24) / self.num_years) # sampled days per year in horizon
        simulation_factor = 365 / sampled_days
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
        # Expand to full-length vector over number_of_edges to broadcast properly with flows, constant cost per hour each year
        year_indices = self._get_year_change_indices() + [self.number_of_edges]
        expanded_costs = np.zeros(self.number_of_edges)
    
        for i, (start, end) in enumerate(zip(year_indices[:-1], year_indices[1:])):
            expanded_costs[start:end] = yearly_costs[i]
    
        # Assign value to pre-defined Parameter for static cost_fun
        self.cost_fun_params["usage_constant_1"].value = expanded_costs

    def get_yearly_usage_costs(self):
        """
        Returns a list of yearly total usage payments (discounted), using hourly flows
        and NPV-adjusted usage costs.
        """
        if "usage_constant_1" not in self.cost_fun_params:
            raise ValueError("Usage cost not defined for this asset.")
    
        hourly_costs = self.cost_fun_params["usage_constant_1"].value  # shape: (number_of_edges,)
        hourly_flows = self.flows.value  # same shape
    
        if hourly_costs is None or hourly_flows is None:
            raise ValueError("Cost or flow values not set.")
    
        total_hourly_costs = hourly_costs * hourly_flows  # element-wise cost per hour
    
        # Final slicing using year_change_indices
        year_indices = self.year_change_indices.copy()
        year_indices += [self.number_of_edges] 
        yearly_costs = [
            np.sum(total_hourly_costs[start:end])
            for start, end in zip(year_indices[:-1], year_indices[1:])
        ]
    
        return yearly_costs  # length = num_years

    
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
        #Update cost parameters based on NPV, simulation sample, and expand for broadcasting asset cost#
        self._update_usage_constants()
        return
    
    def peak_generation(self):
        """Returns yearly peak fossil generation (hourly peak per modelled year)"""
        if self.flows.value is None:
            return None

        year_indices = self._get_year_change_indices()
        year_indices.append(len(self.flows.value))

        yearly_totals = [
            np.max(self.flows.value[start:end])
            for start, end in zip(year_indices[:-1], year_indices[1:])
        ]
        return np.array(yearly_totals)
    
    def get_asset_sizes(self):
        # Returns the size of the asset as a dict #
        asset_size = self.size()
        asset_identity = self.asset_name + r"_location_" + str(self.node_location)
        return {asset_identity: asset_size}
    
    def get_yearly_emissions(self):
        # define indices for emissions edges, hardcoded for this asset
        # self.edges[:self.number_of_edges] corresponds to asset optimisation variable, hourly power flows
        emissions_edges_start = self.number_of_edges
        emissions_edges_end = self.number_of_edges + self.num_years
        annual_emissions = [-self.edges[i].flow.value for i in range(emissions_edges_start,
                                                                    emissions_edges_end)]
        return annual_emissions
    
    def get_yearly_flows(self):
        """
        Returns a list of flow slices split by each year using year_change_indices
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
        year_indices = self.year_change_indices.copy() + [self.number_of_edges] # gets correct length for lcoe calculation
        yearly_flows = [flows_full[start:end] for start, end in zip(year_indices[:-1], year_indices[1:])]
        yearly_flows = [flow for flow in yearly_flows]
        return yearly_flows
                