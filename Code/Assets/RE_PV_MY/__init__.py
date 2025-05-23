#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 11:54:47 2025

@author: Mónica Sagastuy-Breña
Based on PP_CO2_Asset by:
@author: aniqahsan
"""

import os
import numpy as np
import pandas as pd
import cvxpy as cp
from amortization.amount import calculate_amortization_amount as amort
from ..Base_Assets import Asset_STEVFNs
from ...Network import Edge_STEVFNs


class RE_PV_MY_Asset(Asset_STEVFNs):
    """Class of Renewable Energy Sources for multi-year adaptation"""
    asset_name = "RE_PV_MY"
    target_node_type = "EL"
    
    source_node_type_2 = "NULL" # For Edge 2, to constrain maximum capacity
    target_node_type_2 = "RE_PV" # For Edge 2, to constrain maximum capacity

    period = 1
    transport_time = 0
    target_node_time_2 = 0 # For Edge 2, to constrain maximum capacity
    
    @staticmethod
    def conversion_fun_2(flows, params):
        '''Conversion function to limit to maximum capacity vector'''
        return params["maximum_size"] - flows
    
    def build_cost(self):
        '''Re-define build_cost method for this asset to get amortised and discounted cost'''
        self.cost = self._get_amortised_discounted_cost()
        return

    def __init__(self):
        super().__init__()
        # NEW ADDITION: Initialize attributes for multi year modeling
        self.year_change_indices = [0]
        self.power_flows = []
        self.existing_capacity_df = pd.DataFrame()
        # EDITED: Temporary initialization of cost and conversion function parameters,
        # shape defined in structure
        self.cost_fun_params = {"sizing_constant": cp.Parameter(nonneg=True)}
        self.conversion_fun_params = {"existing_capacity": cp.Parameter(nonneg=True)}
        self.conversion_fun_params_2 = {"maximum_size": cp.Parameter(nonneg=True)}
        return
    
    def define_structure(self, asset_structure):
        self.asset_structure = asset_structure
        self.source_node_location = "NULL"
        self.target_node_location = asset_structure["Location_1"]
        # Add node locations for edge 2
        self.source_node_location_2 = "NULL"
        self.target_node_location_2 = asset_structure["Location_1"]
        
        self.target_node_times = np.arange(asset_structure["Start_Time"], 
                                           asset_structure["End_Time"], 
                                           self.period)
        self.number_of_edges = len(self.target_node_times)
        # self.asset_lifetime = int(self.parameters_df.loc["lifespan"] / 8760)
        # Define the number of years in the control horizon
        self.num_years = int(self.network.system_parameters_df.loc["control_horizon", "value"] / 8760)
        self.gen_profile = cp.Parameter(shape = (self.number_of_edges), nonneg=True, name=f"gen_profile_{self.asset_name}")
        # EDITED: set size of RE asset as array of sizes per horizon modeled
        self.flows = cp.Variable(shape=(self.num_years,), nonneg=True, name=f"capacity_{self.asset_name}") # New capacities to install per year
        self.cumulative_capacity = np.zeros(shape=(self.num_years,))
        self.final_capacity = np.zeros(shape=(self.num_years,)) # Initialize dynamic, auxilliary variable with zeros
        self.cost_fun_params = {"sizing_constant": cp.Parameter(shape=(self.num_years,),
                                                                nonneg=True)}
        self.conversion_fun_params = {"existing_capacity": cp.Parameter(shape=(self.num_years,),
                                                                nonneg=True, name=f"existingcap_{self.asset_name}"),}
        self.conversion_fun_params_2 = {"maximum_size": cp.Parameter(shape=(self.num_years,),
                                                                nonneg=True)}
        self.year_change_indices = self._get_year_change_indices()
        self.asset_lifetime = 20
        return
    
    def build_edge(self, edge_number):
        target_node_time = self.target_node_times[edge_number]
        new_edge = Edge_STEVFNs()
        self.edges += [new_edge]
        new_edge.attach_target_node(self.network.extract_node(
            self.target_node_location, self.target_node_type, target_node_time))
        
        # Find correct index_number for year to correctly generate power flow
        index_number = 0
        for i in range(len(self.year_change_indices)):
            if edge_number >= self.year_change_indices[i]:
                index_number = i
            else:
                break
    
        # Lifetime mask: shape (num_years, num_years)
        # Each row corresponds to a year; each column corresponds to when capacity was installed
        lifetime_mask = np.zeros((self.num_years, self.num_years), dtype=int)
        
        for install_year in range(self.num_years):
            start = install_year
            end = min(install_year + self.asset_lifetime, self.num_years)
            lifetime_mask[start:end, install_year] = 1
        
        # Cumulative installed capacity in each year
        # This multiplies each flow by its active years
        self.cumulative_new_installed = cp.matmul(lifetime_mask, self.flows)
        
        # # Rolling accumulation based on asset lifetime
        # for install_year in range(self.num_years):
        #     active_end_year = min(install_year + self.asset_lifetime, self.num_years)
        #     for active_year in range(install_year, active_end_year):
        #         self.cumulative_new_installed[active_year] += self.flows[install_year]
        
        # # Convert to CVXPY expression (column vector)
        # self.cumulative_new_installed = cp.hstack(self.cumulative_new_installed)
        new_edge.flow = (self.cumulative_new_installed[index_number] + self.conversion_fun_params["existing_capacity"][index_number])\
            * self.gen_profile[edge_number]
        
        return
    
    def build_edge_2(self):
        ''' Build a second edge to constrain maximum capacity'''
        source_node_type = "NULL"
        source_node_location = self.source_node_location_2
        source_node_time = 0
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
        new_edge.flow = self.flows # capacities, CVXPY variable
        new_edge.conversion_fun = self.conversion_fun_2
        new_edge.conversion_fun_params = self.conversion_fun_params_2
        return
    
    def build_edges(self):
        self.edges = []
        for counter1 in range(self.number_of_edges):
            self.build_edge(counter1)
        # self.build_edge_2()
        return
    
    def _update_capacities(self):
        """Update existing capacities dynamically for multi-year optimization."""
        
        # Get the historical cumulative capacities
        historic_capacities = self._get_cumulative_capacities_array().tolist()
        final_capacity_expressions = []
        self.cumulative_capacity = historic_capacities.copy()
        
        for year in range(self.num_years):
            new_installed = self.flows[year]  # ✅ Always use current year's flow
            
            # Determine which years this capacity should be active (lifetime)
            start_idx = year
            end_idx = min(year + self.asset_lifetime, self.num_years)
    
            for i in range(start_idx, end_idx):
                self.cumulative_capacity[i] += new_installed  # ✅ Only added once per lifetime window
    
            final_capacity_expressions.append(self.cumulative_capacity[year])
        
        self.final_capacity = cp.hstack(final_capacity_expressions)
    
    def process_csv_values(self,values):
        """Method converts a comma-separated string to a NumPy array of floats or returns
        the original numeric values in an array."""
        if isinstance(values, str):
            return np.array([float(x) for x in values.split(",")], dtype=float)
        return np.array(values, dtype=float)  # Ensure it's always a NumPy array

        
    def _get_amortised_discounted_cost(self):
        '''
        Calculates discounted and amortised costs matrix for installed capacity
        '''
        cost_array = np.array([1.05,1.045,1.04459,1.0439,1.0426,1.03998,1.038,1.026,1.025,1.02366,0.952,0.921,0.8992,0.851,0.836,0.82,0.76,0.714,0.651,0.592,0.488,0.456,0.449,0.426,0.413,0.4,0.3,0.25,0.2,0.1]) # Hard-coded for testing
        num_years = self.num_years
        asset_lifetime = 20 # hard-coded for testing
        interest_rate = float(self.network.system_parameters_df.loc["interest_rate", "value"])
        discount_rate = float(self.network.system_parameters_df.loc["discount_rate", "value"])
        
        # Calculate amortised cost array
        self.amortised_cost = np.array([amort(c, interest_rate, asset_lifetime) for c in cost_array])
        # Create index matrices
        i, j = np.meshgrid(np.arange(num_years), np.arange(num_years), indexing='ij')
        
        # Define discount factor (only upper triangular part matters)
        discount_factor = (1 + discount_rate) ** (j - i)
        valid_mask = j >= i  # Upper triangular mask
        
        # Payments matrix (CVXPY constraints ensure element-wise operations are valid)
        self.payments_M = cp.multiply(self.flows[i], self.amortised_cost[i]) / discount_factor
        
        # Apply mask to ignore lower triangular part
        self.payments_M = cp.multiply(self.payments_M, valid_mask)
        
        return cp.sum(self.payments_M)
        
    
    
    def _update_parameters(self):
        """Updates model parameters efficiently by processing cost projections, max capacities, and min capacities."""
    
        # Update cost function parameters
        for parameter_name, parameter in self.cost_fun_params.items():
            parameter.value = self.process_csv_values(self.parameters_df[parameter_name])
    
        for parameter_name, parameter in self.conversion_fun_params.items():
            parameter.value = self.process_csv_values(self.parameters_df[parameter_name])
    
        for parameter_name, parameter in self.conversion_fun_params_2.items():
            parameter.value = self.process_csv_values(self.parameters_df[parameter_name])
    
        self._load_RE_profile()
    
    def update(self, asset_type):
        self._load_parameters_df(asset_type)
        # self._load_historic_capacity_params()
        self._update_parameters()
        return
    
    def _load_RE_profile(self):
        """This function reads file and updates self.gen_profile """
        lat_lon_df = self.network.lat_lon_df.iloc[self.target_node_location]
        lat = lat_lon_df["lat"]
        lat = np.int64(np.round((lat) / 0.5)) * 0.5
        lat = min(lat,90.0)
        lat = max(lat,-90.0)
        LAT = "{:0.1f}".format(lat)
        lon = lat_lon_df["lon"]
        lon = np.int64(np.round((lon) / 0.625)) * 0.625
        lon = min(lon, 179.375)
        lon = max(lon, -180.0)
        LON = str(lon)
        RE_TYPE = self.parameters_df["RE_type"]
        profile_folder = os.path.join(self.parameters_folder, "profiles", RE_TYPE, r"lat"+LAT)
        profile_filename = RE_TYPE + r"_lat" + LAT + r"_lon" + LON + r".csv"
        profile_filename = os.path.join(profile_folder, profile_filename)
        with open(profile_filename, encoding='utf-8-sig') as f:
            full_profile = np.loadtxt(f)
        set_size = self.parameters_df["set_size"]
        set_number = self.parameters_df["set_number"]
        n_sets = int(np.ceil(self.number_of_edges/set_size))
        gap = int(len(full_profile) / (n_sets * set_size)) * set_size
        offset = set_size * set_number
        new_profile = np.zeros(int(n_sets * set_size))
        for counter1 in range(n_sets):
            old_loc_0 = offset + gap*counter1
            old_loc_1 = old_loc_0 + set_size
            new_loc_0 = set_size * counter1
            new_loc_1 = new_loc_0 + set_size
            new_profile[new_loc_0 : new_loc_1] = full_profile[old_loc_0 : old_loc_1]
        self.gen_profile.value = new_profile[:self.number_of_edges]
        return

    def get_plot_data(self):
        '''
        Gets total power flow data for each timestep, including from existing and
        newly built capacities

        Returns
        -------
        total_flows : list
            List of flows from this asset.

        '''
        total_flows = []
        for edge in self.edges[:self.number_of_edges]:
            total_flows.append(edge.flow.value)
        return total_flows 
    
    def size(self):
        # Returns size of asset for Existing RE, which is a vector #
        return self.flows.value
    
    def asset_size(self):
        # Returns size of asset for Existing RE, which is a vector #
        return self.flows.value
    
    def get_asset_sizes(self):
        # Returns the size of the asset as a dict #
        asset_size = self.size()
        asset_identity = self.asset_name + r"_" + self.parameters_df["RE_type"] + r"_location_" + str(self.target_node_location)
        return {asset_identity: asset_size}
    
    def _get_year_change_indices(self):
        timesteps = self.number_of_edges
        num_years = self.num_years
        total_length = 8760 * num_years
        set_size = 24
        set_number = 0
        n_sets = int(np.ceil(timesteps / set_size))
        gap = int(total_length / (n_sets * set_size)) * set_size
        offset = set_size * set_number
    
        year_change_indices = []
        last_year = -1
    
        for counter1 in range(n_sets):
            old_loc_0 = offset + gap * counter1
            new_loc_0 = set_size * counter1
    
            current_year = old_loc_0 // 8760
            if current_year != last_year:
                year_change_indices.append(new_loc_0)
                last_year = current_year
    
        return year_change_indices
    
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
                
        flows_full = self.get_plot_data()
    
        # Guard against None or unexpected shape
        if flows_full is None:
            raise ValueError("Flow values not assigned yet.")
        
        if not isinstance(flows_full, np.ndarray):
            flows_full = np.array(flows_full)
    
        # Final slicing using year_change_indices
        year_indices = list(self.year_change_indices) + [len(flows_full)]
        yearly_flows = [flows_full[start:end] for start, end in zip(year_indices[:-1], year_indices[1:])]
        
        return yearly_flows