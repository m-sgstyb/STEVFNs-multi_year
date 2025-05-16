#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 16:36:27 2021

@author: aniqahsan
@contributor: Mónica Sagastuy Breña (c) 2024-2025
"""

import os
import numpy as np
import pandas as pd
import cvxpy as cp
from amortization.amount import calculate_amortization_amount as amort
from ..Base_Assets import Asset_STEVFNs
from ...Network import Edge_STEVFNs


class RE_PV_MY_Asset(Asset_STEVFNs):
    """Class of Renewable Energy Sources """
    asset_name = "RE_PV_MY"
    target_node_type = "EL"
    
    source_node_type_2 = "NULL" # For Edge 2, to constrain maximum capacity
    target_node_type_2 = "RE_PV" # For Edge 2, to constrain maximum capacity


    period = 1
    transport_time = 0
    target_node_time_2 = 0 # For Edge 2, to constrain maximum capacity
    
    @staticmethod
    def cost_fun(flows, params):
        return params["sizing_constant"] @ flows # element wise dot product
    
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
        self.gen_profile = cp.Parameter(shape = (self.number_of_edges), nonneg=True)
        # EDITED: set size of RE asset as array of sizes per horizon modeled
        self.flows = cp.Variable(shape=(self.num_years,), nonneg=True) # New capacities to install per year
        self.cumulative_capacity = np.zeros(shape=(self.num_years,))
        self.final_capacity = np.zeros(shape=(self.num_years,)) # Initialize dynamic, auxilliary variable with zeros
        self.cost_fun_params = {"sizing_constant": cp.Parameter(shape=(self.num_years,),
                                                                nonneg=True)}
        self.conversion_fun_params_2 = {"maximum_size": cp.Parameter(shape=(self.num_years,),
                                                                nonneg=True)}
        return
    
    def build_edge(self, edge_number):
        target_node_time = self.target_node_times[edge_number]
        new_edge = Edge_STEVFNs()
        self.edges += [new_edge]
        new_edge.attach_target_node(self.network.extract_node(
            self.target_node_location, self.target_node_type, target_node_time))
        # EDITED:
        # Initialize existing flows as zero if final_capacity is not updated yet to determine flow
        index_number = 0 # Use only initial capacity from final_capacity array at this stage
        
        new_edge.flow = (self.flows[index_number] * self.gen_profile[edge_number])
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
        self.build_edge_2()
        return
    
    def _update_flows(self):
        '''NEW FUNCTION: Allows power flow update for multi-year modeling for RE assets'''
        index_number = 0 # indicates the year for each edge
        edge_counter = 0
        # Loop through all edges from build_edge, not edge_2
        for edge in self.edges[:self.number_of_edges]:
            if edge_counter >= self.year_change_indices[index_number]:
                if index_number < self.num_years-1:
                    if edge_counter == self.year_change_indices[index_number+1]:
                        index_number += 1
                
                # Update value of flow per edge
                edge.flow = self.final_capacity[index_number] * self.gen_profile[edge_counter]
                edge_counter += 1
        return
    
    def _update_capacities(self):
        """Update existing capacities dynamically for multi-year optimization."""
        
        # Get the historical cumulative capacities as a list
        historic_capacities = self._get_cumulative_capacities_array().tolist()  # Ensure list for handling CVXPY variables
        
        # Initialize the final list to hold the updated capacities
        final_capacity_expressions = []
        
        # Initialize the cumulative capacity with the historic values
        self.cumulative_capacity = historic_capacities.copy()
        # Get the minimum capacity to install parameter
        min_capacities = self.process_csv_values(self.parameters_df["minimum_size"])

        for year in range(self.num_years):
            new_installed = self.flows[year]
            min_to_install = min_capacities[year]
            # if year == 0:
            #     # For the first year, start with existing capacities
            #     new_installed = self.flows[year]
            #     final_capacity_expressions.append(cumulative_capacity[year])
            # else:
            #     # For subsequent years, we add the new installed capacity dynamically
            #     new_installed = self.flows[year - 1]  # CVXPY variable for the current year's flow
            #     min_to_install = min_capacities[year]
                
            # Determine the range within which we can update the cumulative capacity
            start_idx = year
            end_idx = min(year + self.asset_lifetime, self.num_years)  # Don't go beyond the control horizon
            
            # Update the cumulative capacity for the current year and the subsequent years up to asset_lifetime
            for i in range(start_idx, end_idx):
                self.cumulative_capacity[i] += new_installed + min_to_install
            
            # Append the updated capacity for the current year
            final_capacity_expressions.append(self.cumulative_capacity[year])
        
        # Convert to a CVXPY expression array
        self.final_capacity = cp.hstack(final_capacity_expressions)  # Concatenates a series of expressions
    
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
        # cost_array = self.cost_fun_params["sizing_constant"].value
        cost_array = np.array([0.8, 0.5, 0.3, 0.1, 0.05]) # Hard-coded for testing
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
    
    
        for parameter_name, parameter in self.conversion_fun_params_2.items():
            parameter.value = self.process_csv_values(self.parameters_df[parameter_name])
        # Update conversion function parameters for max and min capacities
        # for param_dict in [self.conversion_fun_params_2, self.conversion_fun_params_3]:
        #     for parameter_name, parameter in param_dict.items():
        #         parameter.value = self.process_csv_values(self.parameters_df[parameter_name])
    
        self._load_RE_profile()
        # Build and update values for existing capacities
        self._build_existing_capacities_matrix()
        self._get_amortised_discounted_cost()
        self._get_cumulative_capacities_array()
        self._update_capacities()
        self._update_flows()
    
    def update(self, asset_type):
        self._load_parameters_df(asset_type)
        self._load_historic_capacity_params()
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
        # NEW ADDITION: Initialize indices and variables to Build multi-year profiles
        hour_counter = 1
        missing_val = 0
        # --
        for counter1 in range(n_sets):
            old_loc_0 = offset + gap*counter1
            old_loc_1 = old_loc_0 + set_size
            new_loc_0 = set_size * counter1
            new_loc_1 = new_loc_0 + set_size
            new_profile[new_loc_0 : new_loc_1] = full_profile[old_loc_0 : old_loc_1]
            # -- NEW ADDITION: Get indices for change in year when day sampling
            if old_loc_0 > (hour_counter * 8759) + missing_val:
                 self.year_change_indices.append(new_loc_0)
                 hour_counter += 1
                 missing_val += 1
        self.gen_profile.value = new_profile[:self.number_of_edges]
        return
    
    def _build_existing_capacities_matrix(self):
        '''NEW FUNCTION
        Builds the matrix with installed capacities each year based on historic
        total installed capacities'''
        
        if self.historic_capacity_df is None:
            raise ValueError("Error: historic_capacity_df is None. It must be loaded before building the capacities matrix.")
    
        self.historic_capacity_df = self.historic_capacity_df[['year', 'pv_installed_capacity_GW']].dropna()
        # Calculate annual installed capacity
        self.historic_capacity_df['annual_installed_capacity'] = self.historic_capacity_df['pv_installed_capacity_GW'].diff().fillna(0)
        
        # Initialize an empty DataFrame with the Year column
        self.asset_lifetime = int(self.parameters_df["lifespan"] / 8760)
        control_horizon_years = self.num_years
        start_year = int(self.historic_capacity_df['year'].min())
        end_year = int(self.historic_capacity_df['year'].max()) + control_horizon_years
        self.existing_capacity_df = pd.DataFrame({'Year': range(start_year, end_year + 1)})
    
        # Ensure self.existing_capacity_df is not accidentally None
        if self.existing_capacity_df is None or self.existing_capacity_df.empty:
            raise ValueError("Error: self.existing_capacity_df was not created properly.")
    
        for idx, row in self.historic_capacity_df.iterrows():
            year = int(row['year'])
            annual_capacity = float(row['annual_installed_capacity'])
            self._add_new_capacities(year, annual_capacity)
            
        
    def _add_new_capacities(self, current_year, annual_capacity):
        """
        Amended version:
        - If asset_lifetime < control_horizon, fills asset_lifetime years with capacity,
          then fills remaining years with zeros to match control horizon length.
        - If asset_lifetime >= control_horizon, fills asset_lifetime years, possibly exceeding horizon.
        """
        control_horizon_years = self.num_years

        # Ensure the column for current_year exists
        if str(current_year) not in self.existing_capacity_df.columns:
            self.existing_capacity_df[str(current_year)] = pd.Series(0, index=self.existing_capacity_df.index, dtype="float64")
        
        # Find the index of current_year
        if current_year not in self.existing_capacity_df['Year'].values:
            raise ValueError(f"Year {current_year} not found in existing_capacity_df['Year']. Check Year range.")
        
        start_idx = self.existing_capacity_df[self.existing_capacity_df['Year'] == current_year].index[0]
    
        # CASE 1: asset_lifetime < control_horizon
        if self.asset_lifetime < control_horizon_years:
            # Fill capacity for asset_lifetime years
            for row in range(self.asset_lifetime):
                if start_idx + row < len(self.existing_capacity_df):
                    self.existing_capacity_df.at[start_idx + row, str(current_year)] = annual_capacity
            # Fill zeros after lifetime up to control horizon
            for row in range(self.asset_lifetime, control_horizon_years):
                if start_idx + row < len(self.existing_capacity_df):
                    self.existing_capacity_df.at[start_idx + row, str(current_year)] = 0.0
    
        # CASE 2: asset_lifetime >= control_horizon
        else:
            # Extend DataFrame if required
            required_rows = start_idx + self.asset_lifetime
            while len(self.existing_capacity_df) < required_rows:
                new_year = self.existing_capacity_df['Year'].max() + 1
                new_row = {'Year': new_year}
                # Initialize new row with zeros for all capacity columns
                for col in self.existing_capacity_df.columns:
                    if col != 'Year':
                        new_row[col] = 0.0
                self.existing_capacity_df = pd.concat([self.existing_capacity_df, pd.DataFrame([new_row])], ignore_index=True)
            # Fill all asset_lifetime rows
            for row in range(self.asset_lifetime):
                self.existing_capacity_df.at[start_idx + row, str(current_year)] = annual_capacity

    
    def _add_optimized_capacities_to_matrix(self):
        '''NEW FUNCTION
        Adds the optimal capacities from problem to the matrix
        '''
        # Iterate through control horizon
        for year in range(self.num_years):
            # Get start year for scenario
            start_year = int(self.network.system_parameters_df.loc["scenario_start", "value"])
            # Add column for the newly installed capacity being optimised 
            self.existing_capacity_df = self._add_new_capacities(start_year+year, self.flows[year])
        return self.existing_capacity_df 

    def _get_cumulative_capacities_array(self):
        '''
        NEW FUNCTION
        Generates the array for the cumulative installed capacities for the years
        in the control horizon
        '''
        start_year = int(self.network.system_parameters_df.loc["scenario_start", "value"])
        end_year = start_year + int(self.network.system_parameters_df.loc["control_horizon", "value"] / 8760)
    
        # Filter rows within the desired year range
        filtered_df = self.existing_capacity_df[(self.existing_capacity_df['Year'] >= start_year) & 
                                                (self.existing_capacity_df['Year'] < end_year)]

        # Sum all columns for each row (excluding the 'Year' column) to get cumulative capacities array
        cumulative_capacities = filtered_df.iloc[:, 1:].sum(axis=1).to_numpy()

        return cumulative_capacities
    

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