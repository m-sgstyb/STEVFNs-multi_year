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
        self.conversion_fun_params_3 = {"tech_potential": cp.Parameter(nonneg=True,
                                                                       name=f"tech_potential_{self.asset_name}")}
        self.conversion_fun_params_4 = {"embedded_emissions_factor": cp.Parameter(nonneg=True,
                                                                       name=f"embed_emissions_{self.asset_name}")}
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
        self.conversion_fun_params_3 = {"tech_potential": cp.Parameter(nonneg=True,
                                                                       name=f"tech_potential_{self.asset_name}")}
        self.conversion_fun_params_4 = {"embedded_emissions_factor": cp.Parameter(nonneg=True,
                                                                       name=f"embed_emissions_{self.asset_name}")}
        self.year_change_indices = self._get_year_change_indices()
        self.asset_lifetime = 20 # hard-coded for testing
        return
    
    def build_edge(self, edge_number):
        target_node_time = self.target_node_times[edge_number]
        self.flow_edge = Edge_STEVFNs()
        self.edges += [self.flow_edge]
        self.flow_edge.attach_target_node(self.network.extract_node(
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
        
        self.flow_edge.flow = (self.cumulative_new_installed[index_number] + self.conversion_fun_params["existing_capacity"][index_number])\
            * self.gen_profile[edge_number]
        return
    
    def build_max_capacity_edges(self, year_number):
        '''
        Builds edges per year to constrain maximum capacity to be installed per year
        Explicitly sets edge flow
        '''
        source_node_type = self.source_node_type_2
        source_node_location = self.source_node_location_2
        target_node_type = self.target_node_type_2
        target_node_location = source_node_location
        
        source_node_time = 0
        target_node_time = year_number
        
        # Get the installed capacity in that year
        installed_capacity = self.cumulative_new_installed[year_number]
    
        # Create edge with balance = max_capacity - installed_capacity
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
        # Define flow as max capacity minus actual installed capacity
        max_capacity_param = self.conversion_fun_params_2["maximum_size"]
        edge.flow = max_capacity_param[year_number] - installed_capacity
        
    def build_tech_potential_edges(self, year_number):
        source_node_type = "NULL"
        source_node_location = self.source_node_location_2
        target_node_type = "RE_WIND_Tech"
        target_node_location = source_node_location
        
        source_node_time = 0
        target_node_time = year_number
        cumulative_new_installed_in_year = self.cumulative_new_installed[year_number]
        existing_historic = self.conversion_fun_params["existing_capacity"][year_number]
        
        total_available_capacity = cumulative_new_installed_in_year + existing_historic
        
        # Create edge with balance = max_technical_capacity - installed_capacity
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
        # Define flow as max capacity minus actual installed capacity
        technical_capacity = self.conversion_fun_params_3["tech_potential"]
        edge.flow = technical_capacity - total_available_capacity
    
    def build_embedded_emissions_edge(self):
        '''Build edge to calculate embedded emissions over the total generation of the asset'''
        source_node_type = "NULL"
        source_node_location = self.source_node_location
        source_node_time = 0
        target_node_type = "CO2_Budget"
        target_node_location = 0
        target_node_time = 0
        
        self.embed_emissions_edge = Edge_STEVFNs()
        self.edges += [self.embed_emissions_edge]
        if source_node_type != "NULL":
            self.embed_emissions_edge.attach_source_node(self.network.extract_node(
                source_node_location, source_node_type, source_node_time))
        if target_node_type != "NULL":
            self.embed_emissions_edge.attach_target_node(self.network.extract_node(
                target_node_location, target_node_type, target_node_time))
        power_flows = self.flow_edge.flow
        self.embed_emissions_edge.flow = cp.sum(power_flows) * -self.conversion_fun_params_4["embedded_emissions_factor"]
        
        return
    
    
    def build_edges(self):
        self.edges = []
        for hour in range(self.number_of_edges):
            self.build_edge(hour)
        for year in range(self.num_years):
            self.build_max_capacity_edges(year)
            self.build_tech_potential_edges(year)
        self.build_embedded_emissions_edge()
        return
    
    def process_csv_values(self,values):
        """Method converts a comma-separated string to a NumPy array of floats or returns
        the original numeric values in an array."""
        if isinstance(values, str):
            return np.array([float(x) for x in values.split(",")], dtype=float)
        return np.array(values, dtype=float)  # Ensure it's always a NumPy array

    
    def _get_amortised_discounted_cost(self):
        '''
        Calculates total discounted and amortised cost scaled for representative timesteps.
        '''
    
        # Cost per year (learning curve)
        cost_array = np.array([
            0.7060459581718788, 0.654075416391331, 0.6118892459369893, 0.5776453675998691, 0.5498485053888956,
            0.5272848949886316, 0.5089692843969168, 0.4941019125600741, 0.48203358749592495, 0.4722373390489208,
            0.464285408497092, 0.4578305702608631, 0.45259097012331034, 0.4483378179191948, 0.4448853972903205,
            0.4420829562796257, 0.43980812466354835, 0.4379615705876078, 0.43646266318470744, 0.43524595178226116,
            0.4342583079609333, 0.43345660567148603, 0.43280583811079215, 0.432277589129372, 0.43184879242361596,
            0.4315007243321229, 0.431218186256034, 0.4309888410032783, 0.4308026740778189, 0.43065155639078273
        ])  # or pass it in
    
        asset_lifetime = 20  # years
        interest_rate = float(self.network.system_parameters_df.loc["interest_rate", "value"])
        discount_rate = float(self.network.system_parameters_df.loc["discount_rate", "value"])
        project_years = self.num_years
    
        # Amortisation factor for annualised cost
        r = interest_rate
        n = asset_lifetime
        amort_factor = (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    
        amortised_cost = cost_array * amort_factor  # shape (project_years,)
    
        # Create index matrices for payment timing
        i, j = np.meshgrid(np.arange(project_years), np.arange(project_years), indexing='ij')
        # discount_factor = (1 + discount_rate) ** (i - j) # This was discounting to year j, not year 0
        discount_factor = (1 + discount_rate) ** i # This discounts to year 0
        valid_mask = (i >= j) & (i < j + asset_lifetime)
    
        amortised_j = cp.reshape(amortised_cost, (1, project_years))  # shape (1, years)
        flows_j = cp.reshape(self.flows, (1, project_years))  # shape (1, years)
    
        raw_payments = cp.multiply(flows_j, amortised_j) / discount_factor
        self.payments_M = cp.multiply(raw_payments, valid_mask)
        self.yearly_payments = cp.sum(self.payments_M, axis=1)
    
        return cp.sum(self.payments_M)

    
    def _update_parameters(self):
        """Updates model parameters efficiently by processing cost projections,
        max capacities, and technical potential."""
    
        # Update cost function parameters
        for parameter_name, parameter in self.cost_fun_params.items():
            parameter.value = self.process_csv_values(self.parameters_df[parameter_name])
        for parameter_name, parameter in self.conversion_fun_params.items():
            parameter.value = self.process_csv_values(self.parameters_df[parameter_name])
        for parameter_name, parameter in self.conversion_fun_params_2.items():
            parameter.value = self.process_csv_values(self.parameters_df[parameter_name])
        for parameter_name, parameter in self.conversion_fun_params_3.items():
            parameter.value = self.process_csv_values(self.parameters_df[parameter_name])
        for parameter_name, parameter in self.conversion_fun_params_4.items():
            parameter.value = self.process_csv_values(self.parameters_df[parameter_name])
        self._load_RE_profile()
    
    def update(self, asset_type):
        self._load_parameters_df(asset_type)
        self._update_parameters()
        return
    
    def _load_RE_profile(self):
        """Loads renewable profile and resamples to representative days per year"""
        # --- Location-based profile filename selection ---
        lat_lon_df = self.network.lat_lon_df.iloc[self.target_node_location]
        lat = lat_lon_df["lat"]
        lat = np.int64(np.round(lat / 0.5)) * 0.5
        lat = min(lat, 90.0)
        lat = max(lat, -90.0)
        LAT = "{:0.1f}".format(lat)
    
        lon = lat_lon_df["lon"]
        lon = np.int64(np.round(lon / 0.625)) * 0.625
        lon = min(lon, 179.375)
        lon = max(lon, -180.0)
        LON = str(lon)
    
        RE_TYPE = self.parameters_df["RE_type"]
        profile_folder = os.path.join(self.parameters_folder, "profiles", RE_TYPE, r"lat" + LAT)
        profile_filename = os.path.join(profile_folder, RE_TYPE + r"_lat" + LAT + r"_lon" + LON + r".csv")
    
        with open(profile_filename, encoding='utf-8-sig') as f:
            full_profile = np.loadtxt(f)
    
        # --- Sampling parameters ---
        total_hours = len(full_profile)
        hours_per_year = 8760
        n_years = total_hours // hours_per_year # number of years in project
        hours_per_day = 24
        days_per_year = int((self.number_of_edges / hours_per_day) / n_years) # (sampled hours / hours per day) / project life
        # print("Days per year", days_per_year)
        # --- Build new profile ---
        new_profile = []
    
        for year in range(n_years):
            year_start = year * hours_per_year
            for d in range(days_per_year):
                # Spread days evenly across the year
                day_idx = int((d + 0.5) * hours_per_year / days_per_year / hours_per_day)
                hour_idx = year_start + day_idx * hours_per_day
                new_profile.extend(full_profile[hour_idx:hour_idx + hours_per_day])
    
        self.gen_profile.value = np.array(new_profile)
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
        hours_per_day = 24
        num_years = self.num_years
        days_per_year = int((self.number_of_edges / hours_per_day) / num_years)
        hours_per_year = days_per_year * hours_per_day
        self.year_change_indices = [i * hours_per_year for i in range(num_years)]
        return self.year_change_indices
    
    def get_yearly_flows(self):
        """
        Returns a list of flow slices split by each year using year_change_indices.
        """
        sampled_days = int((self.number_of_edges / 24) / self.num_years) # sampled days per year in horizon
        simulation_factor = 365 / sampled_days
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
        yearly_flows = [flow * simulation_factor for flow in yearly_flows]
        return yearly_flows
    
    def get_total_embedded_emissions(self):
        return self.embed_emissions_edge.flow.value