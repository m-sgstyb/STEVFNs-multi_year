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
    # target_node_time_2 = 0 # For Edge 2, to constrain maximum capacity
    
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
        self.cost_fun_params = {"sizing_constant": cp.Parameter(nonneg=True,
                                                                name=f"cost_learning_curve_{self.asset_name}")}
        self.conversion_fun_params = {"existing_capacity": cp.Parameter(nonneg=True,
                                                                        name=f"existing_capacity_{self.asset_name}")}
        self.conversion_fun_params_2 = {"baseline_country_supply": cp.Parameter(nonneg=True,
                                                                               name=f"baseline_country_supply{self.asset_name}"), 
                                        "country_supply_growth": cp.Parameter(nonneg=True,
                                                                               name=f"country_supply_growth{self.asset_name}"),
                                        "global_supply_growth": cp.Parameter(nonneg=True,
                                                                               name=f"global_supply_growth{self.asset_name}"),
                                        "baseline_global_supply": cp.Parameter(nonneg=True,
                                                                               name=f"baseline_global_supply{self.asset_name}"),
                                        "first_year_multiplier": cp.Parameter(nonneg=True,
                                                                               name=f"first_year_multiplier{self.asset_name}"),
                                        "max_global_share": cp.Parameter(shape=(), nonneg=True,
                                                                               name=f"max_global_share{self.asset_name}"),}
        self.conversion_fun_params_3 = {"tech_potential": cp.Parameter(nonneg=True,
                                                                       name=f"tech_potential_{self.asset_name}")}
        return
    
    def define_structure(self, asset_structure):
        self.asset_structure = asset_structure
        self.source_node_location = "NULL"
        self.target_node_location = asset_structure["Location_1"]
        # Add node locations for edge 2 (max capacity)
        self.source_node_location_2 = "NULL"
        self.target_node_location_2 = asset_structure["Location_1"]
        
        self.target_node_times = np.arange(asset_structure["Start_Time"], 
                                           asset_structure["End_Time"], 
                                           self.period)
        self.number_of_edges = len(self.target_node_times)
        # Define the number of years in the control horizon
        self.num_years = int(self.network.system_parameters_df.loc["control_horizon", "value"] / 8760)
        self.gen_profile = cp.Parameter(shape = (self.number_of_edges), nonneg=True, name=f"gen_profile_{self.asset_name}")
        # EDITED: set size of RE asset as array of sizes per horizon modeled
        self.flows = cp.Variable(shape=(self.num_years,), nonneg=True, name=f"new_capacity_{self.asset_name}") # New capacities to install per year
        self.cumulative_capacity = np.zeros(shape=(self.num_years,))
        self.cost_fun_params = {"sizing_constant": cp.Parameter(shape=(self.num_years,),
                                                                nonneg=True,
                                                                name=f"cost_learning_curve_{self.asset_name}")}
        self.conversion_fun_params = {"existing_capacity": cp.Parameter(shape=(self.num_years,),
                                                                nonneg=True, name=f"existing_cap_{self.asset_name}"),}
        self.conversion_fun_params_2 = {"baseline_country_supply": cp.Parameter(shape=(), nonneg=True,
                                                                               name=f"baseline_country_supply{self.asset_name}"), 
                                        "country_supply_growth": cp.Parameter(shape=(), nonneg=True,
                                                                               name=f"country_supply_growth{self.asset_name}"),
                                        "global_supply_growth": cp.Parameter(shape=(), nonneg=True,
                                                                               name=f"global_supply_growth{self.asset_name}"),
                                        "baseline_global_supply": cp.Parameter(shape=(), nonneg=True,
                                                                               name=f"baseline_global_supply{self.asset_name}"),
                                        "first_year_multiplier": cp.Parameter(shape=(), nonneg=True,
                                                                               name=f"first_year_multiplier{self.asset_name}"),
                                        "max_global_share": cp.Parameter(shape=(), nonneg=True,
                                                                               name=f"max_global_share{self.asset_name}"),}
        self.conversion_fun_params_3 = {"tech_potential": cp.Parameter(nonneg=True,
                                                                       name=f"tech_potential_{self.asset_name}")}
        self.year_change_indices = self._get_year_change_indices()
        self.asset_lifetime = 20 # hard-coded for testing
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
        
        new_edge.flow = (self.cumulative_new_installed[index_number] + 
                         self.conversion_fun_params["existing_capacity"][index_number])\
            * self.gen_profile[edge_number]
        return
    
    def build_max_capacity_edges(self):
        """
        Builds the edges to constrain maximum capacity based on the previous year's
        installs, considering both country ramp limits and global supply growth.
        The first-year multiplier affects both the ramp limit and the country’s
        share of the global supply chain. A maximum share of global supply is enforced.
        """
        source_node_type = self.source_node_type_2
        source_node_location = self.source_node_location_2
        target_node_type = self.target_node_type_2
        target_node_location = self.target_node_location_2
    
        source_node_time = 0
        target_node_time = 0
    
        # Initial baselines (GW/year for 2024)
        init_inst_param = self.conversion_fun_params_2["baseline_country_supply"]
        init_global_param = self.conversion_fun_params_2["baseline_global_supply"]
    
        # Growth rates
        country_growth = self.conversion_fun_params_2["country_supply_growth"]
        global_growth = self.conversion_fun_params_2["global_supply_growth"]
    
        # Maximum allowed fraction of global supply chain for the country
        max_share = self.conversion_fun_params_2["max_global_share"]
    
        # Build vector of "previous installs" (init + flows[...])
        prev_installs_components = [init_inst_param]
        prev_installs_components += [self.flows[i] for i in range(self.num_years - 1)]
        self.prev_installs = cp.hstack(prev_installs_components)  # shape (num_years,)
    
        # First year multiplier for ramp
        first_year_multiplier = self.conversion_fun_params_2["first_year_multiplier"]
        first_allowed = first_year_multiplier * init_inst_param
    
        # --- Country ramp-limited installs ---
        country_allowable = []
        for t in range(self.num_years):
            if t == 0:
                country_allowable.append(first_allowed)
            else:
                country_allowable.append((1.0 + country_growth) * self.prev_installs[t])
        country_allowable = cp.hstack(country_allowable)
    
        # --- Global supply path (based on first_allowed share in year 0) ---
        country_share = first_allowed / init_global_param
        global_supply = [init_global_param * country_share]  # year 0 allocation
        for t in range(1, self.num_years):
            global_supply.append(global_supply[-1] * (1.0 + global_growth))
        global_supply = cp.hstack(global_supply)
    
    
        # --- Maximum allowed share of global supply each year (recursive) ---
        share_cap_list = [max_share * init_global_param]  # year 0
        for t in range(1, self.num_years):
            share_cap_list.append(share_cap_list[-1] * (1 + global_growth))
        share_cap = cp.hstack(share_cap_list)
            
        # --- Interaction: min of country ramp, global allocation, and max share ---
        self.allowable_installs = cp.minimum(country_allowable,share_cap)
    
        # Create / append an edge that represents allowable_installs - actual installs
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
    
        # Set the flow expression to allowable - actual installs (vector)
        edge.flow = self.allowable_installs - self.flows

        
    def build_tech_potential_edges(self, year_number):
        source_node_type = "NULL"
        source_node_location = self.source_node_location_2
        target_node_type = "RE_PV_Tech"
        target_node_location = self.target_node_location_2
        
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
    
    def build_edges(self):
        self.edges = []
        for hour in range(self.number_of_edges):
            self.build_edge(hour)
        self.build_max_capacity_edges()
        # for year in range(self.num_years):
        #     self.build_max_capacity_edges(year)
        for year in range(self.num_years):    
            self.build_tech_potential_edges(year)
        
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
        cost_array = self.cost_fun_params["sizing_constant"]
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
        discount_factor = (1 + discount_rate) ** i # Discounting to year 0
        valid_mask = (i >= j) & (i < j + asset_lifetime)
    
        amortised_j = cp.reshape(amortised_cost, (1, project_years), order="F")  # shape (1, years)
        flows_j = cp.reshape(self.flows, (1, project_years), order="F")  # shape (1, years)
    
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
        # Sample evenly spaced days across each sampled year
        for year in range(n_years):
            year_start = year * hours_per_year
            for d in range(days_per_year):
                # Spread days evenly across the sampled year
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
        # Returns size of asset for Total installed RE, which is a vector #
        return self.flows.value
    
    def asset_size(self):
        # Returns size of asset for Total installed RE, which is a vector #
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
        yearly_flows = [flow for flow in yearly_flows]
        return yearly_flows