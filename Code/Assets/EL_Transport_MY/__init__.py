#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 31 16:36:45 2025

@author: Mónica Sagastuy-Breña
Based on EL_Transport_Asset by:
@author: aniqahsan
"""

import os
import numpy as np
import pandas as pd
import cvxpy as cp
from amortization.amount import calculate_amortization_amount as amort
from ..Base_Assets import Asset_STEVFNs
from ...Network import Edge_STEVFNs

class EL_Transport_MY_Asset(Asset_STEVFNs):
    '''
    Class of ELectricity Transport asset for multi-year adaptation of STEVFNs
    '''
    asset_name = "EL_Transport_MY"
    source_node_type = "EL"
    target_node_type = "EL"
    period = 1
    transport_time = 0
    
    @staticmethod
    def conversion_fun(flows, params):
        conversion_factor = params["conversion_factor"]
        return conversion_factor * flows
    
    def build_cost(self):
        '''calculate amortised and discounted payments for this asset'''
        capex = self._get_amortised_sizing_cost()
        opex = self._get_discounted_usage_cost()
        self.cost = capex + opex
        return
    
    def __init__(self):
        super().__init__()
        self.cost_fun_params = {"sizing_constant": cp.Parameter(nonneg=True),
                          "usage_constant": cp.Parameter(nonneg=True)}
        self.conversion_fun_params = {"conversion_factor": cp.Parameter(nonneg=True)}
        return
    
    def define_structure(self, asset_structure):
        self.asset_structure = asset_structure
        self.num_years = int(self.network.system_parameters_df.loc["control_horizon", "value"] / 8760)
        decision_time = asset_structure["Start_Time"]
        horizon_end = asset_structure["End_Time"]
        sampled_length = horizon_end - decision_time
        hours_per_year = sampled_length / self.num_years # Determines the per-year chunks of time sampled per direction
        # Effective operation delay in hours, scaled to sampling
        delay_years = 10
        delay = delay_years * hours_per_year
        self.source_node_location = asset_structure["Location_1"]
        # Define node times with delay of operation 10 years ahead of start_time
        self.source_node_times = np.arange(asset_structure["Start_Time"] + delay, 
                                           asset_structure["End_Time"], 
                                           asset_structure["Period"]).astype(int)
        self.target_node_location = asset_structure["Location_2"]
        self.target_node_times = np.arange(asset_structure["Start_Time"] + delay, 
                                           asset_structure["End_Time"], 
                                           asset_structure["Period"]).astype(int)
        self.number_of_edges = len(self.source_node_times)
        self.target_node_times = self.target_node_times + asset_structure["Transport_Time"]
        self.target_node_times = self.target_node_times % asset_structure["End_Time"]
        start_time = asset_structure["Start_Time"] # For amortised cost
        self.start_year = int(start_time // 8760) # for amortised cost calc
        self.flows = cp.Variable(self.number_of_edges*2, nonneg = True, name=f"flows_{self.asset_name}")
        self.cost_fun_params = {"sizing_constant": cp.Parameter(nonneg=True, name=f"sizing_constant_{self.asset_name}"),
                          "usage_constant": cp.Parameter(nonneg=True, name=f"usage_constant_{self.asset_name}")}
        self.conversion_fun_params = {"conversion_factor": cp.Parameter(nonneg=True, name=f"conv_factor_{self.asset_name}")}
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
        new_edge.flow = self.flows[edge_number]
        new_edge.conversion_fun = self.conversion_fun
        new_edge.conversion_fun_params = self.conversion_fun_params
        return
    
    def build_edge_opposite(self, edge_number):
        """Builds edge in the opposite direction for bi-directional trade"""
        source_node_time = self.source_node_times[edge_number]
        target_node_time = self.target_node_times[edge_number]
        new_edge = Edge_STEVFNs()
        self.edges += [new_edge]
        # Source node becomes target node and vice-versa
        new_edge.attach_source_node(self.network.extract_node(
            self.target_node_location, self.target_node_type, source_node_time))
        new_edge.attach_target_node(self.network.extract_node(
            self.source_node_location, self.source_node_type, target_node_time))
        new_edge.flow = self.flows[self.number_of_edges + edge_number]
        new_edge.conversion_fun = self.conversion_fun
        new_edge.conversion_fun_params = self.conversion_fun_params
        return
    
    def build_edges(self):
        self.edges = []
        for counter1 in range(self.number_of_edges):
            self.build_edge(counter1)
            self.build_edge_opposite(counter1)
        return
    
    def _get_amortised_sizing_cost(self):
        interest_rate = float(self.network.system_parameters_df.loc["interest_rate", "value"])
        discount_rate = float(self.network.system_parameters_df.loc["discount_rate", "value"])
        asset_lifetime = 50
        project_years = self.num_years

        r = interest_rate
        n = asset_lifetime
        amort_factor = (r * (1 + r) ** n) / ((1 + r) ** n - 1)
        try:
            sizing_constant = self.cost_fun_params["sizing_constant"]
            sizing_cost = sizing_constant * cp.max(self.flows)
            annualised_payment = sizing_cost * amort_factor
        except Exception as e:
            print("Could not update sizing constant, sizing cost or annualised payment")
        # print("ANNUALISED PAYMENT", annualised_payment)
        try:
            discount_vector = [(1 + discount_rate) ** -i for i in range(self.start_year, min(self.start_year + n, project_years))]
        except Exception as e:
            print("tried getting discount vector but:", e)
        
        return annualised_payment * sum(discount_vector)
    
    def _get_discounted_usage_cost(self):
        discount_rate = float(self.network.system_parameters_df.loc["discount_rate", "value"])
        usage_constant = self.cost_fun_params["usage_constant"]
        project_years = self.num_years
        print("getting year change indices")
        self._get_year_change_indices()
        self.usage_costs = []
        print("going into loop to determine usage NPV cost per year")
        for y in range(project_years):
            start_idx = self.year_change_indices[y]
            end_idx = self.year_change_indices[y + 1]  # always safe now
            
            forward_flow = cp.sum(self.flows[start_idx:end_idx])
            reverse_start = self.number_of_edges + start_idx
            reverse_end = self.number_of_edges + end_idx
            reverse_flow = cp.sum(self.flows[reverse_start:reverse_end])
            total_flow = forward_flow + reverse_flow
    
            discounted_cost = (usage_constant * total_flow) / ((1 + discount_rate) ** y)
            self.usage_costs.append(discounted_cost)
    
        return cp.sum(self.usage_costs)

    
    def _update_distance(self):
        #Function that calculates approximate distance between the source and target nodes "as the bird flies"#
        lat_lon_0 = self.network.lat_lon_df.iloc[int(self.source_node_location)]
        lat_lon_1 = self.network.lat_lon_df.iloc[int(self.target_node_location)]
        lat_0 = lat_lon_0["lat"]/180 * np.pi
        lat_1 = lat_lon_1["lat"]/180 * np.pi
        lon_d = (lat_lon_1["lon"] - lat_lon_0["lon"])/180 * np.pi
        a = np.sin((lat_1 - lat_0)/2)**2 + np.cos(lat_0) * np.cos(lat_1) * np.sin(lon_d/2)**2
        c = 2 * np.arctan2(a**0.5, (1-a)**0.5)
        R = 6.371 # in Mm radius of the earth
        self.distance = R * c # in Mm
        return 
    
    def _get_year_change_indices(self):
        """
        Computes the indices corresponding to the start of each modelled year,
        based on the original simulation time (not delayed edge times).
        """
        start_time = self.asset_structure["Start_Time"]
        end_time = self.asset_structure["End_Time"]
        period = self.asset_structure["Period"]
        
        total_hours = end_time - start_time
        hours_per_year = total_hours / self.num_years
    
        # Build timepoints from Start_Time using the same resolution as asset_period
        time_points = np.arange(start_time, end_time, period)
    
        # Calculate the model time at the start of each year (relative to start_time)
        self.year_change_indices = []
        for y in range(self.num_years):
            year_start_time = start_time + y * hours_per_year
    
            # Find the first index in time_points that is >= year_start_time
            idx = np.searchsorted(time_points, year_start_time, side='left')
            self.year_change_indices.append(idx)
            
        self.year_change_indices.append(total_hours)
    
    
    def _update_parameters(self):
        self._update_distance()
        for parameter_name, parameter in self.cost_fun_params.items():
            parameter.value = (
                self.parameters_df[parameter_name + r"_1"] + self.parameters_df[parameter_name + r"_2"] * self.distance
            )
        for parameter_name, parameter in self.conversion_fun_params.items():
            parameter.value = 1 - (
                self.parameters_df[parameter_name + r"_1"]
                + self.parameters_df[parameter_name + r"_2"] * self.distance
            )
    
    def get_yearly_flows(self):
        """
        Returns a DataFrame with yearly flows for each direction of the transport asset.
        Skips years beyond the transport asset's defined time window.
        """
        self._get_year_change_indices()
        year_indices = self.year_change_indices.copy()
        source = self.source_node_location
        target = self.target_node_location
        sampled_year_hours = year_indices[1] - year_indices[0]
        reverse_flow_offset = sampled_year_hours * 10 # Assumes always a 10 year lead time for HVDC installation
       # first_operational_hour = self.source_node_times[0] #if len(self.source_node_times) > 0 else float('inf')
    
        data = {}
        lengths = []
        
        # Determine number of actual available years based on flow array length
        max_index = len(self.flows.value) / 2
        year_limits = []
        for y in range(len(year_indices) - 1):
            if year_indices[y] < max_index:
                year_limits.append(y)
        
        for y in range(len(year_indices) - 1):
            real_year = y  # actual model year (0-based)
            start_idx = year_indices[y]
            end_idx = year_indices[y + 1]
        
            if y < 10:  # before operations start, hardcoded, needs to be depending on source node times
                print(f"Start index at {y}:", start_idx)
                print(f"End index at {y}:", end_idx)
                # forward_flow = self.flows[start_idx:end_idx].value
                # reverse_flow = self.flows[int(max_index + start_idx):int(max_index + end_idx)].value
                # print(f"Forward flow value in year {y}", forward_flow)
                # print(f"Year {y}: forward_flow shape {forward_flow.shape}")
                # print(f"Year {y}: reverse_flow shape {reverse_flow.shape}")
                forward_flow = np.full(end_idx - start_idx, 0)
                reverse_flow = np.full(end_idx - start_idx, 0)
            else:
                start_idx = year_indices[y - 10]
                end_idx = year_indices[y - 9]
                print(f"Start index at {y}:", start_idx)
                print(f"End index at {y}:", end_idx)
                forward_flow = self.flows[start_idx:end_idx].value
                reverse_flow = self.flows[int(max_index + start_idx):int(max_index + end_idx)].value
                # print(f"Year {y}: forward_flow shape {forward_flow.shape}")
                # print(f"Year {y}: reverse_flow shape {reverse_flow.shape}")
            data[f"{source}-{target}_year_{real_year}"] = forward_flow
            data[f"{target}-{source}_year_{real_year}"] = reverse_flow
    
            lengths.append(len(forward_flow))
            lengths.append(len(reverse_flow))
    
        # Find max length and pad with NaNs
        max_len = max(lengths)
        for key in data:
            array = data[key]
            if len(array) < max_len:
                data[key] = np.pad(array, (0, max_len - len(array)), constant_values=np.nan)
    
        return pd.DataFrame(data)
