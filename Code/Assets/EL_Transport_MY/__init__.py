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

class EL_Transport_MY_asset(Asset_STEVFNs):
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
        self.cost = self._get_amortised_sizing_cost() + self._get_discounted_usage_cost()
        return
    
    def __init__(self):
        super().__init__()
        self.cost_fun_params = {"sizing_constant": cp.Parameter(nonneg=True),
                          "usage_constant": cp.Parameter(nonneg=True)}
        self.conversion_fun_params = {"conversion_factor": cp.Parameter(nonneg=True)}
        return
    
    def define_structure(self, asset_structure):
        self.asset_structure = asset_structure
        self.source_node_location = asset_structure["Location_1"]
        # Define node times with delay of operation 10 years ahead of start_time
        self.source_node_times = np.arange(asset_structure["Start_Time"] + 87600, 
                                           asset_structure["End_Time"], 
                                           asset_structure["Period"])
        self.target_node_location = asset_structure["Location_2"]
        self.target_node_times = np.arange(asset_structure["Start_Time"] + 87600, 
                                           asset_structure["End_Time"], 
                                           asset_structure["Period"])
        self.number_of_edges = len(self.source_node_times)
        self.target_node_times = self.target_node_times + asset_structure["Transport_Time"]
        self.target_node_times = self.target_node_times % asset_structure["End_Time"]
        self.num_years = int(self.network.system_parameters_df.loc["control_horizon", "value"] / 8760)
        start_time = asset_structure["Start_Time"]
        self.start_year = int(start_time // 8760)
        self.flows = cp.Variable(self.number_of_edges*2, nonneg = True)
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
        for counter1 in self.source_node_times: #apply delay in operation according to Start_Time and End_Time
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

        sizing_constant = self.cost_fun_params["sizing_constant"].value
        sizing_cost = sizing_constant * cp.max(self.flows)
        annualised_payment = sizing_cost * amort_factor

        discount_vector = [(1 + discount_rate) ** -i for i in range(self.start_year, min(self.start_year + n, project_years))]
        return annualised_payment * sum(discount_vector)

    def _get_discounted_usage_cost(self):
        discount_rate = float(self.network.system_parameters_df.loc["discount_rate", "value"])
        usage_constant = self.cost_fun_params["usage_constant"].value
        project_years = self.num_years

        self._get_year_change_indices()
        self.usage_costs = []
        for y in range(project_years):
            start_idx = self.year_change_indices[y]
            end_idx = self.year_change_indices[y + 1] if y + 1 < len(self.year_change_indices) else self.number_of_edges
            
            forward_flow = cp.sum(self.flows[start_idx:end_idx])
            reverse_start = self.number_of_edges + start_idx
            reverse_end = self.number_of_edges + end_idx
            reverse_flow = cp.sum(self.flows[reverse_start:reverse_end])
            total_flow = forward_flow + reverse_flow

            
            # total_flow = cp.sum(self.flows[start_idx:end_idx]) + cp.sum(
            #     self.flows[self.number_of_edges + start_idx : self.number_of_edges + end_idx]
            # )
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
        hours_per_day = 24
        num_years = self.num_years
        days_per_year = int((self.number_of_edges / hours_per_day) / num_years)
        hours_per_year = days_per_year * hours_per_day
        self.year_change_indices = [i * hours_per_year for i in range(num_years)]
        return self.year_change_indices
    
    
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
        Columns:
            - forward_year_0, forward_year_1, ..., forward_year_n
            - reverse_year_0, reverse_year_1, ..., reverse_year_n
        Values are CVXPY expressions (or numbers if problem is solved).
        """
        self._get_year_change_indices()
        project_years = self.num_years
        data = {}
    
        for y in range(project_years):
            start_idx = self.year_change_indices[y]
            end_idx = self.year_change_indices[y + 1] if y + 1 < len(self.year_change_indices) else self.number_of_edges
    
            forward_flow = self.flows[start_idx:end_idx]
            reverse_flow = self.flows[self.number_of_edges + start_idx : self.number_of_edges + end_idx]
    
            data[f"forward_year_{y}"] = [cp.sum(forward_flow)]
            data[f"reverse_year_{y}"] = [cp.sum(reverse_flow)]
    
        return pd.DataFrame(data)
