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
        '''calculate amortised and discounted payments for this asset see coments below '''
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
        self.source_node_times = np.arange(asset_structure["Start_Time"],
                                           asset_structure["End_Time"],
                                           asset_structure["Period"])
        self.target_node_location = asset_structure["Location_2"]
        self.target_node_times = np.arange(asset_structure["Start_Time"],
                                           asset_structure["End_Time"],
                                           asset_structure["Period"])
        self.target_node_times = self.target_node_times + asset_structure["Transport_Time"]
        self.target_node_times = self.target_node_times % asset_structure["End_Time"]
        self.flows = cp.Variable(self.number_of_edges*2, nonneg = True)
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
        super().build_edges()
        for counter1 in range(self.number_of_edges):
            self.build_edge_opposite(counter1)
        return
    
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
    
    # Insert adaptation of cost functions to: 
        # 1. amortise sizing_constant * cp.max(flows) (investment per capacity)
        # 2. Bring amortisation to Net present value
        # 3. Bring usage costs to net present value, not amortised
        # 4. update_sizing and update_usage constants methods should not be required. 
        # 5. Final total cost should be build in def build_cost(self)     
     
    # Insert _get_year_change_indices method as a helper method
    # This should help determine the usage cost net present value
    # If the hvdc cable starts operating in year 10, the trade in its first year operation should be NPV to 10 years before
    # so self.flows[x:y] where x is the sampled first hour of year 10 and y is the last hour of the sampled year 10, the sum of that times the usage cost is the cost in that year. NPV it
    # But be Careful: The flows in the other direction will just follow after the first direction 
    # So the usage cost for that year should match that too. for example:
        # If sampling 4320 hours over 30 years, therefore each year it takes 168 hours as representative
        # the first year's hours for usage cost is [0:168] but that is only in the direction of location 1 to 2
        # To add the usage for location 2 to 1 we need to slice [4320:4320+168]  
    
    def _update_sizing_constant(self):
        N = np.ceil(self.network.system_parameters_df.loc["project_life", "value"]/self.parameters_df["lifespan"])
        r = (1 + self.network.system_parameters_df.loc["discount_rate", "value"])**(-self.parameters_df["lifespan"]/8760)
        NPV_factor = (1-r**N)/(1-r)
        self.cost_fun_params["sizing_constant"].value = self.cost_fun_params["sizing_constant"].value * NPV_factor
        return
    
    def _update_usage_constant(self):
        simulation_factor = 8760/self.network.system_structure_properties["simulated_timesteps"]
        N = np.ceil(self.network.system_parameters_df.loc["project_life", "value"]/8760)
        r = (1 + self.network.system_parameters_df.loc["discount_rate", "value"])**-1
        NPV_factor = (1-r**N)/(1-r)
        self.cost_fun_params["usage_constant"].value = (self.cost_fun_params["usage_constant"].value * 
                                                        NPV_factor * simulation_factor)
        return
    
    def _update_parameters(self):
        #update distance#
        self._update_distance()
        #update parameters using self.parameters_df and self.distance#
        for parameter_name, parameter in self.cost_fun_params.items():
            parameter.value = (self.parameters_df[parameter_name + r"_1"] + 
                               self.parameters_df[parameter_name + r"_2"] * self.distance)
        for parameter_name, parameter in self.conversion_fun_params.items():
            parameter.value = 1 - (self.parameters_df[parameter_name + r"_1"] + 
                               self.parameters_df[parameter_name + r"_2"] * self.distance)
        #Update cost parameters based on NPV#
        self._update_sizing_constant()
        self._update_usage_constant()
        return
    
    # Add a get_yearly_flows method
    # Where it gets, separately, the yearly flows in each direction
    # this is for plotting purposes, remember to use the _get_year_change_indices method to help with this
    
    

    