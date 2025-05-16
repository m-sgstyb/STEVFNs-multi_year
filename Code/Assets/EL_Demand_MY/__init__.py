#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 12:24:49 2025

@author: Mónica Sagastuy-Breña
Based on EL_Demand_Asset by:
@author: aniqahsan
"""

import numpy as np
import cvxpy as cp
import pandas as pd
import os
from ..Base_Assets import Asset_STEVFNs
from ...Network import Edge_STEVFNs

class EL_Demand_MY_Asset(Asset_STEVFNs):
    """Class of Electricity Demand Asset"""
    asset_name = "EL_Demand_MY"
    node_type = "EL"
    def __init__(self):
        super().__init__()
        return
    
    def define_structure(self, asset_structure):
        self.node_location = asset_structure["Location_1"]
        self.node_times = np.arange(
            asset_structure["Start_Time"],
            asset_structure["End_Time"],
            asset_structure["Period"]
        )
        self.number_of_edges = len(self.node_times)
        self.flows = cp.Parameter(shape=self.number_of_edges, nonneg=True)
        return

    def build_costs(self):
        self.cost = cp.Constant(0)
        return

    def build_edge(self, edge_number):
        """Method that Builds Edges for EL_Demand Asset"""
        node_time = self.node_times[edge_number]
        new_edge = Edge_STEVFNs()
        self.edges += [new_edge]
        new_edge.attach_source_node(self.network.extract_node(
            self.node_location, self.node_type, node_time))
        new_edge.flow = self.flows[edge_number]
        new_edge.conversion_fun = lambda x, params: x
        new_edge.conversion_fun_params = {}
        return

    def build_edges(self):
        self.edges = []
        for edge_number in range(self.number_of_edges):
            self.build_edge(edge_number)

    def _update_parameters(self):
        profile_filename = self.parameters_df["profile_filename"] + r".csv"
        profile_path = os.path.join(self.parameters_folder, "profiles", profile_filename)
        profile_df = pd.read_csv(profile_path)

        # Optional: make column name flexible
        demand_column = self.parameters_df.get("profile_column", "Demand")
        full_profile = np.array(profile_df[demand_column])

        set_size = self.parameters_df["set_size"]
        set_number = self.parameters_df["set_number"]
        n_sets = int(np.ceil(self.number_of_edges / set_size))
        gap = int(len(full_profile) / (n_sets * set_size)) * set_size
        offset = set_size * set_number

        new_profile = np.zeros(n_sets * set_size)
        for counter1 in range(n_sets):
            old_loc_0 = offset + gap * counter1
            old_loc_1 = old_loc_0 + set_size
            new_loc_0 = set_size * counter1
            new_loc_1 = new_loc_0 + set_size
            new_profile[new_loc_0:new_loc_1] = full_profile[old_loc_0:old_loc_1]

        self.flows.value = new_profile[:self.number_of_edges]
        return

    def _get_year_change_indices(self):
        timesteps = self.number_of_edges
        num_years = int(self.network.system_parameters_df.loc["control_horizon", "value"] / 8760)
        total_length = 8760 * num_years

        set_size = self.parameters_df["set_size"]
        set_number = self.parameters_df["set_number"]
        n_sets = int(np.ceil(timesteps / set_size))
        gap = int(total_length / (n_sets * set_size)) * set_size
        offset = set_size * set_number

        self.year_change_indices = []
        last_year = -1

        for counter1 in range(n_sets):
            old_loc_0 = offset + gap * counter1
            new_loc_0 = set_size * counter1
            current_year = old_loc_0 // 8760
            if current_year != last_year:
                self.year_change_indices.append(new_loc_0)
                last_year = current_year

        return self.year_change_indices

    def component_size(self):
        """Returns yearly demand totals (e.g., for reporting or budget comparison)"""
        if self.flows.value is None:
            return None

        year_indices = self._get_year_change_indices()
        year_indices.append(len(self.flows.value))

        yearly_totals = [
            np.sum(self.flows.value[start:end])
            for start, end in zip(year_indices[:-1], year_indices[1:])
        ]
        return np.array(yearly_totals)

    def get_asset_sizes(self):
        asset_identity = f"{self.asset_name}_location_{self.node_location}"
        return {asset_identity: self.component_size()}    
    
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