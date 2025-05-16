#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 17:55:37 2025

@author: Mónica Sagastuy-Breña
Based on CO2_Budget_Asset by:
@author: aniqahsan
"""

import os
import numpy as np
import cvxpy as cp
from ..Base_Assets import Asset_STEVFNs
from ...Network import Edge_STEVFNs


class CO2_Budget_MY_Asset(Asset_STEVFNs):
    """Annual CO₂ Budget asset enforcing yearly emissions limit."""
    asset_name = "CO2_Budget_MY"
    source_node_type = "NULL"
    source_node_time = 0
    target_node_type = "CO2_Budget"
    target_node_time = 0
    period = 1
    transport_time = 0

    @staticmethod
    def conversion_fun(flows, params):
        # This returns the annual CO₂ limit directly
        return params["maximum_budget"]

    def __init__(self):
        super().__init__()
        self.num_years = None  # set later
        self.flows = None
        self.conversion_fun_params = {
            "maximum_budget": cp.Parameter(nonneg=True)
        }
        return

    def define_structure(self, asset_structure):
        self.source_node_location = 0
        self.target_node_location = 0
        self.num_years = int(self.network.system_parameters_df.loc["control_horizon", "value"] / 8760)

        # Set a dummy constant; this gets overwritten anyway
        self.flows = cp.Constant(np.zeros(self.num_years))

        # Correct shape for the annual limit
        self.conversion_fun_params["maximum_budget"] = cp.Parameter(
            shape=(self.num_years,), nonneg=True
        )
        return

    def process_csv_values(self, values):
        if isinstance(values, str):
            return np.array([float(x) for x in values.split(",")], dtype=float)
        elif isinstance(values, float) or isinstance(values, int):
            return np.full(self.num_years, values)
        elif isinstance(values, (list, np.ndarray)):
            return np.array(values, dtype=float)
        return np.zeros(self.num_years)

    def build_edge(self):
        self.budget_edge = Edge_STEVFNs()
        self.edges.append(self.budget_edge)

        self.budget_edge.attach_target_node(
            self.network.extract_node(
                self.target_node_location, self.target_node_type, self.target_node_time
            )
        )

        # Flow gets overwritten later using input edge expressions
        self.budget_edge.flow = self.flows
        self.budget_edge.conversion_fun = self.conversion_fun
        self.budget_edge.conversion_fun_params = self.conversion_fun_params
        return

    def build_edges(self):
        self.edges = []
        self.build_edge()

    def _update_parameters(self):
        # Update the max emissions budget per year
        for name, param in self.conversion_fun_params.items():
            param.value = self.process_csv_values(self.parameters_df[name])
            
        self.get_input_emissions_expressions()
        return

    def get_input_emissions_expressions(self):
        # Sum incoming emissions from all attached edges (excluding own budget edge)
        co2_node = self.edges[0].target_node
        emissions_exprs = [
            edge.extract_flow()
            for edge in co2_node.input_edges
            if edge is not self.budget_edge
        ]

        # These should already be (yearly) vectors → sum them
        if emissions_exprs:
            self.flows = cp.sum(cp.vstack(emissions_exprs), axis=0)
        else:
            self.flows = cp.Constant(np.zeros(self.num_years))
        return self.flows

    def component_size(self):
        return -self.flows.value if self.flows is not None else np.zeros(self.num_years)

    def get_plot_data(self):
        return -self.flows.value if self.flows is not None else np.zeros(self.num_years)


    