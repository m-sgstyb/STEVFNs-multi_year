#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 10:49:50 2023

@author: Mónica Sagastuy Breña

The functions in this script are hard-coded to export modelling results within the
case studies in the author's DPhil work. They may serve as guidance on how to 
obtain results from the model but changes to the network structure may require editing
of these functions.

Functions here serve to export asset data such as cost and sizes, along with total
emissions, and the power flows per hour for a given case study/scenario. 

They need to be called upon in main.py
"""

import pandas as pd
import numpy as np


def export_scenario_results(my_network, scenario_name, simulation_factor=1.0):
    print("========= Exporting summary scenario results ========")
    num_years = my_network.assets[0].num_years
    years = list(range(1, num_years + 1))
    # print(f"Expected number of years: {num_years}")
    discount_rate = float(my_network.system_parameters_df.loc["discount_rate", "value"])
    discount_factors = 1 / ((1 + discount_rate) ** np.arange(num_years))
    data = {}

    def safe_assign(key, value):
        try:
            if hasattr(value, '__len__') and not isinstance(value, str):
                val_len = len(value)
                if val_len != num_years:
                    print(f"  ❗ Length mismatch for '{key}': got {val_len}, expected {num_years}")
            data[key] = value
        except Exception as e:
            print(f"❌ Error assigning '{key}': {e}")
            raise

    # Base columns
    safe_assign("year", years)
    safe_assign("scenario", [scenario_name] * num_years)

    # Placeholders for totals
    safe_assign("Annual_Emissions", [0] * num_years)
    safe_assign("Total_Annual_Demand", [0] * num_years)
    safe_assign("Discounted_Annual_Demand", [0] * num_years)

    safe_assign("Total_Annual_Fossil_Gen_GWh", [0] * num_years)
    safe_assign("Discounted_Annual_Fossil_Gen_GWh", [0] * num_years)

    # CAPEX + OPEX columns
    capex_cols = []
    opex_cols = []

    for asset in my_network.assets[1:]:
        name = asset.asset_name
        print(f"\nProcessing asset: {name}")

        # Demand assets
        if hasattr(asset, "peak_demand"):
            total_annual_demand = [d * simulation_factor for d in asset.asset_size()]
            safe_assign("Total_Annual_Demand", total_annual_demand)
            discounted_demand = np.array(total_annual_demand) * discount_factors
            safe_assign("Discounted_Annual_Demand", discounted_demand.tolist())

        # Conventional generation (fossil, etc.)
        if hasattr(asset, "get_yearly_emissions"):
            yearly_emissions = asset.get_yearly_emissions()
            safe_assign("Annual_Emissions", yearly_emissions)

            flows = asset.get_yearly_flows()[:num_years]
            summed_flows = [(sum(year) * simulation_factor) for year in flows]
            safe_assign("Total_Annual_Fossil_Gen_GWh", summed_flows)
            discounted_flows = np.array(summed_flows) * discount_factors
            safe_assign("Discounted_Annual_Fossil_Gen_GWh", discounted_flows.tolist())

            opex_list = asset.get_yearly_usage_costs()
            colname = f"{name}_annual_OPEX_BUSD"
            safe_assign(colname, opex_list)
            opex_cols.append(colname)

        # RE capacity and generation
        if hasattr(asset, "cumulative_new_installed"):
            flows = np.array(getattr(asset.flows, "value", asset.flows), dtype=float)
            cumulative_installed = np.array(asset.cumulative_new_installed.value, dtype=float)
            existing_capacity_val = np.array(asset.conversion_fun_params["existing_capacity"].value, dtype=float)

            safe_assign(f"{name}_new_annual_installed_GWp", flows.tolist())
            safe_assign(f"{name}_total_capacity_GWp", (cumulative_installed + existing_capacity_val).tolist())

            annual_gen = np.sum(asset.get_yearly_flows(), axis=1)
            annual_gen = [g * simulation_factor for g in annual_gen]
            safe_assign(f"{name}_annual_generation_GWh", annual_gen)
            safe_assign(f"{name}_discounted_annual_gen_GWh", (np.array(annual_gen) * discount_factors).tolist())

            payments_M = getattr(asset, "payments_M", None)
            if payments_M is not None:
                payments = getattr(payments_M, "value", payments_M)
                annual_payments = np.sum(payments, axis=1) if payments is not None else [0] * num_years
            else:
                annual_payments = [0] * num_years
            colname = f"{name}_annual_CAPEX_BUSD"
            safe_assign(colname, annual_payments)
            capex_cols.append(colname)

    print("\n✅ All data lengths checked. Creating DataFrame...")
    time_series_df = pd.DataFrame(data)

    # Totals
    total_capex = time_series_df[capex_cols].sum(axis=1) if capex_cols else 0
    total_opex = time_series_df[opex_cols].sum(axis=1) if opex_cols else 0
    total_demand = time_series_df["Discounted_Annual_Demand"]
    gen_cols = [c for c in time_series_df.columns if "_discounted_annual_gen_GWh" in c or "Discounted_Annual_Fossil_Gen_GWh" in c]
    total_gen = time_series_df[gen_cols].sum(axis=1) if gen_cols else 0

    # LCOE & LCUE (USD/MWh → USD/kWh)
    time_series_df["System_LCOE_USD_per_kWh"] = ((total_capex + total_opex) / total_gen) * 1000
    time_series_df["System_LCUE_USD_per_kWh"] = ((total_capex + total_opex) / total_demand) * 1000

    # ===========================
    # SUMMARY DF PER ASSET
    # ===========================
    system_cost = my_network.problem.value
    cost_summary = []
    for asset in my_network.assets[1:]:
        if hasattr(asset, "get_yearly_payments"):
            capex_list = asset.get_yearly_payments()
        else:
            payments_M = getattr(asset, "payments_M", None)
            if payments_M is not None:
                payments = getattr(payments_M, "value", payments_M)
                capex_list = np.sum(payments, axis=1) if payments is not None else [0] * num_years
            else:
                capex_list = [0] * num_years
        opex_list = asset.get_yearly_usage_costs() if hasattr(asset, "get_yearly_usage_costs") else [0] * num_years

        total_capex_asset = sum(capex_list)
        total_opex_asset = sum(opex_list)
        total_cost_asset = total_capex_asset + total_opex_asset

        emissions_list = asset.get_yearly_emissions() if hasattr(asset, "get_yearly_emissions") else [0] * num_years
        total_emissions_asset = sum(emissions_list)

        cost_summary.append({
            "scenario": scenario_name,
            "asset_name": asset.asset_name,
            "total_system_cost_BUSD": system_cost,
            "total_capex_BUSD": total_capex_asset,
            "total_opex_BUSD": total_opex_asset,
            "total_cost_BUSD": total_cost_asset,
            "total_emissions_MtCO2e": total_emissions_asset
        })

    summary_df = pd.DataFrame(cost_summary)

    return time_series_df, summary_df



def export_multi_country_scenario_results(my_network, network_structure_df, scenario_name, simulation_factor):
    print("========= Exporting multi-country scenario results ========")
    num_years = my_network.assets[0].num_years
    years = list(range(1, num_years + 1))
    discount_rate = float(my_network.system_parameters_df.loc["discount_rate", "value"])
    discount_factors = 1 / ((1 + discount_rate) ** np.arange(num_years))

    data = {}

    def safe_assign(key, value):
        try:
            if hasattr(value, '__len__') and not isinstance(value, str):
                val_len = len(value)
                # print(f"Assigning '{key}' with length {val_len}")
                if val_len != num_years:
                    print(f"  ❗ Length mismatch for '{key}': got {val_len}, expected {num_years}")
            else:
                print(f"Assigning '{key}' (non-list-like or scalar)")
            data[key] = value
        except Exception as e:
            print(f"❌ Error assigning '{key}': {e}")
            raise

    # Base info
    safe_assign("year", years)
    safe_assign("scenario", [scenario_name] * num_years)

    # Detect locations
    target_locs = sorted({getattr(a, "target_node_location") for a in my_network.assets[1:] if hasattr(a, "target_node_location")})
    demand_locs = sorted({getattr(a, "node_location") for a in my_network.assets[1:] if hasattr(a, "node_location")})

    # Init per-location columns
    for loc in target_locs:
        safe_assign(f"Fossil_Gen_GWh_loc{loc}", [0] * num_years)
        safe_assign(f"Discounted_Fossil_Gen_GWh_loc{loc}", [0] * num_years)
        safe_assign(f"Annual_Emissions_loc{loc}", [0] * num_years)
    for loc in demand_locs:
        safe_assign(f"Total_Annual_Demand_loc{loc}", [0] * num_years)
        safe_assign(f"Discounted_Annual_Demand_loc{loc}", [0] * num_years)

    # Process assets
    for asset in my_network.assets[1:]:
        name = asset.asset_name
    
        # Demand assets
        if hasattr(asset, "peak_demand") and hasattr(asset, "node_location"):
            loc = asset.node_location
            total_annual_demand = [d * simulation_factor for d in asset.asset_size()]
            safe_assign(f"Total_Annual_Demand_loc{loc}", total_annual_demand)
            discounted_demand = np.array(total_annual_demand) * discount_factors
            safe_assign(f"Discounted_Annual_Demand_loc{loc}", discounted_demand.tolist())
    
        # HVDC assets (or anything with both amortised payments and opex)
        if hasattr(asset, "get_yearly_usage_costs") and hasattr(asset, "get_yearly_payments"):
            loc = getattr(asset, "target_node_location")
    
            # OPEX
            opex_list = asset.get_yearly_usage_costs()
            safe_assign(f"{name}_annual_OPEX_BUSD_target_loc{loc}", opex_list)
    
            # CAPEX payments
            capex_list = asset.get_yearly_payments()
            safe_assign(f"{name}_annual_CAPEX_BUSD_target_loc{loc}", capex_list)
    
        # Conventional Generation-specific data
        if hasattr(asset, "get_yearly_emissions") and hasattr(asset, "target_node_location"):
            loc = asset.target_node_location
            yearly_emissions = asset.get_yearly_emissions()
            safe_assign(f"Annual_Emissions_loc{loc}", yearly_emissions)
    
            flows = asset.get_yearly_flows()[:num_years]
            summed_flows = [(sum(year) * simulation_factor) for year in flows]
            safe_assign(f"Fossil_Gen_GWh_loc{loc}", summed_flows)
            discounted_flows = np.array(summed_flows) * discount_factors
            safe_assign(f"Discounted_Fossil_Gen_GWh_loc{loc}", discounted_flows.tolist())
            # OPEX
            opex_list = asset.get_yearly_usage_costs()
            safe_assign(f"{name}_annual_OPEX_BUSD_loc{loc}", opex_list)
    
        # RE Capacity, generation
        if hasattr(asset, "cumulative_new_installed"):
            loc = getattr(asset, "target_node_location")
            flows = np.array(getattr(asset.flows, "value", asset.flows), dtype=float)
            cumulative_installed = np.array(asset.cumulative_new_installed.value, dtype=float)
            existing_capacity_val = np.array(asset.conversion_fun_params["existing_capacity"].value, dtype=float)
    
            safe_assign(f"{name}_new_annual_installed_GWp_loc{loc}", flows.tolist())
            safe_assign(f"{name}_total_capacity_GWp_loc{loc}", (cumulative_installed + existing_capacity_val).tolist())
    
            annual_gen = np.sum(asset.get_yearly_flows(), axis=1)
            annual_gen = [g * simulation_factor for g in annual_gen]
            safe_assign(f"{name}_annual_generation_GWh_loc{loc}", annual_gen)
            safe_assign(f"{name}_discounted_annual_gen_GWh_loc{loc}", (np.array(annual_gen) * discount_factors).tolist())
            
            payments_M = getattr(asset, "payments_M", None)
            if payments_M is not None:
                payments = getattr(payments_M, "value", payments_M)
                annual_payments = np.sum(payments, axis=1) if payments is not None else [0] * num_years
            else:
                annual_payments = [0] * num_years
            safe_assign(f"{name}_annual_CAPEX_BUSD_loc{loc}", annual_payments)

        # Create dataframe
    time_series_df = pd.DataFrame(data)
    # Collect payment & OPEX columns from new naming convention
    payment_cols = [
        col for col in time_series_df.columns
        if "_annual_CAPEX_BUSD_" in col
    ]
    opex_cols = [
        col for col in time_series_df.columns
        if "_annual_OPEX_BUSD_" in col
    ]

    # Demand and generation columns
    demand_cols = [
        col for col in time_series_df.columns
        if col.startswith("Discounted_Annual_Demand_loc")
    ]
    gen_cols = [
        col for col in time_series_df.columns
        if "_discounted_annual_gen_GWh_loc" in col
        or "Discounted_Fossil_Gen_GWh_loc" in col
    ]

    # Totals per year
    total_gen = time_series_df[gen_cols].sum(axis=1)
    total_capex = time_series_df[payment_cols].sum(axis=1)
    total_opex = time_series_df[opex_cols].sum(axis=1)
    total_demand = time_series_df[demand_cols].sum(axis=1)

    # LCOE & LCUE (USD/MWh → USD/kWh)
    time_series_df["System_LCOE_USD_per_kWh"] = ((total_capex + total_opex) / total_gen) * 1000
    time_series_df["System_LCUE_USD_per_kWh"] = ((total_capex + total_opex) / total_demand) * 1000

    # ===========================
    # SUMMARY DF PER ASSET
    # ===========================
    system_cost = my_network.problem.value
    cost_summary = []
    
    for asset in my_network.assets[1:]:
        # Base info
        if hasattr(asset, "get_yearly_payments"):
            capex_list = asset.get_yearly_payments()
        elif hasattr(asset, "yearly_payments"):
            payments_M = getattr(asset, "payments_M", None)
            if payments_M is not None:
                payments = getattr(payments_M, "value", payments_M)
                annual_payments = np.sum(payments, axis=1) if payments is not None else [0] * num_years
            else:
                annual_payments = [0] * num_years
            capex_list = annual_payments
        elif asset.asset_name == "EL_Demand_MY" or asset.asset_name == "PP_CO2_MY": #No capex assumed for these
            capex_list = [0] * num_years
            
        # capex_list = asset.get_yearly_payments() if hasattr(asset, "get_yearly_payments") else [0] * num_years
        opex_list = asset.get_yearly_usage_costs() if hasattr(asset, "get_yearly_usage_costs") else [0] * num_years

        total_capex_asset = sum(capex_list)
        total_opex_asset = sum(opex_list)
        total_cost_asset = total_capex_asset + total_opex_asset

        emissions_list = asset.get_yearly_emissions() if hasattr(asset, "get_yearly_emissions") else [0] * num_years
        total_emissions_asset = sum(emissions_list)

        cost_summary.append({
            "scenario": scenario_name,
            "asset_name": asset.asset_name,
            "total_system_cost_BUSD": system_cost,
            "total_capex_BUSD": total_capex_asset,
            "total_opex_BUSD": total_opex_asset,
            "total_cost_BUSD": total_cost_asset,
            "total_emissions_MtCO2e": total_emissions_asset
        })

    summary_df = pd.DataFrame(cost_summary)

    return time_series_df, summary_df

def save_yearly_flows_to_csv(network, output_path):
    """
    Saves all asset flows split by year into a CSV file.
    Each column is labeled as assetname_yN.
    Handles different year lengths by padding with NaN.
    """
    flow_data = {}
    
    for asset in network.assets:
        if not hasattr(asset, "get_yearly_flows"):
            continue  # Skip assets without flow chunk method

        try:
            yearly_chunks = asset.get_yearly_flows()
        except Exception as e:
            print(f"[Skip] {asset.asset_name}: {e}")
            continue

        for year_idx, flow_array in enumerate(yearly_chunks):
            col_name = f"{asset.asset_name}_y{year_idx+1}"
            flow_data[col_name] = np.array(flow_array).flatten()

    # Determine max column length for padding
    max_len = max(len(arr) for arr in flow_data.values())

    # Pad all arrays with np.nan to equal length
    for key in flow_data:
        padded = np.full(max_len, np.nan)
        padded[:len(flow_data[key])] = flow_data[key]
        flow_data[key] = padded

    # Create DataFrame and save
    df = pd.DataFrame(flow_data)
    df.to_csv(output_path, index=False)
    print(f"[✓] Yearly flows saved to {output_path}")
    
def save_yearly_flows_to_csv_multiloc(network, location_parameters_df, output_path):
    """
    Saves all asset flows split by year into a CSV file.
    - HVDC transport assets are labeled as HVDC LocA-LocB_yN.
    - Other assets use their asset name.
    - Handles different year lengths by padding with NaN.
    """
    flow_data = {}

    for asset in network.assets:
        if not hasattr(asset, "get_yearly_flows"):
            continue  # Skip assets without flow chunk method

        name = asset.asset_name.lower()

        try:
            yearly_chunks = asset.get_yearly_flows()
        except Exception as e:
            print(f"[Skip] {asset.asset_name}: {e}")
            continue

        # HVDC / transport case
        if "el_transport" in name:
            df = yearly_chunks  # assuming it's a DataFrame with col names like '0-1_year_5'
            for col in df.columns:
                try:
                    real_year = int(col.split("_year_")[1])
                except Exception:
                    continue

                direction = col.split("_year_")[0]
                try:
                    source_id, target_id = map(int, direction.split("-"))
                except Exception as e:
                    print(f"[Skip] Could not parse source/target from '{direction}': {e}")
                    continue

                source_loc_name = location_parameters_df.iloc[source_id]["location_name"]
                target_loc_name = location_parameters_df.iloc[target_id]["location_name"]

                col_name = f"HVDC {source_loc_name}-{target_loc_name}_y{real_year}"
                flow_data[col_name] = np.array(df[col]).flatten()

        else:
            # Non-transport asset: location info optional
            for year_idx, flow_array in enumerate(yearly_chunks):
                col_name = f"{asset.asset_name}_y{year_idx+1}"
                flow_data[col_name] = np.array(flow_array).flatten()

    # Determine max column length for padding
    if not flow_data:
        print("No flows found to save.")
        return

    max_len = max(len(arr) for arr in flow_data.values())

    # Pad all arrays with NaN to equal length
    for key in flow_data:
        padded = np.full(max_len, np.nan)
        padded[:len(flow_data[key])] = flow_data[key]
        flow_data[key] = padded

    # Create DataFrame and save
    df_out = pd.DataFrame(flow_data)
    df_out.to_csv(output_path, index=False)
    print(f"[✓] Yearly flows saved to {output_path}")


def get_lcoe_per_year(network, output_path=None):
    """
    Returns a DataFrame with annual total discounted energy, cost, and LCOE (USD/MWh).
    Optionally saves this data as CSV.
    """
    print("======== Saving LCOE per year calculation =========")
    
    discount_rate = float(network.system_parameters_df.loc["discount_rate", "value"])
    num_years = network.assets[0].num_years

    total_gen_energy = np.zeros(num_years)
    total_discounted_energy = np.zeros(num_years) # total generated energy
    total_discounted_demand = np.zeros(num_years) # total utilised energy
    total_discounted_cost = np.zeros(num_years)

    years = np.arange(num_years)
    discount_factors = 1 / ((1 + discount_rate) ** years)

    for asset in network.assets:
        try:
            # Determine cost and apply discounting
            if hasattr(asset, "get_yearly_usage_costs"):
                cost = asset.get_yearly_usage_costs()
            elif asset.asset_name != "EL_Demand_MY":
                cost = asset.yearly_payments.value
            else:
                cost = None

            if cost is not None:
                total_discounted_cost += cost

            if asset.asset_name != "EL_Demand_MY":
                generation_per_year = np.sum(asset.get_yearly_flows()[:30], axis=1)
                sampled_days = int((asset.number_of_edges / 24) / asset.num_years)
                simulation_factor = 365 / sampled_days
                generation_per_year *= simulation_factor

                discounted_generation = generation_per_year * discount_factors
                total_discounted_energy += discounted_generation
                total_gen_energy += generation_per_year
            else:
                demand_per_year = np.sum(asset.get_yearly_flows()[:30], axis=1)
                sampled_days = int((asset.number_of_edges / 24) / asset.num_years)
                simulation_factor = 365 / sampled_days
                demand_per_year *= simulation_factor
                
                discounted_demand = demand_per_year * discount_factors
                total_discounted_demand += discounted_demand

        except Exception as e:
            print(f"[Skip] {asset.asset_name}: {e}")
            continue

    # Compute LCOE (in USD/MWh)
    lcoe_per_year = np.divide(
        total_discounted_cost,
        total_discounted_energy,
        out=np.zeros_like(total_discounted_cost),
        where=total_discounted_energy != 0,
    ) * 1e3  # BUSD/GWh → USD/kWh
    
    # Compute LCUE (in USD/MWh)
    lcue_per_year = np.divide(
        total_discounted_cost,
        total_discounted_demand,
        out=np.zeros_like(total_discounted_cost),
        where=total_discounted_demand != 0,
    ) * 1e3  # BUSD/GWh → USD/kWh

    # Create DataFrame
    df = pd.DataFrame({
        "total_discounted_energy_GWh": total_discounted_energy,
        "total_discounted_cost_BUSD": total_discounted_cost,
        "total_discounted_demand_GWh": total_discounted_demand,
        "lcoe_USD/kWh": lcoe_per_year,
        "lcue_USD/kWh": lcue_per_year
    })

    # Save if needed
    if output_path:
        df.to_csv(output_path, index=False)

    return df


def get_grid_intensity(network, output_path=None):
    """
    Saves the annual grid intensity
    """
    num_years = network.assets[0].num_years

    total_gen_energy = np.zeros(num_years)
    emissions_per_year = np.zeros(num_years)
    for asset in network.assets:
        try:
            if asset.asset_name != "EL_Demand_MY":
                generation_per_year = np.sum(asset.get_yearly_flows()[:30], axis=1)
            if asset.asset_name == "PP_CO2_MY":
                emissions_per_year = asset.get_yearly_emissions()[:30]
                
            total_gen_energy += generation_per_year
            grid_intensity_per_year = np.divide(emissions_per_year, total_gen_energy,
                                                out=np.zeros_like(generation_per_year),
                                                where=total_gen_energy != 0) # MtCO2/GWh = ktCO2e/MWh
        except Exception as e:
            print(f"[Skip] {asset.asset_name}: {e}")
            continue


    # Optionally save or return
    if output_path:    
        np.savetxt(output_path, grid_intensity_per_year, delimiter=",")
    return grid_intensity_per_year
    

            
    

    