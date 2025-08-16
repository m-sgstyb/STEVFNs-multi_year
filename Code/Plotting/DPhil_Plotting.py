# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 18:43:22 2024

@author: Mónica Sagastuy-Breña
"""

import pandas as pd
import numpy as np
import matplotlib
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from collections import defaultdict
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from Code.Results import get_new_input_params

def logistic_curve(t, K, r, t0):
    '''
    

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    K : TYPE
        DESCRIPTION.
    r : TYPE
        DESCRIPTION.
    t0 : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return K / (1 + np.exp(-r * (t - t0)))

def logistic_curve_derivative(t, K, r, t0):
    '''
    

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    K : TYPE
        DESCRIPTION.
    r : TYPE
        DESCRIPTION.
    t0 : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return (r * K * np.exp(-r * (t - t0))) / ((1 + np.exp(-r * (t - t0)))**2)

def linear_approximation(t, m, b):
    '''
    

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return m * t + b

def fit_s_curves(case_study_folder, tech_lim, assets_folder,
                  goal_capacity, goal_year, historical_data_file):
    '''
    

    Parameters
    ----------
    case_study_folder : TYPE
        DESCRIPTION.
    tech_lim : TYPE
        DESCRIPTION.
    assets_folder : TYPE
        DESCRIPTION.
    goal_capacity : TYPE
        DESCRIPTION.
    goal_year : TYPE
        DESCRIPTION.
    historical_data_file : TYPE
        DESCRIPTION.

    Returns
    -------
    K_fit : TYPE
        DESCRIPTION.
    r_fit : TYPE
        DESCRIPTION.
    t0_fit : TYPE
        DESCRIPTION.
    years : TYPE
        DESCRIPTION.
    capacities : TYPE
        DESCRIPTION.

    '''
    data = pd.read_csv(historical_data_file)
    goal_capacity = get_new_input_params.get_30y_opt_capacities(case_study_folder,
                                                                tech_lim,
                                                                assets_folder)
    years = data['year'].values
    # Identify limited technology for either wind or solar
    if tech_lim[0:5] == 'RE_PV':
        capacities = data['pv_installed_capacity_GW'].values
    elif tech_lim[0:5] == 'RE_WI':
        capacities = data['wind_installed_capacity_GW'].values
        
    # Append the goal data point to the historical data
    years_with_goal = np.append(years, goal_year)
    capacities_with_goal = np.append(capacities, goal_capacity)

    # Initial guesses and bounds to delay inflection
    initial_guess = [goal_capacity, 0.2, 2030]  # K, r, t0
    bounds = ([max(capacities), 0.01, 2030], [goal_capacity * 1.5, 1, 2055])

    # Fit the logistic curve with bounds using the augmented data
    params, _ = curve_fit(
        logistic_curve,
        years_with_goal,
        capacities_with_goal,
        p0=initial_guess,
        bounds=bounds
    )
    K_fit, r_fit, t0_fit = params

    return K_fit, r_fit, t0_fit, years, capacities

def approximate_scurve_derivative(case_study_folder, tech_lim, assets_folder,
                                  goal_capacity, goal_year, historical_data_file):
    '''
    

    Parameters
    ----------
    case_study_folder : TYPE
        DESCRIPTION.
    tech_lim : TYPE
        DESCRIPTION.
    assets_folder : TYPE
        DESCRIPTION.
    goal_capacity : TYPE
        DESCRIPTION.
    goal_year : TYPE
        DESCRIPTION.
    historical_data_file : TYPE
        DESCRIPTION.

    Returns
    -------
    derivative_years : TYPE
        DESCRIPTION.
    derivative_values : TYPE
        DESCRIPTION.

    '''
    K_fit, r_fit, t0_fit, years, capacities = fit_s_curves(case_study_folder, tech_lim, assets_folder,
                                        goal_capacity, goal_year, historical_data_file)
    # Calculate slope (m) at inflection point
    m = logistic_curve_derivative(t0_fit, K_fit, r_fit, t0_fit)
    b = logistic_curve(t0_fit, K_fit, r_fit, t0_fit) - m * t0_fit
    # Generate linear section (±5 years around t0)
    derivative_years = np.arange(int(t0_fit) - 5, int(t0_fit) + 6)
    derivative_values = linear_approximation(derivative_years, m, b)
    return derivative_years, derivative_values
    
def plot_scurves(case_study_folder, tech_lim, assets_folder,
                 goal_capacity, goal_year, historical_data_file):
    '''
    
    Parameters
    ----------
    case_study_folder : TYPE
        DESCRIPTION.
    tech_lim : TYPE
        DESCRIPTION.
    assets_folder : TYPE
        DESCRIPTION.
    goal_capacity : TYPE
        DESCRIPTION.
    goal_year : TYPE
        DESCRIPTION.
    historical_data_file : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    K_fit, r_fit, t0_fit, years, capacities = fit_s_curves(case_study_folder, tech_lim, assets_folder,
                                        goal_capacity, goal_year, historical_data_file)
    r_max = (r_fit * K_fit) / 4
    extended_years = np.arange(years[0], 2056)
    scurve_data = logistic_curve(extended_years, K_fit, r_fit, t0_fit)
    
    derivative_years, derivative_values = approximate_scurve_derivative(case_study_folder, tech_lim, assets_folder,
                                      goal_capacity, goal_year, historical_data_file)
    
    plt.figure(figsize=(12, 8))

    plt.plot(extended_years, scurve_data, color='blue', linestyle='--', label='Projected Logistic Curve')
    plt.scatter(years, capacities, color='red', marker='x', s=70, label='Actual Data')

    # Plot the linear section near the inflection point
    plt.plot(derivative_years, derivative_values, color='green', linestyle='-', label='Linear Approximation of derivative at $t_0$', linewidth=2)

    # Highlight the inflection point
    plt.scatter([t0_fit], [logistic_curve(t0_fit, K_fit, r_fit, t0_fit)], color='black', label='Inflection Point', zorder=5)
    plt.annotate(f"\nt = {t0_fit:.2f},\n$r_{{max}}$ = {r_max:.2f} GW / year", 
                 (t0_fit, logistic_curve(t0_fit, K_fit, r_fit, t0_fit)), 
                 textcoords="offset points", 
                 xytext=(-12, -60),
                 fontsize=12,
                 ha='left', color='black')

    plt.xlabel('Year', fontsize=14)
    plt.xticks(fontsize=12)
    plt.ylabel('Cumulative Installed Wind Capacity (GWp)', fontsize=14)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()
    return

def plot_asset_sizes_stacked(my_network, location_parameters_df, save_path=None):
    '''
    Manually hard-coded for specific assets in system for my thesis, needs to be
    generalised if more assets need to be plotted

    Parameters
    ----------
    my_network : STEVFNs network
        Full network after running a STEVFNs modelling iteration
    location_parameters_df : DataFrame
        Contains coordinates and labels to find the location name and plot
    save_path : PATH, optional
        Path to save the plot to if it needs to be saved. The default is None.

    Returns
    -------
    None.

    '''
    og_df = my_network.system_structure_df.copy()
    asset_sizes_array = np.array([my_network.assets[counter].asset_size() for counter in range(len(og_df))])
    og_df["Asset_Size"] = asset_sizes_array
    max_asset_size = np.max(asset_sizes_array)
    min_asset_size = max_asset_size * 1E-3

    og_df = og_df[og_df["Asset_Size"] >= min_asset_size]
    og_df = og_df[og_df['Asset_Class'] != 'CO2_Budget']
    asset_class_list = np.sort(og_df["Asset_Class"].unique())

    loc_1 = og_df["Location_1"].unique()
    loc_name = location_parameters_df.loc[loc_1[0]]['location_name']

    bars = []  # Collect all bars for the legend
    pv_colors = ['#f35b04', '#f18701']
    wind_colors = ['#126782', '#58B4D1']
    pp_colors = ['#8d99ae']
    bess_colors = ['#226f54', '#87c38f']
    hvdc_color = ['#5e548e']
    
    # PV Capacity
    if "RE_PV_Existing" in asset_class_list:
        existing_pv = float(og_df.query("Asset_Class == 'RE_PV_Existing'")['Asset_Size'].iloc[0])
        bars.append(plt.bar("Total PV", existing_pv, color=pv_colors[0], label="PV Existing", zorder=3))

        if "RE_PV_Openfield_Lim" in asset_class_list:
            new_pv = float(og_df.query("Asset_Class == 'RE_PV_Openfield_Lim'")['Asset_Size'].iloc[0])
            bars.append(plt.bar("Total PV", new_pv, bottom=existing_pv, color=pv_colors[1], label="PV New", zorder=3))

    # Wind Capacity
    if "RE_WIND_Existing" in asset_class_list:
        existing_wind = float(og_df.query("Asset_Class == 'RE_WIND_Existing'")['Asset_Size'].iloc[0])
        bars.append(plt.bar("Total Wind", existing_wind, color=wind_colors[0], label="Wind Existing", zorder=3))

        if "RE_WIND_Onshore_Lim" in asset_class_list:
            new_wind = float(og_df.query("Asset_Class == 'RE_WIND_Onshore_Lim'")['Asset_Size'].iloc[0])
            bars.append(plt.bar("Total Wind", new_wind, bottom=existing_wind, color=wind_colors[1], label="Wind New", zorder=3))

    # Fossil Generation
    if "PP_CO2_Existing" in asset_class_list:
        existing_fossil = float(og_df.query("Asset_Class == 'PP_CO2_Existing'")['Asset_Size'].iloc[0])
        bars.append(plt.bar("Fossil Gen.", existing_fossil, color=pp_colors[0], label="Fossil Existing", zorder=3))
        
    # BESS assets
    if "BESS_Existing" in asset_class_list:
        existing_bess = float(og_df.query("Asset_Class == 'BESS_Existing'")['Asset_Size'].iloc[0])
        bars.append(plt.bar("Total BESS", existing_bess, color=bess_colors[0], label="BESS Existing", zorder=3))
    else:
        existing_bess=0
        
    if "BESS" in asset_class_list:
        new_bess = float(og_df.query("Asset_Class == 'BESS'")['Asset_Size'].iloc[0])
        bars.append(plt.bar("Total BESS", new_bess, bottom=existing_bess, color=bess_colors[1], label="BESS New", zorder=3))
    
    if "EL_Transport" in asset_class_list:
        hvdc_cable = float(og_df.query("Asset_Class == 'EL_Transport'")['Asset_Size'].iloc[0])
        bars.append(plt.bar("EL_Transport", hvdc_cable, color=hvdc_color, label="HVDC Cable", zorder=3))
    
    
    plt.xlabel(loc_name)
    plt.ylabel("Asset Size (GWp)")
    plt.title("Asset Sizes " + my_network.scenario_name)

    # Use only unique labels in the legend to avoid duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.grid(zorder=0)
    plt.legend(unique_labels.values(), unique_labels.keys())
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()
    
    return 

def plot_yearly_flows(network, output_folder):
    """
    Creates a plot per year showing flows from all assets for that year.
    Each technology keeps the same color across plots.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Step 1: Collect all asset flow chunks
    asset_flows_by_year = {}
    for asset in network.assets:
        if not hasattr(asset, "get_yearly_flows"):
            continue
        try:
            yearly_chunks = asset.get_yearly_flows()
            asset_flows_by_year[asset.asset_name] = yearly_chunks
        except Exception as e:
            print(f"[Skip] {asset.asset_name}: {e}")
            continue

    # Step 2: Determine number of years and assign colors
    asset_names = list(asset_flows_by_year.keys())
    num_years = max(len(chunks) for chunks in asset_flows_by_year.values())
    color_map = cm.get_cmap("tab10", len(asset_names))
    color_dict = {name: color_map(i) for i, name in enumerate(asset_names)}

    # Step 3: Plot per year
    for year in range(num_years):
        plt.figure(figsize=(12, 6))
        for asset_name, chunks in asset_flows_by_year.items():
            if year < len(chunks):
                plt.plot(chunks[year], label=asset_name, color=color_dict[asset_name])
        plt.title(f"Year {year + 1} - Hourly Flows")
        plt.xlabel("Hour")
        plt.ylabel("Flow")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, f"flows_year_{year + 1}.png"))
        plt.close()
    
    print(f"[✓] Plots saved to folder: {output_folder}")
    
def plot_yearly_flows_stacked(network, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    tech_order = ["pp", "wind", "pv"]
    tech_colors = {"pp": "#753D0D", "wind": "#009BA1", "pv": "#F0843C"}
    demand_color = "red"
    
    # Collect flows by tech and demand by year
    tech_flows_by_year = {tech: [] for tech in tech_order}
    demand_by_year = []
    
    for asset in network.assets:
        if not hasattr(asset, "get_yearly_flows"):
            continue
        try:
            yearly_chunks = asset.get_yearly_flows()[:30]
        except Exception:
            continue
    
        name = asset.asset_name.lower()
        if "demand" in name:
            demand_by_year = yearly_chunks
        else:
            for tech in tech_order:
                if tech in name:
                    tech_flows_by_year[tech].append(yearly_chunks)
                    break
    print("length of demand by year", len(demand_by_year))
    num_years = len(demand_by_year)
    for year in range(num_years):
        demand = np.array(demand_by_year[year])
        x = np.arange(len(demand))
        remaining_demand = demand.copy()
        bottom = np.zeros_like(demand)
    
        plt.figure(figsize=(14, 6))
    
        for tech in tech_order:
            flows_list = tech_flows_by_year[tech]
            tech_total = np.zeros_like(demand)
    
            for flows in flows_list:
                if year < len(flows):
                    flow_year = np.array(flows[year])
                    if flow_year.shape != demand.shape:
                        print(f"⚠️ Flow shape mismatch in year {year + 1} for tech {tech}: "
                              f"expected {demand.shape}, got {flow_year.shape}")
                    tech_total += flow_year

    
            # Split into used (up to remaining demand) and excess (above demand)
            used = np.minimum(tech_total, remaining_demand)
            excess = tech_total - used
    
            # Plot used part: solid fill
            plt.fill_between(x, bottom, bottom + used,
                             color=tech_colors[tech],
                             label=tech.capitalize(),
                             alpha=1.0,
                             edgecolor='none')
    
            # Plot excess part: transparent fill stacked on top
            plt.fill_between(x, bottom + used, bottom + used + excess,
                             color=tech_colors[tech],
                             alpha=0.3,
                             edgecolor=tech_colors[tech])
    
            # Update bottom and remaining demand
            bottom += tech_total
            remaining_demand -= used
            remaining_demand = np.clip(remaining_demand, 0, None)
    
        # Demand line
        plt.plot(x, demand, color=demand_color, label="Demand",
                 linestyle="--", linewidth=1.5)
    
        plt.title(f"Stacked Generation vs Demand – Year {year + 1}")
        plt.xlabel("Hour")
        plt.ylabel("Power Flow")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"stacked_year_{year + 1}.png"))
        plt.close()

    
    print(f"[✓] Stacked plots saved to: {output_folder}")

def plot_yearly_flows_stacked_by_location(network, case_study_name, location_parameters_df,
                                          output_folder):
    os.makedirs(output_folder, exist_ok=True)
    # --- config ---
    if case_study_name.endswith("_Collab"):
        tech_order = ["pp", "transport", "wind", "pv"]  # 'transport' is an ordering placeholder
    else:
        tech_order = ["pp", "wind", "pv"]

    tech_colors = {"pp": "#753D0D", "wind": "#009BA1", "pv": "#F0843C", "transport": "#047315"}
    demand_color = "red"

    # Use dict for inner level to avoid autovivification
    flows_by_loc = defaultdict(dict)      # loc -> { "pp": [arrs], "HVDC A-B": [arrs], ... }
    demand_by_loc = {}                    # loc -> [year_arrays]

    print("🔍 Collecting yearly flows by location...")

    for asset in network.assets:
        if not hasattr(asset, "get_yearly_flows"):
            continue

        name = getattr(asset, "asset_name", "").lower()

        # Pull data (EL_Transport returns 60 columns; others capped at 30)
        try:
            if case_study_name.endswith("_Collab") and "el_transport" in name:
                yearly_chunks = asset.get_yearly_flows()
            else:
                yearly_chunks = asset.get_yearly_flows()[:30]
        except Exception as e:
            print(f"⚠️ Skipping asset {getattr(asset, 'asset_name', 'unknown')} due to error: {e}")
            continue

        # --- EL_Transport: store by HVDC label per direction ---
        if case_study_name.endswith("_Collab") and "el_transport" in name:
            df = yearly_chunks
            for col in df.columns:
                parts = col.split("_year_")
                if len(parts) != 2:
                    continue
                try:
                    real_year = int(parts[1])
                except Exception:
                    continue

                direction = parts[0]
                try:
                    source_id, target_id = map(int, direction.split("-"))
                except Exception as e:
                    print(f"⚠️ Could not parse source/target from '{direction}': {e}")
                    continue

                src = location_parameters_df.iloc[source_id]["location_name"]
                tgt = location_parameters_df.iloc[target_id]["location_name"]
                label = f"HVDC {src}-{tgt}"

                flow = np.asarray(df[col])
                years_list = flows_by_loc[target_id].setdefault(label, [])
                while len(years_list) <= real_year:
                    years_list.append(np.zeros_like(flow))
                years_list[real_year] = years_list[real_year] + np.nan_to_num(flow)
            continue  # handled

        # --- Non-transport assets ---
        loc = getattr(asset, "target_node_location", getattr(asset, "node_location", None))
        if loc is None:
            continue

        if "demand" in name:
            demand_by_loc[loc] = yearly_chunks
        else:
            # add under first matching tech, but NEVER create a 'transport' key here
            for tech in tech_order:
                if tech == "transport":
                    continue
                if tech in name:
                    flows_by_loc[loc].setdefault(tech, []).append(yearly_chunks)
                    break
        
        
        # Function to get a direction-independent key
    def normalize_hvdc_label(label):
        # label format assumed "HVDC SRC-TGT"
        try:
            _, route = label.split(" ", 1)
            src, tgt = route.split("-")
            return "HVDC " + "-".join(sorted([src.strip(), tgt.strip()]))
        except ValueError:
            return label.strip()
    
    # Collect all unique normalized HVDC routes
    normalized_routes = sorted({
        normalize_hvdc_label(k)
        for loc_data in flows_by_loc.values()
        for k in loc_data
        if k.startswith("HVDC ")
    })
    # Group by normalized route
    hvdc_groups = {}
    # Get all HVDC labels
    hvdc_labels = sorted({
        k
        for loc_data in flows_by_loc.values()
        for k in loc_data
        if k.startswith("HVDC ")
    })
    for lbl in hvdc_labels:
        norm = normalize_hvdc_label(lbl)
        hvdc_groups.setdefault(norm, []).append(lbl)
    # Assign colors
    hvdc_colors = {}
    for norm_route, labels in hvdc_groups.items():
        if len(normalized_routes) > 1:
            # color_cycle = itertools.cycle(matplotlib.cm.tab20.colors)
            shades = ["#047315", "#6DC27A"]  # dark green, light green
            for lbl, shade in zip(sorted(labels), shades):
                hvdc_colors[lbl] = shade
        else:
            # Single link → default transport green
            hvdc_colors[labels[0]] = "#047315"
                    
    # --- Optional summary without creating keys accidentally ---
    # print("\n📊 Data summary per location:")
    # for loc in sorted(demand_by_loc.keys()):
    #     print(f"  Location {loc}:")
    #     print(f"    Demand years: {len(demand_by_loc[loc])}")
    #     for tech in tech_order:
    #         if tech == "transport":
    #             hvdc_count = sum(1 for k in flows_by_loc.get(loc, {}) if k.startswith("HVDC "))
    #             print(f"    transport assets (HVDC links): {hvdc_count}")
    #         else:
    #             print(f"    {tech} assets: {len(flows_by_loc.get(loc, {}).get(tech, []))}")

    # --- Plot per location ---
    for loc, demand_by_year in demand_by_loc.items():
        loc_name = location_parameters_df.iloc[loc]["location_name"]
        tech_flows = flows_by_loc.get(loc, {})
        num_years = len(demand_by_year)

        for year in range(num_years):
            demand = np.array(demand_by_year[year])
            x = np.arange(len(demand))
            remaining_demand = demand.copy()
            bottom = np.zeros_like(demand)

            plt.figure(figsize=(14, 6))
            seen_labels = set()

            for tech in tech_order:
                if tech == "transport":
                    # Plot ALL HVDC labels at this location in the 'transport' slot
                    hvdc_labels = [k for k in tech_flows.keys() if k.startswith("HVDC ")]
                    # stable order
                    hvdc_labels.sort()
                    for label in hvdc_labels:
                        flows_list = tech_flows[label]
                        tech_total = np.zeros_like(demand)
                        if year < len(flows_list):
                            flow_year = np.array(flows_list[year])
                            # pad/trim to match demand
                            if flow_year.shape[0] != demand.shape[0]:
                                if flow_year.shape[0] > demand.shape[0]:
                                    flow_year = flow_year[:demand.shape[0]]
                                else:
                                    flow_year = np.pad(flow_year, (0, demand.shape[0]-flow_year.shape[0]), constant_values=0)
                            tech_total += np.nan_to_num(flow_year)

                        if not np.any(tech_total):
                            continue  # nothing to draw

                        used = np.minimum(tech_total, remaining_demand)
                        excess = tech_total - used

                        lbl = None if label in seen_labels else label
                        plt.fill_between(
                            x, bottom, bottom + used,
                            color = hvdc_colors.get(label, tech_colors["transport"]),
                            label=lbl, alpha=1.0, edgecolor='none'
                        )
                        plt.fill_between(
                            x, bottom + used, bottom + used + excess,
                            color = hvdc_colors.get(label, tech_colors["transport"]),
                            alpha=0.3, edgecolor='none'
                        )

                        seen_labels.add(label)
                        bottom += tech_total
                        remaining_demand = np.clip(remaining_demand - used, 0, None)
                else:
                    # Sum all assets of this tech
                    flows_list_collection = tech_flows.get(tech, [])
                    tech_total = np.zeros_like(demand)
                    for flows_list in flows_list_collection:
                        if year < len(flows_list):
                            flow_year = np.array(flows_list[year])
                            if flow_year.shape[0] != demand.shape[0]:
                                if flow_year.shape[0] > demand.shape[0]:
                                    flow_year = flow_year[:demand.shape[0]]
                                else:
                                    flow_year = np.pad(flow_year, (0, demand.shape[0]-flow_year.shape[0]), constant_values=0)
                            tech_total += np.nan_to_num(flow_year)

                    if not np.any(tech_total):
                        continue

                    used = np.minimum(tech_total, remaining_demand)
                    excess = tech_total - used

                    label = tech.capitalize()
                    lbl = None if label in seen_labels else label
                    plt.fill_between(x, bottom, bottom + used,
                                     color=tech_colors[tech], label=lbl,
                                     alpha=1.0, edgecolor='none')
                    plt.fill_between(x, bottom + used, bottom + used + excess,
                                     color=tech_colors[tech], alpha=0.3, edgecolor='none')

                    seen_labels.add(label)
                    bottom += tech_total
                    remaining_demand = np.clip(remaining_demand - used, 0, None)

            # demand line
            plt.plot(x, demand, color=demand_color, label=None if "Demand" in seen_labels else "Demand",
                     linestyle="--", linewidth=1.5)
            seen_labels.add("Demand")

            plt.title(f"{loc_name} – Stacked Generation vs Demand – Year {year}")
            plt.xlabel("Hour")
            plt.ylabel("Power Flow")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"stacked_{loc_name}_year_{year}.png"))
            plt.close()

        print(f"[✓] Plots for {loc_name} saved to: {output_folder}")

def get_install_pathways(tech_asset, save_path, tech_name="Tech"):
    """
    Plots installed capacity pathways for a technology asset (wind/solar).
    """
    # Ensure optimization has been solved
    if tech_asset.flows.value is None:
        raise ValueError(f"{tech_name}: Optimization problem must be solved before extracting values.")

    # Extract arrays safely
    try:
        new_installed = np.array(tech_asset.flows.value, dtype=float).flatten()
        cumulative_new = np.array(tech_asset.cumulative_new_installed.value, dtype=float).flatten()
        existing = np.array(tech_asset.conversion_fun_params["existing_capacity"].value, dtype=float).flatten()
    except Exception as e:
        raise TypeError(f"Error converting expressions to float: {e}")

    total_existing = cumulative_new + existing
    years = np.arange(len(new_installed))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(years, new_installed, label="New Installed Capacity", color="#025773")
    ax.plot(years, total_existing, color="navy", linewidth=2, marker="o", label="Total Existing Capacity")

    ax.set_title(f"{tech_name} Capacity Installation Pathway")
    ax.set_xlabel("Year")
    ax.set_ylabel("Capacity [GWp]")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(save_path, f"Install_pathways_{tech_name}.png"))
    
def get_dual_install_pathways(tech_asset_1, tech_asset_2, save_path, tech_name_1="Tech 1", tech_name_2="Tech 2"):
    """
    Plots installed capacity pathways for two technology assets using dual y-axes.
    - Bars show new installs (left axis).
    - Lines show cumulative existing capacity (right axis).
    - Legends are separated.
    """
    # Check optimization results
    for tech_asset, name in [(tech_asset_1, tech_name_1), (tech_asset_2, tech_name_2)]:
        if tech_asset.flows.value is None:
            raise ValueError(f"{name}: Optimization must be solved before plotting.")

    # Extract data
    try:
        new_1 = np.array(tech_asset_1.flows.value, dtype=float).flatten()
        cum_1 = np.array(tech_asset_1.cumulative_new_installed.value, dtype=float).flatten()
        exist_1 = np.array(tech_asset_1.conversion_fun_params["existing_capacity"].value, dtype=float).flatten()

        new_2 = np.array(tech_asset_2.flows.value, dtype=float).flatten()
        cum_2 = np.array(tech_asset_2.cumulative_new_installed.value, dtype=float).flatten()
        exist_2 = np.array(tech_asset_2.conversion_fun_params["existing_capacity"].value, dtype=float).flatten()
    except Exception as e:
        raise TypeError(f"Error converting expressions to float: {e}")

    total_1 = cum_1 + exist_1
    total_2 = cum_2 + exist_2
    years = np.arange(len(new_1))
    width = 0.35

    # Setup figure and axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    # Plot bars (new installs)
    bars1 = ax1.bar(years - width/2, new_1, width=width, label=f"{tech_name_1}", color="#F0843C", alpha=0.5)
    bars2 = ax1.bar(years + width/2, new_2, width=width, label=f"{tech_name_2}", color="#025773", alpha=0.5)

    # Plot lines (total capacity)
    line1, = ax2.plot(years, total_1, color="#B14B07", marker="o", linewidth=2, label=f"{tech_name_1}")
    line2, = ax2.plot(years, total_2, color="#02384A", marker="s", linewidth=2, label=f"{tech_name_2}")

    # Axis labels and title
    ax1.set_xlabel("Year")
    ax1.set_ylabel("New Installed Capacity [GWp]", color="gray")
    ax2.set_ylabel("Total Existing Capacity [GWp]", color="black")
    ax1.set_title("Installed Capacity Pathways")
    ax1.set_xticks(years)
    ax1.grid(True, which='major', linestyle='--', alpha=0.5)

    # Separate legends
    bar_legend = ax1.legend(handles=[bars1, bars2], title="New Installs", loc="upper left")
    ax1.add_artist(bar_legend)  # Add bar legend first
    ax2.legend(handles=[line1, line2], title="Total Capacity", bbox_to_anchor=[0.15, 1], loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"Install_pathways_{tech_name_1}_{tech_name_2}.png"))
    
def plot_dual_install_pathways_all_locations(my_network, network_structure_df, tech_class_1, tech_class_2, save_path,
                                             tech_name_1="Tech 1", tech_name_2="Tech 2"):
    """
    Plots installed capacity pathways for two technologies (e.g., PV & Wind) for all locations in the network.
    
    Parameters:
    - my_network: The solved network object containing assets.
    - network_structure_df: DataFrame with 'Asset_Class' and 'Location_1' columns.
    - tech_class_1: Asset class name for first technology (e.g., "RE_PV_MY").
    - tech_class_2: Asset class name for second technology (e.g., "RE_WIND_MY").
    - save_path: Directory where plots will be saved.
    - tech_name_1: Display name for first technology.
    - tech_name_2: Display name for second technology.
    """

    unique_locations = sorted(network_structure_df["Location_1"].unique())

    for loc in unique_locations:
        # Find assets for this location and tech type
        pv_row = network_structure_df[(network_structure_df["Location_1"] == loc) &
                                      (network_structure_df["Asset_Class"] == tech_class_1)]
        wind_row = network_structure_df[(network_structure_df["Location_1"] == loc) &
                                        (network_structure_df["Asset_Class"] == tech_class_2)]

        if pv_row.empty or wind_row.empty:
            print(f"⚠ Skipping location {loc}: Missing {tech_class_1} or {tech_class_2}")
            continue

        pv_asset_num = pv_row["Asset_Number"].iloc[0]
        wind_asset_num = wind_row["Asset_Number"].iloc[0]

        pv_asset = my_network.assets[pv_asset_num]
        wind_asset = my_network.assets[wind_asset_num]

        # --- Extract data ---
        new_pv = np.array(pv_asset.flows.value, dtype=float).flatten()
        cum_pv = np.array(pv_asset.cumulative_new_installed.value, dtype=float).flatten()
        exist_pv = np.array(pv_asset.conversion_fun_params["existing_capacity"].value, dtype=float).flatten()

        new_wind = np.array(wind_asset.flows.value, dtype=float).flatten()
        cum_wind = np.array(wind_asset.cumulative_new_installed.value, dtype=float).flatten()
        exist_wind = np.array(wind_asset.conversion_fun_params["existing_capacity"].value, dtype=float).flatten()

        total_pv = cum_pv + exist_pv
        total_wind = cum_wind + exist_wind
        years = np.arange(len(new_pv))
        width = 0.35

        # --- Plot ---
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()

        bars1 = ax1.bar(years - width/2, new_pv, width=width, label=f"{tech_name_1}", color="#F0843C", alpha=0.5)
        bars2 = ax1.bar(years + width/2, new_wind, width=width, label=f"{tech_name_2}", color="#025773", alpha=0.5)

        line1, = ax2.plot(years, total_pv,  color="#B14B07", marker="o", linewidth=2, label=f"{tech_name_1}")
        line2, = ax2.plot(years, total_wind, color="#02384A", marker="s", linewidth=2, label=f"{tech_name_2}")

        ax1.set_xlabel("Year")
        ax1.set_ylabel("New Installed Capacity [GWp]", color="gray")
        ax2.set_ylabel("Total Existing Capacity [GWp]", color="black")
        ax1.set_title(f"Installed Capacity Pathways - Location {loc}")
        ax1.set_xticks(years)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Legends
        bar_legend = ax1.legend(handles=[bars1, bars2], title="New Installs", loc="upper left")
        ax1.add_artist(bar_legend)
        ax2.legend(handles=[line1, line2], title="Total Capacity", bbox_to_anchor=(0.15, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"Install_pathways_{tech_name_1}_{tech_name_2}_location_{loc}.png"))
        plt.close(fig)  # prevent memory leaks in large runs

        print(f"✅ Saved  installed pathways plot for location {loc}")

    
    