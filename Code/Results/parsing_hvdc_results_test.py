#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 16:50:06 2025

@author: Mónica Sagastuy-Breña
"""

"""
Attempt at processing the flows for trade at hourly rates
process_flows_hourly.py

Usage:
    processed_df = process_flows(
        csv_path="yearly_flows.csv",
        year=1,                        # which year N to pick (y1, y2, ...)
        output_path="processed_hourly_y1.csv"
    )
"""



import re
from collections import defaultdict
import numpy as np
import pandas as pd


def _parse_hvdc_cols(columns, year_token):
    """
    Return dict of hvdc_col -> (source_loc, target_loc)
    matches columns like: "HVDC MEX-CHL_y1" or "HVDC MEX-CHL_y_1"
    """
    hvdc_pattern = re.compile(r"HVDC\s+(.+?)-(.+?)_?" + re.escape(year_token) + r"$", re.IGNORECASE)
    hvdc_map = {}
    for c in columns:
        m = hvdc_pattern.search(c)
        if m:
            src = m.group(1).strip()
            tgt = m.group(2).strip()
            hvdc_map[c] = (src, tgt)
    return hvdc_map


def _parse_asset_loc_cols(columns, year_token):
    """
    Parses columns of type "<asset>_<LOC>_yN" (or similar). Returns dicts mapping:
      - pv_cols[loc] = [col, ...]
      - wind_cols[loc] = [col, ...]
      - plant_cols[loc] = [col, ...]   (fallback for anything that is not pv/wind/demand/hvdc)
      - demand_cols[loc] = [col, ...]
    The function is intentionally permissive about asset names (looks for keywords).
    """
    pv_keywords = ("pv", "solar")
    wind_keywords = ("wind",)
    demand_keywords = ("demand", "el_demand", "load")
    # any others are considered "plant" / "other generation"

    asset_pattern = re.compile(r"^(?P<asset>.+?)_(?P<loc>[^_]+)_" + re.escape(year_token) + r"$", re.IGNORECASE)
    pv_cols = defaultdict(list)
    wind_cols = defaultdict(list)
    plant_cols = defaultdict(list)
    demand_cols = defaultdict(list)
    other_cols = []

    for c in columns:
        m = asset_pattern.search(c)
        if not m:
            # also accept asset names containing spaces e.g. "Solar MEX_y1"
            alt = re.compile(r"^(?P<asset>.+?)\s+(?P<loc>[^_]+)_" + re.escape(year_token) + r"$", re.IGNORECASE)
            m = alt.search(c)
        if not m:
            continue

        asset = m.group("asset").strip().lower()
        loc = m.group("loc").strip()

        if any(k in asset for k in pv_keywords):
            pv_cols[loc].append(c)
        elif any(k in asset for k in wind_keywords):
            wind_cols[loc].append(c)
        elif any(k in asset for k in demand_keywords):
            demand_cols[loc].append(c)
        else:
            # treat as plant/generation
            plant_cols[loc].append(c)

    return pv_cols, wind_cols, plant_cols, demand_cols


def process_flows(csv_path, year=1, output_path=None):
    """
    Main function:
      - csv_path : path to the CSV produced by your exporter.
      - year     : integer N to pick columns ending with _yN (e.g. y1, y2)
      - output_path : if provided, write processed CSV there.

    Returns:
      - processed_df : DataFrame with hourly rows and columns for:
           Demand_LOC, PV_LOC, Wind_LOC, Plant_LOC, ImportsUsed_LOC, ExportsObserved_LOC,
           DeficitUnmet_LOC, SurplusUnused_LOC, plus original HVDC columns preserved.
      - hvdc_summary : dict containing info about simultaneous opposite-direction flows
    """
    df = pd.read_csv(csv_path)
    year_token = f"y{year}"

    # Collect HVDC columns and their (src, tgt)
    hvdc_map = _parse_hvdc_cols(df.columns, year_token)

    # Collect pv/wind/plant/demand columns
    pv_cols, wind_cols, plant_cols, demand_cols = _parse_asset_loc_cols(df.columns, year_token)

    # Build the full location list from all sources found
    all_locs = set()
    all_locs.update(pv_cols.keys(), wind_cols.keys(), plant_cols.keys(), demand_cols.keys())
    # Also include any locations that appear in hvdc_map
    for _, (s, t) in hvdc_map.items():
        all_locs.add(s)
        all_locs.add(t)

    all_locs = sorted(all_locs)

    # Prepare result columns
    results = pd.DataFrame(index=df.index)  # rows = hours

    # Keep HVDC columns as-is for reference (these are the observed flows)
    for hvdc_col in hvdc_map.keys():
        # If col missing from df due to slightly different naming, skip
        if hvdc_col in df.columns:
            results[hvdc_col] = df[hvdc_col]

    # For each location, compute hourly metrics following priority: PV -> Wind -> Plant -> HVDC imports
    for loc in all_locs:
        # sum columns per loc (if multiple PV columns for same loc for any reason)
        pv_series = pd.Series(0.0, index=df.index)
        for c in pv_cols.get(loc, []):
            pv_series = pv_series.add(df[c].fillna(0.0), fill_value=0.0)

        wind_series = pd.Series(0.0, index=df.index)
        for c in wind_cols.get(loc, []):
            wind_series = wind_series.add(df[c].fillna(0.0), fill_value=0.0)

        plant_series = pd.Series(0.0, index=df.index)
        for c in plant_cols.get(loc, []):
            plant_series = plant_series.add(df[c].fillna(0.0), fill_value=0.0)

        demand_series = pd.Series(0.0, index=df.index)
        # if multiple demand columns, sum them (rare but robust)
        for c in demand_cols.get(loc, []):
            demand_series = demand_series.add(df[c].fillna(0.0), fill_value=0.0)

        # compute observed HVDC imports to loc and exports from loc
        hvdc_imports = pd.Series(0.0, index=df.index)
        hvdc_exports = pd.Series(0.0, index=df.index)
        # also keep per-line breakdown if needed
        outgoing_lines = []
        incoming_lines = []
        for col, (src, tgt) in hvdc_map.items():
            if col not in df.columns:
                continue
            vals = df[col].fillna(0.0)
            # interpret positive value as flow from src -> tgt
            if tgt == loc:
                # import to loc (from src)
                hvdc_imports = hvdc_imports.add(vals.clip(lower=0.0), fill_value=0.0)
                incoming_lines.append(col)
            if src == loc:
                # export from loc (to tgt)
                hvdc_exports = hvdc_exports.add(vals.clip(lower=0.0), fill_value=0.0)
                outgoing_lines.append(col)

        # Apply the priority:
        # Local generation used first (PV + Wind + Plant)
        local_gen = pv_series + wind_series + plant_series
        # deficit = demand - local_gen (positive when demand not met by local gen)
        deficit = demand_series - local_gen
        deficit_pos = deficit.clip(lower=0.0)

        # imports available (observed). Only used to the extent needed.
        imports_used = pd.Series(0.0, index=df.index)
        imports_used = imports_used.add(hvdc_imports, fill_value=0.0)  # available imports observed
        # But we cap imports_used to the actual deficit_pos
        imports_used = np.minimum(deficit_pos, imports_used)

        # remaining deficit unmet after using imports
        remaining_deficit_unmet = deficit_pos - imports_used

        # surplus when local_gen > demand
        surplus = (local_gen - demand_series).clip(lower=0.0)

        # observed exports (flows leaving loc) - we treat observed exports as what was actually sent
        exports_observed = hvdc_exports  # flows out, positive

        # surplus_unused: part of surplus not exported (surplus - observed exports out), clipped to >=0
        surplus_unused = (surplus - exports_observed).clip(lower=0.0)

        # Save columns (prefix by metric and loc)
        results[f"Demand_{loc}"] = demand_series
        results[f"PV_{loc}"] = pv_series
        results[f"Wind_{loc}"] = wind_series
        results[f"Plant_{loc}"] = plant_series
        results[f"LocalGen_{loc}"] = local_gen
        results[f"ImportsAvailable_{loc}"] = hvdc_imports
        results[f"ImportsUsed_{loc}"] = imports_used
        results[f"DeficitAfterImports_{loc}"] = remaining_deficit_unmet
        results[f"Surplus_{loc}"] = surplus
        results[f"ExportsObserved_{loc}"] = exports_observed
        results[f"SurplusUnused_{loc}"] = surplus_unused

    # Detect hours where there are simultaneous bi-directional flows on the same pair of locations
    # For every pair of HVDC columns a-b and b-a we check both directions positive at same hour
    pair_bidir_hours = []
    hvdc_cols_list = list(hvdc_map.keys())
    hvdc_pairs = {}  # map frozenset({a,b}) -> [col_from_a_to_b, col_from_b_to_a maybe]
    for col, (src, tgt) in hvdc_map.items():
        k = frozenset({src, tgt})
        hvdc_pairs.setdefault(k, []).append((col, src, tgt))

    # Find hours with simultaneous opposite flows
    bidir_summary = []
    for pair_key, entries in hvdc_pairs.items():
        # look for two entries in opposite directions
        if len(entries) < 2:
            continue
        # try all ordered pairs
        for (col1, s1, t1) in entries:
            for (col2, s2, t2) in entries:
                if s1 == t2 and t1 == s2 and col1 != col2:
                    # compute hours where both col1>0 and col2>0
                    if col1 in df.columns and col2 in df.columns:
                        both_pos = (df[col1].fillna(0.0) > 0) & (df[col2].fillna(0.0) > 0)
                        hours_idx = both_pos[both_pos].index.tolist()
                        if hours_idx:
                            bidir_summary.append({
                                "pair": tuple(pair_key),
                                "line_a": col1,
                                "line_b": col2,
                                "hours_count": len(hours_idx),
                                "hours_indices": hours_idx[:50],  # show first 50 for brevity
                            })

    # Save results
    if output_path:
        results.to_csv(output_path, index=False)

    return results, bidir_summary


if __name__ == "__main__":
    # Example usage:
    import argparse
    parser = argparse.ArgumentParser(description="Process hourly flows and compute local balances by location.")
    parser.add_argument("--input", "-i", required=True, help="Path to the flows CSV")
    parser.add_argument("--year", "-y", type=int, default=1, help="Year index to pick (y1, y2, ...)")
    parser.add_argument("--output", "-o", default="processed_hourly_y{y}.csv".replace("{y}", str(1)), help="Output CSV")
    args = parser.parse_args()

    out_df, bidir = process_flows(args.input, year=args.year, output_path=args.output)
    print(f"Processed hourly dataframe saved to {args.output}")
    if bidir:
        print("Bi-directional HVDC occurrences (sample):")
        for b in bidir[:10]:
            print(b)
