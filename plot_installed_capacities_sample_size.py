import os
import pandas as pd
import matplotlib.pyplot as plt

# base_path = "/Users/mnsgs/Desktop/STEVFNs-DPhil-MSB"
base_path = os.path.dirname(__file__)

case_study_path = os.path.join(base_path, "Data", "Case_Study")
scenario = "linear_red_2055"

# Sample sizes and corresponding total hours
sample_days = [6, 12, 24, 48, 72, 96, 139, 161, 183]
total_hours = [d * 24 * 30 for d in sample_days]

# Store data
pv_by_year = {}
wind_by_year = {}

for hrs in total_hours:
    case_name = f"MEX_30y_MY_{hrs}"
    file_path = os.path.join(case_study_path, case_name, scenario, "Results", "time_series_results.csv")
    
    if not os.path.exists(file_path):
        print(f"Missing file: {file_path}")
        continue
    
    df = pd.read_csv(file_path)

    # Get values
    pv_series = df[['year', 'RE_PV_MY_new_annual_installed_GWp']].dropna().drop_duplicates()
    wind_series = df[['year', 'RE_WIND_MY_new_annual_installed_GWp']].dropna().drop_duplicates()
    
    # Save by sample size
    pv_by_year[hrs] = pv_series.set_index('year').squeeze()
    wind_by_year[hrs] = wind_series.set_index('year').squeeze()

# Plot Solar PV
plt.figure(figsize=(10, 5))
for hrs, series in sorted(pv_by_year.items()):
    plt.plot(series.index, series.values, label=f"{hrs//24//30} days/year")
plt.title("Annual Installed PV Capacity (GWp)")
plt.xlabel("Year")
plt.ylabel("Installed PV Capacity (GWp)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Wind
plt.figure(figsize=(10, 5))
for hrs, series in sorted(wind_by_year.items()):
    plt.plot(series.index, series.values, label=f"{hrs//24//30} days/year")
plt.title("Annual Installed Wind Capacity (GWp)")
plt.xlabel("Year")
plt.ylabel("Installed Wind Capacity (GWp)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Summarize total installed capacities
total_pv = {hrs: series.sum() for hrs, series in pv_by_year.items()}
total_wind = {hrs: series.sum() for hrs, series in wind_by_year.items()}

# Convert to DataFrame for plotting
summary_df = pd.DataFrame({
    'Sample Days per Year': [hrs//24//30 for hrs in total_pv.keys()],
    'PV Total Installed (GWp)': list(total_pv.values()),
    'Wind Total Installed (GWp)': list(total_wind.values())
}).sort_values(by='Sample Days per Year')

# Plot as line chart
plt.figure(figsize=(10, 5))
plt.plot(summary_df['Sample Days per Year'], summary_df['PV Total Installed (GWp)'], marker='o', label='PV')
plt.plot(summary_df['Sample Days per Year'], summary_df['Wind Total Installed (GWp)'], marker='s', label='Wind')
plt.title("Total Installed Capacity vs Sample Size")
plt.xlabel("Sample Days per Year")
plt.ylabel("Total Installed Capacity (GWp)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#
reference_hrs = 183 * 24 * 30
reference_pv = pv_by_year[reference_hrs]
reference_total = reference_pv.sum()
for hrs, series in pv_by_year.items():
    if hrs == reference_hrs:
        continue  # skip the reference
    sample_total = series.sum()
    percent_error = abs(sample_total - reference_total) / reference_total * 100

    plt.figure(figsize=(10, 5))
    plt.plot(series.index, series.values, label=f'{hrs//24//30} days/year', color='blue')
    plt.plot(reference_pv.index, reference_pv.values, label='183 days/year (ref)', color='black', linestyle='--')

    # Interpolate reference to match x-axis if needed
    aligned_ref = reference_pv.reindex(series.index).fillna(method='ffill')

    # Calculate absolute difference
    error = abs(series.values - aligned_ref.values)
    plt.fill_between(series.index, series.values, aligned_ref.values, color='blue', alpha=0.3, label='Error vs ref')

    plt.title(f"PV Capacity: {hrs//24//30} vs 183 Days per Year. % Error: {percent_error:.2f}%")
    plt.xlabel("Year")
    plt.ylabel("Installed PV Capacity (GWp)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




