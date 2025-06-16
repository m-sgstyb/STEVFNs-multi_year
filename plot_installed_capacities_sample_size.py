import os
import pandas as pd
import matplotlib.pyplot as plt

# Base path (adjust if needed)
base_path = os.path.dirname(__file__)
case_study_path = os.path.join(base_path, "Data", "Case_Study")
scenario = "linear_red"

# Sample sizes and corresponding total hours
sample_days = [6, 12, 24, 48, 72, 96, 139, 183]
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
    plt.plot(series.index, series.values, label=f"{hrs//24} days/year")
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
    plt.plot(series.index, series.values, label=f"{hrs//24} days/year")
plt.title("Annual Installed Wind Capacity (GWp)")
plt.xlabel("Year")
plt.ylabel("Installed Wind Capacity (GWp)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()