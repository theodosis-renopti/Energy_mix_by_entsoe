#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 14:56:07 2025

@author: theodosisperifanis
"""
# General imports for data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Scikit-learn imports for preprocessing and metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Meteostat for meteorological data
from meteostat import Point, Hourly

# ENTSOE for European Network of Transmission System Operators for Electricity data
from entsoe import EntsoePandasClient


# PyTorch Lightning for model training enhancements
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


# methods that return Pandas Series
##client.query_day_ahead_prices(country_code, start=start, end=end)
# client.query_net_position(country_code, start=start, end=end, dayahead=True)
# client.query_crossborder_flows(country_code_from, country_code_to, start=start, end=end)
# client.query_scheduled_exchanges(country_code_from, country_code_to, start=start, end=end, dayahead=False)
# client.query_net_transfer_capacity_dayahead(country_code_from, country_code_to, start=start, end=end)
# client.query_net_transfer_capacity_weekahead(country_code_from, country_code_to, start=start, end=end)
# client.query_net_transfer_capacity_monthahead(country_code_from, country_code_to, start=start, end=end)
# client.query_net_transfer_capacity_yearahead(country_code_from, country_code_to, start=start, end=end)
# client.query_intraday_offered_capacity(country_code_from, country_code_to, start=start, end=end, implicit=True)
# client.query_offered_capacity(country_code_from, country_code_to, contract_marketagreement_type, start=start, end=end, implicit=True)
# client.query_aggregate_water_reservoirs_and_hydro_storage(country_code, start=start, end=end)

# methods that return Pandas DataFrames
# client.query_load(country_code, start=start, end=end)
# client.query_load_forecast(country_code, start=start, end=end)
# client.query_load_and_forecast(country_code, start=start, end=end)
# client.query_generation_forecast(country_code, start=start, end=end)
# client.query_wind_and_solar_forecast(country_code, start=start, end=end, psr_type=None)
# client.query_intraday_wind_and_solar_forecast(country_code, start=start, end=end, psr_type=None)
# client.query_generation(country_code, start=start, end=end, psr_type=None)
# client.query_generation_per_plant(country_code, start=start, end=end, psr_type=None, include_eic=False)
# client.query_installed_generation_capacity(country_code, start=start, end=end, psr_type=None)
# client.query_installed_generation_capacity_per_unit(country_code, start=start, end=end, psr_type=None)
# client.query_imbalance_prices(country_code, start=start, end=end, psr_type=None)
# client.query_contracted_reserve_prices(country_code, type_marketagreement_type, start=start, end=end, psr_type=None)
# client.query_contracted_reserve_amount(country_code, type_marketagreement_type, start=start, end=end, psr_type=None)
# client.query_unavailability_of_generation_units(country_code, start=start, end=end, docstatus=None, periodstartupdate=None, periodendupdate=None)
# client.query_unavailability_of_production_units(country_code, start, end, docstatus=None, periodstartupdate=None, periodendupdate=None)
# client.query_unavailability_transmission(country_code_from, country_code_to, start=start, end=end, docstatus=None, periodstartupdate=None, periodendupdate=None)
# client.query_withdrawn_unavailability_of_generation_units(country_code, start, end)
# client.query_physical_crossborder_allborders(country_code, start, end, export=True)
# client.query_generation_import(country_code, start, end)
# client.query_procured_balancing_capacity(country_code, process_type, start=start, end=end, type_marketagreement_type=None)


#### Download ENTSOE files
# Define the URL template and the directory where you want to save the downloaded files:
# Download from ENTSOE
client = EntsoePandasClient(api_key="309f0ee4-71e1-4502-b3ef-7d0bc315513b")
entsoe_download_dir = r"/Users/theodosisperifanis/Documents/dissertation"

# Define the range of dates for which you want to download files. Here, we'll use a start and end date:
start_date = pd.Timestamp("20250101", tz="Europe/Brussels")
end_date = pd.Timestamp.now(
    tz="Europe/Brussels"
)  # We add a day to get etss and entsoe data of the same date

# Convert these dates to UTC
start_date_utc = start_date.tz_convert("UTC")
end_date_utc = end_date.tz_convert("UTC")


# Country codes
# List of multiple country codes
country_codes = [
    "GR",
]

# Create an empty dictionary to store data for each country
data_dict = {}

for country_code in country_codes:
    try:
        load = client.query_load(
            country_code, start=start_date_utc, end=end_date_utc
        )
        data_dict[country_code] = load  # Store the data in the dictionary
    except Exception as e:
        print(f"Failed to retrieve data for {country_code}: {e}")


# Create a DataFrame from the collected data
greek_load = pd.DataFrame(data_dict["GR"])

# Convert the DataFrame index to UTC
greek_load.index = greek_load.index.tz_convert("UTC")


# Resetting the index of greek_load and renaming the index column to 'Timestamp'
greek_load = greek_load.reset_index().rename(columns={"index": "Timestamp"})
# Now, greek_load DataFrame has the timestamp in UTC
# The data are in UTC


greek_load = greek_load.rename(columns={"Actual Load": "load_actual"})

print(greek_load)
#################################
# Greek load forecast
#################################


# Dictionary to hold forecast data
forecast_dict = {}

for country_code in country_codes:
    try:
        forecast = client.query_load_forecast(
            country_code, start=start_date_utc, end=end_date_utc
        )
        forecast_dict[country_code] = forecast
    except Exception as e:
        print(f"Failed to retrieve forecast data for {country_code}: {e}")

# Create a DataFrame from the collected forecast data
greek_load_forecast = pd.DataFrame(forecast_dict["GR"])

# Convert the index to UTC
greek_load_forecast.index = greek_load_forecast.index.tz_convert("UTC")

# Reset index and rename timestamp column
greek_load_forecast = greek_load_forecast.reset_index().rename(
    columns={"index": "Timestamp"}
)


greek_load_forecast = greek_load_forecast.rename(
    columns={"Forecasted Load": "load_forecast"}
)


print(greek_load_forecast.columns)
#################################
# Wind and solar forecast
#################################
# Dictionary to hold wind & solar forecast data
wind_solar_forecast_dict = {}

for country_code in country_codes:
    try:
        wind_solar_forecast = client.query_wind_and_solar_forecast(
            country_code, start=start_date_utc, end=end_date_utc, psr_type=None
        )
        wind_solar_forecast_dict[country_code] = wind_solar_forecast
    except Exception as e:
        print(f"Failed to retrieve wind/solar forecast for {country_code}: {e}")

# Create DataFrame
greek_wind_solar_forecast = pd.DataFrame(wind_solar_forecast_dict["GR"])

# Convert index to UTC
greek_wind_solar_forecast.index = greek_wind_solar_forecast.index.tz_convert(
    "UTC"
)

# Reset index and rename
greek_wind_solar_forecast = greek_wind_solar_forecast.reset_index().rename(
    columns={"index": "Timestamp"}
)

greek_wind_solar_forecast = greek_wind_solar_forecast.rename(
    columns={"Solar": "solar_forecast", "Wind Onshore": "wind_forecast"}
)


print(greek_wind_solar_forecast.tail(10))
#################################
# generation per type
#################################


# Dictionary to hold generation data
generation_dict = {}

for country_code in country_codes:
    try:
        gen = client.query_generation(
            country_code, start=start_date_utc, end=end_date_utc, psr_type=None
        )

        # Normalize to a flat DataFrame
        if isinstance(gen, pd.Series):
            gen = gen.to_frame(name="Generation [MW]")

        # If columns come as a MultiIndex, flatten them
        if isinstance(gen.columns, pd.MultiIndex):
            gen.columns = [
                " | ".join([str(x) for x in col if str(x) != ""])
                for col in gen.columns
            ]

        generation_dict[country_code] = gen
    except Exception as e:
        print(f"Failed to retrieve generation for {country_code}: {e}")

# Create DataFrame for GR
greek_generation = generation_dict["GR"].copy()

# Convert index to UTC and reset index -> 'Timestamp'
greek_generation.index = greek_generation.index.tz_convert("UTC")
greek_generation = greek_generation.reset_index().rename(
    columns={"index": "Timestamp"}
)


greek_generation = greek_generation.rename(
    columns={
        "Fossil Brown coal/Lignite": "lignite",
        "Fossil Gas": "nat_gas",
        "Hydro Pumped Storage": "pump",
        "Hydro Water Reservoir": "hydro",
        "Solar": "solar_actual",
        "Wind Onshore": "wind_actual",
    }
).drop(columns=["Fossil Oil"])


#################################
# df handling
#################################

for df in [
    greek_load,
    greek_load_forecast,
    greek_wind_solar_forecast,
    greek_generation,
]:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.tz_localize(None)


dfs = {
    "greek_load": greek_load,
    "greek_load_forecast": greek_load_forecast,
    "greek_wind_solar_forecast": greek_wind_solar_forecast,
    "greek_generation": greek_generation,
}

for name, df in dfs.items():
    ts = pd.to_datetime(df["Timestamp"], errors="raise")
    bad = ts[
        (ts.dt.minute != 0) | (ts.dt.second != 0) | (ts.dt.microsecond != 0)
    ]
    print(f"{name}: dtype={ts.dtype}, rows={len(ts)}, non-hourly={len(bad)}")


# Merge step by step with inner joins
greek_generation_and_load = (
    greek_load.merge(greek_load_forecast, on="Timestamp", how="inner")
    .merge(greek_wind_solar_forecast, on="Timestamp", how="inner")
    .merge(greek_generation, on="Timestamp", how="inner")
    .sort_values("Timestamp")
    .reset_index(drop=True)
)


#################################
# find missing dates and hours
#################################

df = greek_generation_and_load.copy()
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="raise")

# 1) Build the expected continuous hourly index
start = df["Timestamp"].min()
end = df["Timestamp"].max()
expected = pd.date_range(start=start, end=end, freq="H")

# 2) Find missing hours
present = pd.DatetimeIndex(df["Timestamp"].unique())
missing_ts = expected.difference(present)

print(f"Total hours expected: {len(expected):,}")
print(f"Present hours:        {len(present):,}")
print(f"Missing hours:        {len(missing_ts):,}")

# 3) Nice table of missing dates & hours
missing_df = pd.DataFrame(
    {
        "Timestamp": missing_ts,
    }
)
missing_df["date"] = missing_df["Timestamp"].dt.date
missing_df["hour"] = missing_df["Timestamp"].dt.strftime("%H:00")
missing_df = missing_df.sort_values("Timestamp").reset_index(drop=True)

# Show a quick sample
print(missing_df.head(15))

#################################
# total res
#################################

greek_generation_and_load["res_forecast"] = greek_generation_and_load[
    "solar_forecast"
].fillna(0) + greek_generation_and_load["wind_forecast"].fillna(0)

greek_generation_and_load["res_actual"] = greek_generation_and_load[
    "solar_actual"
].fillna(0) + greek_generation_and_load["wind_actual"].fillna(0)

print(
    greek_generation_and_load[
        ["Timestamp", "res_forecast", "res_actual"]
    ].head()
)

#################################
# Rearrange columns
#################################
# Current columns
cols = greek_generation_and_load.columns.tolist()

# Get current column names
cols = greek_generation_and_load.columns.tolist()

# Group by type
forecast_cols = [c for c in cols if "forecast" in c]
actual_cols = [c for c in cols if "actual" in c]
other_cols = [
    c for c in cols if c not in (["Timestamp"] + forecast_cols + actual_cols)
]

# New order
new_order = ["Timestamp"] + forecast_cols + actual_cols + other_cols

# Apply reordering
greek_generation_and_load = greek_generation_and_load[new_order]

print(greek_generation_and_load)
#################################
# Stats
#################################
import pandas as pd

# Calculate stats
stats_df = greek_generation_and_load.agg(
    [
        "min",
        "max",
        "mean",
        "std",
        "median",
        lambda x: x.quantile(0.20),
        lambda x: x.quantile(0.75),
    ]
).T

# Rename quantile columns
stats_df = stats_df.rename(columns={"<lambda_0>": "q20", "<lambda_1>": "q75"})

# Round for neatness
stats_df = stats_df.round(2)

# Reset index so columns are visible as a column
stats_df = stats_df.reset_index().rename(columns={"index": "column"})

# Display
print(stats_df)


import pandas as pd

df = greek_generation_and_load.copy()
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# restrict to last 30 days
end = df["Timestamp"].max()
start = end - pd.Timedelta(days=30)
last_30d = df.loc[df["Timestamp"].between(start, end)]

# filter hours 08:00 → 13:00
mask = last_30d["Timestamp"].dt.hour.between(8, 17)
work_hours = last_30d.loc[mask]

# group by date and compute daily max res_actual
daily_max_workhours = (
    work_hours.groupby(work_hours["Timestamp"].dt.date)["res_actual"]
    .max()
    .reset_index(name="max_res_actual_8to17")
)

print(daily_max_workhours)
