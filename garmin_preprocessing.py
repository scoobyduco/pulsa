"""
GARMIN DATA PREPROCESSING
Creates master datasets for the performance pipeline.

Outputs
-------
df_daily_master.xlsx
df_running_master.xlsx
"""

import pandas as pd
import numpy as np
import json
import glob
import os

DATA_FOLDER = "./garmin_data"

# =====================================================
# 1 LOAD WELLNESS DATA
# =====================================================

print("Loading wellness data...")

files = glob.glob(os.path.join(DATA_FOLDER, "*UDSFile*.json"))

df_list = []

for file in files:

    with open(file, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        temp = pd.json_normalize(data)
    else:
        temp = pd.json_normalize(data)

    df_list.append(temp)

df = pd.concat(df_list)

# -----------------------------------------------------

df.columns = df.columns.str.replace(".", "_")

df["calendarDate"] = pd.to_datetime(df["calendarDate"])

df = df.sort_values("calendarDate").drop_duplicates("calendarDate")

# -----------------------------------------------------

df["totalDistance_km"] = df["totalDistanceMeters"] / 1000

df["external_daily_load"] = (
    df["activeKilocalories"] +
    df["totalDistanceMeters"] / 100
)

# -----------------------------------------------------
# Rolling baselines
# -----------------------------------------------------

df["rhr_7day_avg"] = df["restingHeartRate"].rolling(7).mean()

df["rhr_delta"] = df["restingHeartRate"] - df["rhr_7day_avg"]

df["rhr_std"] = df["restingHeartRate"].rolling(7).std()

df["rhr_z"] = df["rhr_delta"] / df["rhr_std"]

# -----------------------------------------------------

df_daily = df[
[
"calendarDate",
"restingHeartRate",
"avgRespiration",
"external_daily_load",
"rhr_delta",
"rhr_z"
]
].copy()

df_daily = df_daily.rename(columns={"calendarDate":"Date"})

# =====================================================
# 2 LOAD SLEEP DATA
# =====================================================

print("Loading sleep data...")

sleep_files = glob.glob(os.path.join(DATA_FOLDER, "*sleepData*.json"))

sleep_list = []

for file in sleep_files:

    with open(file) as f:
        data = json.load(f)

    if isinstance(data, list):
        temp = pd.json_normalize(data)
    else:
        temp = pd.json_normalize(data)

    sleep_list.append(temp)

df_sleep = pd.concat(sleep_list)

df_sleep["calendarDate"] = pd.to_datetime(df_sleep["calendarDate"])

df_sleep["total_sleep_hours"] = (
    df_sleep["deepSleepSeconds"] +
    df_sleep["remSleepSeconds"] +
    df_sleep["lightSleepSeconds"]
) / 3600

df_sleep["Date"] = df_sleep["calendarDate"] + pd.Timedelta(days=1)

df_sleep = df_sleep[[
"Date",
"total_sleep_hours",
"avgSleepStress"
]]

# =====================================================
# 3 LOAD CYCLE DATA
# =====================================================

print("Loading cycle data...")

cycle_file = glob.glob(os.path.join(DATA_FOLDER, "*MenstrualCycles*.json"))[0]

with open(cycle_file) as f:
    data = json.load(f)

df_cycles = pd.json_normalize(data)

df_cycles["startDate"] = pd.to_datetime(df_cycles["startDate"])

# -----------------------------------------------------

date_range = pd.date_range(
df_cycles["startDate"].min(),
pd.Timestamp.today(),
freq="D"
)

df_day = pd.DataFrame({"Date":date_range})

df_cycles = df_cycles.sort_values("startDate")

df_cycle = pd.merge_asof(
df_day,
df_cycles,
left_on="Date",
right_on="startDate",
direction="backward"
)

df_cycle["cycleDay"] = (
df_cycle["Date"] - df_cycle["startDate"]
).dt.days + 1

df_cycle["cyclePhase"] = np.select(
[
df_cycle["cycleDay"] <= df_cycle["actualPeriodLength"],
df_cycle["cycleDay"] < df_cycle["fertileWindowStart"],
df_cycle["cycleDay"] <= df_cycle["fertileWindowStart"] + df_cycle["fertileWindowLength"],
df_cycle["cycleDay"] > df_cycle["fertileWindowStart"]
],
[
"Menstrual",
"Follicular",
"Ovulatory",
"Luteal"
],
default="Unknown"
)

# =====================================================
# 4 LOAD ACTIVITIES
# =====================================================

print("Loading activities...")

activity_file = glob.glob(os.path.join(DATA_FOLDER, "*summarizedActivities*.json"))[0]

with open(activity_file) as f:
    data = json.load(f)

df_act = pd.json_normalize(data["summarizedActivitiesExport"])

df_act["startTimeLocal"] = pd.to_datetime(df_act["startTimeLocal"], unit="ms")

df_act["Date"] = df_act["startTimeLocal"].dt.date

df_act["distance_km"] = df_act["distance"] / 100000

df_act["moving_min"] = df_act["movingDuration"] / 1000 / 60

df_act = df_act[df_act["sportType"] == "RUNNING"]

df_activities = df_act[[
"activityId",
"Date",
"distance_km",
"moving_min",
"avgHr",
"maxHr",
"aerobicTrainingEffect",
"anaerobicTrainingEffect",
"avgRunCadence"
]]

# =====================================================
# 5 MERGE DAILY DATASETS
# =====================================================

print("Building daily master dataset...")

df_daily_master = df_daily.merge(
df_sleep,
on="Date",
how="outer"
)

df_daily_master = df_daily_master.merge(
df_cycle,
on="Date",
how="left"
)

# =====================================================
# 6 BUILD RUNNING MASTER
# =====================================================

print("Building running master dataset...")

df_running_master = df_activities.merge(
df_daily_master,
on="Date",
how="left"
)

# =====================================================
# 7 EXPORT
# =====================================================

df_daily_master = df_daily_master.sort_values("Date")

df_running_master = df_running_master.sort_values("Date")

df_daily_master.to_excel("df_daily_master.xlsx", index=False)

df_running_master.to_excel("df_running_master.xlsx", index=False)

print("Preprocessing complete.")
