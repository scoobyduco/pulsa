import pandas as pd
import glob
import os

def build_running_master():

    print("Loading Garmin running activities...")

    DATA_FOLDER = "./garmin_activities"

    files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))

    df_list = []

    for f in files:

        temp = pd.read_csv(f)

        temp["source_file"] = os.path.basename(f)

        df_list.append(temp)

    df = pd.concat(df_list, ignore_index=True)

    df.columns = df.columns.str.strip().str.replace(" ", "_")

    rename_map = {
        "startTimeLocal": "Date",
        "distance": "distance_m",
        "movingDuration": "moving_sec",
        "averageHR": "avgHr",
        "maxHR": "maxHr"
    }

    df = df.rename(columns=rename_map)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df = df.dropna(subset=["Date"])

    if "activityType" in df.columns:

        df = df[df["activityType"].str.contains("run", case=False)]

    df["distance_km"] = df["distance_m"] / 1000
    df["moving_min"] = df["moving_sec"] / 60

    df["pace"] = df["moving_min"] / df["distance_km"]

    df["efficiency_index"] = df["avgHr"] / df["pace"]

    df_running_master = df[[
        "Date",
        "distance_km",
        "moving_min",
        "avgHr",
        "maxHr",
        "efficiency_index"
    ]]

    df_running_master.to_excel("df_running_master.xlsx", index=False)

    return df_running_master
