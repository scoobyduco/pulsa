"""
Garmin running activities preprocessing.
Builds df_running_master from Garmin summarizedActivities JSON exports.

Outputs
-------
df_running_master: one row per running session with performance metrics

Exports
-------
df_running_master.xlsx

Usage
-----
Called by pipeline.py via build_running_master()
"""

import os
import json
import glob
import pandas as pd
import numpy as np

DATA_FOLDER = "./garmin_data"


def _load_activities_json(folder_path: str) -> pd.DataFrame:
    """Load running activities from *summarizedActivities*.json."""
    pattern = os.path.join(folder_path, "*summarizedActivities*.json")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"No summarizedActivities JSON found in {folder_path}")

    df_list = []
    for filepath in files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                if data and isinstance(data[0], dict) and "summarizedActivitiesExport" in data[0]:
                    temp = pd.json_normalize(data[0]["summarizedActivitiesExport"])
                else:
                    temp = pd.json_normalize(data)
            elif isinstance(data, dict):
                if "summarizedActivities" in data:
                    temp = pd.json_normalize(data["summarizedActivities"])
                elif "summarizedActivitiesExport" in data:
                    temp = pd.json_normalize(data["summarizedActivitiesExport"])
                else:
                    temp = pd.json_normalize(data)
            else:
                continue

            if not temp.empty:
                df_list.append(temp)
        except Exception as e:
            print(f"Warning: could not read {filepath}: {e}")

    if not df_list:
        raise ValueError("No activity data could be loaded")

    return pd.concat(df_list, ignore_index=True)


def build_running_master(
    data_folder: str | None = None,
    export_path: str | None = "df_running_master.xlsx",
) -> pd.DataFrame:
    """
    Build df_running_master from Garmin running activity JSON.

    Parameters
    ----------
    data_folder : str, optional
        Path to folder containing Garmin JSON exports. Default: DATA_FOLDER.
    export_path : str, optional
        Path to save Excel. Pass None to skip export.

    Returns
    -------
    pd.DataFrame
        df_running_master: one row per run with distance, pace, HR, cadence, etc.
    """
    folder = data_folder or DATA_FOLDER
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Data folder not found: {folder}")

    print("Loading Garmin running activities...")
    df = _load_activities_json(folder)

    df["startTimeLocal"] = pd.to_datetime(df["startTimeLocal"], unit="ms")
    df = df.sort_values("startTimeLocal").reset_index(drop=True)

    columns_to_keep = [
        "name", "activityId", "sportType", "startTimeLocal",
        "duration", "movingDuration", "distance", "avgSpeed",
        "elevationGain", "avgHr", "maxHr", "avgRunCadence",
        "trainingEffectLabel", "aerobicTrainingEffect", "anaerobicTrainingEffect",
    ]
    columns_to_keep = [c for c in columns_to_keep if c in df.columns]
    df = df[columns_to_keep].dropna(subset=["activityId"]).copy()

    numeric_cols = [
        "duration", "movingDuration", "distance", "avgSpeed",
        "elevationGain", "avgHr", "maxHr", "avgRunCadence",
        "aerobicTrainingEffect", "anaerobicTrainingEffect",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["sportType"] == "RUNNING"].copy()

    df["duration_min"] = df["movingDuration"] / 1000 / 60
    df["distance_km"] = df["distance"] / 100000
    df["pace_min_per_km"] = df["duration_min"] / df["distance_km"]
    df["elevationGain_m"] = df["elevationGain"] / 100
    df["Date"] = pd.to_datetime(df["startTimeLocal"]).dt.date

    df["efficiency_index"] = np.where(
        df["avgHr"] > 0,
        df["pace_min_per_km"] / df["avgHr"],
        np.nan,
    )

    df["distance_km"] = df["distance_km"].round(2)
    df["duration_min"] = df["duration_min"].round(2)
    df["pace_min_per_km"] = df["pace_min_per_km"].round(2)
    df["elevationGain_m"] = df["elevationGain_m"].round(0)

    out_cols = [
        "activityId", "Date", "distance_km", "duration_min", "pace_min_per_km",
        "avgHr", "maxHr", "avgRunCadence", "elevationGain_m",
        "aerobicTrainingEffect", "anaerobicTrainingEffect", "trainingEffectLabel",
        "efficiency_index",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    df_running_master = df[out_cols].copy()

    df_running_master.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_running_master["Date"] = pd.to_datetime(df_running_master["Date"])
    df_running_master = df_running_master.sort_values("Date").reset_index(drop=True)

    if export_path:
        df_running_master.to_excel(export_path, index=False)
        print(f"Exported {export_path}")

    print("Running preprocessing complete.")
    return df_running_master


if __name__ == "__main__":
    build_running_master()
