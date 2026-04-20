"""
Garmin running activities preprocessing.
Builds df_running_master from Garmin activity exports.

Supports:
- summarizedActivities: completed runs (distance, pace, HR)
- workout: running workout plans (estimated distance/duration; HR/pace = NaN)

Layouts: flat (garmin_data/) or nested (DI_CONNECT/DI-Connect-Fitness/)
"""

import os
import json
import glob
import pandas as pd
import numpy as np

DATA_FOLDER = os.environ.get("GARMIN_DATA_PATH", "./garmin_data")


def _discover_activity_files(folder_path: str) -> tuple[list, list]:
    """Discover summarizedActivities and workout JSON files."""
    summaries = glob.glob(os.path.join(folder_path, "*summarizedActivities*.json"))
    workouts = glob.glob(os.path.join(folder_path, "*workout*.json"))
    fitness = os.path.join(folder_path, "DI_CONNECT", "DI-Connect-Fitness")
    if os.path.isdir(fitness):
        summaries = list(set(summaries) | set(glob.glob(os.path.join(fitness, "*summarizedActivities*.json"))))
        workouts = list(set(workouts) | set(glob.glob(os.path.join(fitness, "*workout*.json"))))
    return summaries, workouts


def _load_workout_json(filepath: str) -> pd.DataFrame:
    """Load running workouts from *workout*.json (planned runs; no HR/actual pace)."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list) and data and "workoutList" in data[0]:
        items = data[0]["workoutList"]
    elif isinstance(data, dict) and "workoutList" in data:
        items = data["workoutList"]
    else:
        return pd.DataFrame()
    rows = []
    for w in items:
        sport = w.get("sportType") or {}
        skey = sport.get("sportTypeKey", "") if isinstance(sport, dict) else ""
        if str(skey).lower() != "running":
            continue
        est_dist = w.get("estimatedDistanceInMeters") or 0
        est_dur = w.get("estimatedDurationInSecs") or 0
        created = w.get("createdDate") or w.get("updatedDate") or ""
        if not created:
            continue
        try:
            dt = pd.to_datetime(created)
        except Exception:
            continue
        ms_ts = int(pd.Timestamp(dt).value // 1_000_000)
        rows.append({
            "activityId": w.get("workoutId"),
            "name": w.get("workoutName", ""),
            "sportType": "RUNNING",
            "startTimeLocal": ms_ts,
            "duration": est_dur * 1000 if est_dur else np.nan,
            "movingDuration": est_dur * 1000 if est_dur else np.nan,
            "distance": est_dist * 100 if est_dist else np.nan,
            "avgSpeed": (est_dist / est_dur) if est_dur and est_dist else np.nan,
            "elevationGain": 0,
            "avgHr": np.nan,
            "maxHr": np.nan,
            "avgRunCadence": np.nan,
            "aerobicTrainingEffect": np.nan,
            "anaerobicTrainingEffect": np.nan,
            "trainingEffectLabel": np.nan,
            "_from_workout": True,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _load_activities_json(folder_path: str) -> pd.DataFrame:
    """Load running activities from *summarizedActivities*.json and *workout*.json."""
    summaries, workouts = _discover_activity_files(folder_path)

    if not summaries and not workouts:
        raise FileNotFoundError(
            f"No summarizedActivities or workout JSON found in {folder_path}. "
            "Expected *summarizedActivities*.json or *workout*.json in the folder or DI_CONNECT/DI-Connect-Fitness/"
        )

    df_list: list[pd.DataFrame] = []

    for filepath in summaries:
        try:
            file_mtime = os.path.getmtime(filepath)
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
                temp["_source_mtime"] = file_mtime
                df_list.append(temp)
        except Exception as e:
            print(f"Warning: could not read {filepath}: {e}")

    for filepath in workouts:
        try:
            dw = _load_workout_json(filepath)
            if not dw.empty:
                file_mtime = os.path.getmtime(filepath)
                dw["_source_mtime"] = file_mtime
                df_list.append(dw)
        except Exception as e:
            print(f"Warning: could not read workout {filepath}: {e}")

    if not df_list:
        raise ValueError("No activity or workout data could be loaded")

    df = pd.concat(df_list, ignore_index=True)
    if "activityId" in df.columns and "_source_mtime" in df.columns:
        df = df.sort_values(["activityId", "_source_mtime"], ascending=[True, False]).drop_duplicates(subset=["activityId"], keep="first").reset_index(drop=True)
    df = df.drop(columns=["_source_mtime"], errors="ignore")
    return df


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
