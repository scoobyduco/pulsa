"""
Garmin daily wellness, sleep, and cycle preprocessing.
Builds df_daily_master from raw Garmin JSON exports.

Outputs
-------
df_daily_master: daily physiology + sleep + cycle, merged on Date

Exports
-------
df_daily_master.xlsx

Usage
-----
Called by pipeline.py via build_daily_master()
"""

import os
import json
import glob
import pandas as pd
import numpy as np

DATA_FOLDER = "./garmin_data"


def _load_wellness_uds(folder_path: str) -> pd.DataFrame:
    """Load and preprocess daily wellness data from *UDSFile*.json."""
    file_pattern = os.path.join(folder_path, "*UDSFile*.json")
    files = glob.glob(file_pattern)

    if not files:
        raise FileNotFoundError(f"No UDSFile JSON files found in {folder_path}")

    df_list = []
    common_meta = [
        "calendarDate", "totalKilocalories", "activeKilocalories", "bmrKilocalories",
        "totalSteps", "totalDistanceMeters", "restingHeartRate", "currentDayRestingHeartRate",
        "minHeartRate", "maxHeartRate",
        ["bodyBattery", "chargedValue"],
        ["bodyBattery", "drainedValue"],
        ["respiration", "avgWakingRespirationValue"],
    ]

    for file in files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                temp = pd.json_normalize(data, meta=common_meta, errors="ignore")
            elif isinstance(data, dict):
                if "dailyData" in data and isinstance(data["dailyData"], list):
                    temp = pd.json_normalize(data["dailyData"], meta=common_meta, errors="ignore")
                else:
                    temp = pd.json_normalize(data, meta=common_meta, errors="ignore")
            else:
                continue

            if not temp.empty:
                df_list.append(temp)
        except Exception as e:
            print(f"Warning: could not read {file}: {e}")

    if not df_list:
        raise ValueError("No wellness data could be loaded")

    df = pd.concat(df_list, ignore_index=True)
    df["calendarDate"] = pd.to_datetime(df["calendarDate"], errors="coerce")
    df = df.sort_values("calendarDate").drop_duplicates(subset=["calendarDate"], keep="last").reset_index(drop=True)

    # Column names
    df.columns = df.columns.str.strip().str.replace(".", "_", regex=False)
    rename_map = {
        "respiration_avgWakingRespirationValue": "avgRespiration",
        "bodyBattery_chargedValue": "charged",
        "bodyBattery_drainedValue": "drained",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Select columns
    cols = [
        "calendarDate", "totalKilocalories", "activeKilocalories", "bmrKilocalories",
        "totalSteps", "totalDistanceMeters", "minHeartRate", "maxHeartRate",
        "restingHeartRate", "currentDayRestingHeartRate", "charged", "drained", "avgRespiration",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()

    # Date and numeric cleaning
    df["calendarDate"] = pd.to_datetime(df["calendarDate"], errors="coerce")
    df = df.dropna(subset=["calendarDate"]).sort_values("calendarDate").drop_duplicates(subset=["calendarDate"], keep="last")

    numeric_cols = [c for c in df.columns if c != "calendarDate"]
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Physiological filters
    df = df[df["restingHeartRate"].between(35, 90)]
    if "avgRespiration" in df.columns:
        df = df[df["avgRespiration"].between(8, 25)]
    df = df[df["totalSteps"] >= 0]
    df = df[df["totalDistanceMeters"] >= 0]

    # Engineered daily metrics
    df["totalDistance_km"] = df["totalDistanceMeters"] / 1000
    df["recoveryBalance"] = (df["charged"] - df["drained"]) if "charged" in df.columns and "drained" in df.columns else np.nan
    df["external_daily_load"] = df["activeKilocalories"] + (df["totalDistanceMeters"] / 100)

    if "currentDayRestingHeartRate" in df.columns:
        df["acute_rhr_shift"] = df["currentDayRestingHeartRate"] - df["restingHeartRate"]
    else:
        df["acute_rhr_shift"] = np.nan

    # Rolling baselines
    df = df.sort_values("calendarDate")
    df["rhr_7day_avg"] = df["restingHeartRate"].rolling(7, min_periods=7).mean()
    df["rhr_delta"] = df["restingHeartRate"] - df["rhr_7day_avg"]
    df["rhr_7day_std"] = df["restingHeartRate"].rolling(7, min_periods=7).std()
    df["rhr_z"] = np.where(df["rhr_7day_std"] > 0, df["rhr_delta"] / df["rhr_7day_std"], np.nan)

    if "avgRespiration" in df.columns:
        df["resp_7day_avg"] = df["avgRespiration"].rolling(7, min_periods=7).mean()
        df["resp_delta"] = df["avgRespiration"] - df["resp_7day_avg"]
        df["resp_7day_std"] = df["avgRespiration"].rolling(7, min_periods=7).std()
        df["resp_z"] = np.where(df["resp_7day_std"] > 0, df["resp_delta"] / df["resp_7day_std"], np.nan)
    else:
        df["resp_7day_avg"] = np.nan
        df["resp_delta"] = np.nan
        df["resp_z"] = np.nan

    # Acute state
    if "acute_rhr_shift" in df.columns and "resp_z" in df.columns:
        conditions = [
            (df["acute_rhr_shift"] >= 6) & (df["resp_z"] >= 2),
            (df["acute_rhr_shift"] >= 4) | (df["resp_z"] >= 1),
            (df["acute_rhr_shift"] >= 2),
        ]
        choices = ["Overloaded", "Strained", "Elevated"]
        df["acute_state"] = np.select(conditions, choices, default="Stable")
    else:
        df["acute_state"] = "Unknown"

    # Final columns
    final_cols = [
        "calendarDate", "restingHeartRate", "rhr_7day_avg", "rhr_delta", "rhr_z",
        "avgRespiration", "resp_7day_avg", "resp_delta", "resp_z",
        "recoveryBalance", "external_daily_load", "acute_state",
    ]
    df = df[[c for c in final_cols if c in df.columns]].copy()
    df = df.rename(columns={"calendarDate": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


def _load_sleep(folder_path: str) -> pd.DataFrame:
    """Load and preprocess sleep data from *sleepData*.json."""
    file_pattern = os.path.join(folder_path, "*sleepData*.json")
    files = glob.glob(file_pattern)

    if not files:
        return pd.DataFrame(columns=["Date", "total_sleep_hours", "avgSleepStress"])

    df_list = []
    for file in files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                temp = pd.json_normalize(data)
            elif isinstance(data, dict):
                temp = pd.json_normalize(data["sleepData"]) if "sleepData" in data else pd.json_normalize(data)
            else:
                continue

            temp["source_file"] = os.path.basename(file)
            df_list.append(temp)
        except Exception as e:
            print(f"Warning: could not read {file}: {e}")

    if not df_list:
        return pd.DataFrame(columns=["Date", "total_sleep_hours", "avgSleepStress"])

    df = pd.concat(df_list, ignore_index=True)
    df.columns = df.columns.str.replace(".", "_", regex=False)
    df["calendarDate"] = pd.to_datetime(df["calendarDate"], errors="coerce")
    df = df.sort_values("calendarDate").drop_duplicates(subset=["calendarDate"], keep="last")

    # Filter OFF_WRIST
    if "sleepWindowConfirmationType" in df.columns:
        df = df[df["sleepWindowConfirmationType"] != "OFF_WRIST"]

    cols = ["calendarDate", "deepSleepSeconds", "remSleepSeconds", "lightSleepSeconds", "awakeSleepSeconds", "averageRespiration", "avgSleepStress"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()
    df = df.dropna(subset=["calendarDate"])

    df["total_sleep_sec"] = df["deepSleepSeconds"] + df["remSleepSeconds"] + df["lightSleepSeconds"]
    df["total_sleep_hours"] = df["total_sleep_sec"] / 3600

    df = df[(df["total_sleep_hours"] >= 3) & (df["total_sleep_hours"] <= 12)]
    if "averageRespiration" in df.columns:
        df = df[(df["averageRespiration"] >= 8) & (df["averageRespiration"] <= 25)]

    df["deep_ratio"] = df["deepSleepSeconds"] / df["total_sleep_sec"]
    df["rem_ratio"] = df["remSleepSeconds"] / df["total_sleep_sec"]
    df["fragmentation_ratio"] = df["awakeSleepSeconds"] / (df["total_sleep_sec"] + df["awakeSleepSeconds"])

    df = df.sort_values("calendarDate")
    if "averageRespiration" in df.columns:
        df["resp_7day_avg"] = df["averageRespiration"].rolling(7, min_periods=3).mean()
        df["resp_delta"] = df["averageRespiration"] - df["resp_7day_avg"]
    else:
        df["resp_delta"] = np.nan

    TARGET_SLEEP = 7.5
    df["sleep_debt"] = TARGET_SLEEP - df["total_sleep_hours"]
    df["sleep_debt_3day"] = df["sleep_debt"].rolling(3, min_periods=1).sum()

    df["sleepStress_7day_avg"] = df["avgSleepStress"].rolling(7, min_periods=3).mean()
    df["sleepStress_delta"] = df["avgSleepStress"] - df["sleepStress_7day_avg"]

    df["Date"] = df["calendarDate"] + pd.Timedelta(days=1)

    out_cols = [
        "Date", "total_sleep_hours", "deep_ratio", "rem_ratio", "fragmentation_ratio",
        "resp_delta", "sleep_debt_3day", "avgSleepStress", "sleepStress_delta",
    ]
    if "averageRespiration" in df.columns:
        out_cols.insert(6, "averageRespiration")
    df = df[[c for c in out_cols if c in df.columns]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


def _load_cycle(folder_path: str) -> pd.DataFrame:
    """Load and preprocess menstrual cycle data from *MenstrualCycles*.json."""
    pattern = os.path.join(folder_path, "*MenstrualCycles*.json")
    files = glob.glob(pattern)
    if not files:
        return pd.DataFrame(columns=["Date", "cycleDay", "cyclePhase", "cycleProgress", "daysUntilNextPeriod"])

    with open(files[0], "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        df = pd.json_normalize(data)
    elif isinstance(data, dict):
        if "cycles" in data:
            df = pd.json_normalize(data["cycles"])
        elif "menstrualCycles" in data:
            df = pd.json_normalize(data["menstrualCycles"])
        elif "dailyData" in data:
            df = pd.json_normalize(data["dailyData"])
        else:
            df = pd.json_normalize(data)
    else:
        raise ValueError("Unknown cycle JSON structure")

    cols = ["startDate", "actualCycleLength", "actualPeriodLength", "fertileWindowStart", "fertileWindowLength"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()
    df["startDate"] = pd.to_datetime(df["startDate"], errors="coerce")
    df = df.dropna(subset=["startDate"])

    for col in ["actualCycleLength", "actualPeriodLength", "fertileWindowStart", "fertileWindowLength"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce")

    df = df.sort_values("startDate").reset_index(drop=True)
    start_date = df["startDate"].min()
    end_date = pd.Timestamp.today().normalize()
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    df_day = pd.DataFrame({"Date": date_range})

    merged = pd.merge_asof(
        df_day.sort_values("Date"),
        df,
        left_on="Date",
        right_on="startDate",
        direction="backward",
    )

    merged["cycleEndDate"] = merged["startDate"] + pd.to_timedelta(merged["actualCycleLength"], unit="D")
    merged["cycleDay"] = (merged["Date"] - merged["startDate"]).dt.days + 1
    merged.loc[merged["cycleDay"] <= 0, "cycleDay"] = np.nan

    merged["isPeriod"] = merged["cycleDay"] <= merged["actualPeriodLength"]
    merged["fertileStartDay"] = merged["fertileWindowStart"]
    merged["fertileEndDay"] = merged["fertileWindowStart"] + merged["fertileWindowLength"] - 1
    merged["isFertileWindow"] = (merged["cycleDay"] >= merged["fertileStartDay"]) & (merged["cycleDay"] <= merged["fertileEndDay"])

    conditions = [
        merged["cycleDay"] <= merged["actualPeriodLength"],
        (merged["cycleDay"] > merged["actualPeriodLength"]) & (merged["cycleDay"] < merged["fertileStartDay"]),
        merged["isFertileWindow"],
        merged["cycleDay"] > merged["fertileEndDay"],
    ]
    merged["cyclePhase"] = np.select(conditions, ["Menstrual", "Follicular", "Ovulatory", "Luteal"], default="Unknown")

    merged["cycleProgress"] = np.where(
        merged["actualCycleLength"] > 0,
        (merged["cycleDay"] / merged["actualCycleLength"]).clip(0, 1),
        np.nan,
    )
    merged["daysUntilNextPeriod"] = merged["actualCycleLength"] - merged["cycleDay"]
    merged.loc[merged["daysUntilNextPeriod"] < 0, "daysUntilNextPeriod"] = np.nan

    out_cols = ["Date", "cycleDay", "cyclePhase", "cycleProgress", "daysUntilNextPeriod", "isPeriod", "isFertileWindow"]
    out_cols = [c for c in out_cols if c in merged.columns]
    merged = merged[out_cols].copy()
    merged["Date"] = pd.to_datetime(merged["Date"])

    return merged


def build_daily_master(data_folder: str | None = None, export_path: str | None = "df_daily_master.xlsx") -> pd.DataFrame:
    """
    Build df_daily_master from wellness, sleep, and cycle data.

    Parameters
    ----------
    data_folder : str, optional
        Path to folder containing Garmin JSON exports. Default: DATA_FOLDER.
    export_path : str, optional
        Path to save Excel. Pass None to skip export.

    Returns
    -------
    pd.DataFrame
        df_daily_master: daily physiology + sleep + cycle
    """
    folder = data_folder or DATA_FOLDER
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Data folder not found: {folder}")

    print("Loading wellness data...")
    df_daily = _load_wellness_uds(folder)

    print("Loading sleep data...")
    df_sleep = _load_sleep(folder)

    print("Loading cycle data...")
    df_cycle = _load_cycle(folder)

    print("Building daily master dataset...")
    df_daily_master = df_daily.merge(df_sleep, on="Date", how="outer", suffixes=("_physio", "_sleep"))
    df_daily_master = df_daily_master.merge(df_cycle, on="Date", how="outer")
    df_daily_master = df_daily_master.sort_values("Date").reset_index(drop=True)

    if export_path:
        df_daily_master.to_excel(export_path, index=False)
        print(f"Exported {export_path}")

    print("Preprocessing complete.")
    return df_daily_master


# Allow script execution
if __name__ == "__main__":
    build_daily_master()
