import os
import json
from datetime import date, timedelta

import pandas as pd
import numpy as np

from garmin_daily_preprocessing import build_daily_master, _load_cycle
from garmin_running_preprocessing import build_running_master
from pulsa_engine import run_pulsa_engine
from audit import build_audit_log

# Data folder: set GARMIN_DATA_PATH env var, or use ./garmin_data
DATA_FOLDER = os.environ.get("GARMIN_DATA_PATH", "./garmin_data")



def _maybe_fetch_garmin_csv() -> None:
    if not (os.environ.get("GARMIN_EMAIL", "").strip() and os.environ.get("GARMIN_PASSWORD")):
        return
    if _garmin_csv_ready(DATA_FOLDER) and os.environ.get("GARMIN_REFRESH", "").strip() != "1":
        print(
            "Garmin CSVs already present; skipping API fetch. "
            "Set GARMIN_REFRESH=1 to pull fresh data from Garmin."
        )
        return
    try:
        from fetch_garmin import connect, fetch_activities, fetch_daily_metrics, save_data

        print("Refreshing Garmin data from API (this can take several minutes)...")
        days = int(os.environ.get("GARMIN_FETCH_DAYS", "365"))
        end = date.today()
        start = end - timedelta(days=days)
        api = connect()
        print("Fetching activities...")
        act = fetch_activities(api, start, end)
        print(f"Fetched {len(act)} activities")
        print("Fetching daily metrics...")
        daily = fetch_daily_metrics(api, start, end)
        print(f"Fetched {len(daily)} daily records")
        print("Saving data...")
        save_data(act, daily, DATA_FOLDER)
        print("Garmin CSVs updated.")
    except Exception as e:
        print(f"Garmin API fetch skipped: {e}")


def _garmin_csv_ready(folder: str) -> bool:
    act_p = os.path.join(folder, "activities.csv")
    day_p = os.path.join(folder, "daily.csv")
    if not (os.path.isfile(act_p) and os.path.isfile(day_p)):
        return False
    try:
        if pd.read_csv(act_p).empty or pd.read_csv(day_p).empty:
            return False
    except Exception:
        return False
    return True


def _load_running_master_csv(data_folder: str) -> pd.DataFrame:
    path = os.path.join(data_folder, "activities.csv")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["activityId"] = df["activityId"].astype(str)
    df = df.drop_duplicates(subset=["activityId"], keep="last").sort_values("Date")
    for c in [
        "distance_km",
        "duration_min",
        "pace_min_per_km",
        "avgHr",
        "maxHr",
        "avgRunCadence",
        "elevationGain_m",
        "aerobicTrainingEffect",
        "anaerobicTrainingEffect",
        "efficiency_index",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "trainingEffectLabel" not in df.columns:
        df["trainingEffectLabel"] = np.nan
    out_cols = [
        "activityId",
        "Date",
        "distance_km",
        "duration_min",
        "pace_min_per_km",
        "avgHr",
        "maxHr",
        "avgRunCadence",
        "elevationGain_m",
        "aerobicTrainingEffect",
        "anaerobicTrainingEffect",
        "trainingEffectLabel",
        "efficiency_index",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    df = df[out_cols].replace([np.inf, -np.inf], np.nan)
    return df.reset_index(drop=True)


def _sleep_block_from_daily_csv(raw: pd.DataFrame) -> pd.DataFrame:
    r = raw.copy()
    r["Date"] = pd.to_datetime(r["Date"], errors="coerce")
    r = r.dropna(subset=["Date"]).sort_values("Date")
    for c in ["deepSleepSeconds", "remSleepSeconds", "lightSleepSeconds", "awakeSleepSeconds"]:
        if c not in r.columns:
            r[c] = 0.0
        else:
            r[c] = pd.to_numeric(r[c], errors="coerce").fillna(0)
    r["total_sleep_sec"] = r["deepSleepSeconds"] + r["remSleepSeconds"] + r["lightSleepSeconds"]
    api_tot = pd.to_numeric(r.get("total_sleep_hours"), errors="coerce")
    r["total_sleep_hours"] = np.where(
        r["total_sleep_sec"] > 0,
        r["total_sleep_sec"] / 3600.0,
        api_tot,
    )
    # Keep every calendar day for merge with physio; do not drop partial/in-progress days.
    if "sleepWindowConfirmationType" in r.columns:
        off = r["sleepWindowConfirmationType"].astype(str) == "OFF_WRIST"
        r.loc[off, "total_sleep_hours"] = np.nan
    _sleep_ok = r["total_sleep_hours"].isna() | ((r["total_sleep_hours"] >= 3) & (r["total_sleep_hours"] <= 12))
    r.loc[~_sleep_ok, "total_sleep_hours"] = np.nan
    r["deep_ratio"] = np.where(r["total_sleep_sec"] > 0, r["deepSleepSeconds"] / r["total_sleep_sec"], np.nan)
    r["rem_ratio"] = np.where(r["total_sleep_sec"] > 0, r["remSleepSeconds"] / r["total_sleep_sec"], np.nan)
    den = r["total_sleep_sec"] + r["awakeSleepSeconds"]
    r["fragmentation_ratio"] = np.where(den > 0, r["awakeSleepSeconds"] / den, np.nan)
    if "averageRespiration_sleep" in r.columns:
        r["averageRespiration"] = pd.to_numeric(r["averageRespiration_sleep"], errors="coerce")
    else:
        r["averageRespiration"] = np.nan
    r = r.sort_values("Date")
    if r["averageRespiration"].notna().any():
        r["resp_7day_avg"] = r["averageRespiration"].rolling(7, min_periods=3).mean()
        r["resp_delta"] = r["averageRespiration"] - r["resp_7day_avg"]
    else:
        r["resp_delta"] = np.nan
    TARGET_SLEEP = 7.5
    r["sleep_debt"] = TARGET_SLEEP - r["total_sleep_hours"]
    r["sleep_debt_3day"] = r["sleep_debt"].rolling(3, min_periods=1).sum()
    r["avgSleepStress"] = pd.to_numeric(r.get("avgSleepStress"), errors="coerce")
    if r["avgSleepStress"].notna().any():
        r["sleepStress_7day_avg"] = r["avgSleepStress"].rolling(7, min_periods=3).mean()
        r["sleepStress_delta"] = r["avgSleepStress"] - r["sleepStress_7day_avg"]
    else:
        r["avgSleepStress"] = np.nan
        r["sleepStress_7day_avg"] = np.nan
        r["sleepStress_delta"] = np.nan
    out_cols = [
        "Date",
        "total_sleep_hours",
        "deep_ratio",
        "rem_ratio",
        "fragmentation_ratio",
        "resp_delta",
        "sleep_debt_3day",
        "avgSleepStress",
        "sleepStress_delta",
    ]
    if r["averageRespiration"].notna().any():
        out_cols.insert(6, "averageRespiration")
    out_cols = [c for c in out_cols if c in r.columns]
    out = r[out_cols].copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def _physio_block_from_daily_csv(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    for c in ["activeKilocalories", "totalSteps", "totalDistanceMeters", "avgRespiration", "restingHeartRate"]:
        if c not in df.columns:
            df[c] = np.nan
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    m = df["restingHeartRate"].notna()
    df = df[(~m) | (df["restingHeartRate"].between(35, 90))]
    if "avgRespiration" in df.columns:
        ar = df["avgRespiration"].notna()
        df = df[(~ar) | (df["avgRespiration"].between(8, 25))]
    if "totalSteps" in df.columns:
        ts = df["totalSteps"].notna()
        df = df[(~ts) | (df["totalSteps"] >= 0)]
    if "totalDistanceMeters" in df.columns:
        td = df["totalDistanceMeters"].notna()
        df = df[(~td) | (df["totalDistanceMeters"] >= 0)]
    df["recoveryBalance"] = np.nan
    df["external_daily_load"] = df["activeKilocalories"] + (df["totalDistanceMeters"] / 100)
    df["acute_rhr_shift"] = np.nan
    df = df.sort_values("Date")
    df["rhr_7day_avg"] = df["restingHeartRate"].rolling(7, min_periods=3).mean()
    df["rhr_delta"] = df["restingHeartRate"] - df["rhr_7day_avg"]
    df["rhr_7day_std"] = df["restingHeartRate"].rolling(7, min_periods=3).std()
    df["rhr_z"] = np.where(df["rhr_7day_std"] > 0, df["rhr_delta"] / df["rhr_7day_std"], np.nan)
    if "avgRespiration" in df.columns and df["avgRespiration"].notna().any():
        df["resp_7day_avg"] = df["avgRespiration"].rolling(7, min_periods=3).mean()
        df["resp_delta"] = df["avgRespiration"] - df["resp_7day_avg"]
        df["resp_7day_std"] = df["avgRespiration"].rolling(7, min_periods=3).std()
        df["resp_z"] = np.where(df["resp_7day_std"] > 0, df["resp_delta"] / df["resp_7day_std"], np.nan)
    else:
        df["resp_7day_avg"] = np.nan
        df["resp_delta"] = np.nan
        df["resp_z"] = np.nan
        df["avgRespiration"] = np.nan
    conditions = [
        (df["acute_rhr_shift"] >= 6) & (df["resp_z"] >= 2),
        (df["acute_rhr_shift"] >= 4) | (df["resp_z"] >= 1),
        (df["acute_rhr_shift"] >= 2),
    ]
    choices = ["Overloaded", "Strained", "Elevated"]
    df["acute_state"] = np.select(conditions, choices, default="Stable")
    final_cols = [
        "Date",
        "restingHeartRate",
        "rhr_7day_avg",
        "rhr_delta",
        "rhr_z",
        "avgRespiration",
        "resp_7day_avg",
        "resp_delta",
        "resp_z",
        "recoveryBalance",
        "external_daily_load",
        "acute_state",
    ]
    final_cols = [c for c in final_cols if c in df.columns]
    out = df[final_cols].copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def _load_daily_master_csv(data_folder: str) -> pd.DataFrame:
    path = os.path.join(data_folder, "daily.csv")
    raw = pd.read_csv(path)
    df_daily = _physio_block_from_daily_csv(raw)
    df_sleep = _sleep_block_from_daily_csv(raw)
    df_cycle = _load_cycle(data_folder)
    df_daily_master = df_daily.merge(df_sleep, on="Date", how="outer", suffixes=("_physio", "_sleep"))
    # Left join cycle only: do not add calendar days that have no Garmin daily row (avoids empty "today")
    df_daily_master = df_daily_master.merge(df_cycle, on="Date", how="left")
    df_daily_master = df_daily_master.sort_values("Date").reset_index(drop=True)
    return df_daily_master


def main():
    """Run the full Pulsa data pipeline."""
    print("Starting full pipeline...")
    print(f"Data folder: {DATA_FOLDER}")

    _maybe_fetch_garmin_csv()

    # Always use JSON preprocessing (handles unit conversions correctly).
    # CSV path from fetch_garmin has known unit bugs (pace in wrong scale).
    print("Loading from Garmin JSON exports...")
    df_daily_master = build_daily_master(data_folder=DATA_FOLDER)
    df_running_master = build_running_master(data_folder=DATA_FOLDER)

    df_daily_master["Date"] = pd.to_datetime(df_daily_master["Date"])
    df_running_master["Date"] = pd.to_datetime(df_running_master["Date"])

    print("Computing training load...")

    # load_1d is the canonical name; drop external_daily_load after copying
    df_daily_master["load_1d"] = df_daily_master["external_daily_load"]
    df_daily_master.drop(columns=["external_daily_load"], inplace=True, errors="ignore")

    df_daily_master["load_3d"] = df_daily_master["load_1d"].rolling(3).sum()

    df_daily_master["load_7d"] = df_daily_master["load_1d"].rolling(7).sum()

    df_daily_master["load_28d"] = df_daily_master["load_1d"].rolling(28).mean()

    df_daily_master["acwr"] = df_daily_master["load_7d"].div(
        df_daily_master["load_28d"].replace(0, np.nan)
    )

    print("Computing readiness...")

    score = 100

    df_daily_master["readiness_score"] = (
        score
        - (df_daily_master["load_3d"] > 2000) * 20
        - (df_daily_master["rhr_z"] > 1.5) * 20
        - (df_daily_master["resp_z"] > 1.5) * 10
    )

    df_daily_master["readiness_score"] = df_daily_master["readiness_score"].clip(0, 100)

    df_daily_master["readiness_state"] = pd.cut(
        df_daily_master["readiness_score"],
        bins=[0, 50, 70, 85, 100],
        labels=["Recovery Needed", "Fatigued", "Normal", "Optimal"],
    )

    print("Computing running efficiency trends...")

    df_running_master = df_running_master.sort_values("Date")

    # Use whichever efficiency column exists (efficiency_index pre-engine, efficiency post-engine)
    _eff_col = "efficiency_index" if "efficiency_index" in df_running_master.columns else "efficiency"
    df_running_master["efficiency_14d"] = (
        df_running_master[_eff_col].shift(1).rolling(14).mean()
    )
    df_running_master["efficiency_delta"] = df_running_master[_eff_col] - df_running_master["efficiency_14d"]

    print("Merging datasets...")

    # Use daily as base so we have ALL days (including non-run days) — app "Today" needs readiness/RHR through today
    # One run per day when multiple; run columns are NaN on non-run days
    runs_one_per_day = df_running_master.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    df_run_daily = df_daily_master.merge(runs_one_per_day, on="Date", how="left")

    print("Interpreting sessions...")

    df_run_daily["session_flag"] = np.select(
        [
            (df_run_daily["efficiency_delta"] > 0.02) & (df_run_daily["load_3d"] > 2000),
            (df_run_daily["efficiency_delta"] > 0.02),
        ],
        [
            "Fatigue driven",
            "Underperforming",
        ],
        default="Normal",
    )

    print("Running Pulsa engine...")
    df_run_daily, insights, training_windows = run_pulsa_engine(df_run_daily)

    _audit = build_audit_log(
        df_run_daily,
        {"insights": insights, "training_windows": training_windows},
    )
    with open("audit.json", "w", encoding="utf-8") as f:
        json.dump(_audit, f, indent=2, ensure_ascii=False)

    df_daily_master.to_excel("daily_engine_output.xlsx", index=False)
    df_run_daily.to_excel("running_engine_output.xlsx", index=False)

    # Freshness check: warn if output data trails today by more than 2 days.
    # Helps catch silent failures where pipeline pointed at a stale folder.
    try:
        _latest = pd.to_datetime(df_daily_master["Date"]).max().date()
        _lag = (date.today() - _latest).days
        if _lag > 2:
            print(
                f"WARNING: engine output's latest date is {_latest} "
                f"({_lag} days behind today). "
                f"Check that GARMIN_DATA_PATH (or ./garmin_data) has the newest export."
            )
        else:
            print(f"Engine output is fresh: latest date {_latest} ({_lag}d lag).")
    except Exception:
        pass

    with open("insights.json", "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2, ensure_ascii=False)
    with open("training_windows.json", "w", encoding="utf-8") as f:
        json.dump(training_windows, f, indent=2)

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
