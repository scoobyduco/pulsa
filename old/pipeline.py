"""
Cycle-Aware Running Performance Pipeline
Simple version

Required files in same folder:
- df_daily_master.xlsx
- df_running_master.xlsx
"""

# =========================
# 1. IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 2. LOAD DATA
# =========================

print("Loading datasets...")

df_daily_master = pd.read_excel("df_daily_master.xlsx")
df_running_master = pd.read_excel("df_running_master.xlsx")

df_daily_master["Date"] = pd.to_datetime(df_daily_master["Date"])
df_running_master["Date"] = pd.to_datetime(df_running_master["Date"])


# =========================
# 3. DAILY PHYSIOLOGY ENGINE
# =========================

print("Computing physiology metrics...")

df_daily_master = df_daily_master.sort_values("Date")

# rolling baselines
df_daily_master["rhr_7d_avg"] = (
    df_daily_master["restingHeartRate"]
    .rolling(7, min_periods=3)
    .mean()
)

df_daily_master["resp_7d_avg"] = (
    df_daily_master["avgRespiration"]
    .rolling(7, min_periods=3)
    .mean()
)

# deviations
df_daily_master["rhr_delta"] = (
    df_daily_master["restingHeartRate"] -
    df_daily_master["rhr_7d_avg"]
)

df_daily_master["resp_delta"] = (
    df_daily_master["avgRespiration"] -
    df_daily_master["resp_7d_avg"]
)

# z-scores
df_daily_master["rhr_z"] = (
    (df_daily_master["restingHeartRate"] -
     df_daily_master["rhr_7d_avg"]) /
    df_daily_master["restingHeartRate"].rolling(7).std()
)

df_daily_master["resp_z"] = (
    (df_daily_master["avgRespiration"] -
     df_daily_master["resp_7d_avg"]) /
    df_daily_master["avgRespiration"].rolling(7).std()
)


# =========================
# 4. TRAINING LOAD ENGINE
# =========================

print("Computing training load...")

df_daily_master["load_1d"] = df_daily_master["external_daily_load"]

df_daily_master["load_3d"] = (
    df_daily_master["load_1d"]
    .rolling(3, min_periods=1)
    .sum()
)

df_daily_master["load_7d"] = (
    df_daily_master["load_1d"]
    .rolling(7, min_periods=1)
    .sum()
)

df_daily_master["load_28d"] = (
    df_daily_master["load_1d"]
    .rolling(28, min_periods=7)
    .mean()
)

df_daily_master["acwr"] = (
    df_daily_master["load_7d"] /
    df_daily_master["load_28d"]
)


# =========================
# 5. SLEEP ENGINE
# =========================

print("Computing sleep metrics...")

df_daily_master["sleep_debt_3d"] = (
    (8 - df_daily_master["total_sleep_hours"])
    .rolling(3, min_periods=1)
    .sum()
)


# =========================
# 6. READINESS SCORE
# =========================

print("Calculating readiness score...")


# =========================
# 6. READINESS SCORE (OPTIMIZED)
# =========================

print("Calculating readiness score...")


def compute_readiness(row):

    score = 100

    # -------------------------
    # Training Load Fatigue
    # -------------------------

    if row["load_3d"] > 2000:
        score -= 20

    # Chronic overload
    if "acwr" in row and row["acwr"] > 1.5:
        score -= 10

    # -------------------------
    # Sleep Recovery
    # -------------------------

    if row["sleep_debt_3d"] > 3:
        score -= 15

    if row["sleep_debt_3d"] > 5:
        score -= 10

    # -------------------------
    # Autonomic Stress
    # -------------------------

    if pd.notna(row["rhr_z"]) and row["rhr_z"] > 1.5:
        score -= 20

    if pd.notna(row["resp_z"]) and row["resp_z"] > 1.5:
        score -= 10

    # -------------------------
    # Menstrual Cycle Physiology
    # -------------------------

    if "cyclePhase" in row and pd.notna(row["cyclePhase"]):

        if row["cyclePhase"] == "Luteal":
            score -= 5

        if row["cyclePhase"] == "Menstrual":
            score -= 5

    # -------------------------
    # Bound score
    # -------------------------

    return max(min(score, 100), 0)


df_daily_master["readiness_score"] = df_daily_master.apply(
    compute_readiness,
    axis=1
)


# -------------------------
# Readiness Labels
# -------------------------

def readiness_label(score):

    if score >= 85:
        return "Optimal"

    elif score >= 70:
        return "Normal"

    elif score >= 50:
        return "Fatigued"

    else:
        return "Recovery Needed"


df_daily_master["readiness_state"] = df_daily_master[
    "readiness_score"
].apply(readiness_label)

# =========================
# 7. RUNNING PERFORMANCE
# =========================

print("Computing running performance metrics...")

df_running_master = df_running_master.sort_values("Date")

df_running_master["efficiency_14d"] = (
    df_running_master["efficiency_index"]
    .shift(1)
    .rolling(14)
    .mean()
)

df_running_master["efficiency_delta"] = (
    df_running_master["efficiency_index"] -
    df_running_master["efficiency_14d"]
)


# =========================
# 8. MERGE DATASETS
# =========================

print("Merging running + daily data...")

# Remove cycle columns from running dataset
cycle_cols = [
    "cyclePhase",
    "cycleDay",
    "actualCycleLength",
    "actualPeriodLength",
    "fertileWindowStart",
    "fertileWindowLength",
    "cycleEndDate",
    "isPeriod",
    "fertileStartDay",
    "fertileEndDay",
    "isFertileWindow",
    "ovulationDay",
    "isOvulationDay",
    "cycleProgress",
    "daysUntilNextPeriod"
]

df_running_clean = df_running_master.drop(
    columns=[c for c in cycle_cols if c in df_running_master.columns],
    errors="ignore"
)

# Select only useful columns from daily dataset
daily_cols = [
    "Date",
    "cyclePhase",
    "cycleDay",
    "restingHeartRate",
    "avgRespiration",
    "external_daily_load",
    "total_sleep_hours",
    "load_1d",
    "load_3d",
    "load_7d",
    "load_28d",
    "acwr",
    "sleep_debt_3d",
    "readiness_score",
    "readiness_state"
]

df_daily_clean = df_daily_master[daily_cols]

# Merge
df_run_daily = df_running_clean.merge(
    df_daily_clean,
    on="Date",
    how="left"
)

# =========================
# 9. SESSION INTERPRETATION
# =========================

print("Interpreting sessions...")


def session_interpretation(row):

    if row["efficiency_delta"] > 0.02 and row["load_3d"] > 2000:
        return "Fatigue driven"

    if (
        row["efficiency_delta"] > 0.02
        and pd.notna(row["cyclePhase"])
        and row["cyclePhase"] == "Luteal"
    ):
        return "Cycle physiology"

    return "Normal performance"

df_run_daily["session_flag"] = df_run_daily.apply(
    session_interpretation,
    axis=1
)


# =========================
# 10. EXPORT RESULTS
# =========================

print("Exporting results...")

df_daily_master.to_excel(
    "daily_engine_output.xlsx",
    index=False
)

df_run_daily.to_excel(
    "running_engine_output.xlsx",
    index=False
)

# =========================
# 11. SIMPLE PLOT
# =========================

print("Generating plot...")

plt.figure(figsize=(8,4))
plt.plot(
    df_daily_master["Date"],
    df_daily_master["readiness_score"]
)

plt.title("Readiness Score Over Time")
plt.xlabel("Date")
plt.ylabel("Readiness Score")

plt.tight_layout()

plt.savefig("readiness_plot.png")
plt.close()
