import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from garmin_preprocessing import build_daily_master
from garmin_running_preprocessing import build_running_master

print("Starting full pipeline...")

df_daily_master = build_daily_master()
df_running_master = build_running_master()

df_daily_master["Date"] = pd.to_datetime(df_daily_master["Date"])
df_running_master["Date"] = pd.to_datetime(df_running_master["Date"])

print("Computing training load...")

df_daily_master["load_1d"] = df_daily_master["external_daily_load"]

df_daily_master["load_3d"] = df_daily_master["load_1d"].rolling(3).sum()

df_daily_master["load_7d"] = df_daily_master["load_1d"].rolling(7).sum()

df_daily_master["load_28d"] = df_daily_master["load_1d"].rolling(28).mean()

df_daily_master["acwr"] = df_daily_master["load_7d"] / df_daily_master["load_28d"]

print("Computing readiness...")

score = 100

df_daily_master["readiness_score"] = (
    score
    - (df_daily_master["load_3d"] > 2000)*20
    - (df_daily_master["rhr_z"] > 1.5)*20
    - (df_daily_master["resp_z"] > 1.5)*10
)

df_daily_master["readiness_score"] = df_daily_master["readiness_score"].clip(0,100)

df_daily_master["readiness_state"] = pd.cut(
    df_daily_master["readiness_score"],
    bins=[0,50,70,85,100],
    labels=["Recovery Needed","Fatigued","Normal","Optimal"]
)

print("Computing running efficiency trends...")

df_running_master = df_running_master.sort_values("Date")

df_running_master["efficiency_14d"] = (
    df_running_master["efficiency_index"]
    .shift(1)
    .rolling(14)
    .mean()
)

df_running_master["efficiency_delta"] = (
    df_running_master["efficiency_index"]
    - df_running_master["efficiency_14d"]
)

print("Merging datasets...")

df_run_daily = df_running_master.merge(
    df_daily_master,
    on="Date",
    how="left"
)

print("Interpreting sessions...")

df_run_daily["session_flag"] = np.select(
    [
        (df_run_daily["efficiency_delta"] > 0.02) & (df_run_daily["load_3d"] > 2000),
        (df_run_daily["efficiency_delta"] > 0.02)
    ],
    [
        "Fatigue driven",
        "Underperforming"
    ],
    default="Normal"
)

df_daily_master.to_excel("daily_engine_output.xlsx", index=False)

df_run_daily.to_excel("running_engine_output.xlsx", index=False)

plt.plot(df_daily_master["Date"], df_daily_master["readiness_score"])
plt.title("Readiness Over Time")
plt.savefig("readiness_plot.png")

print("Pipeline completed successfully.")
