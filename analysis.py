"""
Batch analysis: generates static figures from running_engine_output.xlsx.
Aligns with app.py visualizations for reports/export.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

print("Running analysis...")

df = pd.read_excel("running_engine_output.xlsx")
df["Date"] = pd.to_datetime(df["Date"])

os.makedirs("figures", exist_ok=True)

# 1 Readiness over time
plt.figure(figsize=(10, 5))
plt.plot(df["Date"], df["readiness_score"])
plt.title("Readiness Score Over Time")
plt.xlabel("Date")
plt.ylabel("Readiness")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/readiness_over_time.png")
plt.close()

# 2 Resting HR over time
plt.figure(figsize=(10, 5))
plt.plot(df["Date"], df["restingHeartRate"])
plt.title("Resting Heart Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Resting HR")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/resting_hr_over_time.png")
plt.close()

# 3 Load vs HR
plt.figure(figsize=(8, 5))
plt.scatter(df["load_3d"], df["avgHr"])
plt.title("Training Load vs HR")
plt.xlabel("3-Day Load")
plt.ylabel("Average HR")
plt.tight_layout()
plt.savefig("figures/load_vs_hr.png")
plt.close()

# 4 Efficiency over time
plt.figure(figsize=(10, 5))
plt.plot(df["Date"], df["efficiency_index"])
plt.title("Running Efficiency Over Time")
plt.xlabel("Date")
plt.ylabel("Efficiency Index")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/efficiency_over_time.png")
plt.close()

# 5 Resting HR by cycle phase
plt.figure(figsize=(8, 5))
df.boxplot(column="restingHeartRate", by="cyclePhase")
plt.title("Resting HR by Cycle Phase")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("figures/rhr_by_cycle_phase.png")
plt.close()

# 6 Sleep by cycle phase (if column exists)
if "total_sleep_hours" in df.columns:
    plt.figure(figsize=(8, 5))
    df.boxplot(column="total_sleep_hours", by="cyclePhase")
    plt.title("Sleep Duration by Cycle Phase")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("figures/sleep_by_cycle_phase.png")
    plt.close()

# 7 Efficiency by cycle phase
plt.figure(figsize=(8, 5))
df.boxplot(column="efficiency_index", by="cyclePhase")
plt.title("Efficiency by Cycle Phase")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("figures/efficiency_by_cycle_phase.png")
plt.close()

# 8 Session classification
plt.figure(figsize=(6, 4))
df["session_flag"].value_counts().plot(kind="bar")
plt.title("Run Session Classification")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("figures/session_types.png")
plt.close()

print("Analysis complete. Figures saved to figures/")
