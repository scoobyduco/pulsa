import pandas as pd
import matplotlib.pyplot as plt
import os

print("Running analysis...")

df = pd.read_excel("running_engine_output.xlsx")

df["Date"] = pd.to_datetime(df["Date"])

os.makedirs("figures", exist_ok=True)

# -----------------------------
# 1 Readiness over time
# -----------------------------

plt.figure(figsize=(10,5))

plt.plot(df["Date"], df["readiness_score"])

plt.title("Readiness Score Over Time")
plt.xlabel("Date")
plt.ylabel("Readiness")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("figures/readiness_over_time.png")

plt.close()

# -----------------------------
# 2 Load vs HR
# -----------------------------

plt.figure(figsize=(8,5))

plt.scatter(df["load_3d"], df["avgHr"])

plt.title("Training Load vs HR")
plt.xlabel("3 Day Load")
plt.ylabel("Average HR")

plt.tight_layout()

plt.savefig("figures/load_vs_hr.png")

plt.close()

# -----------------------------
# 3 Efficiency trend
# -----------------------------

plt.figure(figsize=(10,5))

plt.plot(df["Date"], df["efficiency_index"])

plt.title("Running Efficiency Over Time")

plt.tight_layout()

plt.savefig("figures/efficiency_trend.png")

plt.close()

# -----------------------------
# 4 Efficiency deviation
# -----------------------------

plt.figure(figsize=(10,5))

plt.plot(df["Date"], df["efficiency_delta"])

plt.title("Efficiency Deviation")

plt.tight_layout()

plt.savefig("figures/efficiency_delta.png")

plt.close()

# -----------------------------
# 5 Session classification
# -----------------------------

session_counts = df["session_flag"].value_counts()

plt.figure(figsize=(6,4))

session_counts.plot(kind="bar")

plt.title("Run Session Classification")

plt.tight_layout()

plt.savefig("figures/session_types.png")

plt.close()

print("Analysis complete. Figures saved.")
